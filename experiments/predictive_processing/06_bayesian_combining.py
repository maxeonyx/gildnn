import math
from dataclasses import dataclass

import torch


DTYPE = torch.float64
SEED = 42
BATCH = 512
DIM = 16
CHAIN_STEPS = 20
LOG_SIGMA_MIN = math.log(0.05)
LOG_SIGMA_MAX = math.log(2.0)
CASE_A_LAMBDA = 0.5
LIMIT_LAMBDA = 1.0
HUGE_SIGMA = 1e12
VALID_TOL = 1e-12
LIMIT_TOL = 1e-9
ANALYTIC_TOL = 1e-12
FIXED_POINT_TOL = 3e-2
TAIL_DELTA_TOL = 5e-3
MIN_BOUNDED_SIGMA = 1e-2
MAX_REASONABLE_SIGMA = 5.0
SURPRISE_MEAN_OFFSETS = (0.0, 0.25, 0.5, 1.0, 2.0)
SURPRISE_SIGMA_VALUES = (1.0, 1.1, 1.25, 1.5, 2.0)


@dataclass(frozen=True)
class GaussianBatch:
    mu: torch.Tensor
    sigma: torch.Tensor


@dataclass(frozen=True)
class CheckRecord:
    label: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class CaseAResult:
    checks: list[CheckRecord]
    mean_stream_weight: float
    mean_pred_weight: float
    stream_dominant_dims: int
    pred_dominant_dims: int


@dataclass(frozen=True)
class CaseBResult:
    checks: list[CheckRecord]
    step_sigmas: list[float]
    analytic_sigmas: list[float]


@dataclass(frozen=True)
class BoundedConfig:
    lambda_weight: float
    process_noise: float


@dataclass(frozen=True)
class BoundedRunSummary:
    lambda_weight: float
    process_noise: float
    analytic_fixed_point: float
    final_sigma: float
    final_error: float
    min_sigma: float
    max_sigma: float
    tail_delta: float


@dataclass(frozen=True)
class CaseCResult:
    checks: list[CheckRecord]
    rows: list[BoundedRunSummary]


@dataclass(frozen=True)
class SurpriseRow:
    mismatch: float
    surprise: float


@dataclass(frozen=True)
class CaseDResult:
    checks: list[CheckRecord]
    mean_rows: list[SurpriseRow]
    sigma_rows: list[SurpriseRow]


CASE_C_CONFIGS = (
    BoundedConfig(lambda_weight=1.0, process_noise=0.3),
    BoundedConfig(lambda_weight=0.5, process_noise=0.2),
    BoundedConfig(lambda_weight=0.25, process_noise=0.1),
)


def sample_diag_gaussian(batch: int = BATCH, dim: int = DIM) -> GaussianBatch:
    mu = torch.randn(batch, dim, dtype=DTYPE)
    log_sigma = torch.empty(batch, dim, dtype=DTYPE).uniform_(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
    sigma = log_sigma.exp()
    return GaussianBatch(mu=mu, sigma=sigma)


def validate_gaussian(state: GaussianBatch, *, name: str) -> None:
    if state.mu.shape != state.sigma.shape:
        raise ValueError(f"{name} mu/sigma shape mismatch: {state.mu.shape} vs {state.sigma.shape}")
    if not torch.isfinite(state.mu).all().item():
        raise ValueError(f"{name} mean contains non-finite values")
    if not torch.isfinite(state.sigma).all().item():
        raise ValueError(f"{name} sigma contains non-finite values")
    if not (state.sigma > 0).all().item():
        raise ValueError(f"{name} sigma must be strictly positive")


def w2_sq(state_a: GaussianBatch, state_b: GaussianBatch) -> torch.Tensor:
    validate_gaussian(state_a, name="state_a")
    validate_gaussian(state_b, name="state_b")
    if state_a.mu.shape != state_b.mu.shape:
        raise ValueError(f"W2 shape mismatch: {state_a.mu.shape} vs {state_b.mu.shape}")
    return ((state_a.mu - state_b.mu).square() + (state_a.sigma - state_b.sigma).square()).sum(dim=-1)


def tempered_poe_update(stream: GaussianBatch, prediction: GaussianBatch, *, lambda_weight: float) -> GaussianBatch:
    validate_gaussian(stream, name="stream")
    validate_gaussian(prediction, name="prediction")
    if stream.mu.shape != prediction.mu.shape:
        raise ValueError(f"stream/prediction shape mismatch: {stream.mu.shape} vs {prediction.mu.shape}")
    if not math.isfinite(lambda_weight) or lambda_weight < 0.0:
        raise ValueError(f"lambda_weight must be finite and non-negative, got {lambda_weight}")

    tau_stream = stream.sigma.reciprocal().square()
    tau_pred = prediction.sigma.reciprocal().square()
    tau_new = tau_stream + lambda_weight * tau_pred
    if not (tau_new > 0).all().item():
        raise ValueError("Combined precision became non-positive")

    mu_new = (tau_stream * stream.mu + lambda_weight * tau_pred * prediction.mu) / tau_new
    sigma_new = tau_new.rsqrt()
    result = GaussianBatch(mu=mu_new, sigma=sigma_new)
    validate_gaussian(result, name="poe_update")
    return result


def propagate_process_noise(state: GaussianBatch, *, process_noise: float) -> GaussianBatch:
    validate_gaussian(state, name="state_before_noise")
    if not math.isfinite(process_noise) or process_noise < 0.0:
        raise ValueError(f"process_noise must be finite and non-negative, got {process_noise}")
    sigma = torch.sqrt(state.sigma.square() + process_noise**2)
    result = GaussianBatch(mu=state.mu.clone(), sigma=sigma)
    validate_gaussian(result, name="state_after_noise")
    return result


def combine_step(stream: GaussianBatch, prediction: GaussianBatch, *, lambda_weight: float, process_noise: float) -> GaussianBatch:
    updated = tempered_poe_update(stream, prediction, lambda_weight=lambda_weight)
    return propagate_process_noise(updated, process_noise=process_noise)


def make_check(label: str, passed: bool, detail: str) -> CheckRecord:
    return CheckRecord(label=label, passed=passed, detail=detail)


def monotone_nonincreasing(values: list[float], *, tol: float = 0.0) -> bool:
    return all(current <= previous + tol for previous, current in zip(values, values[1:], strict=False))


def fixed_point_sigma(pred_sigma: float, *, lambda_weight: float, process_noise: float) -> float:
    if pred_sigma <= 0.0:
        raise ValueError(f"pred_sigma must be positive, got {pred_sigma}")
    if lambda_weight <= 0.0:
        raise ValueError(f"lambda_weight must be positive for the fixed-point formula, got {lambda_weight}")
    if process_noise < 0.0:
        raise ValueError(f"process_noise must be non-negative, got {process_noise}")
    if process_noise == 0.0:
        return 0.0

    q_sq = process_noise**2
    discriminant = q_sq**2 + 4.0 * q_sq * pred_sigma**2 / lambda_weight
    sigma_sq = 0.5 * (q_sq + math.sqrt(discriminant))
    return math.sqrt(sigma_sq)


def run_constant_chain(
    *,
    initial_mu: float,
    initial_sigma: float,
    pred_sigma: float,
    lambda_weight: float,
    process_noise: float,
    steps: int,
    prediction_matches_stream_mean: bool,
) -> tuple[list[float], list[float]]:
    stream = GaussianBatch(
        mu=torch.full((1, 1), initial_mu, dtype=DTYPE),
        sigma=torch.full((1, 1), initial_sigma, dtype=DTYPE),
    )
    sigmas: list[float] = []
    mu_abs: list[float] = []

    for _ in range(steps):
        pred_mu = stream.mu.clone() if prediction_matches_stream_mean else torch.zeros_like(stream.mu)
        prediction = GaussianBatch(mu=pred_mu, sigma=torch.full_like(stream.sigma, pred_sigma))
        stream = combine_step(stream, prediction, lambda_weight=lambda_weight, process_noise=process_noise)
        sigmas.append(stream.sigma.item())
        mu_abs.append(stream.mu.abs().max().item())

    return sigmas, mu_abs


def run_case_a() -> CaseAResult:
    stream = sample_diag_gaussian(batch=BATCH, dim=DIM)
    prediction = sample_diag_gaussian(batch=BATCH, dim=DIM)
    combined = tempered_poe_update(stream, prediction, lambda_weight=CASE_A_LAMBDA)

    tau_stream = stream.sigma.reciprocal().square()
    tau_pred = prediction.sigma.reciprocal().square()
    tau_new = tau_stream + CASE_A_LAMBDA * tau_pred
    stream_weight = tau_stream / tau_new
    pred_weight = CASE_A_LAMBDA * tau_pred / tau_new
    mu_expected = stream_weight * stream.mu + pred_weight * prediction.mu

    lower_bound = torch.minimum(stream.mu, prediction.mu)
    upper_bound = torch.maximum(stream.mu, prediction.mu)
    between_error = torch.maximum(lower_bound - combined.mu, combined.mu - upper_bound).clamp_min(0.0).max().item()
    weight_error = (combined.mu - mu_expected).abs().max().item()

    gap = (stream.mu - prediction.mu).abs()
    dist_to_stream = (combined.mu - stream.mu).abs()
    dist_to_pred = (combined.mu - prediction.mu).abs()
    stream_dominant_mask = (tau_stream > CASE_A_LAMBDA * tau_pred + VALID_TOL) & (gap > VALID_TOL)
    pred_dominant_mask = (CASE_A_LAMBDA * tau_pred > tau_stream + VALID_TOL) & (gap > VALID_TOL)

    stream_routing_ok = bool(stream_dominant_mask.any().item()) and bool(
        (dist_to_stream[stream_dominant_mask] < dist_to_pred[stream_dominant_mask]).all().item()
    )
    pred_routing_ok = bool(pred_dominant_mask.any().item()) and bool(
        (dist_to_pred[pred_dominant_mask] < dist_to_stream[pred_dominant_mask]).all().item()
    )

    uninformative_prediction = GaussianBatch(mu=prediction.mu.clone(), sigma=torch.full_like(prediction.sigma, HUGE_SIGMA))
    limit_to_stream = tempered_poe_update(stream, uninformative_prediction, lambda_weight=LIMIT_LAMBDA)
    stream_limit_error = max(
        (limit_to_stream.mu - stream.mu).abs().max().item(),
        (limit_to_stream.sigma - stream.sigma).abs().max().item(),
    )

    uninformative_stream = GaussianBatch(mu=stream.mu.clone(), sigma=torch.full_like(stream.sigma, HUGE_SIGMA))
    limit_to_prediction = tempered_poe_update(uninformative_stream, prediction, lambda_weight=LIMIT_LAMBDA)
    prediction_limit_error = max(
        (limit_to_prediction.mu - prediction.mu).abs().max().item(),
        (limit_to_prediction.sigma - prediction.sigma).abs().max().item(),
    )

    checks = [
        make_check(
            "A1 valid output",
            bool(torch.isfinite(combined.mu).all().item() and torch.isfinite(combined.sigma).all().item() and (combined.sigma > 0).all().item()),
            f"sigma_min={combined.sigma.min().item():.6e}, sigma_max={combined.sigma.max().item():.6e}",
        ),
        make_check("A2 mean stays between inputs", between_error <= ANALYTIC_TOL, f"max_overshoot={between_error:.6e}"),
        make_check("A3 precision-weighted mean formula", weight_error <= ANALYTIC_TOL, f"max_abs_error={weight_error:.6e}"),
        make_check(
            "A4 lower effective sigma gets more weight",
            stream_routing_ok and pred_routing_ok,
            f"stream_dominant_dims={int(stream_dominant_mask.sum().item())}, pred_dominant_dims={int(pred_dominant_mask.sum().item())}",
        ),
        make_check("A5 sigma_pred -> inf recovers stream", stream_limit_error <= LIMIT_TOL, f"max_abs_error={stream_limit_error:.6e}"),
        make_check(
            "A6 sigma_stream -> inf recovers prediction",
            prediction_limit_error <= LIMIT_TOL,
            f"max_abs_error={prediction_limit_error:.6e}",
        ),
    ]
    return CaseAResult(
        checks=checks,
        mean_stream_weight=stream_weight.mean().item(),
        mean_pred_weight=pred_weight.mean().item(),
        stream_dominant_dims=int(stream_dominant_mask.sum().item()),
        pred_dominant_dims=int(pred_dominant_mask.sum().item()),
    )


def run_case_b() -> CaseBResult:
    step_sigmas, mu_abs = run_constant_chain(
        initial_mu=0.0,
        initial_sigma=1.0,
        pred_sigma=1.0,
        lambda_weight=1.0,
        process_noise=0.0,
        steps=CHAIN_STEPS,
        prediction_matches_stream_mean=True,
    )
    analytic_sigmas = [1.0 / math.sqrt(step + 1) for step in range(1, CHAIN_STEPS + 1)]
    sigma_error = max(abs(observed - expected) for observed, expected in zip(step_sigmas, analytic_sigmas, strict=True))
    final_sigma = step_sigmas[-1]
    checks = [
        make_check("B1 sigma shrinks monotonically", monotone_nonincreasing(step_sigmas, tol=ANALYTIC_TOL), f"final_sigma={final_sigma:.6f}"),
        make_check("B2 mean stays fixed", max(mu_abs) <= ANALYTIC_TOL, f"max_abs_mu={max(mu_abs):.6e}"),
        make_check("B3 matches analytic collapse curve", sigma_error <= ANALYTIC_TOL, f"max_abs_error={sigma_error:.6e}"),
        make_check("B4 collapse is visible", final_sigma < 0.25, f"final_sigma={final_sigma:.6f}"),
    ]
    return CaseBResult(checks=checks, step_sigmas=step_sigmas, analytic_sigmas=analytic_sigmas)


def run_case_c() -> CaseCResult:
    rows: list[BoundedRunSummary] = []
    checks: list[CheckRecord] = []

    for config in CASE_C_CONFIGS:
        sigmas, _mu_abs = run_constant_chain(
            initial_mu=0.0,
            initial_sigma=1.0,
            pred_sigma=0.8,
            lambda_weight=config.lambda_weight,
            process_noise=config.process_noise,
            steps=CHAIN_STEPS,
            prediction_matches_stream_mean=True,
        )
        analytic = fixed_point_sigma(0.8, lambda_weight=config.lambda_weight, process_noise=config.process_noise)
        final_sigma = sigmas[-1]
        min_sigma = min(sigmas)
        max_sigma = max(sigmas)
        tail_delta = abs(sigmas[-1] - sigmas[-2])
        final_error = abs(final_sigma - analytic)
        label_prefix = f"lambda={config.lambda_weight:.2f}, q={config.process_noise:.2f}"

        rows.append(
            BoundedRunSummary(
                lambda_weight=config.lambda_weight,
                process_noise=config.process_noise,
                analytic_fixed_point=analytic,
                final_sigma=final_sigma,
                final_error=final_error,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                tail_delta=tail_delta,
            )
        )

        checks.extend(
            [
                make_check(f"C {label_prefix} stays finite", math.isfinite(final_sigma) and max_sigma < MAX_REASONABLE_SIGMA, f"max_sigma={max_sigma:.6f}"),
                make_check(f"C {label_prefix} avoids collapse", final_sigma > MIN_BOUNDED_SIGMA, f"final_sigma={final_sigma:.6f}"),
                make_check(f"C {label_prefix} matches analytic fixed point", final_error <= FIXED_POINT_TOL, f"abs_error={final_error:.6e}"),
                make_check(f"C {label_prefix} settles by step 20", tail_delta <= TAIL_DELTA_TOL, f"tail_delta={tail_delta:.6e}"),
            ]
        )

    return CaseCResult(checks=checks, rows=rows)


def run_case_d() -> CaseDResult:
    stream = GaussianBatch(mu=torch.zeros(1, DIM, dtype=DTYPE), sigma=torch.ones(1, DIM, dtype=DTYPE))

    mean_rows = [
        SurpriseRow(
            mismatch=offset,
            surprise=w2_sq(stream, GaussianBatch(mu=torch.full((1, DIM), offset, dtype=DTYPE), sigma=torch.ones(1, DIM, dtype=DTYPE))).item(),
        )
        for offset in SURPRISE_MEAN_OFFSETS
    ]
    sigma_rows = [
        SurpriseRow(
            mismatch=abs(sigma_value - 1.0),
            surprise=w2_sq(
                stream,
                GaussianBatch(mu=torch.zeros(1, DIM, dtype=DTYPE), sigma=torch.full((1, DIM), sigma_value, dtype=DTYPE)),
            ).item(),
        )
        for sigma_value in SURPRISE_SIGMA_VALUES
    ]

    mean_surprises = [row.surprise for row in mean_rows]
    sigma_surprises = [row.surprise for row in sigma_rows]

    checks = [
        make_check("D1 perfect prediction has zero surprise", mean_rows[0].surprise <= ANALYTIC_TOL, f"surprise={mean_rows[0].surprise:.6e}"),
        make_check(
            "D2 mean mismatch increases surprise monotonically",
            monotone_nonincreasing([-value for value in mean_surprises], tol=ANALYTIC_TOL),
            f"surprises={[round(value, 6) for value in mean_surprises]}",
        ),
        make_check(
            "D3 sigma mismatch increases surprise monotonically",
            monotone_nonincreasing([-value for value in sigma_surprises], tol=ANALYTIC_TOL),
            f"surprises={[round(value, 6) for value in sigma_surprises]}",
        ),
        make_check(
            "D4 any mismatch gives positive surprise",
            all(row.surprise > 0.0 for row in mean_rows[1:] + sigma_rows[1:]),
            f"smallest_nonzero={min(row.surprise for row in mean_rows[1:] + sigma_rows[1:]):.6e}",
        ),
    ]
    return CaseDResult(checks=checks, mean_rows=mean_rows, sigma_rows=sigma_rows)


def print_checks(checks: list[CheckRecord]) -> None:
    print("Check                                              Status  Detail")
    print("-------------------------------------------------  ------  -----------------------------------------------")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"{check.label:49}  {status:6}  {check.detail}")


def print_case_a(result: CaseAResult) -> None:
    print("=== Case A: Algebra sanity ===")
    print(f"lambda={CASE_A_LAMBDA:.2f} for weighting checks; lambda={LIMIT_LAMBDA:.2f} for uninformative-input limits")
    print(
        f"mean_stream_weight={result.mean_stream_weight:.6f}  mean_pred_weight={result.mean_pred_weight:.6f}  "
        f"stream_dominant_dims={result.stream_dominant_dims}  pred_dominant_dims={result.pred_dominant_dims}"
    )
    print_checks(result.checks)
    print()


def print_case_b(result: CaseBResult) -> None:
    print("=== Case B: Collapse control (pure PoE, no noise) ===")
    print("step  empirical_sigma  analytic_sigma  abs_error")
    print("----  ---------------  --------------  ---------")
    for step in (1, 5, 10, 20):
        empirical = result.step_sigmas[step - 1]
        analytic = result.analytic_sigmas[step - 1]
        print(f"{step:4d}  {empirical:15.6f}  {analytic:14.6f}  {abs(empirical - analytic):9.2e}")
    print_checks(result.checks)
    print()


def print_case_c(result: CaseCResult) -> None:
    print("=== Case C: Bounded chain (PoE + process noise) ===")
    print("lambda   q      sigma*     sigma_20    abs_err    min_sigma  max_sigma  tail_delta")
    print("------  -----  ---------  ---------  ---------  ---------  ---------  ----------")
    for row in result.rows:
        print(
            f"{row.lambda_weight:6.2f}  {row.process_noise:5.2f}  {row.analytic_fixed_point:9.6f}  "
            f"{row.final_sigma:9.6f}  {row.final_error:9.2e}  {row.min_sigma:9.6f}  {row.max_sigma:9.6f}  {row.tail_delta:10.2e}"
        )
    print_checks(result.checks)
    print()


def print_case_d(result: CaseDResult) -> None:
    print("=== Case D: Surprise as side-channel ===")
    print("Mean mismatch sweep")
    print("mismatch  W2_sq")
    print("--------  --------")
    for row in result.mean_rows:
        print(f"{row.mismatch:8.2f}  {row.surprise:8.4f}")
    print()
    print("Sigma mismatch sweep")
    print("mismatch  W2_sq")
    print("--------  --------")
    for row in result.sigma_rows:
        print(f"{row.mismatch:8.2f}  {row.surprise:8.4f}")
    print_checks(result.checks)
    print()


def main() -> int:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    case_a = run_case_a()
    case_b = run_case_b()
    case_c = run_case_c()
    case_d = run_case_d()

    print(f"seed={SEED}  dtype={DTYPE}  batch={BATCH}  dim={DIM}  chain_steps={CHAIN_STEPS}")
    print()
    print_case_a(case_a)
    print_case_b(case_b)
    print_case_c(case_c)
    print_case_d(case_d)

    all_checks = case_a.checks + case_b.checks + case_c.checks + case_d.checks
    failures = [check.label for check in all_checks if not check.passed]
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for label in failures:
            print(f"  - {label}")
        return 1

    print(f"All checks passed ({len(all_checks)} total).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
