import math
from dataclasses import dataclass

import torch


DTYPE = torch.float64
SEED = 42
N = 20
BATCH = 512
DIM = 32
LOG_SIGMA_MIN = math.log(0.05)
LOG_SIGMA_MAX = math.log(2.0)
TAU_EPS = 1e-9
ETA_ZERO_TOL = 1e-12
VALID_TOL = 1e-12
SILENCE_TOL = 1e-6

# The request said sigma / 0.8 and sigma / 0.5, but also said those explain
# 80% and 50% of precision. Precision is 1 / sigma^2, so the precision-faithful
# interpretation is sigma / sqrt(q).
PARTIAL_QUALITIES = {
    "partial_80": 0.8,
    "partial_50": 0.5,
}


@dataclass
class StepMetrics:
    valid: bool
    stream_magnitude: float
    mean_energy: float
    mean_log_sigma: float
    mismatch_magnitude: float


@dataclass
class RegimeResult:
    steps: list[StepMetrics]
    valid_all: bool
    recon_error: float | None
    stream_silences: bool


def sample_diag_gaussian(batch: int = BATCH, dim: int = DIM) -> tuple[torch.Tensor, torch.Tensor]:
    mu = torch.randn(batch, dim, dtype=DTYPE)
    log_sigma = torch.empty(batch, dim, dtype=DTYPE).uniform_(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
    sigma = log_sigma.exp()
    return mu, sigma


def w2_sq(mu_1: torch.Tensor, sigma_1: torch.Tensor, mu_2: torch.Tensor, sigma_2: torch.Tensor) -> torch.Tensor:
    return ((mu_1 - mu_2).square() + (sigma_1 - sigma_2).square()).sum(dim=-1)


def gaussian_to_natural(mu: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tau = sigma.reciprocal().square()
    eta = mu * tau
    return tau, eta


def natural_to_gaussian_for_stream(tau: torch.Tensor, eta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tau_safe = tau.clamp_min(TAU_EPS)
    sigma = tau_safe.rsqrt()
    mu = eta / tau_safe
    zero_info_mask = (tau <= TAU_EPS) & (eta.abs() <= ETA_ZERO_TOL)
    sigma = torch.where(zero_info_mask, torch.ones_like(sigma), sigma)
    mu = torch.where(zero_info_mask, torch.zeros_like(mu), mu)
    return mu, sigma


def natural_to_gaussian_for_prediction(tau: torch.Tensor, eta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tau_safe = tau.clamp_min(TAU_EPS)
    sigma = tau_safe.rsqrt()
    mu = eta / tau_safe
    zero_info_mask = (tau <= TAU_EPS) & (eta.abs() <= ETA_ZERO_TOL)
    sigma = torch.where(zero_info_mask, torch.full_like(sigma, 1e9), sigma)
    mu = torch.where(zero_info_mask, torch.zeros_like(mu), mu)
    return mu, sigma


def raw_chart_norm(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(mu.square().sum(dim=-1) + sigma.square().sum(dim=-1))


def gaussian_stream_magnitude(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    null_mu = torch.zeros(1, DIM, dtype=DTYPE)
    null_sigma = torch.ones(1, DIM, dtype=DTYPE)
    return w2_sq(mu, sigma, null_mu, null_sigma).sqrt()


def natural_stream_magnitude(tau: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(tau.square().sum(dim=-1) + eta.square().sum(dim=-1))


def mean_log_sigma(sigma: torch.Tensor) -> float:
    if not bool((sigma > 0).all()):
        return float("nan")
    return sigma.log().mean().item()


def make_gaussian_prediction(regime: str, stream_mu: torch.Tensor, stream_sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if regime == "perfect":
        return stream_mu.clone(), stream_sigma.clone()

    if regime in PARTIAL_QUALITIES:
        quality = PARTIAL_QUALITIES[regime]
        return stream_mu.clone(), stream_sigma / math.sqrt(quality)

    if regime == "random":
        return sample_diag_gaussian(batch=stream_mu.shape[0], dim=stream_mu.shape[1])

    if regime == "overconf_wrong":
        return stream_mu + 2.0, torch.full_like(stream_sigma, 0.1)

    raise ValueError(f"Unknown regime: {regime}")


def make_natural_prediction(
    regime: str,
    stream_tau: torch.Tensor,
    stream_eta: torch.Tensor,
    stream_mu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if regime == "perfect":
        pred_tau = stream_tau.clone()
        pred_eta = stream_eta.clone()
    elif regime in PARTIAL_QUALITIES:
        quality = PARTIAL_QUALITIES[regime]
        pred_tau = quality * stream_tau
        pred_eta = quality * stream_eta
    elif regime == "random":
        pred_mu, pred_sigma = sample_diag_gaussian(batch=stream_mu.shape[0], dim=stream_mu.shape[1])
        pred_tau, pred_eta = gaussian_to_natural(pred_mu, pred_sigma)
        return pred_mu, pred_sigma, pred_tau, pred_eta
    elif regime == "overconf_wrong":
        pred_mu = stream_mu + 2.0
        pred_sigma = torch.full_like(stream_mu, 0.1)
        pred_tau, pred_eta = gaussian_to_natural(pred_mu, pred_sigma)
        return pred_mu, pred_sigma, pred_tau, pred_eta
    else:
        raise ValueError(f"Unknown regime: {regime}")

    pred_mu, pred_sigma = natural_to_gaussian_for_prediction(pred_tau, pred_eta)
    return pred_mu, pred_sigma, pred_tau, pred_eta


def metrics_from_gaussian_step(
    next_mu: torch.Tensor,
    next_sigma: torch.Tensor,
    mismatch: torch.Tensor,
    *,
    stream_magnitude: torch.Tensor,
    valid: bool,
) -> StepMetrics:
    return StepMetrics(
        valid=valid,
        stream_magnitude=stream_magnitude.mean().item(),
        mean_energy=next_mu.square().sum(dim=-1).mean().item(),
        mean_log_sigma=mean_log_sigma(next_sigma),
        mismatch_magnitude=mismatch.mean().item(),
    )


def run_raw_w2_residual(regime: str, initial_mu: torch.Tensor, initial_sigma: torch.Tensor) -> RegimeResult:
    stream_mu = initial_mu.clone()
    stream_sigma = initial_sigma.clone()
    steps: list[StepMetrics] = []
    predicted_mu_sum = torch.zeros_like(initial_mu)
    predicted_sigma_sum = torch.zeros_like(initial_sigma)

    for _ in range(N):
        pred_mu, pred_sigma = make_gaussian_prediction(regime, stream_mu, stream_sigma)
        next_mu = stream_mu - pred_mu
        next_sigma = stream_sigma - pred_sigma
        mismatch = raw_chart_norm(next_mu, next_sigma)
        valid = bool((next_sigma > VALID_TOL).all())
        steps.append(
            metrics_from_gaussian_step(
                next_mu,
                next_sigma,
                mismatch,
                stream_magnitude=mismatch,
                valid=valid,
            )
        )
        predicted_mu_sum = predicted_mu_sum + pred_mu
        predicted_sigma_sum = predicted_sigma_sum + pred_sigma
        stream_mu = next_mu
        stream_sigma = next_sigma

    recon_mu = predicted_mu_sum + stream_mu
    recon_sigma = predicted_sigma_sum + stream_sigma
    recon_error = max(
        (recon_mu - initial_mu).abs().max().item(),
        (recon_sigma - initial_sigma).abs().max().item(),
    )
    return RegimeResult(
        steps=steps,
        valid_all=all(step.valid for step in steps),
        recon_error=recon_error,
        stream_silences=all(step.stream_magnitude <= SILENCE_TOL for step in steps),
    )


def run_passthrough_actual(regime: str, initial_mu: torch.Tensor, initial_sigma: torch.Tensor) -> RegimeResult:
    stream_mu = initial_mu.clone()
    stream_sigma = initial_sigma.clone()
    steps: list[StepMetrics] = []

    for _ in range(N):
        pred_mu, pred_sigma = make_gaussian_prediction(regime, stream_mu, stream_sigma)
        mismatch = raw_chart_norm(stream_mu - pred_mu, stream_sigma - pred_sigma)
        steps.append(
            metrics_from_gaussian_step(
                stream_mu,
                stream_sigma,
                mismatch,
                stream_magnitude=gaussian_stream_magnitude(stream_mu, stream_sigma),
                valid=True,
            )
        )

    return RegimeResult(
        steps=steps,
        valid_all=True,
        recon_error=None,
        stream_silences=all(step.stream_magnitude <= SILENCE_TOL for step in steps),
    )


def run_precision_weighted_residual(regime: str, initial_mu: torch.Tensor, initial_sigma: torch.Tensor) -> RegimeResult:
    stream_tau, stream_eta = gaussian_to_natural(initial_mu, initial_sigma)
    stream_mu = initial_mu.clone()
    stream_sigma = initial_sigma.clone()
    steps: list[StepMetrics] = []
    explained_tau_sum = torch.zeros_like(initial_sigma)
    explained_eta_sum = torch.zeros_like(initial_mu)

    for _ in range(N):
        pred_mu, pred_sigma, pred_tau, _pred_eta = make_natural_prediction(regime, stream_tau, stream_eta, stream_mu)
        mismatch = w2_sq(stream_mu, stream_sigma, pred_mu, pred_sigma).sqrt()
        tau_explained = torch.minimum(pred_tau, stream_tau)
        eta_explained = tau_explained * pred_mu
        tau_residual = stream_tau - tau_explained
        eta_residual = stream_eta - eta_explained
        next_mu, next_sigma = natural_to_gaussian_for_stream(tau_residual, eta_residual)
        valid = bool((tau_residual >= -VALID_TOL).all())
        steps.append(
            metrics_from_gaussian_step(
                next_mu,
                next_sigma,
                mismatch,
                stream_magnitude=natural_stream_magnitude(tau_residual, eta_residual),
                valid=valid,
            )
        )
        explained_tau_sum = explained_tau_sum + tau_explained
        explained_eta_sum = explained_eta_sum + eta_explained
        stream_tau = tau_residual
        stream_eta = eta_residual
        stream_mu = next_mu
        stream_sigma = next_sigma

    initial_tau, initial_eta = gaussian_to_natural(initial_mu, initial_sigma)
    recon_tau = explained_tau_sum + stream_tau
    recon_eta = explained_eta_sum + stream_eta
    recon_error = max(
        (recon_tau - initial_tau).abs().max().item(),
        (recon_eta - initial_eta).abs().max().item(),
    )
    return RegimeResult(
        steps=steps,
        valid_all=all(step.valid for step in steps),
        recon_error=recon_error,
        stream_silences=all(step.stream_magnitude <= SILENCE_TOL for step in steps),
    )


def fmt_bool(value: bool) -> str:
    return "YES" if value else "no"


def fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return f"{value:.3f}"


def print_operator_summary(operator_name: str, results: dict[str, RegimeResult]) -> None:
    print(f"=== Operator: {operator_name} ===")
    print(
        "Regime          Valid?  Step1_mag  Step5_mag  Step10_mag  Step20_mag  "
        "MeanE20  LogSigma20  Recon_err"
    )
    for regime_name in ["perfect", "partial_80", "partial_50", "random", "overconf_wrong"]:
        result = results[regime_name]
        step_1 = result.steps[0]
        step_5 = result.steps[4]
        step_10 = result.steps[9]
        step_20 = result.steps[19]
        print(
            f"{regime_name:15} {fmt_bool(result.valid_all):6}  {fmt_float(step_1.stream_magnitude):9}  "
            f"{fmt_float(step_5.stream_magnitude):9}  {fmt_float(step_10.stream_magnitude):10}  {fmt_float(step_20.stream_magnitude):10}  "
            f"{fmt_float(step_20.mean_energy):7}  {fmt_float(step_20.mean_log_sigma):10}  {fmt_float(result.recon_error)}"
        )

    perfect = results["perfect"]
    random_case = results["random"]
    wrong_case = results["overconf_wrong"]
    print(
        "Properties: "
        f"perfect_silences_stream={fmt_bool(perfect.stream_silences)}, "
        f"wrong_gt_random_step1={fmt_bool(wrong_case.steps[0].mismatch_magnitude > random_case.steps[0].mismatch_magnitude)} "
        f"({fmt_float(wrong_case.steps[0].mismatch_magnitude)} vs {fmt_float(random_case.steps[0].mismatch_magnitude)})"
    )
    print()


def main() -> int:
    torch.manual_seed(SEED)
    initial_mu, initial_sigma = sample_diag_gaussian()
    regimes = ["perfect", "partial_80", "partial_50", "random", "overconf_wrong"]

    operators = {
        "raw_w2_residual": run_raw_w2_residual,
        "passthrough_actual": run_passthrough_actual,
        "precision_weighted_residual": run_precision_weighted_residual,
    }

    print("Chain setup:")
    print(f"  seed={SEED}, N={N}, batch={BATCH}, dim={DIM}, dtype={DTYPE}")
    print("  partial regimes use sigma / sqrt(q) so q really means explained precision")
    print("  Step*_mag is operator-native stream magnitude: chart norm (A), W2-to-null (B), natural-parameter norm (C)")
    print("  wrong_gt_random_step1 compares prediction mismatch magnitude, not stream magnitude")
    print()

    all_results: dict[str, dict[str, RegimeResult]] = {}
    for operator_name, operator_runner in operators.items():
        operator_results = {regime: operator_runner(regime, initial_mu, initial_sigma) for regime in regimes}
        all_results[operator_name] = operator_results
        print_operator_summary(operator_name, operator_results)

    valid_everywhere = [
        operator_name for operator_name, results in all_results.items() if all(result.valid_all for result in results.values())
    ]
    print(f"Operators valid across all regimes: {', '.join(valid_everywhere) if valid_everywhere else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
