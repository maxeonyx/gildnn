import math
from dataclasses import dataclass

import torch


DTYPE = torch.float64
SEED = 42
BATCH = 512
DIM = 32
LOG_SIGMA_MIN = math.log(0.05)
LOG_SIGMA_MAX = math.log(2.0)


@dataclass
class DecompositionOutput:
    surprise_magnitude: torch.Tensor
    reconstruction_error: float


def sample_diag_gaussian(batch: int = BATCH, dim: int = DIM) -> tuple[torch.Tensor, torch.Tensor]:
    mu = torch.randn(batch, dim, dtype=DTYPE)
    log_sigma = torch.empty(batch, dim, dtype=DTYPE).uniform_(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
    sigma = log_sigma.exp()
    return mu, sigma


def w2_sq(mu_1: torch.Tensor, sigma_1: torch.Tensor, mu_2: torch.Tensor, sigma_2: torch.Tensor) -> torch.Tensor:
    return ((mu_1 - mu_2).square() + (sigma_1 - sigma_2).square()).sum(dim=-1)


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt(x_centered.square().sum() * y_centered.square().sum())
    return (x_centered * y_centered).sum().div(denom).item()


def candidate_b(mu_pred: torch.Tensor, sigma_pred: torch.Tensor, mu_actual: torch.Tensor, sigma_actual: torch.Tensor) -> DecompositionOutput:
    up_mu = mu_actual - mu_pred
    up_sigma = sigma_actual - sigma_pred
    recon_mu = mu_pred + up_mu
    recon_sigma = sigma_pred + up_sigma
    error = max((recon_mu - mu_actual).abs().max().item(), (recon_sigma - sigma_actual).abs().max().item())
    surprise = torch.sqrt(up_mu.square().sum(dim=-1) + up_sigma.square().sum(dim=-1))
    return DecompositionOutput(surprise_magnitude=surprise, reconstruction_error=error)


def candidate_c(mu_pred: torch.Tensor, sigma_pred: torch.Tensor, mu_actual: torch.Tensor, sigma_actual: torch.Tensor) -> DecompositionOutput:
    lambda_pred = sigma_pred.reciprocal().square()
    lambda_actual = sigma_actual.reciprocal().square()
    eta_actual = lambda_actual * mu_actual

    lambda_explained = torch.minimum(lambda_pred, lambda_actual)
    eta_explained = lambda_explained * mu_pred
    lambda_surprise = lambda_actual - lambda_explained
    eta_surprise = eta_actual - eta_explained

    lambda_recon = lambda_explained + lambda_surprise
    eta_recon = eta_explained + eta_surprise
    sigma_recon = lambda_recon.rsqrt()
    mu_recon = eta_recon / lambda_recon
    error = max((mu_recon - mu_actual).abs().max().item(), (sigma_recon - sigma_actual).abs().max().item())
    surprise = torch.sqrt(lambda_surprise.square().sum(dim=-1) + eta_surprise.square().sum(dim=-1))
    return DecompositionOutput(surprise_magnitude=surprise, reconstruction_error=error)


def monotonic_non_decreasing(values: list[float], tol: float = 1e-9) -> bool:
    return all(next_value + tol >= current for current, next_value in zip(values, values[1:]))


def record(label: str, passed: bool, detail: str, failures: list[str]) -> None:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {label}: {detail}")
    if not passed:
        failures.append(label)


def check_observation(
    label: str,
    observed: bool,
    expected: bool,
    detail: str,
    failures: list[str],
) -> None:
    passed = observed == expected
    status = "[PASS]" if passed else "[FAIL]"
    expectation = "meets" if expected else "fails"
    print(f"{status} {label}: observed={observed}, expected={expectation} ({detail})")
    if not passed:
        failures.append(label)


def main() -> int:
    torch.manual_seed(SEED)
    failures: list[str] = []

    candidates = {
        "w2_residual_decomposition": candidate_b,
        "clipped_natural_residual": candidate_c,
    }
    expected = {
        "w2_residual_decomposition": {
            "conservation": True,
            "identity": True,
            "correlation": True,
            "mean_monotonic": True,
            "uncertainty_monotonic": True,
        },
        "clipped_natural_residual": {
            "conservation": True,
            "identity": True,
            "correlation": False,
            "mean_monotonic": True,
            "uncertainty_monotonic": True,
        },
    }

    mu_pred, sigma_pred = sample_diag_gaussian()
    mu_actual, sigma_actual = sample_diag_gaussian()

    for candidate_name, candidate_fn in candidates.items():
        output = candidate_fn(mu_pred, sigma_pred, mu_actual, sigma_actual)
        expected_candidate = expected[candidate_name]
        conservation_ok = output.reconstruction_error <= 1e-9
        check_observation(
            f"{candidate_name} conservation",
            conservation_ok,
            expected_candidate["conservation"],
            f"max_error={output.reconstruction_error:.6e}",
            failures,
        )

        identity_output = candidate_fn(mu_actual, sigma_actual, mu_actual, sigma_actual)
        identity_mag = identity_output.surprise_magnitude.max().item()
        identity_ok = identity_mag <= 1e-9
        check_observation(
            f"{candidate_name} zero on identity",
            identity_ok,
            expected_candidate["identity"],
            f"max_surprise={identity_mag:.6e}",
            failures,
        )

        correlation = pearson(w2_sq(mu_pred, sigma_pred, mu_actual, sigma_actual), output.surprise_magnitude)
        correlation_ok = correlation >= 0.85
        check_observation(
            f"{candidate_name} loss correlation",
            correlation_ok,
            expected_candidate["correlation"],
            f"pearson={correlation:.6f}",
            failures,
        )

        mean_base = torch.zeros(BATCH, DIM, dtype=DTYPE)
        sigma_base = torch.full((BATCH, DIM), 0.3, dtype=DTYPE)
        delta = torch.linspace(-0.8, 0.8, DIM, dtype=DTYPE).unsqueeze(0).expand(BATCH, -1)
        mean_alphas = [0.0, 0.5, 1.0, 2.0, 4.0]
        mean_sweep = [candidate_fn(mean_base + alpha * delta, sigma_base, mean_base, sigma_base).surprise_magnitude.mean().item() for alpha in mean_alphas]
        mean_monotonic_ok = monotonic_non_decreasing(mean_sweep)
        check_observation(
            f"{candidate_name} mean-offset monotonicity",
            mean_monotonic_ok,
            expected_candidate["mean_monotonic"],
            f"alphas={mean_alphas}, mags={[round(value, 6) for value in mean_sweep]}",
            failures,
        )

        sigma_actual_sweep = torch.full((BATCH, DIM), 0.1, dtype=DTYPE)
        sigma_values = [0.1, 0.2, 0.4, 0.8, 1.6, 2.0]
        uncertainty_sweep = [
            candidate_fn(
                mean_base,
                torch.full((BATCH, DIM), sigma_value, dtype=DTYPE),
                mean_base,
                sigma_actual_sweep,
            ).surprise_magnitude.mean().item()
            for sigma_value in sigma_values
        ]
        uncertainty_monotonic_ok = monotonic_non_decreasing(uncertainty_sweep)
        check_observation(
            f"{candidate_name} uncertainty sweep",
            uncertainty_monotonic_ok,
            expected_candidate["uncertainty_monotonic"],
            f"sigma_pred={sigma_values}, mags={[round(value, 6) for value in uncertainty_sweep]}",
            failures,
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
