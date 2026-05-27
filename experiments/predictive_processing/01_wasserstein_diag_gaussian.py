import math
import sys

import torch


DTYPE = torch.float64
SEED = 42
BATCH = 512
DIM = 32
LOG_SIGMA_MIN = math.log(0.05)
LOG_SIGMA_MAX = math.log(2.0)
TOL = 1e-10


def sample_diag_gaussian(batch: int = BATCH, dim: int = DIM) -> tuple[torch.Tensor, torch.Tensor]:
    mu = torch.randn(batch, dim, dtype=DTYPE)
    log_sigma = torch.empty(batch, dim, dtype=DTYPE).uniform_(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
    sigma = log_sigma.exp()
    return mu, sigma


def w2_sq(mu_1: torch.Tensor, sigma_1: torch.Tensor, mu_2: torch.Tensor, sigma_2: torch.Tensor) -> torch.Tensor:
    return ((mu_1 - mu_2).square() + (sigma_1 - sigma_2).square()).sum(dim=-1)


def record(label: str, passed: bool, detail: str, failures: list[str]) -> None:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {label}: {detail}")
    if not passed:
        failures.append(label)


def main() -> int:
    torch.manual_seed(SEED)
    failures: list[str] = []

    mu_a, sigma_a = sample_diag_gaussian()
    mu_b, sigma_b = sample_diag_gaussian()
    mu_c, sigma_c = sample_diag_gaussian()

    d_ab_sq = w2_sq(mu_a, sigma_a, mu_b, sigma_b)
    d_ba_sq = w2_sq(mu_b, sigma_b, mu_a, sigma_a)
    d_ac_sq = w2_sq(mu_a, sigma_a, mu_c, sigma_c)
    d_bc_sq = w2_sq(mu_b, sigma_b, mu_c, sigma_c)

    record("A1 Nonnegativity", bool((d_ab_sq >= -TOL).all()), f"min={d_ab_sq.min().item():.6e}", failures)

    same_sq = w2_sq(mu_a, sigma_a, mu_a, sigma_a)
    different_sq = d_ab_sq
    identity_ok = bool(torch.allclose(same_sq, torch.zeros_like(same_sq), atol=TOL, rtol=0.0))
    distinct_ok = bool((different_sq > 1e-8).all())
    record(
        "A2 Identity of indiscernibles",
        identity_ok and distinct_ok,
        f"same_max={same_sq.max().item():.6e}, different_min={different_sq.min().item():.6e}",
        failures,
    )

    symmetry_error = (d_ab_sq - d_ba_sq).abs().max().item()
    record("A3 Symmetry", symmetry_error <= TOL, f"max_abs_diff={symmetry_error:.6e}", failures)

    d_ab = d_ab_sq.sqrt()
    d_bc = d_bc_sq.sqrt()
    d_ac = d_ac_sq.sqrt()
    triangle_slack = (d_ac - (d_ab + d_bc)).max().item()
    record("B4 Triangle inequality for W2", triangle_slack <= 1e-9, f"max_slack={triangle_slack:.6e}", failures)

    shift = torch.randn(DIM, dtype=DTYPE)
    shifted_sq = w2_sq(mu_a + shift, sigma_a, mu_b + shift, sigma_b)
    translation_error = (shifted_sq - d_ab_sq).abs().max().item()
    record("C6 Mean translation invariance", translation_error <= TOL, f"max_abs_diff={translation_error:.6e}", failures)

    scale = torch.tensor(3.25, dtype=DTYPE)
    scaled_sq = w2_sq(scale * mu_a, scale * sigma_a, scale * mu_b, scale * sigma_b)
    scale_error = (scaled_sq - scale.square() * d_ab_sq).abs().max().item()
    record("C7 Scale homogeneity", scale_error <= 1e-9, f"max_abs_diff={scale_error:.6e}", failures)

    mu_pred = mu_a.clone().requires_grad_(True)
    sigma_pred = sigma_a.clone().requires_grad_(True)
    analytic_mu = 2.0 * (mu_pred.detach() - mu_b)
    analytic_sigma = 2.0 * (sigma_pred.detach() - sigma_b)
    autograd_loss = w2_sq(mu_pred, sigma_pred, mu_b, sigma_b).sum()
    autograd_loss.backward()
    mu_grad_error = (mu_pred.grad - analytic_mu).abs().max().item()
    sigma_grad_error = (sigma_pred.grad - analytic_sigma).abs().max().item()
    record(
        "D8 Autograd matches analytic gradient",
        mu_grad_error <= TOL and sigma_grad_error <= TOL,
        f"mu_err={mu_grad_error:.6e}, sigma_err={sigma_grad_error:.6e}",
        failures,
    )

    step_mu = mu_a.clone().requires_grad_(True)
    step_sigma = sigma_a.clone().requires_grad_(True)
    before = w2_sq(step_mu, step_sigma, mu_b, sigma_b).mean().item()
    step_loss = w2_sq(step_mu, step_sigma, mu_b, sigma_b).mean()
    step_loss.backward()
    learning_rate = 0.1
    with torch.no_grad():
        updated_mu = step_mu - learning_rate * step_mu.grad
        updated_sigma = step_sigma - learning_rate * step_sigma.grad
    after = w2_sq(updated_mu, updated_sigma, mu_b, sigma_b).mean().item()
    record("D9 Gradient descent step reduces loss", after < before, f"before={before:.6f}, after={after:.6f}", failures)

    mu_target = torch.zeros(1, DIM, dtype=DTYPE)
    sigma_target = torch.ones(1, DIM, dtype=DTYPE)
    direction_mu = torch.linspace(-0.3, 0.5, DIM, dtype=DTYPE).unsqueeze(0)
    direction_sigma = torch.linspace(0.05, 0.25, DIM, dtype=DTYPE).unsqueeze(0)

    near_mu = (mu_target + 0.25 * direction_mu).clone().requires_grad_(True)
    near_sigma = (sigma_target + 0.25 * direction_sigma).clone().requires_grad_(True)
    near_loss = w2_sq(near_mu, near_sigma, mu_target, sigma_target).sum()
    near_loss.backward()
    near_grad_norm = torch.sqrt(near_mu.grad.square().sum() + near_sigma.grad.square().sum()).item()

    far_mu = (mu_target + 1.0 * direction_mu).clone().requires_grad_(True)
    far_sigma = (sigma_target + 1.0 * direction_sigma).clone().requires_grad_(True)
    far_loss = w2_sq(far_mu, far_sigma, mu_target, sigma_target).sum()
    far_loss.backward()
    far_grad_norm = torch.sqrt(far_mu.grad.square().sum() + far_sigma.grad.square().sum()).item()

    record(
        "D10 Gradient magnitude grows with distance",
        far_grad_norm > near_grad_norm * 3.5,
        f"near={near_grad_norm:.6f}, far={far_grad_norm:.6f}, ratio={far_grad_norm / near_grad_norm:.3f}",
        failures,
    )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
