import math
from dataclasses import dataclass

import torch


DTYPE = torch.float64
SEED = 42
BATCH = 512
DIM = 32
LOG_SIGMA_MIN = math.log(0.05)
LOG_SIGMA_MAX = math.log(2.0)
TOL = 1e-9


@dataclass
class CandidateOutput:
    surprise_magnitude: torch.Tensor
    valid: bool
    reconstructable: bool
    reconstruction_error: float | None
    note: str


def sample_diag_gaussian(batch: int = BATCH, dim: int = DIM) -> tuple[torch.Tensor, torch.Tensor]:
    mu = torch.randn(batch, dim, dtype=DTYPE)
    log_sigma = torch.empty(batch, dim, dtype=DTYPE).uniform_(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
    sigma = log_sigma.exp()
    return mu, sigma


def w2_sq(mu_1: torch.Tensor, sigma_1: torch.Tensor, mu_2: torch.Tensor, sigma_2: torch.Tensor) -> torch.Tensor:
    return ((mu_1 - mu_2).square() + (sigma_1 - sigma_2).square()).sum(dim=-1)


def poe_posterior(mu_pred: torch.Tensor, sigma_pred: torch.Tensor, mu_actual: torch.Tensor, sigma_actual: torch.Tensor) -> CandidateOutput:
    tau_pred = sigma_pred.reciprocal().square()
    tau_actual = sigma_actual.reciprocal().square()
    tau_combined = tau_pred + tau_actual
    mu_combined = (tau_pred * mu_pred + tau_actual * mu_actual) / tau_combined
    sigma_combined = tau_combined.rsqrt()
    residual = w2_sq(mu_combined, sigma_combined, mu_actual, sigma_actual).sqrt()
    valid = bool(torch.isfinite(mu_combined).all() and torch.isfinite(sigma_combined).all() and (sigma_combined > 0).all())
    return CandidateOutput(
        surprise_magnitude=residual,
        valid=valid,
        reconstructable=False,
        reconstruction_error=None,
        note="fusion only; no exact down/up decomposition",
    )


def w2_residual_decomposition(mu_pred: torch.Tensor, sigma_pred: torch.Tensor, mu_actual: torch.Tensor, sigma_actual: torch.Tensor) -> CandidateOutput:
    up_mu = mu_actual - mu_pred
    up_sigma = sigma_actual - sigma_pred
    recon_mu = mu_pred + up_mu
    recon_sigma = sigma_pred + up_sigma
    reconstruction_error = max(
        (recon_mu - mu_actual).abs().max().item(),
        (recon_sigma - sigma_actual).abs().max().item(),
    )
    residual = torch.sqrt(up_mu.square().sum(dim=-1) + up_sigma.square().sum(dim=-1))
    valid = bool(
        torch.isfinite(mu_pred).all()
        and torch.isfinite(sigma_pred).all()
        and torch.isfinite(up_mu).all()
        and torch.isfinite(up_sigma).all()
        and (sigma_pred > 0).all()
    )
    return CandidateOutput(
        surprise_magnitude=residual,
        valid=valid,
        reconstructable=True,
        reconstruction_error=reconstruction_error,
        note="exact chart-space residual; surprise sigma may be negative",
    )


def clipped_natural_residual(mu_pred: torch.Tensor, sigma_pred: torch.Tensor, mu_actual: torch.Tensor, sigma_actual: torch.Tensor) -> CandidateOutput:
    lambda_pred = sigma_pred.reciprocal().square()
    lambda_actual = sigma_actual.reciprocal().square()
    eta_pred = lambda_pred * mu_pred
    eta_actual = lambda_actual * mu_actual

    lambda_explained = torch.minimum(lambda_pred, lambda_actual)
    eta_explained = lambda_explained * mu_pred
    lambda_surprise = lambda_actual - lambda_explained
    eta_surprise = eta_actual - eta_explained

    lambda_recon = lambda_explained + lambda_surprise
    eta_recon = eta_explained + eta_surprise
    sigma_recon = lambda_recon.rsqrt()
    mu_recon = eta_recon / lambda_recon
    reconstruction_error = max(
        (mu_recon - mu_actual).abs().max().item(),
        (sigma_recon - sigma_actual).abs().max().item(),
    )

    invalid_gaussian_mask = (lambda_surprise <= TOL) & (eta_surprise.abs() > 1e-8)
    valid = bool(
        torch.isfinite(lambda_explained).all()
        and torch.isfinite(eta_explained).all()
        and torch.isfinite(lambda_surprise).all()
        and torch.isfinite(eta_surprise).all()
        and not invalid_gaussian_mask.any().item()
    )
    residual = torch.sqrt(lambda_surprise.square().sum(dim=-1) + eta_surprise.square().sum(dim=-1))
    note = "exact in natural parameters; can produce non-Gaussian surprise when lambda_surprise=0 but eta_surprise!=0"
    return CandidateOutput(
        surprise_magnitude=residual,
        valid=valid,
        reconstructable=True,
        reconstruction_error=reconstruction_error,
        note=note,
    )


def make_constant_scenario(mean_offset: float, sigma_pred_value: float, sigma_actual_value: float = 0.3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mu_actual = torch.zeros(BATCH, DIM, dtype=DTYPE)
    sigma_actual = torch.full((BATCH, DIM), sigma_actual_value, dtype=DTYPE)
    mu_pred = torch.full((BATCH, DIM), mean_offset, dtype=DTYPE)
    sigma_pred = torch.full((BATCH, DIM), sigma_pred_value, dtype=DTYPE)
    return mu_pred, sigma_pred, mu_actual, sigma_actual


def classify(validity: str, reconstruction: str, perfect: bool, routing: bool, wrong_confidence: bool) -> str:
    if reconstruction == "N/A":
        return "not suitable"
    if validity == "PASS" and perfect and routing and wrong_confidence:
        return "promising"
    if perfect and routing and wrong_confidence:
        return "mixed"
    return "weak"


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

    candidates = {
        "poe_posterior": poe_posterior,
        "w2_residual_decomposition": w2_residual_decomposition,
        "clipped_natural_residual": clipped_natural_residual,
    }

    mu_rand_pred, sigma_rand_pred = sample_diag_gaussian()
    mu_rand_actual, sigma_rand_actual = sample_diag_gaussian()

    scenarios = {
        "perfect_prediction": make_constant_scenario(mean_offset=0.0, sigma_pred_value=0.3),
        "uncertain_correct": make_constant_scenario(mean_offset=0.0, sigma_pred_value=1.5),
        "confident_correct": make_constant_scenario(mean_offset=0.0, sigma_pred_value=0.31),
        "confident_wrong": make_constant_scenario(mean_offset=1.5, sigma_pred_value=0.1),
        "uncertain_wrong": make_constant_scenario(mean_offset=1.5, sigma_pred_value=1.5),
        "random_batch": (mu_rand_pred, sigma_rand_pred, mu_rand_actual, sigma_rand_actual),
    }

    failures: list[str] = []
    summary_rows: list[tuple[str, str, str, str, str, str, str]] = []
    expected = {
        "poe_posterior": {
            "validity": True,
            "reconstruction": False,
            "perfect": False,
            "routing": False,
            "wrong_confidence": True,
        },
        "w2_residual_decomposition": {
            "validity": True,
            "reconstruction": True,
            "perfect": True,
            "routing": True,
            "wrong_confidence": True,
        },
        "clipped_natural_residual": {
            "validity": False,
            "reconstruction": True,
            "perfect": True,
            "routing": True,
            "wrong_confidence": True,
        },
    }

    for candidate_name, candidate_fn in candidates.items():
        outputs = {scenario_name: candidate_fn(*scenario) for scenario_name, scenario in scenarios.items()}

        validity_ok = all(output.valid for output in outputs.values())
        reconstruction_label = "N/A"
        reconstruction_ok = True
        if outputs["random_batch"].reconstructable:
            reconstruction_error = max(
                output.reconstruction_error for output in outputs.values() if output.reconstruction_error is not None
            )
            reconstruction_ok = reconstruction_error <= 1e-9
            reconstruction_label = "PASS" if reconstruction_ok else "FAIL"
        perfect_mag = outputs["perfect_prediction"].surprise_magnitude.mean().item()
        uncertain_correct_mag = outputs["uncertain_correct"].surprise_magnitude.mean().item()
        confident_correct_mag = outputs["confident_correct"].surprise_magnitude.mean().item()
        confident_wrong_mag = outputs["confident_wrong"].surprise_magnitude.mean().item()
        uncertain_wrong_mag = outputs["uncertain_wrong"].surprise_magnitude.mean().item()

        perfect_ok = perfect_mag <= 1e-8
        routing_ok = confident_correct_mag + 1e-8 < uncertain_correct_mag
        wrong_confidence_ok = confident_wrong_mag > max(confident_correct_mag * 10.0, 1.0)

        expected_candidate = expected[candidate_name]
        check_observation(
            f"{candidate_name} validity",
            validity_ok,
            expected_candidate["validity"],
            outputs["random_batch"].note,
            failures,
        )
        check_observation(
            f"{candidate_name} reconstruction",
            outputs["random_batch"].reconstructable and reconstruction_ok,
            expected_candidate["reconstruction"],
            f"label={reconstruction_label}",
            failures,
        )
        check_observation(
            f"{candidate_name} perfect-prediction criterion",
            perfect_ok,
            expected_candidate["perfect"],
            f"perfect_mean={perfect_mag:.6f}",
            failures,
        )
        check_observation(
            f"{candidate_name} confidence-routing criterion",
            routing_ok,
            expected_candidate["routing"],
            f"confident_correct={confident_correct_mag:.6f}, uncertain_correct={uncertain_correct_mag:.6f}",
            failures,
        )
        check_observation(
            f"{candidate_name} wrong-confidence criterion",
            wrong_confidence_ok,
            expected_candidate["wrong_confidence"],
            f"confident_wrong={confident_wrong_mag:.6f}, uncertain_wrong={uncertain_wrong_mag:.6f}",
            failures,
        )

        print(
            "    "
            f"surprise means: perfect={perfect_mag:.6f}, confident_correct={confident_correct_mag:.6f}, "
            f"uncertain_correct={uncertain_correct_mag:.6f}, confident_wrong={confident_wrong_mag:.6f}, "
            f"uncertain_wrong={uncertain_wrong_mag:.6f}"
        )
        print(f"    note: {outputs['random_batch'].note}")

        validity_label = "PASS" if validity_ok else "FAIL"
        perfect_label = "PASS" if perfect_ok else "FAIL"
        routing_label = "PASS" if routing_ok else "FAIL"
        wrong_label = "PASS" if wrong_confidence_ok else "FAIL"
        classification = classify(validity_label, reconstruction_label, perfect_ok, routing_ok, wrong_confidence_ok)
        summary_rows.append(
            (
                candidate_name,
                validity_label,
                reconstruction_label,
                perfect_label,
                routing_label,
                wrong_label,
                classification,
            )
        )

    print()
    print("Summary")
    print("candidate                     validity  reconstruct  perfect  routing  wrong-conf  class")
    print("---------------------------- ---------- ------------ -------- -------- ----------- ----------")
    for row in summary_rows:
        print(f"{row[0]:28} {row[1]:10} {row[2]:12} {row[3]:8} {row[4]:8} {row[5]:11} {row[6]:10}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
