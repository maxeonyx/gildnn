import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


DTYPE = torch.float64
SEED = 42
TRAIN_STEPS = 1000
LOG_EVERY = 100
BATCH = 64
DIM = 16
SEQ_LEN = 50
HIDDEN = 64
LR = 1e-3
WEIGHT_DECAY = 1e-2
MU_STEP_SCALE = 0.1
LOG_SIGMA_STEP_SCALE = 0.02
LOG_SIGMA_MIN = math.log(0.1)
LOG_SIGMA_MAX = math.log(1.0)
EVAL_BATCHES = 64
GRAD_NORM_EXPLOSION_THRESHOLD = 1e3
MU_RESTORING_STRENGTH = 0.2
LOG_SIGMA_RESTORING_STRENGTH = 0.2
LOG_SIGMA_CENTER = 0.5 * (LOG_SIGMA_MIN + LOG_SIGMA_MAX)


@dataclass
class StepRecord:
    step: int
    w2_loss: float
    mse_loss: float
    copy_loss: float
    grad_norm_w2: float
    grad_norm_mse: float
    min_sigma_w2: float
    min_sigma_mse: float
    loss_diff: float
    param_diff: float


@dataclass
class EvalResult:
    trained_w2_loss: float
    trained_mse_loss: float
    copy_loss: float
    random_loss: float
    min_sigma_w2: float
    min_sigma_mse: float
    max_model_output_diff: float


@dataclass(frozen=True)
class Scenario:
    name: str
    predictable_drift: bool


SCENARIOS = [
    Scenario(name="requested_pure_random_walk", predictable_drift=False),
    Scenario(name="state_dependent_drift", predictable_drift=True),
]


def generate_diagonal_gaussian_sequences(
    *,
    batch: int,
    seq_len: int,
    dim: int,
    generator: torch.Generator,
    predictable_drift: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    mu = torch.empty(batch, seq_len, dim, dtype=DTYPE)
    log_sigma = torch.empty(batch, seq_len, dim, dtype=DTYPE)

    mu[:, 0] = torch.randn(batch, dim, generator=generator, dtype=DTYPE)
    log_sigma[:, 0] = torch.empty(batch, dim, dtype=DTYPE).uniform_(LOG_SIGMA_MIN, LOG_SIGMA_MAX, generator=generator)

    for t in range(seq_len - 1):
        mu_noise = torch.randn(batch, dim, generator=generator, dtype=DTYPE)
        log_sigma_noise = torch.randn(batch, dim, generator=generator, dtype=DTYPE)
        if predictable_drift:
            mu_drift = -MU_RESTORING_STRENGTH * torch.tanh(mu[:, t])
            log_sigma_drift = -LOG_SIGMA_RESTORING_STRENGTH * (log_sigma[:, t] - LOG_SIGMA_CENTER)
        else:
            mu_drift = torch.zeros_like(mu[:, t])
            log_sigma_drift = torch.zeros_like(log_sigma[:, t])
        mu[:, t + 1] = mu[:, t] + mu_drift + MU_STEP_SCALE * mu_noise
        log_sigma[:, t + 1] = log_sigma[:, t] + log_sigma_drift + LOG_SIGMA_STEP_SCALE * log_sigma_noise

    sigma = log_sigma.exp()
    return mu, sigma


def build_training_pairs(mu: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = torch.cat([mu[:, :-1], sigma[:, :-1]], dim=-1).reshape(-1, 2 * DIM)
    target_mu = mu[:, 1:].reshape(-1, DIM)
    target_sigma = sigma[:, 1:].reshape(-1, DIM)
    return inputs, target_mu, target_sigma


def copy_predictor_loss(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return ((mu[:, :-1] - mu[:, 1:]).square() + (sigma[:, :-1] - sigma[:, 1:]).square()).mean()


def random_predictor_loss(mu_target: torch.Tensor, sigma_target: torch.Tensor, *, generator: torch.Generator) -> torch.Tensor:
    mu_random = torch.randn(mu_target.shape, generator=generator, dtype=DTYPE)
    log_sigma_random = torch.empty(sigma_target.shape, dtype=DTYPE).uniform_(LOG_SIGMA_MIN, LOG_SIGMA_MAX, generator=generator)
    sigma_random = log_sigma_random.exp()
    return wasserstein_2_sq_loss(mu_random, sigma_random, mu_target, sigma_target)


def wasserstein_2_sq_loss(
    mu_pred: torch.Tensor,
    sigma_pred: torch.Tensor,
    mu_target: torch.Tensor,
    sigma_target: torch.Tensor,
) -> torch.Tensor:
    return ((mu_pred - mu_target).square() + (sigma_pred - sigma_target).square()).mean()


def parameter_mse_loss(
    mu_pred: torch.Tensor,
    sigma_pred: torch.Tensor,
    mu_target: torch.Tensor,
    sigma_target: torch.Tensor,
) -> torch.Tensor:
    pred = torch.cat([mu_pred, sigma_pred], dim=-1)
    target = torch.cat([mu_target, sigma_target], dim=-1)
    return 2.0 * F.mse_loss(pred, target)


def global_grad_norm(parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().square().sum().item()
        total += grad_norm
    return math.sqrt(total)


def max_parameter_difference(model_a: nn.Module, model_b: nn.Module) -> float:
    return max(
        (param_a.detach() - param_b.detach()).abs().max().item()
        for param_a, param_b in zip(model_a.parameters(), model_b.parameters(), strict=True)
    )


class PredictorBlock(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * dim),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.net(inputs)
        mu_pred, sigma_raw = output.chunk(2, dim=-1)
        sigma_pred = F.softplus(sigma_raw)
        return mu_pred, sigma_pred


def train_step(
    model: PredictorBlock,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    target_mu: torch.Tensor,
    target_sigma: torch.Tensor,
    *,
    loss_kind: str,
) -> tuple[float, float, float]:
    optimizer.zero_grad(set_to_none=True)
    mu_pred, sigma_pred = model(inputs)

    if loss_kind == "w2":
        loss = wasserstein_2_sq_loss(mu_pred, sigma_pred, target_mu, target_sigma)
    elif loss_kind == "mse":
        loss = parameter_mse_loss(mu_pred, sigma_pred, target_mu, target_sigma)
    else:
        raise ValueError(f"Unknown loss kind: {loss_kind}")

    loss.backward()
    grad_norm = global_grad_norm(list(model.parameters()))
    optimizer.step()
    return loss.item(), grad_norm, sigma_pred.min().item()


def monotone_nonincreasing(values: list[float], *, tolerance: float = 0.0) -> bool:
    return all(current <= previous + tolerance for previous, current in zip(values, values[1:], strict=False))


def run_scenario(scenario: Scenario, *, seed: int) -> bool:
    train_generator = torch.Generator().manual_seed(seed)
    eval_generator = torch.Generator().manual_seed(seed + 10_000)

    model_w2 = PredictorBlock(dim=DIM, hidden=HIDDEN).to(dtype=DTYPE)
    model_mse = PredictorBlock(dim=DIM, hidden=HIDDEN).to(dtype=DTYPE)
    model_mse.load_state_dict(model_w2.state_dict())

    optimizer_w2 = torch.optim.AdamW(model_w2.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer_mse = torch.optim.AdamW(model_mse.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    records: list[StepRecord] = []
    loss_diffs: list[float] = []
    grad_problem = False
    sigma_problem = False

    for step in range(1, TRAIN_STEPS + 1):
        mu, sigma = generate_diagonal_gaussian_sequences(
            batch=BATCH,
            seq_len=SEQ_LEN,
            dim=DIM,
            generator=train_generator,
            predictable_drift=scenario.predictable_drift,
        )
        inputs, target_mu, target_sigma = build_training_pairs(mu, sigma)
        copy_loss = copy_predictor_loss(mu, sigma).item()

        w2_loss, grad_norm_w2, min_sigma_w2 = train_step(
            model_w2,
            optimizer_w2,
            inputs,
            target_mu,
            target_sigma,
            loss_kind="w2",
        )
        mse_loss, grad_norm_mse, min_sigma_mse = train_step(
            model_mse,
            optimizer_mse,
            inputs,
            target_mu,
            target_sigma,
            loss_kind="mse",
        )

        loss_diff = abs(w2_loss - mse_loss)
        param_diff = max_parameter_difference(model_w2, model_mse)
        loss_diffs.append(loss_diff)

        gradients_finite = all(math.isfinite(value) for value in (grad_norm_w2, grad_norm_mse))
        gradients_small = grad_norm_w2 < GRAD_NORM_EXPLOSION_THRESHOLD and grad_norm_mse < GRAD_NORM_EXPLOSION_THRESHOLD
        grad_problem = grad_problem or (not gradients_finite) or (not gradients_small)
        sigma_problem = sigma_problem or min_sigma_w2 <= 0.0 or min_sigma_mse <= 0.0

        record = StepRecord(
            step=step,
            w2_loss=w2_loss,
            mse_loss=mse_loss,
            copy_loss=copy_loss,
            grad_norm_w2=grad_norm_w2,
            grad_norm_mse=grad_norm_mse,
            min_sigma_w2=min_sigma_w2,
            min_sigma_mse=min_sigma_mse,
            loss_diff=loss_diff,
            param_diff=param_diff,
        )
        records.append(record)

        if step == 1 or step % LOG_EVERY == 0:
            print(
                f"scenario={scenario.name} "
                f"step={step:4d} "
                f"w2_loss={w2_loss:.6f} "
                f"mse_loss={mse_loss:.6f} "
                f"copy={copy_loss:.6f} "
                f"grad_w2={grad_norm_w2:.6f} "
                f"grad_mse={grad_norm_mse:.6f} "
                f"sigma_min_w2={min_sigma_w2:.6f} "
                f"sigma_min_mse={min_sigma_mse:.6f} "
                f"loss_diff={loss_diff:.3e} "
                f"param_diff={param_diff:.3e}"
            )

    evaluation = evaluate_models(
        model_w2,
        model_mse,
        batches=EVAL_BATCHES,
        generator=eval_generator,
        predictable_drift=scenario.predictable_drift,
    )

    w2_curve = [record.w2_loss for record in records]
    mse_curve = [record.mse_loss for record in records]
    strict_monotone_w2 = monotone_nonincreasing(w2_curve, tolerance=1e-12)
    strict_monotone_mse = monotone_nonincreasing(mse_curve, tolerance=1e-12)
    w2_drop = w2_curve[-1] - w2_curve[0]
    mse_drop = mse_curve[-1] - mse_curve[0]
    max_grad_w2 = max(record.grad_norm_w2 for record in records)
    max_grad_mse = max(record.grad_norm_mse for record in records)
    min_sigma_w2_seen = min(record.min_sigma_w2 for record in records)
    min_sigma_mse_seen = min(record.min_sigma_mse for record in records)
    max_loss_diff = max(loss_diffs)
    max_param_diff = max(record.param_diff for record in records)

    print()
    print(f"final_metrics scenario={scenario.name}")
    print(f"train_start_w2={w2_curve[0]:.6f}")
    print(f"train_end_w2={w2_curve[-1]:.6f}")
    print(f"train_delta_w2={w2_drop:.6f}")
    print(f"train_start_mse={mse_curve[0]:.6f}")
    print(f"train_end_mse={mse_curve[-1]:.6f}")
    print(f"train_delta_mse={mse_drop:.6f}")
    print(f"strict_monotone_w2={strict_monotone_w2}")
    print(f"strict_monotone_mse={strict_monotone_mse}")
    print(f"eval_trained_w2={evaluation.trained_w2_loss:.6f}")
    print(f"eval_trained_mse={evaluation.trained_mse_loss:.6f}")
    print(f"eval_copy={evaluation.copy_loss:.6f}")
    print(f"eval_random={evaluation.random_loss:.6f}")
    print(f"min_sigma_w2_seen={min_sigma_w2_seen:.6f}")
    print(f"min_sigma_mse_seen={min_sigma_mse_seen:.6f}")
    print(f"eval_min_sigma_w2={evaluation.min_sigma_w2:.6f}")
    print(f"eval_min_sigma_mse={evaluation.min_sigma_mse:.6f}")
    print(f"max_grad_w2={max_grad_w2:.6f}")
    print(f"max_grad_mse={max_grad_mse:.6f}")
    print(f"max_loss_diff={max_loss_diff:.6e}")
    print(f"max_param_diff={max_param_diff:.6e}")
    print(f"max_model_output_diff={evaluation.max_model_output_diff:.6e}")
    print(f"grad_problem={grad_problem}")
    print(f"sigma_problem={sigma_problem}")
    print(f"trained_beats_copy={evaluation.trained_w2_loss < evaluation.copy_loss}")
    print(f"trained_beats_random={evaluation.trained_w2_loss < evaluation.random_loss}")

    failure = False
    if not w2_curve[-1] < w2_curve[0]:
        print("FAIL: W2 loss did not decrease overall.")
        failure = True
    if not mse_curve[-1] < mse_curve[0]:
        print("FAIL: MSE loss did not decrease overall.")
        failure = True
    if sigma_problem:
        print("FAIL: predicted sigma became non-positive.")
        failure = True
    if grad_problem:
        print("FAIL: gradient norms were non-finite or exploded.")
        failure = True
    if max_loss_diff > 1e-12 or max_param_diff > 1e-12 or evaluation.max_model_output_diff > 1e-12:
        print("FAIL: W2 and parameter-MSE training diverged numerically.")
        failure = True
    if scenario.predictable_drift:
        if not evaluation.trained_w2_loss < evaluation.copy_loss:
            print("FAIL: predictable-drift model did not beat copy baseline.")
            failure = True
        if not evaluation.trained_mse_loss < evaluation.copy_loss:
            print("FAIL: predictable-drift MSE model did not beat copy baseline.")
            failure = True
    else:
        copy_gap = abs(evaluation.trained_w2_loss - evaluation.copy_loss)
        print(f"pure_random_walk_copy_gap={copy_gap:.6f}")
        print("note=pure random walk makes copy the Bayes-optimal one-step predictor; beating copy is not expected")

    return not failure


@torch.no_grad()
def evaluate_models(
    model_w2: PredictorBlock,
    model_mse: PredictorBlock,
    *,
    batches: int,
    generator: torch.Generator,
    predictable_drift: bool,
) -> EvalResult:
    trained_w2_losses: list[float] = []
    trained_mse_losses: list[float] = []
    copy_losses: list[float] = []
    random_losses: list[float] = []
    min_sigma_w2 = float("inf")
    min_sigma_mse = float("inf")
    max_model_output_diff = 0.0

    for _ in range(batches):
        mu, sigma = generate_diagonal_gaussian_sequences(
            batch=BATCH,
            seq_len=SEQ_LEN,
            dim=DIM,
            generator=generator,
            predictable_drift=predictable_drift,
        )
        inputs, target_mu, target_sigma = build_training_pairs(mu, sigma)

        mu_pred_w2, sigma_pred_w2 = model_w2(inputs)
        mu_pred_mse, sigma_pred_mse = model_mse(inputs)

        trained_w2_losses.append(wasserstein_2_sq_loss(mu_pred_w2, sigma_pred_w2, target_mu, target_sigma).item())
        trained_mse_losses.append(wasserstein_2_sq_loss(mu_pred_mse, sigma_pred_mse, target_mu, target_sigma).item())
        copy_losses.append(copy_predictor_loss(mu, sigma).item())
        random_losses.append(random_predictor_loss(target_mu, target_sigma, generator=generator).item())

        min_sigma_w2 = min(min_sigma_w2, sigma_pred_w2.min().item())
        min_sigma_mse = min(min_sigma_mse, sigma_pred_mse.min().item())
        max_model_output_diff = max(
            max_model_output_diff,
            (mu_pred_w2 - mu_pred_mse).abs().max().item(),
            (sigma_pred_w2 - sigma_pred_mse).abs().max().item(),
        )

    return EvalResult(
        trained_w2_loss=sum(trained_w2_losses) / len(trained_w2_losses),
        trained_mse_loss=sum(trained_mse_losses) / len(trained_mse_losses),
        copy_loss=sum(copy_losses) / len(copy_losses),
        random_loss=sum(random_losses) / len(random_losses),
        min_sigma_w2=min_sigma_w2,
        min_sigma_mse=min_sigma_mse,
        max_model_output_diff=max_model_output_diff,
    )


def main() -> int:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    passed_all = True
    for index, scenario in enumerate(SCENARIOS):
        if index > 0:
            print()
            print("-" * 80)
            print()
        scenario_passed = run_scenario(scenario, seed=SEED + 1_000 * index)
        passed_all = passed_all and scenario_passed

    return 0 if passed_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
