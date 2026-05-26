from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    raise RuntimeError(
        "Run from the repo root with `.\\.venv\\Scripts\\python.exe -m runs.halting_regression ...` so `core` imports resolve cleanly."
    )

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import load_dataset, set_seed
from core.recurrent_depth import RecurrentDepthConfig, RecurrentDepthLM
from core.run_utils import append_log, prepare_output_paths, redirect_sanity_check_paths, register_active_lock, resolve_device

CONTEXT_SIZE = 128
RECURRENT_ITERATIONS = 8
TRAINING_STEPS = 10_000
SANITY_CHECK_STEPS = 10
TRAIN_BATCH_SIZE = 64
SANITY_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 2_048
LEARNING_RATE = 3e-4
DEFAULT_SEED = 42
TRAIN_CHARACTERS = 900_000
VAL_CHARACTERS = 20_000
HALT_LAMBDA = 0.1
HALT_WARMUP_FRACTION = 0.1
PRINT_INTERVAL = 1_000
EPSILON_SWEEP = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
FIXED_DEPTH_BASELINES = (4, 5, 6, 7)
BEST_TRADEOFF_MAX_LOSS_HIT = 0.02


@dataclass(frozen=True)
class ExperimentConfig:
    context_size: int = CONTEXT_SIZE
    d_model: int = 128
    n_heads: int = 4
    ff_dim: int = 512
    recurrent_iterations: int = RECURRENT_ITERATIONS
    temperature: float = 0.07
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingSummary:
    steps: int
    batch_size: int
    final_train_loss: float
    final_lm_loss: float
    final_halt_loss: float
    final_halt_lambda: float
    running_gain_pearson: tuple[float, ...]
    wall_seconds: float


@dataclass(frozen=True)
class OracleMetrics:
    mean_loss_per_depth: tuple[float, ...]
    mean_gain_per_depth: tuple[float, ...]
    oracle_best_loss: float
    oracle_depth_histogram: tuple[int, ...]
    fraction_harmed_by_full_depth: float
    total_examples: int


@dataclass(frozen=True)
class ThresholdMetrics:
    epsilon: float
    avg_depth: float
    val_loss: float
    loss_hit_vs_full_depth: float
    learned_speedup: float
    oracle_efficiency: float
    oracle_avg_depth: float
    oracle_speedup: float


@dataclass(frozen=True)
class FixedDepthMetrics:
    depth: int
    avg_depth: float
    val_loss: float
    loss_hit_vs_full_depth: float
    learned_speedup: float


@dataclass(frozen=True)
class EvaluationSummary:
    oracle: OracleMetrics
    full_depth_val_loss: float
    gain_pearson_per_depth: tuple[float, ...]
    epsilon_sweep: tuple[ThresholdMetrics, ...]
    fixed_depth_baselines: tuple[FixedDepthMetrics, ...]
    best_tradeoff: ThresholdMetrics | None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "tinyshakespeare" / "artifacts" / "halting_regression"
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--steps", type=positive_int, default=TRAINING_STEPS)
    parser.add_argument("--batch-size", type=positive_int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=positive_int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--train-characters", type=positive_int, default=TRAIN_CHARACTERS)
    parser.add_argument("--val-characters", type=positive_int, default=VAL_CHARACTERS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--recurrent-iterations", type=positive_int, default=RECURRENT_ITERATIONS)
    parser.add_argument("--d-model", type=positive_int, default=128)
    parser.add_argument("--halt-lambda", type=float, default=HALT_LAMBDA)
    parser.add_argument("--halt-warmup-fraction", type=float, default=HALT_WARMUP_FRACTION)
    parser.add_argument("--sanity-check-only", action="store_true")
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true", dest="save_checkpoint")
    parser.add_argument("--no-save-checkpoint", action="store_false", dest="save_checkpoint")
    parser.set_defaults(save_checkpoint=True)
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.jsonl")
    return parser.parse_args()


def resolve_training_steps(args: argparse.Namespace) -> int:
    return SANITY_CHECK_STEPS if args.sanity_check_only else args.steps


def resolve_batch_size(args: argparse.Namespace) -> int:
    return SANITY_BATCH_SIZE if args.sanity_check_only else args.batch_size


def resolve_print_interval(steps: int) -> int:
    if steps <= 20:
        return 1
    return PRINT_INTERVAL


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def dataset_to_device(dataset: tuple[Tensor, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    inputs, targets = dataset
    return inputs.to(device), targets.to(device)


def sample_batch(dataset: tuple[Tensor, Tensor], *, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    inputs, targets = dataset
    indices = torch.randint(0, targets.shape[0], (batch_size,), device=device)
    return inputs[indices], targets[indices]


def pearson_correlation(predictions: Tensor, targets: Tensor) -> float:
    flat_predictions = predictions.detach().flatten().double().cpu()
    flat_targets = targets.detach().flatten().double().cpu()
    centered_predictions = flat_predictions - flat_predictions.mean()
    centered_targets = flat_targets - flat_targets.mean()
    denominator = torch.linalg.vector_norm(centered_predictions) * torch.linalg.vector_norm(centered_targets)
    if float(denominator.item()) <= 0.0:
        return float("nan")
    return float((centered_predictions @ centered_targets / denominator).item())


def build_model(*, vocab_size: int, config: ExperimentConfig) -> RecurrentDepthLM:
    return RecurrentDepthLM(
        vocab_size=vocab_size,
        config=RecurrentDepthConfig(
            context_size=config.context_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            ff_dim=config.ff_dim,
            iterations=config.recurrent_iterations,
            temperature=config.temperature,
            dropout=config.dropout,
        ),
    )


def build_optimizer(model: nn.Module, *, learning_rate: float, device: torch.device) -> torch.optim.Optimizer:
    optimizer_kwargs: dict[str, object] = {"lr": learning_rate}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    return torch.optim.Adam(model.parameters(), **optimizer_kwargs)


def current_halt_lambda(*, step: int, steps: int, halt_lambda: float, halt_warmup_fraction: float) -> float:
    warmup_steps = int(math.ceil(steps * halt_warmup_fraction))
    if warmup_steps <= 0:
        return halt_lambda
    return halt_lambda * min(step / warmup_steps, 1.0)


def compute_depth_outputs(
    model: RecurrentDepthLM,
    batch_inputs: Tensor,
    batch_targets: Tensor,
) -> tuple[Tensor, Tensor]:
    _, iteration_states = model.iteration_states(batch_inputs, collect_iteration_states=True)
    per_depth_losses: list[Tensor] = []
    predicted_gains: list[Tensor] = []
    for depth_index, iteration_hidden in enumerate(iteration_states, start=1):
        last_hidden = iteration_hidden[:, -1, :]
        logits = model.lm_logits_from_hidden(last_hidden)
        per_depth_losses.append(F.cross_entropy(logits.float(), batch_targets, reduction="none"))
        if depth_index < model.config.recurrent_iterations:
            predicted_gains.append(model.predicted_gain_from_hidden(last_hidden, depth_index=depth_index))
    return torch.stack(per_depth_losses, dim=1), torch.stack(predicted_gains, dim=1)


def compute_actual_gains(per_depth_losses: Tensor) -> Tensor:
    return per_depth_losses[:, :-1] - per_depth_losses[:, -1:]


def choose_depth_indices(predicted_gains: Tensor, *, epsilon: float, full_depth_index: int) -> Tensor:
    early_halt_mask = predicted_gains < epsilon
    any_halt = early_halt_mask.any(dim=1)
    return torch.where(
        any_halt,
        early_halt_mask.float().argmax(dim=1),
        torch.full((predicted_gains.shape[0],), full_depth_index, dtype=torch.int64),
    )


def train_model(
    model: RecurrentDepthLM,
    train_dataset: tuple[Tensor, Tensor],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    halt_lambda: float,
    halt_warmup_fraction: float,
    device: torch.device,
    log_path: Path,
) -> TrainingSummary:
    optimizer = build_optimizer(model, learning_rate=learning_rate, device=device)
    depth_count = model.config.recurrent_iterations
    running_predictions: list[list[Tensor]] = [[] for _ in range(depth_count - 1)]
    running_targets: list[list[Tensor]] = [[] for _ in range(depth_count - 1)]
    started_at = perf_counter()
    last_total_loss = float("nan")
    last_lm_loss = float("nan")
    last_halt_loss = float("nan")
    last_lambda = 0.0
    last_running_gain_pearson = tuple(float("nan") for _ in range(depth_count - 1))
    print_interval = resolve_print_interval(steps)

    for step in range(1, steps + 1):
        model.train()
        batch_inputs, batch_targets = sample_batch(train_dataset, batch_size=batch_size, device=device)
        step_halt_lambda = current_halt_lambda(
            step=step,
            steps=steps,
            halt_lambda=halt_lambda,
            halt_warmup_fraction=halt_warmup_fraction,
        )

        with autocast_context(device):
            per_depth_losses, predicted_gains = compute_depth_outputs(model, batch_inputs, batch_targets)
            actual_gains = compute_actual_gains(per_depth_losses.detach())
            lm_loss = per_depth_losses[:, -1].mean()
            halt_loss = F.mse_loss(predicted_gains.float(), actual_gains.float())
            total_loss = lm_loss + step_halt_lambda * halt_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        last_total_loss = float(total_loss.item())
        last_lm_loss = float(lm_loss.item())
        last_halt_loss = float(halt_loss.item())
        last_lambda = step_halt_lambda

        detached_predictions = predicted_gains.detach().float()
        detached_targets = actual_gains.detach().float()
        for depth_index in range(depth_count - 1):
            running_predictions[depth_index].append(detached_predictions[:, depth_index].cpu())
            running_targets[depth_index].append(detached_targets[:, depth_index].cpu())

        if step % print_interval != 0 and step != steps:
            continue

        last_running_gain_pearson = tuple(
            pearson_correlation(torch.cat(running_predictions[depth_index]), torch.cat(running_targets[depth_index]))
            for depth_index in range(depth_count - 1)
        )
        print(
            f"step={step:05d}/{steps} train_loss={last_total_loss:.4f} lm_loss={last_lm_loss:.4f} "
            f"halt_loss={last_halt_loss:.4f} halt_lambda={last_lambda:.4f}",
            flush=True,
        )
        for depth_index, correlation in enumerate(last_running_gain_pearson, start=1):
            print(f"  depth {depth_index}: running_gain_pearson={correlation:.4f}", flush=True)
        append_log(
            log_path,
            {
                "stage": "train_progress",
                "step": step,
                "steps": steps,
                "train_loss": round(last_total_loss, 6),
                "lm_loss": round(last_lm_loss, 6),
                "halt_loss": round(last_halt_loss, 6),
                "halt_lambda": round(last_lambda, 6),
                "running_gain_pearson": [
                    round(value, 6) if math.isfinite(value) else None for value in last_running_gain_pearson
                ],
            },
        )

    return TrainingSummary(
        steps=steps,
        batch_size=batch_size,
        final_train_loss=last_total_loss,
        final_lm_loss=last_lm_loss,
        final_halt_loss=last_halt_loss,
        final_halt_lambda=last_lambda,
        running_gain_pearson=last_running_gain_pearson,
        wall_seconds=perf_counter() - started_at,
    )


def compute_oracle_metrics(per_depth_losses: Tensor) -> OracleMetrics:
    depth_count = per_depth_losses.shape[1]
    total_examples = per_depth_losses.shape[0]
    if total_examples == 0:
        raise RuntimeError("validation dataset was empty")

    actual_gains = compute_actual_gains(per_depth_losses)
    mean_loss_per_depth = tuple(float(value) for value in per_depth_losses.mean(dim=0).tolist())
    mean_gain_per_depth = tuple(float(value) for value in actual_gains.mean(dim=0).tolist())
    oracle_best_losses, oracle_depth_indices = per_depth_losses.min(dim=1)
    full_depth_losses = per_depth_losses[:, -1]
    return OracleMetrics(
        mean_loss_per_depth=mean_loss_per_depth,
        mean_gain_per_depth=mean_gain_per_depth,
        oracle_best_loss=float(oracle_best_losses.mean().item()),
        oracle_depth_histogram=tuple(
            int(value) for value in torch.bincount(oracle_depth_indices.cpu(), minlength=depth_count).tolist()
        ),
        fraction_harmed_by_full_depth=float((full_depth_losses > oracle_best_losses + 1e-12).float().mean().item()),
        total_examples=total_examples,
    )


def summarize_fixed_depth_baselines(per_depth_losses: Tensor) -> tuple[FixedDepthMetrics, ...]:
    depth_count = per_depth_losses.shape[1]
    full_depth_val_loss = float(per_depth_losses[:, -1].mean().item())
    baselines: list[FixedDepthMetrics] = []
    for depth in FIXED_DEPTH_BASELINES:
        if depth > depth_count:
            continue
        chosen_losses = per_depth_losses[:, depth - 1]
        val_loss = float(chosen_losses.mean().item())
        baselines.append(
            FixedDepthMetrics(
                depth=depth,
                avg_depth=float(depth),
                val_loss=val_loss,
                loss_hit_vs_full_depth=val_loss - full_depth_val_loss,
                learned_speedup=depth_count / depth,
            )
        )
    return tuple(baselines)


@torch.inference_mode()
def evaluate_model(
    model: RecurrentDepthLM,
    dataset: tuple[Tensor, Tensor],
    *,
    eval_batch_size: int,
    device: torch.device,
) -> EvaluationSummary:
    model.eval()
    inputs, targets = dataset
    all_per_depth_losses: list[Tensor] = []
    all_predicted_gains: list[Tensor] = []

    for start in range(0, targets.shape[0], eval_batch_size):
        stop = min(start + eval_batch_size, targets.shape[0])
        batch_inputs = inputs[start:stop]
        batch_targets = targets[start:stop]
        with autocast_context(device):
            per_depth_losses, predicted_gains = compute_depth_outputs(model, batch_inputs, batch_targets)
        all_per_depth_losses.append(per_depth_losses.float().cpu())
        all_predicted_gains.append(predicted_gains.float().cpu())

    per_depth_losses = torch.cat(all_per_depth_losses, dim=0)
    predicted_gains = torch.cat(all_predicted_gains, dim=0)
    actual_gains = compute_actual_gains(per_depth_losses)
    oracle = compute_oracle_metrics(per_depth_losses)
    full_depth_val_loss = float(per_depth_losses[:, -1].mean().item())
    gain_pearson_per_depth = tuple(
        pearson_correlation(predicted_gains[:, depth_index], actual_gains[:, depth_index])
        for depth_index in range(predicted_gains.shape[1])
    )

    epsilon_sweep: list[ThresholdMetrics] = []
    depth_count = model.config.recurrent_iterations
    full_depth_index = depth_count - 1
    example_indices = torch.arange(per_depth_losses.shape[0], dtype=torch.int64)
    for epsilon in EPSILON_SWEEP:
        chosen_depth_indices = choose_depth_indices(predicted_gains, epsilon=epsilon, full_depth_index=full_depth_index)
        chosen_depths = chosen_depth_indices + 1
        halted_losses = per_depth_losses[example_indices, chosen_depth_indices]
        avg_depth = float(chosen_depths.float().mean().item())
        val_loss = float(halted_losses.mean().item())
        learned_speedup = depth_count / avg_depth

        oracle_depth_indices = choose_depth_indices(actual_gains, epsilon=epsilon, full_depth_index=full_depth_index)
        oracle_avg_depth = float((oracle_depth_indices + 1).float().mean().item())
        oracle_speedup = depth_count / oracle_avg_depth
        oracle_efficiency = 0.0
        if oracle_speedup > 1.0:
            oracle_efficiency = (learned_speedup - 1.0) / (oracle_speedup - 1.0)

        epsilon_sweep.append(
            ThresholdMetrics(
                epsilon=epsilon,
                avg_depth=avg_depth,
                val_loss=val_loss,
                loss_hit_vs_full_depth=val_loss - full_depth_val_loss,
                learned_speedup=learned_speedup,
                oracle_efficiency=oracle_efficiency,
                oracle_avg_depth=oracle_avg_depth,
                oracle_speedup=oracle_speedup,
            )
        )

    acceptable_thresholds = [entry for entry in epsilon_sweep if entry.loss_hit_vs_full_depth <= BEST_TRADEOFF_MAX_LOSS_HIT + 1e-12]
    best_tradeoff = None
    if len(acceptable_thresholds) > 0:
        best_tradeoff = max(
            acceptable_thresholds,
            key=lambda entry: (entry.learned_speedup, -entry.loss_hit_vs_full_depth, -entry.epsilon),
        )

    return EvaluationSummary(
        oracle=oracle,
        full_depth_val_loss=full_depth_val_loss,
        gain_pearson_per_depth=gain_pearson_per_depth,
        epsilon_sweep=tuple(epsilon_sweep),
        fixed_depth_baselines=summarize_fixed_depth_baselines(per_depth_losses),
        best_tradeoff=best_tradeoff,
    )


def print_evaluation(summary: EvaluationSummary) -> None:
    oracle = summary.oracle
    print("=== Halting regression evaluation ===", flush=True)
    print("Mean loss per depth:", flush=True)
    for depth, loss in enumerate(oracle.mean_loss_per_depth, start=1):
        print(f"  depth {depth}: {loss:.6f}", flush=True)
    print("Mean actual gain vs full depth:", flush=True)
    for depth, gain in enumerate(oracle.mean_gain_per_depth, start=1):
        print(f"  depth {depth}: {gain:.6f}", flush=True)
    print(f"Full-depth val loss: {summary.full_depth_val_loss:.6f}", flush=True)
    print(f"Oracle-best loss: {oracle.oracle_best_loss:.6f}", flush=True)
    print(f"Fraction harmed by full depth: {oracle.fraction_harmed_by_full_depth:.6f}", flush=True)
    print("Per-depth gain Pearson correlation:", flush=True)
    for depth, correlation in enumerate(summary.gain_pearson_per_depth, start=1):
        print(f"  depth {depth}: {correlation:.6f}", flush=True)
    print("Fixed-depth baselines:", flush=True)
    for entry in summary.fixed_depth_baselines:
        print(
            f"  depth={entry.depth} avg_depth={entry.avg_depth:.4f} val_loss={entry.val_loss:.6f} "
            f"loss_hit={entry.loss_hit_vs_full_depth:.6f} learned_speedup={entry.learned_speedup:.6f}x",
            flush=True,
        )
    print("Epsilon sweep:", flush=True)
    for entry in summary.epsilon_sweep:
        print(
            f"  epsilon={entry.epsilon:.3f} avg_depth={entry.avg_depth:.4f} val_loss={entry.val_loss:.6f} "
            f"loss_hit={entry.loss_hit_vs_full_depth:.6f} learned_speedup={entry.learned_speedup:.6f}x "
            f"oracle_efficiency={entry.oracle_efficiency:.6f} oracle_speedup={entry.oracle_speedup:.6f}x",
            flush=True,
        )
    if summary.best_tradeoff is None:
        print(f"Best tradeoff (<={BEST_TRADEOFF_MAX_LOSS_HIT:.2f} nats loss hit): none", flush=True)
    else:
        entry = summary.best_tradeoff
        print(
            f"Best tradeoff (<={BEST_TRADEOFF_MAX_LOSS_HIT:.2f} nats loss hit): epsilon={entry.epsilon:.3f} "
            f"avg_depth={entry.avg_depth:.4f} val_loss={entry.val_loss:.6f} "
            f"loss_hit={entry.loss_hit_vs_full_depth:.6f} learned_speedup={entry.learned_speedup:.6f}x",
            flush=True,
        )


def threshold_metrics_payload(entry: ThresholdMetrics) -> dict[str, float]:
    return {
        "epsilon": round(entry.epsilon, 6),
        "avg_depth": round(entry.avg_depth, 6),
        "val_loss": round(entry.val_loss, 6),
        "loss_hit_vs_full_depth": round(entry.loss_hit_vs_full_depth, 6),
        "learned_speedup": round(entry.learned_speedup, 6),
        "oracle_efficiency": round(entry.oracle_efficiency, 6),
        "oracle_avg_depth": round(entry.oracle_avg_depth, 6),
        "oracle_speedup": round(entry.oracle_speedup, 6),
    }


def fixed_depth_metrics_payload(entry: FixedDepthMetrics) -> dict[str, float | int]:
    return {
        "depth": entry.depth,
        "avg_depth": round(entry.avg_depth, 6),
        "val_loss": round(entry.val_loss, 6),
        "loss_hit_vs_full_depth": round(entry.loss_hit_vs_full_depth, 6),
        "learned_speedup": round(entry.learned_speedup, 6),
    }


def write_report(
    report_path: Path,
    *,
    args: argparse.Namespace,
    config: ExperimentConfig,
    training: TrainingSummary,
    evaluation: EvaluationSummary,
    device: torch.device,
) -> None:
    payload = {
        "seed": args.seed,
        "device": device.type,
        "sanity_check_only": args.sanity_check_only,
        "config": asdict(config),
        "train_characters": args.train_characters,
        "val_characters": args.val_characters,
        "steps": training.steps,
        "batch_size": training.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "halt_lambda": args.halt_lambda,
        "halt_warmup_fraction": args.halt_warmup_fraction,
        "epsilon_sweep": list(EPSILON_SWEEP),
        "training": {
            "final_train_loss": round(training.final_train_loss, 6),
            "final_lm_loss": round(training.final_lm_loss, 6),
            "final_halt_loss": round(training.final_halt_loss, 6),
            "final_halt_lambda": round(training.final_halt_lambda, 6),
            "running_gain_pearson": [
                round(value, 6) if math.isfinite(value) else None for value in training.running_gain_pearson
            ],
            "wall_seconds": round(training.wall_seconds, 6),
        },
        "evaluation": {
            "full_depth_val_loss": round(evaluation.full_depth_val_loss, 6),
            "oracle": {
                "mean_loss_per_depth": [round(value, 6) for value in evaluation.oracle.mean_loss_per_depth],
                "mean_gain_per_depth": [round(value, 6) for value in evaluation.oracle.mean_gain_per_depth],
                "oracle_best_loss": round(evaluation.oracle.oracle_best_loss, 6),
                "oracle_depth_histogram": list(evaluation.oracle.oracle_depth_histogram),
                "fraction_harmed_by_full_depth": round(evaluation.oracle.fraction_harmed_by_full_depth, 6),
                "total_examples": evaluation.oracle.total_examples,
            },
            "gain_pearson_per_depth": [
                round(value, 6) if math.isfinite(value) else None for value in evaluation.gain_pearson_per_depth
            ],
            "fixed_depth_baselines": [
                fixed_depth_metrics_payload(entry) for entry in evaluation.fixed_depth_baselines
            ],
            "epsilon_sweep": [threshold_metrics_payload(entry) for entry in evaluation.epsilon_sweep],
            "best_tradeoff": None
            if evaluation.best_tradeoff is None
            else threshold_metrics_payload(evaluation.best_tradeoff),
        },
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(
    checkpoint_path: Path,
    *,
    model: RecurrentDepthLM,
    config: ExperimentConfig,
    args: argparse.Namespace,
    vocab_size: int,
) -> None:
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "seed": args.seed,
        "vocab_size": vocab_size,
        "train_characters": args.train_characters,
        "val_characters": args.val_characters,
    }
    torch.save(checkpoint_payload, checkpoint_path)


def main() -> None:
    args = parse_args()
    if args.learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {args.learning_rate}")
    if args.halt_lambda < 0.0:
        raise ValueError(f"halt_lambda must be non-negative, got {args.halt_lambda}")
    if not 0.0 <= args.halt_warmup_fraction <= 1.0:
        raise ValueError(
            f"halt_warmup_fraction must be between 0 and 1 inclusive, got {args.halt_warmup_fraction}"
        )
    if args.recurrent_iterations < 2:
        raise ValueError(f"recurrent_iterations must be at least 2, got {args.recurrent_iterations}")

    config = ExperimentConfig(recurrent_iterations=args.recurrent_iterations, d_model=args.d_model, ff_dim=args.d_model * 4)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.sanity_check_only:
        redirect_sanity_check_paths(args)

    prepare_output_paths(report_path=args.report_path, log_path=args.log_path)
    if not args.sanity_check_only:
        register_active_lock(
            experiment_name="halting_regression",
            variants=[f"recurrent_{config.recurrent_iterations}"],
            enabled=not args.no_lock,
        )

    append_log(
        args.log_path,
        {
            "stage": "run_started",
            "seed": args.seed,
            "device": device.type,
            "sanity_check_only": args.sanity_check_only,
            "steps": resolve_training_steps(args),
            "batch_size": resolve_batch_size(args),
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "halt_lambda": args.halt_lambda,
            "halt_warmup_fraction": args.halt_warmup_fraction,
            "epsilon_sweep": list(EPSILON_SWEEP),
            "train_characters": args.train_characters,
            "val_characters": args.val_characters,
            "config": asdict(config),
        },
    )

    set_seed(args.seed)
    train_dataset, val_dataset, vocab_size = load_dataset(
        context_size=config.context_size,
        train_characters=args.train_characters,
        val_characters=args.val_characters,
    )
    train_dataset = dataset_to_device(train_dataset, device)
    val_dataset = dataset_to_device(val_dataset, device)

    append_log(
        args.log_path,
        {
            "stage": "dataset_loaded",
            "train_examples": train_dataset[1].shape[0],
            "val_examples": val_dataset[1].shape[0],
            "vocab_size": vocab_size,
        },
    )

    model = build_model(vocab_size=vocab_size, config=config).to(device)
    training = train_model(
        model,
        train_dataset,
        steps=resolve_training_steps(args),
        batch_size=resolve_batch_size(args),
        learning_rate=args.learning_rate,
        halt_lambda=args.halt_lambda,
        halt_warmup_fraction=args.halt_warmup_fraction,
        device=device,
        log_path=args.log_path,
    )
    evaluation = evaluate_model(
        model,
        val_dataset,
        eval_batch_size=args.eval_batch_size,
        device=device,
    )

    print_evaluation(evaluation)
    write_report(
        args.report_path,
        args=args,
        config=config,
        training=training,
        evaluation=evaluation,
        device=device,
    )
    checkpoint_path = args.report_path.parent / "checkpoint.pt"
    if args.save_checkpoint:
        save_checkpoint(
            checkpoint_path,
            model=model,
            config=config,
            args=args,
            vocab_size=vocab_size,
        )
    append_log(
        args.log_path,
        {
            "stage": "run_completed",
            "report_path": str(args.report_path),
            "checkpoint_path": str(checkpoint_path) if args.save_checkpoint else None,
            "training": {
                "final_train_loss": round(training.final_train_loss, 6),
                "final_lm_loss": round(training.final_lm_loss, 6),
                "final_halt_loss": round(training.final_halt_loss, 6),
                "final_halt_lambda": round(training.final_halt_lambda, 6),
                "running_gain_pearson": [
                    round(value, 6) if math.isfinite(value) else None for value in training.running_gain_pearson
                ],
            },
            "evaluation": {
                "full_depth_val_loss": round(evaluation.full_depth_val_loss, 6),
                "gain_pearson_per_depth": [
                    round(value, 6) if math.isfinite(value) else None for value in evaluation.gain_pearson_per_depth
                ],
                "fixed_depth_baselines": [
                    fixed_depth_metrics_payload(entry) for entry in evaluation.fixed_depth_baselines
                ],
                "best_tradeoff": None
                if evaluation.best_tradeoff is None
                else threshold_metrics_payload(evaluation.best_tradeoff),
            },
        },
    )


if __name__ == "__main__":
    main()
