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
        "Run from the repo root with `.\\.venv\\Scripts\\python.exe -m runs.halting_aware_training ...` so `core` imports resolve cleanly."
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
DELTA = 0.01
PRINT_INTERVAL = 1_000
POSITIVE_RATE_EMA_DECAY = 0.95


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
    halt_positive_rate_ema: tuple[float, ...]
    wall_seconds: float


@dataclass(frozen=True)
class OracleMetrics:
    mean_loss_per_depth: tuple[float, ...]
    oracle_best_loss: float
    oracle_depth_histogram: tuple[int, ...]
    no_regret_depth_histogram: tuple[int, ...]
    mean_no_regret_depth: float
    oracle_speedup: float
    fraction_harmed_by_full_depth: float
    total_examples: int


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    avg_depth: float
    val_loss: float
    loss_hit_vs_full_depth: float
    learned_speedup: float
    oracle_efficiency: float


@dataclass(frozen=True)
class EvaluationSummary:
    oracle: OracleMetrics
    full_depth_val_loss: float
    halt_auroc_per_depth: tuple[float, ...]
    threshold_sweep: tuple[ThresholdMetrics, ...]
    best_tradeoff: ThresholdMetrics | None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "tinyshakespeare" / "artifacts" / "halting_aware_training"
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
    parser.add_argument("--halt-lambda", type=float, default=HALT_LAMBDA)
    parser.add_argument("--halt-warmup-fraction", type=float, default=HALT_WARMUP_FRACTION)
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--sanity-check-only", action="store_true")
    parser.add_argument("--no-lock", action="store_true")
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


def binary_auroc(scores: Tensor, labels: Tensor) -> float:
    flat_scores = scores.detach().flatten().float().cpu()
    flat_labels = labels.detach().flatten().to(torch.bool).cpu()
    positives = int(flat_labels.sum().item())
    negatives = int(flat_labels.numel() - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = torch.argsort(flat_scores, descending=True)
    sorted_scores = flat_scores[order]
    sorted_labels = flat_labels[order].to(torch.int64)
    distinct_indices = torch.where(sorted_scores[1:] != sorted_scores[:-1])[0]
    threshold_indices = torch.cat(
        [distinct_indices, torch.tensor([sorted_labels.numel() - 1], dtype=torch.int64)]
    )
    true_positives = torch.cumsum(sorted_labels, dim=0)[threshold_indices].to(torch.float64)
    false_positives = (threshold_indices + 1).to(torch.float64) - true_positives
    true_positive_rate = torch.cat([torch.zeros(1, dtype=torch.float64), true_positives / positives])
    false_positive_rate = torch.cat([torch.zeros(1, dtype=torch.float64), false_positives / negatives])
    return float(torch.trapz(true_positive_rate, false_positive_rate).item())


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
    halt_logits: list[Tensor] = []
    for depth_index, iteration_hidden in enumerate(iteration_states, start=1):
        last_hidden = iteration_hidden[:, -1, :]
        logits = model.lm_logits_from_hidden(last_hidden)
        per_depth_losses.append(F.cross_entropy(logits.float(), batch_targets, reduction="none"))
        if depth_index < model.config.recurrent_iterations:
            halt_logits.append(model.predicted_gain_from_hidden(last_hidden, depth_index=depth_index))
    return torch.stack(per_depth_losses, dim=1), torch.stack(halt_logits, dim=1)


def train_model(
    model: RecurrentDepthLM,
    train_dataset: tuple[Tensor, Tensor],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    halt_lambda: float,
    halt_warmup_fraction: float,
    delta: float,
    device: torch.device,
    log_path: Path,
) -> TrainingSummary:
    optimizer = build_optimizer(model, learning_rate=learning_rate, device=device)
    depth_count = model.config.recurrent_iterations
    positive_rate_ema = torch.full((depth_count - 1,), 0.5, device=device)
    running_scores: list[list[Tensor]] = [[] for _ in range(depth_count - 1)]
    running_labels: list[list[Tensor]] = [[] for _ in range(depth_count - 1)]
    started_at = perf_counter()
    last_total_loss = float("nan")
    last_lm_loss = float("nan")
    last_halt_loss = float("nan")
    last_lambda = 0.0
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
            per_depth_losses, halt_logits = compute_depth_outputs(model, batch_inputs, batch_targets)
            detached_losses = per_depth_losses.detach()
            full_depth_losses = detached_losses[:, -1:]
            halt_labels = (detached_losses[:, :-1] <= (full_depth_losses + delta)).float()
            batch_positive_rate = halt_labels.mean(dim=0)
            positive_rate_ema = (
                POSITIVE_RATE_EMA_DECAY * positive_rate_ema + (1.0 - POSITIVE_RATE_EMA_DECAY) * batch_positive_rate
            )
            clamped_positive_rate = positive_rate_ema.clamp(1e-3, 1.0 - 1e-3)
            pos_weight = ((1.0 - clamped_positive_rate) / clamped_positive_rate).detach()
            lm_loss = per_depth_losses[:, -1].mean()
            halt_loss = F.binary_cross_entropy_with_logits(halt_logits.float(), halt_labels, pos_weight=pos_weight)
            total_loss = lm_loss + step_halt_lambda * halt_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        last_total_loss = float(total_loss.item())
        last_lm_loss = float(lm_loss.item())
        last_halt_loss = float(halt_loss.item())
        last_lambda = step_halt_lambda

        halt_probabilities = torch.sigmoid(halt_logits.detach().float())
        for depth_index in range(depth_count - 1):
            running_scores[depth_index].append(halt_probabilities[:, depth_index].cpu())
            running_labels[depth_index].append(halt_labels[:, depth_index].cpu())

        if step % print_interval != 0 and step != steps:
            continue

        running_auroc = [
            binary_auroc(torch.cat(running_scores[depth_index]), torch.cat(running_labels[depth_index]))
            for depth_index in range(depth_count - 1)
        ]
        print(
            f"step={step:05d}/{steps} train_loss={last_total_loss:.4f} lm_loss={last_lm_loss:.4f} "
            f"halt_loss={last_halt_loss:.4f} halt_lambda={last_lambda:.4f}",
            flush=True,
        )
        for depth_index, (rate, auroc) in enumerate(zip(positive_rate_ema.tolist(), running_auroc, strict=True), start=1):
            print(
                f"  depth {depth_index}: halt_pos_rate_ema={rate:.4f} running_auroc={auroc:.4f}",
                flush=True,
            )
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
                "halt_positive_rate_ema": [round(value, 6) for value in positive_rate_ema.tolist()],
                "running_halt_auroc": [round(value, 6) if math.isfinite(value) else None for value in running_auroc],
            },
        )

    return TrainingSummary(
        steps=steps,
        batch_size=batch_size,
        final_train_loss=last_total_loss,
        final_lm_loss=last_lm_loss,
        final_halt_loss=last_halt_loss,
        final_halt_lambda=last_lambda,
        halt_positive_rate_ema=tuple(float(value) for value in positive_rate_ema.tolist()),
        wall_seconds=perf_counter() - started_at,
    )


def compute_oracle_metrics(per_depth_losses: Tensor, *, delta: float) -> OracleMetrics:
    depth_count = per_depth_losses.shape[1]
    total_examples = per_depth_losses.shape[0]
    if total_examples == 0:
        raise RuntimeError("validation dataset was empty")

    mean_loss_per_depth = tuple(float(value) for value in per_depth_losses.mean(dim=0).tolist())
    oracle_best_losses, oracle_depth_indices = per_depth_losses.min(dim=1)
    full_depth_losses = per_depth_losses[:, -1]
    no_regret_mask = per_depth_losses <= (full_depth_losses.unsqueeze(1) + delta)
    if not bool(no_regret_mask[:, -1].all().item()):
        raise RuntimeError("full depth must always satisfy the no-regret condition")
    no_regret_depth_indices = no_regret_mask.float().argmax(dim=1)
    mean_no_regret_depth = float((no_regret_depth_indices + 1).float().mean().item())
    return OracleMetrics(
        mean_loss_per_depth=mean_loss_per_depth,
        oracle_best_loss=float(oracle_best_losses.mean().item()),
        oracle_depth_histogram=tuple(
            int(value) for value in torch.bincount(oracle_depth_indices.cpu(), minlength=depth_count).tolist()
        ),
        no_regret_depth_histogram=tuple(
            int(value) for value in torch.bincount(no_regret_depth_indices.cpu(), minlength=depth_count).tolist()
        ),
        mean_no_regret_depth=mean_no_regret_depth,
        oracle_speedup=depth_count / mean_no_regret_depth,
        fraction_harmed_by_full_depth=float((full_depth_losses > oracle_best_losses + 1e-12).float().mean().item()),
        total_examples=total_examples,
    )


@torch.inference_mode()
def evaluate_model(
    model: RecurrentDepthLM,
    dataset: tuple[Tensor, Tensor],
    *,
    eval_batch_size: int,
    delta: float,
    device: torch.device,
) -> EvaluationSummary:
    model.eval()
    inputs, targets = dataset
    all_per_depth_losses: list[Tensor] = []
    all_halt_probabilities: list[Tensor] = []

    for start in range(0, targets.shape[0], eval_batch_size):
        stop = min(start + eval_batch_size, targets.shape[0])
        batch_inputs = inputs[start:stop]
        batch_targets = targets[start:stop]
        with autocast_context(device):
            per_depth_losses, halt_logits = compute_depth_outputs(model, batch_inputs, batch_targets)
        all_per_depth_losses.append(per_depth_losses.float().cpu())
        all_halt_probabilities.append(torch.sigmoid(halt_logits.float()).cpu())

    per_depth_losses = torch.cat(all_per_depth_losses, dim=0)
    halt_probabilities = torch.cat(all_halt_probabilities, dim=0)
    oracle = compute_oracle_metrics(per_depth_losses, delta=delta)
    full_depth_val_loss = float(per_depth_losses[:, -1].mean().item())
    halt_labels = (per_depth_losses[:, :-1] <= (per_depth_losses[:, -1:] + delta)).float()
    halt_auroc_per_depth = tuple(
        binary_auroc(halt_probabilities[:, depth_index], halt_labels[:, depth_index])
        for depth_index in range(halt_probabilities.shape[1])
    )

    threshold_sweep: list[ThresholdMetrics] = []
    depth_count = model.config.recurrent_iterations
    example_indices = torch.arange(per_depth_losses.shape[0], dtype=torch.int64)
    for threshold_step in range(5, 100, 5):
        threshold = threshold_step / 100.0
        early_halt_mask = halt_probabilities >= threshold
        any_halt = early_halt_mask.any(dim=1)
        chosen_depth_indices = torch.where(
            any_halt,
            early_halt_mask.float().argmax(dim=1),
            torch.full((per_depth_losses.shape[0],), depth_count - 1, dtype=torch.int64),
        )
        chosen_depths = chosen_depth_indices + 1
        halted_losses = per_depth_losses[example_indices, chosen_depth_indices]
        avg_depth = float(chosen_depths.float().mean().item())
        val_loss = float(halted_losses.mean().item())
        learned_speedup = depth_count / avg_depth
        oracle_efficiency = 0.0
        if oracle.oracle_speedup > 1.0:
            oracle_efficiency = (learned_speedup - 1.0) / (oracle.oracle_speedup - 1.0)
        threshold_sweep.append(
            ThresholdMetrics(
                threshold=threshold,
                avg_depth=avg_depth,
                val_loss=val_loss,
                loss_hit_vs_full_depth=val_loss - full_depth_val_loss,
                learned_speedup=learned_speedup,
                oracle_efficiency=oracle_efficiency,
            )
        )

    acceptable_thresholds = [entry for entry in threshold_sweep if entry.loss_hit_vs_full_depth <= 0.02 + 1e-12]
    best_tradeoff = None
    if len(acceptable_thresholds) > 0:
        best_tradeoff = max(
            acceptable_thresholds,
            key=lambda entry: (entry.learned_speedup, -entry.loss_hit_vs_full_depth, -entry.threshold),
        )

    return EvaluationSummary(
        oracle=oracle,
        full_depth_val_loss=full_depth_val_loss,
        halt_auroc_per_depth=halt_auroc_per_depth,
        threshold_sweep=tuple(threshold_sweep),
        best_tradeoff=best_tradeoff,
    )


def print_evaluation(summary: EvaluationSummary, *, delta: float) -> None:
    oracle = summary.oracle
    print("=== Halting-aware evaluation ===", flush=True)
    print("Mean loss per depth:", flush=True)
    for depth, loss in enumerate(oracle.mean_loss_per_depth, start=1):
        print(f"  depth {depth}: {loss:.6f}", flush=True)
    print(f"Full-depth val loss: {summary.full_depth_val_loss:.6f}", flush=True)
    print(f"Oracle-best loss: {oracle.oracle_best_loss:.6f}", flush=True)
    print(f"Oracle speedup at delta={delta:.2f}: {oracle.oracle_speedup:.6f}x", flush=True)
    print(f"Fraction harmed by full depth: {oracle.fraction_harmed_by_full_depth:.6f}", flush=True)
    print("Per-depth halt AUROC:", flush=True)
    for depth, auroc in enumerate(summary.halt_auroc_per_depth, start=1):
        print(f"  depth {depth}: {auroc:.6f}", flush=True)
    print("Threshold sweep:", flush=True)
    for entry in summary.threshold_sweep:
        print(
            f"  tau={entry.threshold:.2f} avg_depth={entry.avg_depth:.4f} val_loss={entry.val_loss:.6f} "
            f"loss_hit={entry.loss_hit_vs_full_depth:.6f} learned_speedup={entry.learned_speedup:.6f}x "
            f"oracle_efficiency={entry.oracle_efficiency:.6f}",
            flush=True,
        )
    if summary.best_tradeoff is None:
        print("Best tradeoff (<=0.02 nats loss hit): none", flush=True)
    else:
        entry = summary.best_tradeoff
        print(
            f"Best tradeoff (<=0.02 nats loss hit): tau={entry.threshold:.2f} avg_depth={entry.avg_depth:.4f} "
            f"val_loss={entry.val_loss:.6f} loss_hit={entry.loss_hit_vs_full_depth:.6f} "
            f"learned_speedup={entry.learned_speedup:.6f}x",
            flush=True,
        )


def threshold_metrics_payload(entry: ThresholdMetrics) -> dict[str, float]:
    return {
        "threshold": round(entry.threshold, 6),
        "avg_depth": round(entry.avg_depth, 6),
        "val_loss": round(entry.val_loss, 6),
        "loss_hit_vs_full_depth": round(entry.loss_hit_vs_full_depth, 6),
        "learned_speedup": round(entry.learned_speedup, 6),
        "oracle_efficiency": round(entry.oracle_efficiency, 6),
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
        "delta": args.delta,
        "training": {
            "final_train_loss": round(training.final_train_loss, 6),
            "final_lm_loss": round(training.final_lm_loss, 6),
            "final_halt_loss": round(training.final_halt_loss, 6),
            "final_halt_lambda": round(training.final_halt_lambda, 6),
            "halt_positive_rate_ema": [round(value, 6) for value in training.halt_positive_rate_ema],
            "wall_seconds": round(training.wall_seconds, 6),
        },
        "evaluation": {
            "full_depth_val_loss": round(evaluation.full_depth_val_loss, 6),
            "oracle": {
                "mean_loss_per_depth": [round(value, 6) for value in evaluation.oracle.mean_loss_per_depth],
                "oracle_best_loss": round(evaluation.oracle.oracle_best_loss, 6),
                "oracle_depth_histogram": list(evaluation.oracle.oracle_depth_histogram),
                "no_regret_depth_histogram": list(evaluation.oracle.no_regret_depth_histogram),
                "mean_no_regret_depth": round(evaluation.oracle.mean_no_regret_depth, 6),
                "oracle_speedup": round(evaluation.oracle.oracle_speedup, 6),
                "fraction_harmed_by_full_depth": round(evaluation.oracle.fraction_harmed_by_full_depth, 6),
                "total_examples": evaluation.oracle.total_examples,
            },
            "halt_auroc_per_depth": [
                round(value, 6) if math.isfinite(value) else None for value in evaluation.halt_auroc_per_depth
            ],
            "threshold_sweep": [threshold_metrics_payload(entry) for entry in evaluation.threshold_sweep],
            "best_tradeoff": None
            if evaluation.best_tradeoff is None
            else threshold_metrics_payload(evaluation.best_tradeoff),
        },
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    if args.delta < 0.0:
        raise ValueError(f"delta must be non-negative, got {args.delta}")
    if args.recurrent_iterations < 2:
        raise ValueError(f"recurrent_iterations must be at least 2, got {args.recurrent_iterations}")

    config = ExperimentConfig(recurrent_iterations=args.recurrent_iterations)
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
            experiment_name="halting_aware_training",
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
            "delta": args.delta,
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
        delta=args.delta,
        device=device,
        log_path=args.log_path,
    )
    evaluation = evaluate_model(
        model,
        val_dataset,
        eval_batch_size=args.eval_batch_size,
        delta=args.delta,
        device=device,
    )

    print_evaluation(evaluation, delta=args.delta)
    write_report(
        args.report_path,
        args=args,
        config=config,
        training=training,
        evaluation=evaluation,
        device=device,
    )
    append_log(
        args.log_path,
        {
            "stage": "run_completed",
            "report_path": str(args.report_path),
            "training": {
                "final_train_loss": round(training.final_train_loss, 6),
                "final_lm_loss": round(training.final_lm_loss, 6),
                "final_halt_loss": round(training.final_halt_loss, 6),
                "final_halt_lambda": round(training.final_halt_lambda, 6),
            },
            "evaluation": {
                "full_depth_val_loss": round(evaluation.full_depth_val_loss, 6),
                "oracle_speedup": round(evaluation.oracle.oracle_speedup, 6),
                "halt_auroc_per_depth": [
                    round(value, 6) if math.isfinite(value) else None for value in evaluation.halt_auroc_per_depth
                ],
                "best_tradeoff": None
                if evaluation.best_tradeoff is None
                else threshold_metrics_payload(evaluation.best_tradeoff),
            },
        },
    )


if __name__ == "__main__":
    main()
