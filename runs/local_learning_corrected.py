from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from core.fixed_window_char import load_dataset, set_seed
from core.model import ParallelDiagonalModel, count_parameters
from core.training import (
    GraphTrainer,
    capturable_adamw,
    current_git_sha,
    current_git_status_short,
    evaluate_model,
    fixed_step_indices,
    write_json,
)

CONTEXT_SIZE = 32
VOCAB_SIZE = 67
TRAINING_STEPS = 20_000
EVAL_INTERVAL = 1_000
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 3e-4
DEFAULT_SEEDS = (42, 43, 44)
WARMUP_STEPS = 3
NUM_BLOCKS = 4
D_MODEL = 256
FEEDFORWARD_DIM = 512
ABLATION_LOGIT = -100.0


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    rates: tuple[int, ...]
    detach_lateral: bool
    d_model: int
    feedforward_dim: int
    num_blocks: int
    readout_mode: str
    token_injection: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = (
        repo_root
        / "experiments"
        / "fixed_multi_rate"
        / "artifacts"
        / "local_learning_corrected"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.jsonl")
    return parser.parse_args()


def variant_specs() -> dict[str, VariantSpec]:
    return {
        "rate1_full": VariantSpec(
            key="rate1_full",
            label="local_learning_corrected_rate1111_full",
            rates=(1, 1, 1, 1),
            detach_lateral=False,
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=NUM_BLOCKS,
            readout_mode="all",
            token_injection="block0",
        ),
        "rate1_detached": VariantSpec(
            key="rate1_detached",
            label="local_learning_corrected_rate1111_detached",
            rates=(1, 1, 1, 1),
            detach_lateral=True,
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=NUM_BLOCKS,
            readout_mode="all",
            token_injection="block0",
        ),
        "multirate_full": VariantSpec(
            key="multirate_full",
            label="local_learning_corrected_rate1248_full",
            rates=(1, 2, 4, 8),
            detach_lateral=False,
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=NUM_BLOCKS,
            readout_mode="all",
            token_injection="block0",
        ),
        "multirate_detached": VariantSpec(
            key="multirate_detached",
            label="local_learning_corrected_rate1248_detached",
            rates=(1, 2, 4, 8),
            detach_lateral=True,
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=NUM_BLOCKS,
            readout_mode="all",
            token_injection="block0",
        ),
    }


def build_model(*, device: torch.device, spec: VariantSpec) -> ParallelDiagonalModel:
    return ParallelDiagonalModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=spec.d_model,
        feedforward_dim=spec.feedforward_dim,
        num_blocks=spec.num_blocks,
        rates=spec.rates,
        readout_mode=spec.readout_mode,
        token_injection=spec.token_injection,
        detach_lateral=spec.detach_lateral,
    ).to(device)


def materialize_batch(
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return train_inputs[indices], train_targets[indices]


def readout_weights(model: ParallelDiagonalModel) -> list[float]:
    weights = model.mix_coefficients()["readout_weights"]
    if weights is None:
        raise RuntimeError("Expected readout weights for readout_mode='all'.")
    return [round(float(weight), 6) for weight in weights]


def evaluate_with_block_ablated(
    model: ParallelDiagonalModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    block_index: int,
) -> dict[str, float]:
    if model.readout_logits is None:
        raise RuntimeError("Expected readout_logits for block ablation.")
    with torch.no_grad():
        original_logits = model.readout_logits.detach().clone()
        model.readout_logits.copy_(original_logits)
        model.readout_logits[block_index] = ABLATION_LOGIT
    try:
        return evaluate_model(model, inputs, targets, batch_size=batch_size)
    finally:
        with torch.no_grad():
            model.readout_logits.copy_(original_logits)


def ablation_losses(
    model: ParallelDiagonalModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
) -> list[float]:
    return [
        round(
            evaluate_with_block_ablated(
                model,
                inputs,
                targets,
                batch_size=batch_size,
                block_index=block_index,
            )["loss"],
            6,
        )
        for block_index in range(model.num_blocks)
    ]


def checkpoint_metrics(
    model: ParallelDiagonalModel,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    batch_size: int,
    step: int,
) -> dict[str, float | int | list[float]]:
    metrics = evaluate_model(model, val_inputs, val_targets, batch_size=batch_size)
    return {
        "step": step,
        "val_loss": round(metrics["loss"], 6),
        "val_accuracy": round(metrics["accuracy"], 6),
        "readout_weights": readout_weights(model),
        "ablation_losses": ablation_losses(
            model,
            val_inputs,
            val_targets,
            batch_size=batch_size,
        ),
    }


def append_log(log_path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def mean_rounded(values: list[float]) -> float:
    return round(mean(values), 6)


def mean_vector_rounded(vectors: list[list[float]]) -> list[float]:
    if len(vectors) == 0:
        raise ValueError("Expected at least one vector.")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ValueError("All vectors must have the same length.")
    return [round(mean(vector[index] for vector in vectors), 6) for index in range(width)]


def train_single_variant(
    *,
    seed: int,
    variant_key: str,
    spec: VariantSpec,
    args: argparse.Namespace,
    device: torch.device,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
) -> dict[str, object]:
    batch_schedule = fixed_step_indices(
        train_inputs.shape[0],
        steps=args.training_steps,
        batch_size=args.batch_size,
        seed=seed,
        device=device,
    )
    warmup_batches = [
        materialize_batch(train_inputs, train_targets, indices)
        for indices in batch_schedule[:WARMUP_STEPS]
    ]

    set_seed(seed)
    model = build_model(device=device, spec=spec)
    optimizer = capturable_adamw(model, lr=args.learning_rate)
    trainer = GraphTrainer(
        model,
        optimizer,
        batch_size=args.batch_size,
        seq_len=CONTEXT_SIZE,
        device=device,
    )

    initial_checkpoint = checkpoint_metrics(
        model,
        val_inputs,
        val_targets,
        batch_size=args.eval_batch_size,
        step=0,
    )
    checkpoints = [initial_checkpoint]
    parameter_count = count_parameters(model)
    append_log(
        args.log_path,
        {
            "stage": "variant_started",
            "seed": seed,
            "variant": variant_key,
            "label": spec.label,
            "parameter_count": parameter_count,
            "initial_checkpoint": initial_checkpoint,
        },
    )

    started_at = perf_counter()
    trainer.capture(warmup_batches)
    last_loss = trainer.static_loss.detach().clone()

    for zero_based_index, indices in enumerate(batch_schedule[WARMUP_STEPS:], start=WARMUP_STEPS):
        batch_input, batch_target = materialize_batch(train_inputs, train_targets, indices)
        last_loss = trainer.step(batch_input, batch_target)
        step = zero_based_index + 1
        if step % args.eval_interval != 0 and step != args.training_steps:
            continue

        trainer.synchronize()
        checkpoint = checkpoint_metrics(
            model,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=step,
        )
        checkpoints.append(checkpoint)
        append_log(
            args.log_path,
            {
                "stage": "checkpoint",
                "seed": seed,
                "variant": variant_key,
                **checkpoint,
            },
        )

    trainer.synchronize()
    wall_seconds = perf_counter() - started_at
    result = {
        "seed": seed,
        "variant": variant_key,
        "label": spec.label,
        "class_name": "ParallelDiagonalModel",
        "d_model": spec.d_model,
        "feedforward_dim": spec.feedforward_dim,
        "num_blocks": spec.num_blocks,
        "rates": list(spec.rates),
        "readout_mode": spec.readout_mode,
        "token_injection": spec.token_injection,
        "detach_lateral": spec.detach_lateral,
        "parameter_count": parameter_count,
        "initial_checkpoint": initial_checkpoint,
        "checkpoints": checkpoints,
        "best_checkpoint": min(checkpoints, key=lambda checkpoint: checkpoint["val_loss"]),
        "final_checkpoint": checkpoints[-1],
        "final_training_loss": round(last_loss.item(), 6),
        "wall_seconds": round(wall_seconds, 6),
        "mix_coefficients": model.mix_coefficients(),
    }
    append_log(
        args.log_path,
        {
            "stage": "variant_done",
            "seed": seed,
            "variant": variant_key,
            "final_checkpoint": result["final_checkpoint"],
            "final_training_loss": result["final_training_loss"],
            "wall_seconds": result["wall_seconds"],
        },
    )

    del trainer
    del optimizer
    del model
    del warmup_batches
    del batch_schedule
    gc.collect()
    torch.cuda.empty_cache()
    return result


def summarize_results(
    *,
    per_seed_results: list[dict[str, object]],
    specs: dict[str, VariantSpec],
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {key: [] for key in specs}
    for result in per_seed_results:
        grouped[result["variant"]].append(result)

    summary: dict[str, object] = {}
    for key, runs in grouped.items():
        final_losses = [run["final_checkpoint"]["val_loss"] for run in runs]
        final_accuracies = [run["final_checkpoint"]["val_accuracy"] for run in runs]
        best_losses = [run["best_checkpoint"]["val_loss"] for run in runs]
        best_accuracies = [run["best_checkpoint"]["val_accuracy"] for run in runs]
        final_training_losses = [run["final_training_loss"] for run in runs]
        wall_seconds = [run["wall_seconds"] for run in runs]
        final_readout_weights = [run["final_checkpoint"]["readout_weights"] for run in runs]
        final_ablation_losses = [run["final_checkpoint"]["ablation_losses"] for run in runs]
        summary[key] = {
            "label": specs[key].label,
            "rates": list(specs[key].rates),
            "detach_lateral": specs[key].detach_lateral,
            "num_runs": len(runs),
            "mean_final_val_loss": mean_rounded(final_losses),
            "mean_final_val_accuracy": mean_rounded(final_accuracies),
            "mean_best_val_loss": mean_rounded(best_losses),
            "mean_best_val_accuracy": mean_rounded(best_accuracies),
            "mean_final_training_loss": mean_rounded(final_training_losses),
            "mean_wall_seconds": mean_rounded(wall_seconds),
            "mean_final_readout_weights": mean_vector_rounded(final_readout_weights),
            "mean_final_ablation_losses": mean_vector_rounded(final_ablation_losses),
            "runs": runs,
        }
    return summary


def comparison(summary_by_variant: dict[str, object]) -> dict[str, float]:
    rate1_full = summary_by_variant["rate1_full"]
    rate1_detached = summary_by_variant["rate1_detached"]
    multirate_full = summary_by_variant["multirate_full"]
    multirate_detached = summary_by_variant["multirate_detached"]
    return {
        "mean_final_val_loss_delta_rate1_detached_minus_rate1_full": round(
            rate1_detached["mean_final_val_loss"] - rate1_full["mean_final_val_loss"],
            6,
        ),
        "mean_final_val_loss_delta_multirate_detached_minus_multirate_full": round(
            multirate_detached["mean_final_val_loss"] - multirate_full["mean_final_val_loss"],
            6,
        ),
        "mean_final_val_accuracy_delta_rate1_detached_minus_rate1_full": round(
            rate1_detached["mean_final_val_accuracy"] - rate1_full["mean_final_val_accuracy"],
            6,
        ),
        "mean_final_val_accuracy_delta_multirate_detached_minus_multirate_full": round(
            multirate_detached["mean_final_val_accuracy"] - multirate_full["mean_final_val_accuracy"],
            6,
        ),
    }


def main() -> int:
    args = parse_args()
    if len(args.seeds) == 0:
        raise ValueError("At least one seed is required.")
    if args.training_steps < WARMUP_STEPS:
        raise ValueError(
            f"training_steps must be at least {WARMUP_STEPS} so GraphTrainer.capture() can warm up."
        )
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}.")
    if args.eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {args.eval_batch_size}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/local_learning_corrected.py.")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text("", encoding="utf-8")

    device = torch.device("cuda")
    set_seed(args.seeds[0])
    (train_inputs, train_targets), (val_inputs, val_targets), vocab_size = load_dataset(
        context_size=CONTEXT_SIZE
    )
    if vocab_size > VOCAB_SIZE:
        raise ValueError(
            f"Loaded dataset vocab_size {vocab_size} exceeds configured model vocab_size {VOCAB_SIZE}."
        )

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    val_inputs = val_inputs.to(device)
    val_targets = val_targets.to(device)

    specs = variant_specs()
    append_log(
        args.log_path,
        {
            "stage": "experiment_started",
            "seeds": args.seeds,
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "device": str(device),
        },
    )

    overall_started_at = perf_counter()
    per_seed_results: list[dict[str, object]] = []
    for seed in args.seeds:
        append_log(args.log_path, {"stage": "seed_started", "seed": seed})
        for variant_key, spec in specs.items():
            per_seed_results.append(
                train_single_variant(
                    seed=seed,
                    variant_key=variant_key,
                    spec=spec,
                    args=args,
                    device=device,
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    val_inputs=val_inputs,
                    val_targets=val_targets,
                )
            )
        append_log(args.log_path, {"stage": "seed_done", "seed": seed})

    wall_seconds = perf_counter() - overall_started_at
    git_status_short = current_git_status_short()
    summary_by_variant = summarize_results(per_seed_results=per_seed_results, specs=specs)
    report = {
        "config": {
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
            "warmup_steps": WARMUP_STEPS,
            "context_size": CONTEXT_SIZE,
            "dataset_vocab_size": vocab_size,
            "model_vocab_size": VOCAB_SIZE,
            "trainer": "GraphTrainer",
            "ablation_logit": ABLATION_LOGIT,
        },
        "environment": {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": git_status_short == [],
            "git_status_short": git_status_short,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
        },
        "timing": {
            "overall_wall_seconds": round(wall_seconds, 6),
        },
        "variants": {key: asdict(spec) for key, spec in specs.items()},
        "per_seed_results": per_seed_results,
        "summary_by_variant": summary_by_variant,
        "comparison": comparison(summary_by_variant),
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "done",
            "report_path": str(args.report_path),
            "summary_by_variant": {
                key: {
                    "mean_final_val_loss": value["mean_final_val_loss"],
                    "mean_final_val_accuracy": value["mean_final_val_accuracy"],
                    "mean_final_readout_weights": value["mean_final_readout_weights"],
                    "mean_final_ablation_losses": value["mean_final_ablation_losses"],
                }
                for key, value in summary_by_variant.items()
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
