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
D_MODEL = 256
FEEDFORWARD_DIM = 512


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    d_model: int
    feedforward_dim: int
    num_blocks: int
    rates: tuple[int, ...]
    readout_mode: str
    token_injection: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = (
        repo_root
        / "experiments"
        / "fixed_multi_rate"
        / "artifacts"
        / "token_injection_sanity"
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


def build_single_block(*, device: torch.device) -> ParallelDiagonalModel:
    return ParallelDiagonalModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        num_blocks=1,
        rates=(1,),
        readout_mode="last",
        token_injection="block0",
    ).to(device)


def build_four_block_all(*, device: torch.device) -> ParallelDiagonalModel:
    return ParallelDiagonalModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        num_blocks=4,
        rates=(1, 1, 1, 1),
        readout_mode="all",
        token_injection="all",
    ).to(device)


def build_four_block_block0(*, device: torch.device) -> ParallelDiagonalModel:
    return ParallelDiagonalModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        num_blocks=4,
        rates=(1, 1, 1, 1),
        readout_mode="all",
        token_injection="block0",
    ).to(device)


def variant_specs() -> dict[str, VariantSpec]:
    return {
        "single_block": VariantSpec(
            key="single_block",
            label="parallel_diagonal_1x256_ff512_rate1_block0",
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=1,
            rates=(1,),
            readout_mode="last",
            token_injection="block0",
        ),
        "four_block_all": VariantSpec(
            key="four_block_all",
            label="parallel_diagonal_4x256_ff512_rate1111_all",
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=4,
            rates=(1, 1, 1, 1),
            readout_mode="all",
            token_injection="all",
        ),
        "four_block_block0": VariantSpec(
            key="four_block_block0",
            label="parallel_diagonal_4x256_ff512_rate1111_block0",
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=4,
            rates=(1, 1, 1, 1),
            readout_mode="all",
            token_injection="block0",
        ),
    }


def materialize_batch(
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return train_inputs[indices], train_targets[indices]


def checkpoint_metrics(
    model: torch.nn.Module,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    batch_size: int,
    step: int,
) -> dict[str, float | int]:
    metrics = evaluate_model(model, val_inputs, val_targets, batch_size=batch_size)
    return {
        "step": step,
        "val_loss": round(metrics["loss"], 6),
        "val_accuracy": round(metrics["accuracy"], 6),
    }


def append_log(log_path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def mean_rounded(values: list[float]) -> float:
    return round(mean(values), 6)


def train_single_variant(
    *,
    seed: int,
    variant_key: str,
    spec: VariantSpec,
    builder,
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
    model = builder(device=device)
    optimizer = capturable_adamw(model, lr=args.learning_rate)
    trainer = GraphTrainer(
        model,
        optimizer,
        batch_size=args.batch_size,
        seq_len=CONTEXT_SIZE,
        device=device,
    )

    checkpoints = [
        checkpoint_metrics(
            model,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=0,
        )
    ]
    parameter_count = count_parameters(model)
    append_log(
        args.log_path,
        {
            "stage": "variant_started",
            "seed": seed,
            "variant": variant_key,
            "label": spec.label,
            "parameter_count": parameter_count,
            "initial_checkpoint": checkpoints[-1],
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
        "parameter_count": parameter_count,
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
        wall_seconds = [run["wall_seconds"] for run in runs]
        summary[key] = {
            "label": specs[key].label,
            "token_injection": specs[key].token_injection,
            "num_runs": len(runs),
            "mean_final_val_loss": mean_rounded(final_losses),
            "mean_final_val_accuracy": mean_rounded(final_accuracies),
            "mean_best_val_loss": mean_rounded(best_losses),
            "mean_best_val_accuracy": mean_rounded(best_accuracies),
            "mean_wall_seconds": mean_rounded(wall_seconds),
            "runs": runs,
        }
    return summary


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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/token_injection_sanity.py.")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text("", encoding="utf-8")

    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}.")
    if args.eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {args.eval_batch_size}.")

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

    builders = {
        "single_block": build_single_block,
        "four_block_all": build_four_block_all,
        "four_block_block0": build_four_block_block0,
    }
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
        for variant_key, builder in builders.items():
            per_seed_results.append(
                train_single_variant(
                    seed=seed,
                    variant_key=variant_key,
                    spec=specs[variant_key],
                    builder=builder,
                    args=args,
                    device=device,
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    val_inputs=val_inputs,
                    val_targets=val_targets,
                )
            )

    wall_seconds = perf_counter() - overall_started_at
    git_status_short = current_git_status_short()
    by_variant = summarize_results(per_seed_results=per_seed_results, specs=specs)
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
        "summary_by_variant": by_variant,
        "comparison": {
            "mean_final_val_loss_delta_C_minus_A": round(
                by_variant["four_block_block0"]["mean_final_val_loss"]
                - by_variant["single_block"]["mean_final_val_loss"],
                6,
            ),
            "mean_final_val_accuracy_delta_C_minus_A": round(
                by_variant["four_block_block0"]["mean_final_val_accuracy"]
                - by_variant["single_block"]["mean_final_val_accuracy"],
                6,
            ),
            "mean_final_val_loss_delta_C_minus_B": round(
                by_variant["four_block_block0"]["mean_final_val_loss"]
                - by_variant["four_block_all"]["mean_final_val_loss"],
                6,
            ),
            "mean_final_val_accuracy_delta_C_minus_B": round(
                by_variant["four_block_block0"]["mean_final_val_accuracy"]
                - by_variant["four_block_all"]["mean_final_val_accuracy"],
                6,
            ),
        },
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
                }
                for key, value in by_variant.items()
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
