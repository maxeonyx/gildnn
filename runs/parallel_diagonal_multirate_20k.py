from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch

from core.fixed_window_char import load_dataset, set_seed
from core.model import MultiRateResidualModel, ParallelDiagonalModel, count_parameters
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
SEED = 42
WARMUP_STEPS = 3
NUM_BLOCKS = 4
RATES = (1, 2, 4, 8)
D_MODEL = 128
FEEDFORWARD_DIM = 256


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    class_name: str
    d_model: int
    feedforward_dim: int
    num_blocks: int
    rates: tuple[int, ...]
    readout_mode: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = (
        repo_root
        / "experiments"
        / "fixed_multi_rate"
        / "artifacts"
        / "parallel_diagonal_multirate_20k"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.jsonl")
    return parser.parse_args()


def build_control_model(*, device: torch.device) -> MultiRateResidualModel:
    return MultiRateResidualModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        temporal_window=4,
        num_heads=4,
        rates=RATES,
    ).to(device)


def build_parallel_all_rate_1_model(*, device: torch.device) -> ParallelDiagonalModel:
    return ParallelDiagonalModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        num_blocks=NUM_BLOCKS,
        rates=(1, 1, 1, 1),
        readout_mode="all",
    ).to(device)


def build_parallel_multirate_model(*, device: torch.device) -> ParallelDiagonalModel:
    return ParallelDiagonalModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        num_blocks=NUM_BLOCKS,
        rates=RATES,
        readout_mode="all",
    ).to(device)


def variant_specs() -> dict[str, VariantSpec]:
    return {
        "control": VariantSpec(
            key="control",
            label="multi_rate_4x128_ff256_1248_control",
            class_name="MultiRateResidualModel",
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=NUM_BLOCKS,
            rates=RATES,
            readout_mode="last",
        ),
        "parallel_diagonal_all_rate_1": VariantSpec(
            key="parallel_diagonal_all_rate_1",
            label="parallel_diagonal_4x128_ff256_all_rate1111",
            class_name="ParallelDiagonalModel",
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=NUM_BLOCKS,
            rates=(1, 1, 1, 1),
            readout_mode="all",
        ),
        "parallel_diagonal_multirate": VariantSpec(
            key="parallel_diagonal_multirate",
            label="parallel_diagonal_4x128_ff256_all_rate1248",
            class_name="ParallelDiagonalModel",
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=NUM_BLOCKS,
            rates=RATES,
            readout_mode="all",
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


def block_eval_stats(*, context_size: int, rates: tuple[int, ...]) -> dict[str, float | int | list[int]]:
    evals_per_block = [sum(1 for time_index in range(context_size) if time_index % rate == 0) for rate in rates]
    total_block_evals = sum(evals_per_block)
    average_block_evals_per_token = total_block_evals / context_size
    full_rate_1_evals_per_token = len(rates)
    return {
        "rates": list(rates),
        "evals_per_block": evals_per_block,
        "total_block_evals_per_forward": total_block_evals,
        "average_block_evals_per_token": round(average_block_evals_per_token, 6),
        "savings_vs_all_rate_1": round(1.0 - (average_block_evals_per_token / full_rate_1_evals_per_token), 6),
    }


def main() -> int:
    args = parse_args()
    if args.training_steps < WARMUP_STEPS:
        raise ValueError(
            f"training_steps must be at least {WARMUP_STEPS} so GraphTrainer.capture() can warm up."
        )
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/parallel_diagonal_multirate_20k.py.")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text("", encoding="utf-8")

    device = torch.device("cuda")
    set_seed(args.seed)
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

    batch_schedule = fixed_step_indices(
        train_inputs.shape[0],
        steps=args.training_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    warmup_batches = [
        materialize_batch(train_inputs, train_targets, indices)
        for indices in batch_schedule[:WARMUP_STEPS]
    ]

    builders = {
        "control": build_control_model,
        "parallel_diagonal_all_rate_1": build_parallel_all_rate_1_model,
        "parallel_diagonal_multirate": build_parallel_multirate_model,
    }
    specs = variant_specs()
    models: dict[str, torch.nn.Module] = {}
    trainers: dict[str, GraphTrainer] = {}
    checkpoints: dict[str, list[dict[str, float | int]]] = {}
    last_losses: dict[str, torch.Tensor] = {}

    for key, builder in builders.items():
        set_seed(args.seed)
        model = builder(device=device)
        optimizer = capturable_adamw(model, lr=args.learning_rate)
        trainer = GraphTrainer(
            model,
            optimizer,
            batch_size=args.batch_size,
            seq_len=CONTEXT_SIZE,
            device=device,
        )
        models[key] = model
        trainers[key] = trainer
        checkpoints[key] = [
            checkpoint_metrics(
                model,
                val_inputs,
                val_targets,
                batch_size=args.eval_batch_size,
                step=0,
            )
        ]

    append_log(
        args.log_path,
        {
            "stage": "checkpoint",
            "step": 0,
            **{key: variant_checkpoints[-1] for key, variant_checkpoints in checkpoints.items()},
        },
    )

    overall_started_at = perf_counter()
    for trainer in trainers.values():
        trainer.capture(warmup_batches)
    for key, trainer in trainers.items():
        last_losses[key] = trainer.static_loss.detach().clone()

    for zero_based_index, indices in enumerate(batch_schedule[WARMUP_STEPS:], start=WARMUP_STEPS):
        batch_input, batch_target = materialize_batch(train_inputs, train_targets, indices)
        for key, trainer in trainers.items():
            last_losses[key] = trainer.step(batch_input, batch_target)

        step = zero_based_index + 1
        if step % args.eval_interval != 0 and step != args.training_steps:
            continue

        for trainer in trainers.values():
            trainer.synchronize()
        for key, model in models.items():
            checkpoints[key].append(
                checkpoint_metrics(
                    model,
                    val_inputs,
                    val_targets,
                    batch_size=args.eval_batch_size,
                    step=step,
                )
            )
        append_log(
            args.log_path,
            {
                "stage": "checkpoint",
                "step": step,
                **{key: variant_checkpoints[-1] for key, variant_checkpoints in checkpoints.items()},
            },
        )

    for trainer in trainers.values():
        trainer.synchronize()
    wall_seconds = perf_counter() - overall_started_at

    git_status_short = current_git_status_short()
    report_models: dict[str, object] = {}
    for key, spec in specs.items():
        model_report = {
            "label": spec.label,
            "class_name": spec.class_name,
            "d_model": spec.d_model,
            "feedforward_dim": spec.feedforward_dim,
            "num_blocks": spec.num_blocks,
            "parameter_count": count_parameters(models[key]),
            "rates": list(spec.rates),
            "readout_mode": spec.readout_mode,
            "checkpoints": checkpoints[key],
            "best_checkpoint": min(checkpoints[key], key=lambda checkpoint: checkpoint["val_loss"]),
            "final_checkpoint": checkpoints[key][-1],
            "final_training_loss": round(last_losses[key].item(), 6),
            "block_eval_stats": block_eval_stats(context_size=CONTEXT_SIZE, rates=spec.rates),
        }
        if isinstance(models[key], ParallelDiagonalModel):
            model_report["mix_coefficients"] = models[key].mix_coefficients()
        report_models[key] = model_report

    control_final = report_models["control"]["final_checkpoint"]
    parallel_rate_1_final = report_models["parallel_diagonal_all_rate_1"]["final_checkpoint"]
    parallel_multirate_final = report_models["parallel_diagonal_multirate"]["final_checkpoint"]
    report = {
        "config": {
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "warmup_steps": WARMUP_STEPS,
            "context_size": CONTEXT_SIZE,
            "dataset_vocab_size": vocab_size,
            "model_vocab_size": VOCAB_SIZE,
            "schedule_seed": args.seed,
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
        "models": report_models,
        "comparison": {
            "delta_control_minus_parallel_diagonal_all_rate_1": {
                "final_val_loss": round(control_final["val_loss"] - parallel_rate_1_final["val_loss"], 6),
                "final_val_accuracy": round(
                    control_final["val_accuracy"] - parallel_rate_1_final["val_accuracy"],
                    6,
                ),
            },
            "delta_control_minus_parallel_diagonal_multirate": {
                "final_val_loss": round(control_final["val_loss"] - parallel_multirate_final["val_loss"], 6),
                "final_val_accuracy": round(
                    control_final["val_accuracy"] - parallel_multirate_final["val_accuracy"],
                    6,
                ),
            },
            "delta_parallel_diagonal_multirate_minus_parallel_diagonal_all_rate_1": {
                "final_val_loss": round(
                    parallel_multirate_final["val_loss"] - parallel_rate_1_final["val_loss"],
                    6,
                ),
                "final_val_accuracy": round(
                    parallel_multirate_final["val_accuracy"] - parallel_rate_1_final["val_accuracy"],
                    6,
                ),
            },
            "block_eval_savings_parallel_diagonal_multirate_vs_all_rate_1": report_models[
                "parallel_diagonal_multirate"
            ]["block_eval_stats"],
        },
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "done",
            "report_path": str(args.report_path),
            "final_checkpoints": {
                key: report_models[key]["final_checkpoint"] for key in report_models
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
