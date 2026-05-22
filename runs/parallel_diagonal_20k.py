from __future__ import annotations

import argparse
import json
import sys
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
CONTROL_D_MODEL = 128
CONTROL_FEEDFORWARD_DIM = 256
PARALLEL_DIAGONAL_D_MODEL = 128
PARALLEL_DIAGONAL_FEEDFORWARD_DIM = 320
NUM_BLOCKS = 4
CONTROL_RATES = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "fixed_multi_rate" / "artifacts" / "parallel_diagonal_20k"
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.log")
    return parser.parse_args()


def build_control_model(*, device: torch.device) -> MultiRateResidualModel:
    return MultiRateResidualModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=CONTROL_D_MODEL,
        feedforward_dim=CONTROL_FEEDFORWARD_DIM,
        temporal_window=4,
        num_heads=4,
        rates=CONTROL_RATES,
    ).to(device)


def build_parallel_diagonal_model(*, device: torch.device) -> ParallelDiagonalModel:
    return ParallelDiagonalModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=PARALLEL_DIAGONAL_D_MODEL,
        feedforward_dim=PARALLEL_DIAGONAL_FEEDFORWARD_DIM,
        num_blocks=NUM_BLOCKS,
    ).to(device)


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


def append_log(log_path: Path, line: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def main() -> int:
    args = parse_args()
    if args.training_steps < WARMUP_STEPS:
        raise ValueError(
            f"training_steps must be at least {WARMUP_STEPS} so GraphTrainer.capture() can warm up."
        )
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/parallel_diagonal_20k.py.")

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

    set_seed(args.seed)
    control_model = build_control_model(device=device)
    control_optimizer = capturable_adamw(control_model, lr=args.learning_rate)
    control_trainer = GraphTrainer(
        control_model,
        control_optimizer,
        batch_size=args.batch_size,
        seq_len=CONTEXT_SIZE,
        device=device,
    )

    set_seed(args.seed)
    parallel_diagonal_model = build_parallel_diagonal_model(device=device)
    parallel_diagonal_optimizer = capturable_adamw(parallel_diagonal_model, lr=args.learning_rate)
    parallel_diagonal_trainer = GraphTrainer(
        parallel_diagonal_model,
        parallel_diagonal_optimizer,
        batch_size=args.batch_size,
        seq_len=CONTEXT_SIZE,
        device=device,
    )

    control_checkpoints = [
        checkpoint_metrics(
            control_model,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=0,
        )
    ]
    parallel_diagonal_checkpoints = [
        checkpoint_metrics(
            parallel_diagonal_model,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=0,
        )
    ]

    append_log(
        args.log_path,
        json.dumps(
            {
                "stage": "checkpoint",
                "step": 0,
                "control": control_checkpoints[-1],
                "parallel_diagonal": parallel_diagonal_checkpoints[-1],
            }
        ),
    )

    overall_started_at = perf_counter()
    control_trainer.capture(warmup_batches)
    parallel_diagonal_trainer.capture(warmup_batches)

    last_loss_control = control_trainer.static_loss.detach().clone()
    last_loss_parallel_diagonal = parallel_diagonal_trainer.static_loss.detach().clone()
    for zero_based_index, indices in enumerate(batch_schedule[WARMUP_STEPS:], start=WARMUP_STEPS):
        batch_input, batch_target = materialize_batch(train_inputs, train_targets, indices)
        last_loss_control = control_trainer.step(batch_input, batch_target)
        last_loss_parallel_diagonal = parallel_diagonal_trainer.step(batch_input, batch_target)

        step = zero_based_index + 1
        if step % args.eval_interval != 0 and step != args.training_steps:
            continue

        control_trainer.synchronize()
        parallel_diagonal_trainer.synchronize()
        control_checkpoint = checkpoint_metrics(
            control_model,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=step,
        )
        parallel_diagonal_checkpoint = checkpoint_metrics(
            parallel_diagonal_model,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=step,
        )
        control_checkpoints.append(control_checkpoint)
        parallel_diagonal_checkpoints.append(parallel_diagonal_checkpoint)
        append_log(
            args.log_path,
            json.dumps(
                {
                    "stage": "checkpoint",
                    "step": step,
                    "control": control_checkpoint,
                    "parallel_diagonal": parallel_diagonal_checkpoint,
                }
            ),
        )

    control_trainer.synchronize()
    parallel_diagonal_trainer.synchronize()
    wall_seconds = perf_counter() - overall_started_at

    git_status_short = current_git_status_short()
    control_parameter_count = count_parameters(control_model)
    parallel_diagonal_parameter_count = count_parameters(parallel_diagonal_model)
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
        "models": {
            "control": {
                "label": "multi_rate_4x128_1248_control",
                "class_name": "MultiRateResidualModel",
                "d_model": CONTROL_D_MODEL,
                "feedforward_dim": CONTROL_FEEDFORWARD_DIM,
                "rates": list(CONTROL_RATES),
                "parameter_count": control_parameter_count,
                "checkpoints": control_checkpoints,
                "best_checkpoint": min(control_checkpoints, key=lambda checkpoint: checkpoint["val_loss"]),
                "final_checkpoint": control_checkpoints[-1],
                "final_training_loss": round(last_loss_control.item(), 6),
            },
            "parallel_diagonal": {
                "label": "parallel_diagonal_4x128_ff320",
                "class_name": "ParallelDiagonalModel",
                "d_model": PARALLEL_DIAGONAL_D_MODEL,
                "feedforward_dim": PARALLEL_DIAGONAL_FEEDFORWARD_DIM,
                "num_blocks": NUM_BLOCKS,
                "neighbor_direction": "previous_block_only",
                "state_update": "mix(previous_same_block_state, current_token_embedding), average_with_previous_block_state, residual_ff",
                "readout": "last_block_state",
                "parameter_count": parallel_diagonal_parameter_count,
                "checkpoints": parallel_diagonal_checkpoints,
                "best_checkpoint": min(
                    parallel_diagonal_checkpoints,
                    key=lambda checkpoint: checkpoint["val_loss"],
                ),
                "final_checkpoint": parallel_diagonal_checkpoints[-1],
                "final_training_loss": round(last_loss_parallel_diagonal.item(), 6),
            },
        },
        "delta_control_minus_parallel_diagonal": {
            "final_val_loss": round(
                control_checkpoints[-1]["val_loss"] - parallel_diagonal_checkpoints[-1]["val_loss"],
                6,
            ),
            "final_val_accuracy": round(
                control_checkpoints[-1]["val_accuracy"] - parallel_diagonal_checkpoints[-1]["val_accuracy"],
                6,
            ),
            "best_val_loss": round(
                min(checkpoint["val_loss"] for checkpoint in control_checkpoints)
                - min(checkpoint["val_loss"] for checkpoint in parallel_diagonal_checkpoints),
                6,
            ),
            "parameter_count": control_parameter_count - parallel_diagonal_parameter_count,
        },
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        json.dumps(
            {
                "stage": "done",
                "report_path": str(args.report_path),
                "final_control": control_checkpoints[-1],
                "final_parallel_diagonal": parallel_diagonal_checkpoints[-1],
            }
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
