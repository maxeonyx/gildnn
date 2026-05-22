from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from core.fixed_window_char import load_dataset, set_seed
from core.model import MultiRateResidualModel, count_parameters
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
DIVERGENCE_THRESHOLD = 10.0


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=repo_root / "experiments" / "fixed_multi_rate" / "artifacts" / "diagonal_20k" / "report.json",
    )
    return parser.parse_args()


def build_model(*, device: torch.device, diagonal_enabled: bool) -> MultiRateResidualModel:
    return MultiRateResidualModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=128,
        feedforward_dim=256,
        temporal_window=4,
        num_heads=4,
        rates=(1, 2, 4, 8),
        diagonal_enabled=diagonal_enabled,
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


def log_checkpoint(*, step: int, control_loss: float, diagonal_loss: float) -> None:
    print(
        f"Step {step}: control val_loss={control_loss:.6f} diagonal val_loss={diagonal_loss:.6f}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if args.training_steps < WARMUP_STEPS:
        raise ValueError(
            f"training_steps must be at least {WARMUP_STEPS} so GraphTrainer.capture() can warm up."
        )
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/diagonal_20k.py.")

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
    control_model = build_model(device=device, diagonal_enabled=False)
    control_optimizer = capturable_adamw(control_model, lr=args.learning_rate)
    control_trainer = GraphTrainer(
        control_model,
        control_optimizer,
        batch_size=args.batch_size,
        seq_len=CONTEXT_SIZE,
        device=device,
    )

    set_seed(args.seed)
    diagonal_model = build_model(device=device, diagonal_enabled=True)
    diagonal_optimizer = capturable_adamw(diagonal_model, lr=args.learning_rate)
    diagonal_trainer = GraphTrainer(
        diagonal_model,
        diagonal_optimizer,
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
    diagonal_checkpoints = [
        checkpoint_metrics(
            diagonal_model,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=0,
        )
    ]
    log_checkpoint(
        step=0,
        control_loss=control_checkpoints[-1]["val_loss"],
        diagonal_loss=diagonal_checkpoints[-1]["val_loss"],
    )

    control_diverged = False
    diagonal_diverged = False
    control_diverged_at_step: int | None = None
    diagonal_diverged_at_step: int | None = None

    overall_started_at = perf_counter()
    control_trainer.capture(warmup_batches)
    diagonal_trainer.capture(warmup_batches)

    last_loss_control = control_trainer.static_loss.detach().clone()
    last_loss_diagonal = diagonal_trainer.static_loss.detach().clone()
    for zero_based_index, indices in enumerate(batch_schedule[WARMUP_STEPS:], start=WARMUP_STEPS):
        batch_input, batch_target = materialize_batch(train_inputs, train_targets, indices)
        if not control_diverged:
            last_loss_control = control_trainer.step(batch_input, batch_target)
        if not diagonal_diverged:
            last_loss_diagonal = diagonal_trainer.step(batch_input, batch_target)

        step = zero_based_index + 1
        if step % args.eval_interval != 0 and step != args.training_steps:
            continue

        if not control_diverged:
            control_trainer.synchronize()
            control_checkpoint = checkpoint_metrics(
                control_model,
                val_inputs,
                val_targets,
                batch_size=args.eval_batch_size,
                step=step,
            )
            control_checkpoints.append(control_checkpoint)
            if control_checkpoint["val_loss"] > DIVERGENCE_THRESHOLD:
                control_diverged = True
                control_diverged_at_step = step
                print(
                    f"DIVERGENCE: control val_loss={control_checkpoint['val_loss']:.6f} exceeded {DIVERGENCE_THRESHOLD:.1f} at step {step}",
                    flush=True,
                )
        else:
            control_checkpoint = control_checkpoints[-1]

        if not diagonal_diverged:
            diagonal_trainer.synchronize()
            diagonal_checkpoint = checkpoint_metrics(
                diagonal_model,
                val_inputs,
                val_targets,
                batch_size=args.eval_batch_size,
                step=step,
            )
            diagonal_checkpoints.append(diagonal_checkpoint)
            if diagonal_checkpoint["val_loss"] > DIVERGENCE_THRESHOLD:
                diagonal_diverged = True
                diagonal_diverged_at_step = step
                print(
                    f"DIVERGENCE: diagonal val_loss={diagonal_checkpoint['val_loss']:.6f} exceeded {DIVERGENCE_THRESHOLD:.1f} at step {step}",
                    flush=True,
                )
        else:
            diagonal_checkpoint = diagonal_checkpoints[-1]

        log_checkpoint(
            step=step,
            control_loss=control_checkpoint["val_loss"],
            diagonal_loss=diagonal_checkpoint["val_loss"],
        )

        if control_diverged and diagonal_diverged:
            break

    control_trainer.synchronize()
    diagonal_trainer.synchronize()
    wall_seconds = perf_counter() - overall_started_at

    git_status_short = current_git_status_short()
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
            "divergence_threshold": DIVERGENCE_THRESHOLD,
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
        "diverged_models": [
            model_name
            for model_name, diverged in (("control", control_diverged), ("diagonal", diagonal_diverged))
            if diverged
        ],
        "models": {
            "control": {
                "label": "multi_rate_4x128_1248_control",
                "d_model": 128,
                "feedforward_dim": 256,
                "rates": [1, 2, 4, 8],
                "diagonal_enabled": False,
                "parameter_count": count_parameters(control_model),
                "diverged": control_diverged,
                "diverged_at_step": control_diverged_at_step,
                "checkpoints": control_checkpoints,
                "best_checkpoint": min(control_checkpoints, key=lambda checkpoint: checkpoint["val_loss"]),
                "final_checkpoint": control_checkpoints[-1],
                "final_training_loss": round(last_loss_control.item(), 6),
            },
            "diagonal": {
                "label": "multi_rate_4x128_1248_diagonal",
                "d_model": 128,
                "feedforward_dim": 256,
                "rates": [1, 2, 4, 8],
                "diagonal_enabled": True,
                "parameter_count": count_parameters(diagonal_model),
                "diverged": diagonal_diverged,
                "diverged_at_step": diagonal_diverged_at_step,
                "checkpoints": diagonal_checkpoints,
                "best_checkpoint": min(diagonal_checkpoints, key=lambda checkpoint: checkpoint["val_loss"]),
                "final_checkpoint": diagonal_checkpoints[-1],
                "final_training_loss": round(last_loss_diagonal.item(), 6),
            },
        },
        "delta_control_minus_diagonal": {
            "final_val_loss": round(
                control_checkpoints[-1]["val_loss"] - diagonal_checkpoints[-1]["val_loss"],
                6,
            ),
            "final_val_accuracy": round(
                control_checkpoints[-1]["val_accuracy"] - diagonal_checkpoints[-1]["val_accuracy"],
                6,
            ),
            "best_val_loss": round(
                min(checkpoint["val_loss"] for checkpoint in control_checkpoints)
                - min(checkpoint["val_loss"] for checkpoint in diagonal_checkpoints),
                6,
            ),
            "parameter_count": count_parameters(control_model) - count_parameters(diagonal_model),
        },
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report_path, report)
    print(f"Wrote report to {args.report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
