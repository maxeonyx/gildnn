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
EVAL_INTERVAL = 2_000
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 3e-4
SEED = 42
WARMUP_STEPS = 3


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
        default=repo_root / "experiments" / "fixed_multi_rate" / "artifacts" / "matched_flop" / "report_20k.json",
    )
    return parser.parse_args()


def build_model_a(*, device: torch.device) -> MultiRateResidualModel:
    return MultiRateResidualModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=128,
        feedforward_dim=256,
        temporal_window=4,
        num_heads=4,
        rates=(1, 2, 4, 8),
        diagonal_enabled=False,
    ).to(device)


def build_model_b(*, device: torch.device) -> MultiRateResidualModel:
    return MultiRateResidualModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=112,
        feedforward_dim=224,
        temporal_window=4,
        num_heads=4,
        rates=(1, 1, 1, 1),
        diagonal_enabled=False,
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


def main() -> int:
    args = parse_args()
    if args.training_steps < WARMUP_STEPS:
        raise ValueError(
            f"training_steps must be at least {WARMUP_STEPS} so GraphTrainer.capture() can warm up."
        )
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/matched_flop_20k.py.")

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
    model_a = build_model_a(device=device)
    optimizer_a = capturable_adamw(model_a, lr=args.learning_rate)
    trainer_a = GraphTrainer(
        model_a,
        optimizer_a,
        batch_size=args.batch_size,
        seq_len=CONTEXT_SIZE,
        device=device,
    )

    set_seed(args.seed)
    model_b = build_model_b(device=device)
    optimizer_b = capturable_adamw(model_b, lr=args.learning_rate)
    trainer_b = GraphTrainer(
        model_b,
        optimizer_b,
        batch_size=args.batch_size,
        seq_len=CONTEXT_SIZE,
        device=device,
    )

    checkpoints_a = [
        checkpoint_metrics(
            model_a,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=0,
        )
    ]
    checkpoints_b = [
        checkpoint_metrics(
            model_b,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=0,
        )
    ]
    print(
        "Step 0: "
        f"model_a val_loss={checkpoints_a[-1]['val_loss']:.6f} "
        f"model_b val_loss={checkpoints_b[-1]['val_loss']:.6f}",
        flush=True,
    )

    overall_started_at = perf_counter()
    trainer_a.capture(warmup_batches)
    trainer_b.capture(warmup_batches)

    last_loss_a = trainer_a.static_loss.detach().clone()
    last_loss_b = trainer_b.static_loss.detach().clone()
    for zero_based_index, indices in enumerate(batch_schedule[WARMUP_STEPS:], start=WARMUP_STEPS):
        batch_input, batch_target = materialize_batch(train_inputs, train_targets, indices)
        last_loss_a = trainer_a.step(batch_input, batch_target)
        last_loss_b = trainer_b.step(batch_input, batch_target)
        step = zero_based_index + 1
        if step % args.eval_interval != 0 and step != args.training_steps:
            continue

        trainer_a.synchronize()
        trainer_b.synchronize()
        checkpoint_a = checkpoint_metrics(
            model_a,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=step,
        )
        checkpoint_b = checkpoint_metrics(
            model_b,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=step,
        )
        checkpoints_a.append(checkpoint_a)
        checkpoints_b.append(checkpoint_b)
        print(
            f"Step {step}: "
            f"model_a val_loss={checkpoint_a['val_loss']:.6f} "
            f"model_b val_loss={checkpoint_b['val_loss']:.6f}",
            flush=True,
        )

    trainer_a.synchronize()
    trainer_b.synchronize()
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
            "model_a": {
                "label": "multi_rate_4x128_1248",
                "d_model": 128,
                "feedforward_dim": 256,
                "rates": [1, 2, 4, 8],
                "diagonal_enabled": False,
                "parameter_count": count_parameters(model_a),
                "checkpoints": checkpoints_a,
                "best_checkpoint": min(checkpoints_a, key=lambda checkpoint: checkpoint["val_loss"]),
                "final_checkpoint": checkpoints_a[-1],
                "final_training_loss": round(last_loss_a.item(), 6),
            },
            "model_b": {
                "label": "matched_all_rate_1_4x112",
                "d_model": 112,
                "feedforward_dim": 224,
                "rates": [1, 1, 1, 1],
                "diagonal_enabled": False,
                "parameter_count": count_parameters(model_b),
                "checkpoints": checkpoints_b,
                "best_checkpoint": min(checkpoints_b, key=lambda checkpoint: checkpoint["val_loss"]),
                "final_checkpoint": checkpoints_b[-1],
                "final_training_loss": round(last_loss_b.item(), 6),
            },
        },
        "delta_model_a_minus_model_b": {
            "final_val_loss": round(checkpoints_a[-1]["val_loss"] - checkpoints_b[-1]["val_loss"], 6),
            "final_val_accuracy": round(
                checkpoints_a[-1]["val_accuracy"] - checkpoints_b[-1]["val_accuracy"],
                6,
            ),
            "best_val_loss": round(
                min(checkpoint["val_loss"] for checkpoint in checkpoints_a)
                - min(checkpoint["val_loss"] for checkpoint in checkpoints_b),
                6,
            ),
            "parameter_count": count_parameters(model_a) - count_parameters(model_b),
        },
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report_path, report)
    print(f"Wrote report to {args.report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
