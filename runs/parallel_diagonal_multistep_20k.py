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
CONTROL_RATES = (1, 2, 4, 8)
CONTROL_D_MODEL = 128
CONTROL_FEEDFORWARD_DIM = 256
PARALLEL_SINGLE_STEP_D_MODEL = 128
PARALLEL_SINGLE_STEP_FEEDFORWARD_DIM = 256
PARALLEL_MULTI_STEP_D_MODEL = 128
PARALLEL_MULTI_STEP_FEEDFORWARD_DIM = 64
PARALLEL_MULTI_STEP_INTERNAL_STEPS = 2


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    class_name: str
    d_model: int
    feedforward_dim: int
    internal_steps: int | None
    num_blocks: int
    rates: tuple[int, ...] | None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = (
        repo_root
        / "experiments"
        / "fixed_multi_rate"
        / "artifacts"
        / "parallel_diagonal_multistep_20k"
    )
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


def build_parallel_single_step_model(*, device: torch.device) -> ParallelDiagonalModel:
    return ParallelDiagonalModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=PARALLEL_SINGLE_STEP_D_MODEL,
        feedforward_dim=PARALLEL_SINGLE_STEP_FEEDFORWARD_DIM,
        num_blocks=NUM_BLOCKS,
        internal_steps=1,
    ).to(device)


def build_parallel_multi_step_model(*, device: torch.device) -> ParallelDiagonalModel:
    return ParallelDiagonalModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=PARALLEL_MULTI_STEP_D_MODEL,
        feedforward_dim=PARALLEL_MULTI_STEP_FEEDFORWARD_DIM,
        num_blocks=NUM_BLOCKS,
        internal_steps=PARALLEL_MULTI_STEP_INTERNAL_STEPS,
    ).to(device)


def variant_specs() -> dict[str, VariantSpec]:
    return {
        "control": VariantSpec(
            key="control",
            label="multi_rate_4x128_1248_control",
            class_name="MultiRateResidualModel",
            d_model=CONTROL_D_MODEL,
            feedforward_dim=CONTROL_FEEDFORWARD_DIM,
            internal_steps=None,
            num_blocks=len(CONTROL_RATES),
            rates=CONTROL_RATES,
        ),
        "parallel_diagonal_1step": VariantSpec(
            key="parallel_diagonal_1step",
            label="parallel_diagonal_4x128_ff256_step1",
            class_name="ParallelDiagonalModel",
            d_model=PARALLEL_SINGLE_STEP_D_MODEL,
            feedforward_dim=PARALLEL_SINGLE_STEP_FEEDFORWARD_DIM,
            internal_steps=1,
            num_blocks=NUM_BLOCKS,
            rates=None,
        ),
        "parallel_diagonal_2step": VariantSpec(
            key="parallel_diagonal_2step",
            label="parallel_diagonal_4x128_ff64_step2",
            class_name="ParallelDiagonalModel",
            d_model=PARALLEL_MULTI_STEP_D_MODEL,
            feedforward_dim=PARALLEL_MULTI_STEP_FEEDFORWARD_DIM,
            internal_steps=PARALLEL_MULTI_STEP_INTERNAL_STEPS,
            num_blocks=NUM_BLOCKS,
            rates=None,
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


def control_block_flop_units_per_token() -> float:
    return sum(1.0 / rate for rate in CONTROL_RATES) * CONTROL_D_MODEL * CONTROL_FEEDFORWARD_DIM


def parallel_block_flop_units_per_token(*, d_model: int, feedforward_dim: int, internal_steps: int) -> int:
    return NUM_BLOCKS * internal_steps * d_model * feedforward_dim


def main() -> int:
    args = parse_args()
    if args.training_steps < WARMUP_STEPS:
        raise ValueError(
            f"training_steps must be at least {WARMUP_STEPS} so GraphTrainer.capture() can warm up."
        )
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/parallel_diagonal_multistep_20k.py.")

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
        "parallel_diagonal_1step": build_parallel_single_step_model,
        "parallel_diagonal_2step": build_parallel_multi_step_model,
    }
    specs = variant_specs()
    models: dict[str, torch.nn.Module] = {}
    optimizers: dict[str, torch.optim.AdamW] = {}
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
        optimizers[key] = optimizer
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
    control_units = control_block_flop_units_per_token()
    report_models: dict[str, object] = {}
    for key, spec in specs.items():
        parameter_count = count_parameters(models[key])
        model_report = {
            "label": spec.label,
            "class_name": spec.class_name,
            "d_model": spec.d_model,
            "feedforward_dim": spec.feedforward_dim,
            "num_blocks": spec.num_blocks,
            "parameter_count": parameter_count,
            "checkpoints": checkpoints[key],
            "best_checkpoint": min(checkpoints[key], key=lambda checkpoint: checkpoint["val_loss"]),
            "final_checkpoint": checkpoints[key][-1],
            "final_training_loss": round(last_losses[key].item(), 6),
            "approx_block_flop_units_per_token": round(
                parallel_block_flop_units_per_token(
                    d_model=spec.d_model,
                    feedforward_dim=spec.feedforward_dim,
                    internal_steps=spec.internal_steps or 1,
                )
                if spec.internal_steps is not None
                else control_units,
                6,
            ),
            "approx_block_flop_ratio_vs_control": round(
                (
                    parallel_block_flop_units_per_token(
                        d_model=spec.d_model,
                        feedforward_dim=spec.feedforward_dim,
                        internal_steps=spec.internal_steps or 1,
                    )
                    / control_units
                )
                if spec.internal_steps is not None
                else 1.0,
                6,
            ),
        }
        if spec.rates is not None:
            model_report["rates"] = list(spec.rates)
        if spec.internal_steps is not None:
            model_report["internal_steps"] = spec.internal_steps
            model_report["neighbor_direction"] = "previous_block_only"
            model_report["state_update"] = (
                "seed each block once from token+same-block state, then repeat parallel neighbor propagation"
            )
            model_report["readout"] = "last_block_state"
        report_models[key] = model_report

    final_val_losses = {
        key: report_models[key]["final_checkpoint"]["val_loss"] for key in report_models
    }
    sorted_by_final_loss = sorted(final_val_losses.items(), key=lambda item: item[1])
    winner_key, winner_loss = sorted_by_final_loss[0]
    runner_up_key, runner_up_loss = sorted_by_final_loss[1]

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
        "flop_matching": {
            "method": "approximate dominant FFN block FLOPs per token",
            "note": (
                "This ignores control temporal-attention FLOPs, so the 2-step match is approximate rather than exact."
            ),
            "control_avg_active_blocks_per_token": round(sum(1.0 / rate for rate in CONTROL_RATES), 6),
            "control_block_flop_units_per_token": round(control_units, 6),
            "parallel_single_step_block_flop_units_per_token": parallel_block_flop_units_per_token(
                d_model=PARALLEL_SINGLE_STEP_D_MODEL,
                feedforward_dim=PARALLEL_SINGLE_STEP_FEEDFORWARD_DIM,
                internal_steps=1,
            ),
            "parallel_multi_step_block_flop_units_per_token": parallel_block_flop_units_per_token(
                d_model=PARALLEL_MULTI_STEP_D_MODEL,
                feedforward_dim=PARALLEL_MULTI_STEP_FEEDFORWARD_DIM,
                internal_steps=PARALLEL_MULTI_STEP_INTERNAL_STEPS,
            ),
        },
        "variants": {key: asdict(spec) for key, spec in specs.items()},
        "models": report_models,
        "comparison": {
            "winner_by_final_val_loss": winner_key,
            "winner_final_val_loss": round(winner_loss, 6),
            "margin_vs_runner_up": round(runner_up_loss - winner_loss, 6),
            "delta_control_minus_parallel_diagonal_1step": {
                "final_val_loss": round(
                    report_models["control"]["final_checkpoint"]["val_loss"]
                    - report_models["parallel_diagonal_1step"]["final_checkpoint"]["val_loss"],
                    6,
                ),
                "final_val_accuracy": round(
                    report_models["control"]["final_checkpoint"]["val_accuracy"]
                    - report_models["parallel_diagonal_1step"]["final_checkpoint"]["val_accuracy"],
                    6,
                ),
            },
            "delta_control_minus_parallel_diagonal_2step": {
                "final_val_loss": round(
                    report_models["control"]["final_checkpoint"]["val_loss"]
                    - report_models["parallel_diagonal_2step"]["final_checkpoint"]["val_loss"],
                    6,
                ),
                "final_val_accuracy": round(
                    report_models["control"]["final_checkpoint"]["val_accuracy"]
                    - report_models["parallel_diagonal_2step"]["final_checkpoint"]["val_accuracy"],
                    6,
                ),
            },
            "delta_parallel_diagonal_2step_minus_parallel_diagonal_1step": {
                "final_val_loss": round(
                    report_models["parallel_diagonal_2step"]["final_checkpoint"]["val_loss"]
                    - report_models["parallel_diagonal_1step"]["final_checkpoint"]["val_loss"],
                    6,
                ),
                "final_val_accuracy": round(
                    report_models["parallel_diagonal_2step"]["final_checkpoint"]["val_accuracy"]
                    - report_models["parallel_diagonal_1step"]["final_checkpoint"]["val_accuracy"],
                    6,
                ),
            },
        },
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "done",
            "report_path": str(args.report_path),
            "winner": winner_key,
            "final_checkpoints": {
                key: report_models[key]["final_checkpoint"] for key in report_models
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
