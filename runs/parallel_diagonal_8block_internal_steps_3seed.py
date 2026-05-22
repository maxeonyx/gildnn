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
DEFAULT_SEEDS = (42, 43, 44)
WARMUP_STEPS = 3
NUM_BLOCKS = 8
CONTROL_RATES = (1, 1, 1, 1, 1, 1, 1, 1)
PARALLEL_RATES = (1, 1, 2, 2, 4, 4, 8, 8)

CONTROL_D_MODEL = 128
CONTROL_FEEDFORWARD_DIM = 256
CONTROL_INTERNAL_STEPS = 1
PARALLEL_D_MODEL = 128
PARALLEL_INTERNAL_STEPS = 2
PARALLEL_FEEDFORWARD_DIM = 273


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    class_name: str
    d_model: int
    feedforward_dim: int
    num_blocks: int
    rates: tuple[int, ...]
    internal_steps: int
    readout_mode: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = (
        repo_root
        / "experiments"
        / "fixed_multi_rate"
        / "artifacts"
        / "parallel_diagonal_8block_internal_steps_3seed"
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
    parser.add_argument("--dry-run", action="store_true")
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


def build_parallel_internal_steps_model(*, device: torch.device) -> ParallelDiagonalModel:
    return ParallelDiagonalModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=PARALLEL_D_MODEL,
        feedforward_dim=PARALLEL_FEEDFORWARD_DIM,
        num_blocks=NUM_BLOCKS,
        rates=PARALLEL_RATES,
        internal_steps=PARALLEL_INTERNAL_STEPS,
        readout_mode="all",
    ).to(device)


def variant_specs() -> dict[str, VariantSpec]:
    return {
        "control": VariantSpec(
            key="control",
            label="multi_rate_8x128_ff256_11111111_control_fixed",
            class_name="MultiRateResidualModel",
            d_model=CONTROL_D_MODEL,
            feedforward_dim=CONTROL_FEEDFORWARD_DIM,
            num_blocks=NUM_BLOCKS,
            rates=CONTROL_RATES,
            internal_steps=CONTROL_INTERNAL_STEPS,
            readout_mode="last",
        ),
        "parallel_diagonal_8block_internal_steps": VariantSpec(
            key="parallel_diagonal_8block_internal_steps",
            label="parallel_diagonal_8x128_ff273_step2_all_rate11224488_matched_flop",
            class_name="ParallelDiagonalModel",
            d_model=PARALLEL_D_MODEL,
            feedforward_dim=PARALLEL_FEEDFORWARD_DIM,
            num_blocks=NUM_BLOCKS,
            rates=PARALLEL_RATES,
            internal_steps=PARALLEL_INTERNAL_STEPS,
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


def per_token_block_flops(
    *,
    d_model: int,
    feedforward_dim: int,
    average_block_evals_per_token: float,
    internal_steps: int,
) -> int:
    return round(average_block_evals_per_token * internal_steps * 2 * d_model * feedforward_dim)


def matched_flop_design() -> dict[str, object]:
    control_stats = block_eval_stats(context_size=CONTEXT_SIZE, rates=CONTROL_RATES)
    parallel_stats = block_eval_stats(context_size=CONTEXT_SIZE, rates=PARALLEL_RATES)
    control_average = float(control_stats["average_block_evals_per_token"])
    parallel_average = float(parallel_stats["average_block_evals_per_token"])
    control_flops = per_token_block_flops(
        d_model=CONTROL_D_MODEL,
        feedforward_dim=CONTROL_FEEDFORWARD_DIM,
        average_block_evals_per_token=control_average,
        internal_steps=CONTROL_INTERNAL_STEPS,
    )
    parallel_flops = per_token_block_flops(
        d_model=PARALLEL_D_MODEL,
        feedforward_dim=PARALLEL_FEEDFORWARD_DIM,
        average_block_evals_per_token=parallel_average,
        internal_steps=PARALLEL_INTERNAL_STEPS,
    )
    return {
        "control_rates": list(CONTROL_RATES),
        "parallel_rates": list(PARALLEL_RATES),
        "control_internal_steps": CONTROL_INTERNAL_STEPS,
        "parallel_internal_steps": PARALLEL_INTERNAL_STEPS,
        "control_average_block_evals_per_token": control_stats["average_block_evals_per_token"],
        "parallel_average_block_evals_per_token": parallel_stats["average_block_evals_per_token"],
        "control_effective_ff_passes_per_token": round(control_average * CONTROL_INTERNAL_STEPS, 6),
        "parallel_effective_ff_passes_per_token": round(parallel_average * PARALLEL_INTERNAL_STEPS, 6),
        "control_per_token_block_flops": control_flops,
        "parallel_per_token_block_flops": parallel_flops,
        "parallel_block_flop_delta_vs_control": parallel_flops - control_flops,
    }


def mean_rounded(values: list[float]) -> float:
    return round(mean(values), 6)


def build_dry_run_report(
    *,
    args: argparse.Namespace,
    device: torch.device,
    vocab_size: int,
    design: dict[str, object],
    specs: dict[str, VariantSpec],
    control_parameter_count: int,
    parallel_parameter_count: int,
    control_output_shape: list[int],
    parallel_output_shape: list[int],
) -> dict[str, object]:
    git_status_short = current_git_status_short()
    return {
        "dry_run": True,
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
        "matched_flop_design": design,
        "variants": {key: asdict(spec) for key, spec in specs.items()},
        "model_checks": {
            "control": {
                "parameter_count": control_parameter_count,
                "output_shape": control_output_shape,
            },
            "parallel_diagonal_8block_internal_steps": {
                "parameter_count": parallel_parameter_count,
                "output_shape": parallel_output_shape,
            },
        },
    }


def run_dry_run(*, args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/parallel_diagonal_8block_internal_steps_3seed.py.")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text("", encoding="utf-8")

    device = torch.device("cuda")
    set_seed(args.seeds[0])
    (train_inputs, _), (_, _), vocab_size = load_dataset(context_size=CONTEXT_SIZE)
    if vocab_size > VOCAB_SIZE:
        raise ValueError(
            f"Loaded dataset vocab_size {vocab_size} exceeds configured model vocab_size {VOCAB_SIZE}."
        )

    sample_tokens = train_inputs[:2].to(device)
    control_model = build_control_model(device=device)
    parallel_model = build_parallel_internal_steps_model(device=device)
    with torch.no_grad():
        control_output = control_model(sample_tokens)
        parallel_output = parallel_model(sample_tokens)
    specs = variant_specs()
    design = matched_flop_design()
    report = build_dry_run_report(
        args=args,
        device=device,
        vocab_size=vocab_size,
        design=design,
        specs=specs,
        control_parameter_count=count_parameters(control_model),
        parallel_parameter_count=count_parameters(parallel_model),
        control_output_shape=list(control_output.shape),
        parallel_output_shape=list(parallel_output.shape),
    )
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "dry_run",
            "report_path": str(args.report_path),
            "matched_flop_design": design,
            "control_parameter_count": report["model_checks"]["control"]["parameter_count"],
            "parallel_parameter_count": report["model_checks"]["parallel_diagonal_8block_internal_steps"][
                "parameter_count"
            ],
        },
    )
    return 0


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
    average_block_evals_per_token = block_eval_stats(
        context_size=CONTEXT_SIZE,
        rates=spec.rates,
    )["average_block_evals_per_token"]

    result = {
        "label": spec.label,
        "class_name": spec.class_name,
        "d_model": spec.d_model,
        "feedforward_dim": spec.feedforward_dim,
        "num_blocks": spec.num_blocks,
        "rates": list(spec.rates),
        "internal_steps": spec.internal_steps,
        "readout_mode": spec.readout_mode,
        "parameter_count": parameter_count,
        "checkpoints": checkpoints,
        "best_checkpoint": min(checkpoints, key=lambda checkpoint: checkpoint["val_loss"]),
        "final_checkpoint": checkpoints[-1],
        "final_training_loss": round(last_loss.item(), 6),
        "wall_seconds": round(wall_seconds, 6),
        "block_eval_stats": block_eval_stats(context_size=CONTEXT_SIZE, rates=spec.rates),
        "per_token_block_flops": per_token_block_flops(
            d_model=spec.d_model,
            feedforward_dim=spec.feedforward_dim,
            average_block_evals_per_token=float(average_block_evals_per_token),
            internal_steps=spec.internal_steps,
        ),
    }
    if isinstance(model, ParallelDiagonalModel):
        result["mix_coefficients"] = model.mix_coefficients()

    append_log(
        args.log_path,
        {
            "stage": "variant_finished",
            "seed": seed,
            "variant": variant_key,
            "final_checkpoint": result["final_checkpoint"],
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


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return run_dry_run(args=args)
    if args.training_steps < WARMUP_STEPS:
        raise ValueError(
            f"training_steps must be at least {WARMUP_STEPS} so GraphTrainer.capture() can warm up."
        )
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    if len(args.seeds) == 0:
        raise ValueError("At least one seed is required.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/parallel_diagonal_8block_internal_steps_3seed.py.")

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
    builders = {
        "control": build_control_model,
        "parallel_diagonal_8block_internal_steps": build_parallel_internal_steps_model,
    }
    design = matched_flop_design()

    append_log(
        args.log_path,
        {
            "stage": "run_started",
            "seeds": args.seeds,
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "trainer": "GraphTrainer",
            "matched_flop_design": design,
        },
    )

    seed_results: list[dict[str, object]] = []
    overall_started_at = perf_counter()
    for seed in args.seeds:
        append_log(args.log_path, {"stage": "seed_started", "seed": seed})
        control_result = train_single_variant(
            seed=seed,
            variant_key="control",
            spec=specs["control"],
            builder=builders["control"],
            args=args,
            device=device,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
        )
        parallel_result = train_single_variant(
            seed=seed,
            variant_key="parallel_diagonal_8block_internal_steps",
            spec=specs["parallel_diagonal_8block_internal_steps"],
            builder=builders["parallel_diagonal_8block_internal_steps"],
            args=args,
            device=device,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
        )
        delta = {
            "final_val_loss": round(
                control_result["final_checkpoint"]["val_loss"]
                - parallel_result["final_checkpoint"]["val_loss"],
                6,
            ),
            "final_val_accuracy": round(
                control_result["final_checkpoint"]["val_accuracy"]
                - parallel_result["final_checkpoint"]["val_accuracy"],
                6,
            ),
            "best_val_loss": round(
                control_result["best_checkpoint"]["val_loss"]
                - parallel_result["best_checkpoint"]["val_loss"],
                6,
            ),
            "parameter_count": control_result["parameter_count"] - parallel_result["parameter_count"],
        }
        seed_result = {
            "seed": seed,
            "control": control_result,
            "parallel_diagonal_8block_internal_steps": parallel_result,
            "delta_control_minus_parallel_diagonal_8block_internal_steps": delta,
        }
        seed_results.append(seed_result)
        append_log(
            args.log_path,
            {
                "stage": "seed_finished",
                "seed": seed,
                "delta_control_minus_parallel_diagonal_8block_internal_steps": delta,
            },
        )

    overall_wall_seconds = perf_counter() - overall_started_at
    git_status_short = current_git_status_short()
    averages = {
        "control": {
            "final_val_loss": mean_rounded(
                [result["control"]["final_checkpoint"]["val_loss"] for result in seed_results]
            ),
            "final_val_accuracy": mean_rounded(
                [result["control"]["final_checkpoint"]["val_accuracy"] for result in seed_results]
            ),
            "best_val_loss": mean_rounded(
                [result["control"]["best_checkpoint"]["val_loss"] for result in seed_results]
            ),
            "final_training_loss": mean_rounded(
                [result["control"]["final_training_loss"] for result in seed_results]
            ),
            "wall_seconds": mean_rounded([result["control"]["wall_seconds"] for result in seed_results]),
        },
        "parallel_diagonal_8block_internal_steps": {
            "final_val_loss": mean_rounded(
                [
                    result["parallel_diagonal_8block_internal_steps"]["final_checkpoint"]["val_loss"]
                    for result in seed_results
                ]
            ),
            "final_val_accuracy": mean_rounded(
                [
                    result["parallel_diagonal_8block_internal_steps"]["final_checkpoint"]["val_accuracy"]
                    for result in seed_results
                ]
            ),
            "best_val_loss": mean_rounded(
                [
                    result["parallel_diagonal_8block_internal_steps"]["best_checkpoint"]["val_loss"]
                    for result in seed_results
                ]
            ),
            "final_training_loss": mean_rounded(
                [result["parallel_diagonal_8block_internal_steps"]["final_training_loss"] for result in seed_results]
            ),
            "wall_seconds": mean_rounded(
                [result["parallel_diagonal_8block_internal_steps"]["wall_seconds"] for result in seed_results]
            ),
        },
        "delta_control_minus_parallel_diagonal_8block_internal_steps": {
            "final_val_loss": mean_rounded(
                [
                    result["delta_control_minus_parallel_diagonal_8block_internal_steps"]["final_val_loss"]
                    for result in seed_results
                ]
            ),
            "final_val_accuracy": mean_rounded(
                [
                    result["delta_control_minus_parallel_diagonal_8block_internal_steps"]["final_val_accuracy"]
                    for result in seed_results
                ]
            ),
            "best_val_loss": mean_rounded(
                [
                    result["delta_control_minus_parallel_diagonal_8block_internal_steps"]["best_val_loss"]
                    for result in seed_results
                ]
            ),
            "parameter_count": mean_rounded(
                [
                    float(result["delta_control_minus_parallel_diagonal_8block_internal_steps"]["parameter_count"])
                    for result in seed_results
                ]
            ),
        },
    }
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
            "overall_wall_seconds": round(overall_wall_seconds, 6),
        },
        "matched_flop_design": design,
        "variants": {key: asdict(spec) for key, spec in specs.items()},
        "seed_results": seed_results,
        "averages": averages,
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "done",
            "report_path": str(args.report_path),
            "average_delta_control_minus_parallel_diagonal_8block_internal_steps": averages[
                "delta_control_minus_parallel_diagonal_8block_internal_steps"
            ],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
