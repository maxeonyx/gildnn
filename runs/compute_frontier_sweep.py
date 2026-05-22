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
TEMPORAL_WINDOW = 4
NUM_HEADS = 4
D_MODEL = 128
FEEDFORWARD_DIM = 256


@dataclass(frozen=True)
class VariantSpec:
    key: str
    family: str
    label: str
    class_name: str
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
        / "compute_frontier_sweep"
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


def variant_specs() -> list[VariantSpec]:
    return [
        VariantSpec(
            key="sequential_2block_rate11",
            family="sequential_all_rate_1",
            label="sequential_2block_rate11",
            class_name="MultiRateResidualModel",
            num_blocks=2,
            rates=(1, 1),
            readout_mode="last",
        ),
        VariantSpec(
            key="parallel_2block_rate12",
            family="parallel_multi_rate",
            label="parallel_2block_rate12_all",
            class_name="ParallelDiagonalModel",
            num_blocks=2,
            rates=(1, 2),
            readout_mode="all",
        ),
        VariantSpec(
            key="sequential_4block_rate1",
            family="sequential_all_rate_1",
            label="sequential_4block_rate1111",
            class_name="MultiRateResidualModel",
            num_blocks=4,
            rates=(1, 1, 1, 1),
            readout_mode="last",
        ),
        VariantSpec(
            key="parallel_4block_rate1248",
            family="parallel_multi_rate",
            label="parallel_4block_rate1248_all",
            class_name="ParallelDiagonalModel",
            num_blocks=4,
            rates=(1, 2, 4, 8),
            readout_mode="all",
        ),
        VariantSpec(
            key="sequential_6block_rate1",
            family="sequential_all_rate_1",
            label="sequential_6block_rate111111",
            class_name="MultiRateResidualModel",
            num_blocks=6,
            rates=(1, 1, 1, 1, 1, 1),
            readout_mode="last",
        ),
        VariantSpec(
            key="parallel_6block_rate112248",
            family="parallel_multi_rate",
            label="parallel_6block_rate112248_all",
            class_name="ParallelDiagonalModel",
            num_blocks=6,
            rates=(1, 1, 2, 2, 4, 8),
            readout_mode="all",
        ),
        VariantSpec(
            key="sequential_8block_rate1",
            family="sequential_all_rate_1",
            label="sequential_8block_rate11111111",
            class_name="MultiRateResidualModel",
            num_blocks=8,
            rates=(1, 1, 1, 1, 1, 1, 1, 1),
            readout_mode="last",
        ),
        VariantSpec(
            key="parallel_8block_rate11224488",
            family="parallel_multi_rate",
            label="parallel_8block_rate11224488_all",
            class_name="ParallelDiagonalModel",
            num_blocks=8,
            rates=(1, 1, 2, 2, 4, 4, 8, 8),
            readout_mode="all",
        ),
    ]


def build_model(*, spec: VariantSpec, device: torch.device) -> torch.nn.Module:
    if spec.class_name == "MultiRateResidualModel":
        return MultiRateResidualModel(
            vocab_size=VOCAB_SIZE,
            context_size=CONTEXT_SIZE,
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            temporal_window=TEMPORAL_WINDOW,
            num_heads=NUM_HEADS,
            rates=spec.rates,
        ).to(device)
    if spec.class_name == "ParallelDiagonalModel":
        return ParallelDiagonalModel(
            vocab_size=VOCAB_SIZE,
            context_size=CONTEXT_SIZE,
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=spec.num_blocks,
            rates=spec.rates,
            readout_mode=spec.readout_mode,
        ).to(device)
    raise ValueError(f"Unsupported class_name {spec.class_name!r}.")


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


def evals_per_block(*, context_size: int, rates: tuple[int, ...]) -> list[int]:
    return [sum(1 for time_index in range(context_size) if time_index % rate == 0) for rate in rates]


def average_evals_per_token(*, context_size: int, rates: tuple[int, ...]) -> float:
    return sum(evals_per_block(context_size=context_size, rates=rates)) / context_size


def flops_proxy(*, avg_evals_per_token: float, d_model: int, feedforward_dim: int) -> float:
    return avg_evals_per_token * 2.0 * d_model * feedforward_dim


def run_variant(
    spec: VariantSpec,
    *,
    args: argparse.Namespace,
    device: torch.device,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    batch_schedule: list[torch.Tensor],
    log_path: Path,
) -> dict[str, object]:
    set_seed(args.seed)
    model = build_model(spec=spec, device=device)
    optimizer = capturable_adamw(model, lr=args.learning_rate)
    trainer = GraphTrainer(
        model,
        optimizer,
        batch_size=args.batch_size,
        seq_len=CONTEXT_SIZE,
        device=device,
    )

    warmup_batches = [
        materialize_batch(train_inputs, train_targets, indices)
        for indices in batch_schedule[:WARMUP_STEPS]
    ]
    checkpoints = [
        checkpoint_metrics(
            model,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=0,
        )
    ]
    avg_eval_per_token = average_evals_per_token(context_size=CONTEXT_SIZE, rates=spec.rates)
    variant_started_at = perf_counter()

    append_log(
        log_path,
        {
            "stage": "variant_start",
            "variant": spec.key,
            "family": spec.family,
            "num_blocks": spec.num_blocks,
            "rates": list(spec.rates),
            "readout_mode": spec.readout_mode,
            "avg_evals_per_token": round(avg_eval_per_token, 6),
            "flops_proxy": round(
                flops_proxy(
                    avg_evals_per_token=avg_eval_per_token,
                    d_model=D_MODEL,
                    feedforward_dim=FEEDFORWARD_DIM,
                ),
                6,
            ),
        },
    )
    append_log(
        log_path,
        {
            "stage": "checkpoint",
            "variant": spec.key,
            **checkpoints[-1],
        },
    )

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
            log_path,
            {
                "stage": "checkpoint",
                "variant": spec.key,
                **checkpoint,
            },
        )

    trainer.synchronize()
    wall_seconds = perf_counter() - variant_started_at
    eval_counts = evals_per_block(context_size=CONTEXT_SIZE, rates=spec.rates)
    result = {
        **asdict(spec),
        "d_model": D_MODEL,
        "feedforward_dim": FEEDFORWARD_DIM,
        "parameter_count": count_parameters(model),
        "evals_per_block": eval_counts,
        "total_block_evals_per_forward": sum(eval_counts),
        "avg_evals_per_token": round(avg_eval_per_token, 6),
        "flops_proxy": round(
            flops_proxy(
                avg_evals_per_token=avg_eval_per_token,
                d_model=D_MODEL,
                feedforward_dim=FEEDFORWARD_DIM,
            ),
            6,
        ),
        "checkpoints": checkpoints,
        "best_checkpoint": min(checkpoints, key=lambda checkpoint: checkpoint["val_loss"]),
        "final_checkpoint": checkpoints[-1],
        "final_training_loss": round(last_loss.item(), 6),
        "wall_seconds": round(wall_seconds, 6),
    }
    append_log(
        log_path,
        {
            "stage": "variant_done",
            "variant": spec.key,
            "final_checkpoint": result["final_checkpoint"],
            "wall_seconds": result["wall_seconds"],
        },
    )

    del trainer
    del optimizer
    del model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if args.training_steps < WARMUP_STEPS:
        raise ValueError(
            f"training_steps must be at least {WARMUP_STEPS} so GraphTrainer.capture() can warm up."
        )
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/compute_frontier_sweep.py.")

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
    specs = variant_specs()

    append_log(
        args.log_path,
        {
            "stage": "start",
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "variant_count": len(specs),
        },
    )

    overall_started_at = perf_counter()
    results = [
        run_variant(
            spec,
            args=args,
            device=device,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            batch_schedule=batch_schedule,
            log_path=args.log_path,
        )
        for spec in specs
    ]
    overall_wall_seconds = perf_counter() - overall_started_at

    git_status_short = current_git_status_short()
    family_results = {
        family: [result for result in results if result["family"] == family]
        for family in sorted({result["family"] for result in results})
    }
    frontier_points = [
        {
            "variant": result["key"],
            "family": result["family"],
            "num_blocks": result["num_blocks"],
            "rates": result["rates"],
            "avg_evals_per_token": result["avg_evals_per_token"],
            "flops_proxy": result["flops_proxy"],
            "final_val_loss": result["final_checkpoint"]["val_loss"],
            "final_val_accuracy": result["final_checkpoint"]["val_accuracy"],
        }
        for result in results
    ]
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
            "d_model": D_MODEL,
            "feedforward_dim": FEEDFORWARD_DIM,
            "temporal_window": TEMPORAL_WINDOW,
            "num_heads": NUM_HEADS,
            "flops_proxy_formula": "avg_evals_per_token * 2 * d_model * feedforward_dim",
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
        "variants": [asdict(spec) for spec in specs],
        "results": results,
        "by_family": family_results,
        "frontier_points": frontier_points,
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "done",
            "report_path": str(args.report_path),
            "overall_wall_seconds": round(overall_wall_seconds, 6),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
