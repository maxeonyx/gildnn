from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

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

CONTEXT_SIZE = 128
VOCAB_SIZE = 67
TRAINING_STEPS = 20_000
EVAL_INTERVAL = 2_000
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
SEED = 42
WARMUP_STEPS = 3
NUM_HEADS = 4
RATES = (1, 4, 16, 32)
D_MODEL = 256
FEEDFORWARD_DIM = 512


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    temporal_window: int
    rates: tuple[int, ...]
    d_model: int
    feedforward_dim: int

    @property
    def num_blocks(self) -> int:
        return len(self.rates)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = (
        repo_root
        / "experiments"
        / "fixed_multi_rate"
        / "artifacts"
        / "temporal_window_scaling"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.jsonl")
    return parser.parse_args()


def variant_specs() -> list[VariantSpec]:
    return [
        VariantSpec(
            key="multirate_rate141632_d256_ff512_ctx128_tw4",
            label="multirate_rate141632_d256_ff512_ctx128_tw4",
            temporal_window=4,
            rates=RATES,
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
        ),
        VariantSpec(
            key="multirate_rate141632_d256_ff512_ctx128_tw16",
            label="multirate_rate141632_d256_ff512_ctx128_tw16",
            temporal_window=16,
            rates=RATES,
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
        ),
        VariantSpec(
            key="multirate_rate141632_d256_ff512_ctx128_tw32",
            label="multirate_rate141632_d256_ff512_ctx128_tw32",
            temporal_window=32,
            rates=RATES,
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
        ),
        VariantSpec(
            key="multirate_rate141632_d256_ff512_ctx128_tw64",
            label="multirate_rate141632_d256_ff512_ctx128_tw64",
            temporal_window=64,
            rates=RATES,
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
        ),
        VariantSpec(
            key="multirate_rate141632_d256_ff512_ctx128_tw128",
            label="multirate_rate141632_d256_ff512_ctx128_tw128",
            temporal_window=128,
            rates=RATES,
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
        ),
    ]


def build_model(*, spec: VariantSpec, device: torch.device) -> MultiRateResidualModel:
    return MultiRateResidualModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=spec.d_model,
        feedforward_dim=spec.feedforward_dim,
        temporal_window=spec.temporal_window,
        num_heads=NUM_HEADS,
        rates=spec.rates,
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
    wall_seconds_at_eval: float,
) -> dict[str, float | int]:
    metrics = evaluate_model(model, val_inputs, val_targets, batch_size=batch_size)
    return {
        "step": step,
        "val_loss": round(metrics["loss"], 6),
        "val_accuracy": round(metrics["accuracy"], 6),
        "wall_seconds_at_eval": round(wall_seconds_at_eval, 6),
    }


def evaluate_checkpoint(
    model: torch.nn.Module,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    batch_size: int,
    step: int,
    device: torch.device,
    run_started_at: float,
) -> dict[str, float | int]:
    checkpoint = checkpoint_metrics(
        model,
        val_inputs,
        val_targets,
        batch_size=batch_size,
        step=step,
        wall_seconds_at_eval=0.0,
    )
    torch.cuda.synchronize(device)
    checkpoint["wall_seconds_at_eval"] = round(perf_counter() - run_started_at, 6)
    return checkpoint


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


def is_oom_error(error: BaseException) -> bool:
    if hasattr(torch, "OutOfMemoryError") and isinstance(error, torch.OutOfMemoryError):
        return True
    if not isinstance(error, RuntimeError):
        return False
    return "out of memory" in str(error).lower()


def skip_result(
    spec: VariantSpec,
    *,
    parameter_count: int | None,
    flops_proxy_value: float,
    error: BaseException,
    wall_seconds: float,
) -> dict[str, object]:
    eval_counts = evals_per_block(context_size=CONTEXT_SIZE, rates=spec.rates)
    return {
        **asdict(spec),
        "status": "oom_skipped",
        "skip_reason": str(error),
        "context_size": CONTEXT_SIZE,
        "num_blocks": spec.num_blocks,
        "parameter_count": parameter_count,
        "evals_per_block": eval_counts,
        "total_block_evals_per_forward": sum(eval_counts),
        "avg_evals_per_token": round(
            average_evals_per_token(context_size=CONTEXT_SIZE, rates=spec.rates),
            6,
        ),
        "flops_proxy": round(flops_proxy_value, 6),
        "checkpoints": [],
        "best_checkpoint": None,
        "final_checkpoint": None,
        "final_training_loss": None,
        "wall_seconds": round(wall_seconds, 6),
    }


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
    avg_eval_per_token = average_evals_per_token(context_size=CONTEXT_SIZE, rates=spec.rates)
    flops_proxy_value = flops_proxy(
        avg_evals_per_token=avg_eval_per_token,
        d_model=spec.d_model,
        feedforward_dim=spec.feedforward_dim,
    )
    variant_started_at = perf_counter()

    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    trainer: GraphTrainer | None = None
    parameter_count: int | None = None

    try:
        model = build_model(spec=spec, device=device)
        parameter_count = count_parameters(model)
        optimizer = capturable_adamw(
            model,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
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
            evaluate_checkpoint(
                model,
                val_inputs,
                val_targets,
                batch_size=args.eval_batch_size,
                step=0,
                device=device,
                run_started_at=variant_started_at,
            )
        ]

        append_log(
            log_path,
            {
                "stage": "variant_start",
                "variant": spec.key,
                "context_size": CONTEXT_SIZE,
                "num_blocks": spec.num_blocks,
                "rates": list(spec.rates),
                "d_model": spec.d_model,
                "feedforward_dim": spec.feedforward_dim,
                "temporal_window": spec.temporal_window,
                "parameter_count": parameter_count,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "avg_evals_per_token": round(avg_eval_per_token, 6),
                "flops_proxy": round(flops_proxy_value, 6),
            },
        )
        append_log(
            log_path,
            {
                "stage": "checkpoint",
                "variant": spec.key,
                "train_loss": None,
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
            checkpoint = evaluate_checkpoint(
                model,
                val_inputs,
                val_targets,
                batch_size=args.eval_batch_size,
                step=step,
                device=device,
                run_started_at=variant_started_at,
            )
            checkpoints.append(checkpoint)
            append_log(
                log_path,
                {
                    "stage": "checkpoint",
                    "variant": spec.key,
                    "train_loss": round(last_loss.item(), 6),
                    **checkpoint,
                },
            )

        trainer.synchronize()
        wall_seconds = perf_counter() - variant_started_at
        eval_counts = evals_per_block(context_size=CONTEXT_SIZE, rates=spec.rates)
        result = {
            **asdict(spec),
            "status": "ok",
            "context_size": CONTEXT_SIZE,
            "num_blocks": spec.num_blocks,
            "parameter_count": parameter_count,
            "evals_per_block": eval_counts,
            "total_block_evals_per_forward": sum(eval_counts),
            "avg_evals_per_token": round(avg_eval_per_token, 6),
            "flops_proxy": round(flops_proxy_value, 6),
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
                "status": result["status"],
                "final_checkpoint": result["final_checkpoint"],
                "best_checkpoint": result["best_checkpoint"],
                "final_training_loss": result["final_training_loss"],
                "wall_seconds": result["wall_seconds"],
            },
        )
        return result
    except BaseException as error:
        if not is_oom_error(error):
            raise
        wall_seconds = perf_counter() - variant_started_at
        result = skip_result(
            spec,
            parameter_count=parameter_count,
            flops_proxy_value=flops_proxy_value,
            error=error,
            wall_seconds=wall_seconds,
        )
        append_log(
            log_path,
            {
                "stage": "variant_skipped",
                "variant": spec.key,
                "status": result["status"],
                "skip_reason": result["skip_reason"],
                "wall_seconds": result["wall_seconds"],
            },
        )
        return result
    finally:
        del trainer
        del optimizer
        del model
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    if args.training_steps < WARMUP_STEPS:
        raise ValueError(
            f"training_steps must be at least {WARMUP_STEPS} so GraphTrainer.capture() can warm up."
        )
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/temporal_window_scaling.py.")

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
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "context_size": CONTEXT_SIZE,
            "variant_count": len(specs),
            "temporal_windows": [spec.temporal_window for spec in specs],
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
    successful_results = [result for result in results if result["status"] == "ok"]
    skipped_results = [result for result in results if result["status"] != "ok"]
    report = {
        "config": {
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "warmup_steps": WARMUP_STEPS,
            "context_size": CONTEXT_SIZE,
            "dataset_vocab_size": vocab_size,
            "model_vocab_size": VOCAB_SIZE,
            "schedule_seed": args.seed,
            "trainer": "GraphTrainer",
            "num_heads": NUM_HEADS,
            "rates": list(RATES),
            "d_model": D_MODEL,
            "feedforward_dim": FEEDFORWARD_DIM,
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
        "variants": [
            {
                **asdict(spec),
                "context_size": CONTEXT_SIZE,
                "num_blocks": spec.num_blocks,
                "avg_evals_per_token": round(
                    average_evals_per_token(context_size=CONTEXT_SIZE, rates=spec.rates),
                    6,
                ),
                "flops_proxy": round(
                    flops_proxy(
                        avg_evals_per_token=average_evals_per_token(
                            context_size=CONTEXT_SIZE,
                            rates=spec.rates,
                        ),
                        d_model=spec.d_model,
                        feedforward_dim=spec.feedforward_dim,
                    ),
                    6,
                ),
            }
            for spec in specs
        ],
        "results": results,
        "successful_results": successful_results,
        "skipped_results": skipped_results,
        "comparison": {
            "baseline_key": "multirate_rate141632_d256_ff512_ctx128_tw4",
            "sweep_keys": [spec.key for spec in specs],
        },
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "done",
            "report_path": str(args.report_path),
            "overall_wall_seconds": round(overall_wall_seconds, 6),
            "successful_variants": len(successful_results),
            "skipped_variants": len(skipped_results),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
