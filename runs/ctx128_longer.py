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

CONTEXT_SIZE = 128
VOCAB_SIZE = 67
TRAINING_STEPS = 50_000
EVAL_INTERVAL = 5_000
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
PEAK_LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 0.0
WEIGHT_DECAY = 1e-2
SEED = 42
GRAPH_WARMUP_STEPS = 3
COSINE_WARMUP_STEPS = 1_000
TEMPORAL_WINDOW = 4
NUM_HEADS = 4


@dataclass(frozen=True)
class VariantSpec:
    key: str
    family: str
    label: str
    class_name: str
    rates: tuple[int, ...]
    readout_mode: str
    d_model: int
    feedforward_dim: int
    trainer: str
    scheduler: str

    @property
    def num_blocks(self) -> int:
        return len(self.rates)


class WarmupThenCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        peak_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be positive, got {warmup_steps}.")
        if total_steps <= warmup_steps:
            raise ValueError(
                f"total_steps must exceed warmup_steps, got total_steps={total_steps}, warmup_steps={warmup_steps}."
            )

        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_index = 0
        self.cosine_scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None = None
        self._set_lr(0.0)

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    @property
    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def step(self) -> float:
        self.step_index += 1
        if self.step_index <= self.warmup_steps:
            self._set_lr(self.peak_lr * (self.step_index / self.warmup_steps))
            return self.current_lr

        if self.cosine_scheduler is None:
            self._set_lr(self.peak_lr)
            self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_steps - self.warmup_steps,
                eta_min=self.min_lr,
            )

        self.cosine_scheduler.step()
        return self.current_lr


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "fixed_multi_rate" / "artifacts" / "ctx128_longer"
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=PEAK_LEARNING_RATE)
    parser.add_argument("--min-learning-rate", type=float, default=MIN_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-steps", type=int, default=COSINE_WARMUP_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.jsonl")
    return parser.parse_args()


def variant_specs() -> list[VariantSpec]:
    return [
        VariantSpec(
            key="parallel_4block_rate141632_d256_ff512_ctx128_50k_constant",
            family="parallel_multi_rate",
            label="parallel_4block_rate141632_all_d256_ff512_ctx128_50k_constant",
            class_name="ParallelDiagonalModel",
            rates=(1, 4, 16, 32),
            readout_mode="all",
            d_model=256,
            feedforward_dim=512,
            trainer="GraphTrainer",
            scheduler="constant",
        ),
        VariantSpec(
            key="parallel_4block_rate141632_d256_ff512_ctx128_50k_cosine",
            family="parallel_multi_rate",
            label="parallel_4block_rate141632_all_d256_ff512_ctx128_50k_cosine",
            class_name="ParallelDiagonalModel",
            rates=(1, 4, 16, 32),
            readout_mode="all",
            d_model=256,
            feedforward_dim=512,
            trainer="manual_adamw",
            scheduler="warmup_cosine",
        ),
        VariantSpec(
            key="sequential_6block_rate1_d256_ff512_ctx128_50k_cosine",
            family="sequential_all_rate_1",
            label="sequential_6block_rate111111_d256_ff512_ctx128_50k_cosine",
            class_name="MultiRateResidualModel",
            rates=(1, 1, 1, 1, 1, 1),
            readout_mode="last",
            d_model=256,
            feedforward_dim=512,
            trainer="manual_adamw",
            scheduler="warmup_cosine",
        ),
    ]


def build_model(*, spec: VariantSpec, device: torch.device) -> torch.nn.Module:
    if spec.class_name == "MultiRateResidualModel":
        return MultiRateResidualModel(
            vocab_size=VOCAB_SIZE,
            context_size=CONTEXT_SIZE,
            d_model=spec.d_model,
            feedforward_dim=spec.feedforward_dim,
            temporal_window=TEMPORAL_WINDOW,
            num_heads=NUM_HEADS,
            rates=spec.rates,
        ).to(device)
    if spec.class_name == "ParallelDiagonalModel":
        return ParallelDiagonalModel(
            vocab_size=VOCAB_SIZE,
            context_size=CONTEXT_SIZE,
            d_model=spec.d_model,
            feedforward_dim=spec.feedforward_dim,
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


def checkpoint_metrics(
    model: torch.nn.Module,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    batch_size: int,
    step: int,
    wall_seconds_at_eval: float,
    current_lr: float,
) -> dict[str, float | int]:
    metrics = evaluate_model(model, val_inputs, val_targets, batch_size=batch_size)
    return {
        "step": step,
        "val_loss": round(metrics["loss"], 6),
        "val_accuracy": round(metrics["accuracy"], 6),
        "wall_seconds_at_eval": round(wall_seconds_at_eval, 6),
        "current_lr": round(current_lr, 12),
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
    current_lr: float,
) -> dict[str, float | int]:
    checkpoint = checkpoint_metrics(
        model,
        val_inputs,
        val_targets,
        batch_size=batch_size,
        step=step,
        wall_seconds_at_eval=0.0,
        current_lr=current_lr,
    )
    torch.cuda.synchronize(device)
    checkpoint["wall_seconds_at_eval"] = round(perf_counter() - run_started_at, 6)
    return checkpoint


def run_graph_variant(
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
    optimizer = capturable_adamw(model, lr=args.learning_rate, weight_decay=args.weight_decay)
    trainer = GraphTrainer(
        model,
        optimizer,
        batch_size=args.batch_size,
        seq_len=CONTEXT_SIZE,
        device=device,
    )

    warmup_batches = [
        materialize_batch(train_inputs, train_targets, indices)
        for indices in batch_schedule[:GRAPH_WARMUP_STEPS]
    ]
    avg_eval_per_token = average_evals_per_token(context_size=CONTEXT_SIZE, rates=spec.rates)
    variant_started_at = perf_counter()
    checkpoints = [
        evaluate_checkpoint(
            model,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=0,
            device=device,
            run_started_at=variant_started_at,
            current_lr=float(args.learning_rate),
        )
    ]

    append_log(
        log_path,
        {
            "stage": "variant_start",
            "variant": spec.key,
            "family": spec.family,
            "class_name": spec.class_name,
            "training_steps": args.training_steps,
            "seed": args.seed,
            "context_size": CONTEXT_SIZE,
            "num_blocks": spec.num_blocks,
            "rates": list(spec.rates),
            "readout_mode": spec.readout_mode,
            "d_model": spec.d_model,
            "feedforward_dim": spec.feedforward_dim,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "avg_evals_per_token": round(avg_eval_per_token, 6),
            "flops_proxy": round(
                flops_proxy(
                    avg_evals_per_token=avg_eval_per_token,
                    d_model=spec.d_model,
                    feedforward_dim=spec.feedforward_dim,
                ),
                6,
            ),
            "trainer": spec.trainer,
            "scheduler": spec.scheduler,
        },
    )
    append_log(
        log_path,
        {
            "stage": "checkpoint",
            "variant": spec.key,
            "family": spec.family,
            "training_steps": args.training_steps,
            "seed": args.seed,
            "train_loss": None,
            **checkpoints[-1],
        },
    )

    trainer.capture(warmup_batches)
    last_loss = trainer.static_loss.detach().clone()

    for zero_based_index, indices in enumerate(batch_schedule[GRAPH_WARMUP_STEPS:], start=GRAPH_WARMUP_STEPS):
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
            current_lr=float(optimizer.param_groups[0]["lr"]),
        )
        checkpoints.append(checkpoint)
        append_log(
            log_path,
            {
                "stage": "checkpoint",
                "variant": spec.key,
                "family": spec.family,
                "training_steps": args.training_steps,
                "seed": args.seed,
                "train_loss": round(last_loss.item(), 6),
                **checkpoint,
            },
        )

    trainer.synchronize()
    wall_seconds = perf_counter() - variant_started_at
    eval_counts = evals_per_block(context_size=CONTEXT_SIZE, rates=spec.rates)
    result = {
        **asdict(spec),
        "training_steps": args.training_steps,
        "seed": args.seed,
        "context_size": CONTEXT_SIZE,
        "parameter_count": count_parameters(model),
        "evals_per_block": eval_counts,
        "total_block_evals_per_forward": sum(eval_counts),
        "avg_evals_per_token": round(avg_eval_per_token, 6),
        "flops_proxy": round(
            flops_proxy(
                avg_evals_per_token=avg_eval_per_token,
                d_model=spec.d_model,
                feedforward_dim=spec.feedforward_dim,
            ),
            6,
        ),
        "checkpoints": checkpoints,
        "best_checkpoint": min(checkpoints, key=lambda checkpoint: checkpoint["val_loss"]),
        "final_checkpoint": checkpoints[-1],
        "final_training_loss": round(last_loss.item(), 6),
        "final_learning_rate": round(float(optimizer.param_groups[0]["lr"]), 12),
        "wall_seconds": round(wall_seconds, 6),
    }
    append_log(
        log_path,
        {
            "stage": "variant_done",
            "variant": spec.key,
            "family": spec.family,
            "training_steps": args.training_steps,
            "seed": args.seed,
            "final_checkpoint": result["final_checkpoint"],
            "best_checkpoint": result["best_checkpoint"],
            "final_training_loss": result["final_training_loss"],
            "wall_seconds": result["wall_seconds"],
        },
    )

    del trainer
    del optimizer
    del model
    torch.cuda.empty_cache()
    return result


def run_manual_variant(
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,
        weight_decay=args.weight_decay,
    )
    scheduler = WarmupThenCosineScheduler(
        optimizer,
        peak_lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        total_steps=args.training_steps,
    )

    avg_eval_per_token = average_evals_per_token(context_size=CONTEXT_SIZE, rates=spec.rates)
    variant_started_at = perf_counter()
    checkpoints = [
        evaluate_checkpoint(
            model,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=0,
            device=device,
            run_started_at=variant_started_at,
            current_lr=scheduler.current_lr,
        )
    ]
    last_loss = torch.zeros((), device=device)

    append_log(
        log_path,
        {
            "stage": "variant_start",
            "variant": spec.key,
            "family": spec.family,
            "class_name": spec.class_name,
            "training_steps": args.training_steps,
            "seed": args.seed,
            "context_size": CONTEXT_SIZE,
            "num_blocks": spec.num_blocks,
            "rates": list(spec.rates),
            "readout_mode": spec.readout_mode,
            "d_model": spec.d_model,
            "feedforward_dim": spec.feedforward_dim,
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "avg_evals_per_token": round(avg_eval_per_token, 6),
            "flops_proxy": round(
                flops_proxy(
                    avg_evals_per_token=avg_eval_per_token,
                    d_model=spec.d_model,
                    feedforward_dim=spec.feedforward_dim,
                ),
                6,
            ),
            "trainer": spec.trainer,
            "scheduler": spec.scheduler,
        },
    )
    append_log(
        log_path,
        {
            "stage": "checkpoint",
            "variant": spec.key,
            "family": spec.family,
            "training_steps": args.training_steps,
            "seed": args.seed,
            "train_loss": None,
            **checkpoints[-1],
        },
    )

    model.train()
    for zero_based_index, indices in enumerate(batch_schedule):
        batch_input, batch_target = materialize_batch(train_inputs, train_targets, indices)
        step = zero_based_index + 1

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_input)
        loss = torch.nn.functional.cross_entropy(logits, batch_target)
        loss.backward()
        optimizer.step()
        last_loss = loss.detach()

        if step % args.eval_interval != 0 and step != args.training_steps:
            continue

        checkpoint = evaluate_checkpoint(
            model,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=step,
            device=device,
            run_started_at=variant_started_at,
            current_lr=scheduler.current_lr,
        )
        checkpoints.append(checkpoint)
        append_log(
            log_path,
            {
                "stage": "checkpoint",
                "variant": spec.key,
                "family": spec.family,
                "training_steps": args.training_steps,
                "seed": args.seed,
                "train_loss": round(last_loss.item(), 6),
                **checkpoint,
            },
        )

    torch.cuda.synchronize(device)
    wall_seconds = perf_counter() - variant_started_at
    eval_counts = evals_per_block(context_size=CONTEXT_SIZE, rates=spec.rates)
    result = {
        **asdict(spec),
        "training_steps": args.training_steps,
        "seed": args.seed,
        "context_size": CONTEXT_SIZE,
        "parameter_count": count_parameters(model),
        "evals_per_block": eval_counts,
        "total_block_evals_per_forward": sum(eval_counts),
        "avg_evals_per_token": round(avg_eval_per_token, 6),
        "flops_proxy": round(
            flops_proxy(
                avg_evals_per_token=avg_eval_per_token,
                d_model=spec.d_model,
                feedforward_dim=spec.feedforward_dim,
            ),
            6,
        ),
        "checkpoints": checkpoints,
        "best_checkpoint": min(checkpoints, key=lambda checkpoint: checkpoint["val_loss"]),
        "final_checkpoint": checkpoints[-1],
        "final_training_loss": round(last_loss.item(), 6),
        "final_learning_rate": round(scheduler.current_lr, 12),
        "wall_seconds": round(wall_seconds, 6),
    }
    append_log(
        log_path,
        {
            "stage": "variant_done",
            "variant": spec.key,
            "family": spec.family,
            "training_steps": args.training_steps,
            "seed": args.seed,
            "final_checkpoint": result["final_checkpoint"],
            "best_checkpoint": result["best_checkpoint"],
            "final_training_loss": result["final_training_loss"],
            "wall_seconds": result["wall_seconds"],
        },
    )

    del optimizer
    del model
    torch.cuda.empty_cache()
    return result


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
    if spec.trainer == "GraphTrainer":
        return run_graph_variant(
            spec,
            args=args,
            device=device,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            batch_schedule=batch_schedule,
            log_path=log_path,
        )
    if spec.trainer == "manual_adamw":
        return run_manual_variant(
            spec,
            args=args,
            device=device,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            batch_schedule=batch_schedule,
            log_path=log_path,
        )
    raise ValueError(f"Unsupported trainer {spec.trainer!r}.")


def main() -> int:
    args = parse_args()
    if args.training_steps <= 0:
        raise ValueError(f"training_steps must be positive, got {args.training_steps}.")
    if args.training_steps < GRAPH_WARMUP_STEPS:
        raise ValueError(
            f"training_steps must be at least {GRAPH_WARMUP_STEPS} so GraphTrainer.capture() can warm up."
        )
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    if args.warmup_steps <= 0:
        raise ValueError(f"warmup_steps must be positive, got {args.warmup_steps}.")
    if args.training_steps <= args.warmup_steps:
        raise ValueError(
            f"training_steps must exceed warmup_steps, got training_steps={args.training_steps}, warmup_steps={args.warmup_steps}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/ctx128_longer.py.")

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
            "min_learning_rate": args.min_learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "seed": args.seed,
            "context_size": CONTEXT_SIZE,
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
    by_family = {
        family: [result for result in results if result["family"] == family]
        for family in sorted({result["family"] for result in results})
    }
    report = {
        "config": {
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "graph_warmup_steps": GRAPH_WARMUP_STEPS,
            "seed": args.seed,
            "context_size": CONTEXT_SIZE,
            "dataset_vocab_size": vocab_size,
            "model_vocab_size": VOCAB_SIZE,
            "schedule_seed": args.seed,
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
        "by_family": by_family,
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
