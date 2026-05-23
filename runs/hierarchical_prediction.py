from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from torch import Tensor

from core.fixed_window_char import load_dataset, set_seed
from core.model import ParallelDiagonalModel, count_parameters
from core.training import (
    GraphTrainer,
    capturable_adamw,
    current_git_sha,
    current_git_status_short,
    fixed_step_indices,
    write_json,
)
from experiments.fixed_multi_rate.hierarchical_prediction_aux import (
    AuxLossTensors,
    HierarchicalPredictionAux,
    HierarchicalPredictionGraphTrainer,
    evaluate_hierarchical_prediction,
)

CONTEXT_SIZE = 32
VOCAB_SIZE = 65
TRAINING_STEPS = 20_000
EVAL_INTERVAL = 5_000
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 3e-4
DEFAULT_SEEDS = (42, 43, 44)
WARMUP_STEPS = 3
NUM_BLOCKS = 4
RATES = (1, 2, 4, 8)
D_MODEL = 256
FEEDFORWARD_DIM = 512
D_Z = 32
LAMBDA_PROBE = 0.1
LAMBDA_HIER = 0.1
AUX_WARMUP_STEPS = 1_000
SANITY_STEPS = 100
OVERFIT_STEPS = 600
OVERFIT_BATCH_SIZE = 64


class Condition(StrEnum):
    BASELINE = "a_baseline"
    PROBE_ONLY = "b_probe_only"
    HIERARCHICAL = "c_hierarchical"


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    label: str
    include_probe: bool
    include_hierarchical: bool
    lambda_probe: float
    lambda_hier: float


def artifact_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "experiments" / "fixed_multi_rate" / "artifacts" / "hierarchical_prediction"


def append_log(log_path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def mean_rounded(values: list[float]) -> float:
    return round(mean(values), 6)


def round_list(values: list[float], digits: int = 6) -> list[float]:
    return [round(value, digits) for value in values]


def specs() -> dict[str, ConditionSpec]:
    return {
        Condition.BASELINE.value: ConditionSpec(
            key=Condition.BASELINE.value,
            label="hierarchical_prediction_a_baseline",
            include_probe=False,
            include_hierarchical=False,
            lambda_probe=0.0,
            lambda_hier=0.0,
        ),
        Condition.PROBE_ONLY.value: ConditionSpec(
            key=Condition.PROBE_ONLY.value,
            label="hierarchical_prediction_b_probe_only",
            include_probe=True,
            include_hierarchical=False,
            lambda_probe=LAMBDA_PROBE,
            lambda_hier=0.0,
        ),
        Condition.HIERARCHICAL.value: ConditionSpec(
            key=Condition.HIERARCHICAL.value,
            label="hierarchical_prediction_c_hierarchical",
            include_probe=True,
            include_hierarchical=True,
            lambda_probe=LAMBDA_PROBE,
            lambda_hier=LAMBDA_HIER,
        ),
    }


def parse_args() -> argparse.Namespace:
    default_artifact_dir = artifact_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sanity", "overfit", "full"], required=True)
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=[condition.value for condition in Condition],
    )
    parser.add_argument("--report-path", type=Path, default=default_artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=default_artifact_dir / "run.jsonl")
    return parser.parse_args()


def build_model(*, device: torch.device) -> ParallelDiagonalModel:
    return ParallelDiagonalModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        num_blocks=NUM_BLOCKS,
        rates=RATES,
        readout_mode="all",
        detach_lateral=False,
    ).to(device)


def build_aux_module(*, device: torch.device, spec: ConditionSpec) -> HierarchicalPredictionAux | None:
    if not spec.include_probe and not spec.include_hierarchical:
        return None
    return HierarchicalPredictionAux(
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        d_z=D_Z,
        vocab_size=VOCAB_SIZE,
        rates=RATES,
        include_hierarchical=spec.include_hierarchical,
    ).to(device)


def materialize_batch(
    inputs: Tensor,
    targets: Tensor,
    indices: Tensor,
) -> tuple[Tensor, Tensor]:
    return inputs[indices], targets[indices]


def batch_metrics(losses: AuxLossTensors) -> dict[str, float | list[float]]:
    return {
        "task_loss": round(losses.task_loss.item(), 6),
        "probe_loss": round(losses.probe_loss.item(), 6),
        "hier_loss": round(losses.hier_loss.item(), 6),
        "total_loss": round(losses.total_loss.item(), 6),
        "per_block_probe_losses": round_list(losses.per_block_probe_losses.cpu().tolist()),
        "per_pair_hier_losses": round_list(losses.per_pair_hier_losses.cpu().tolist()),
        "per_block_latent_dim_var": round_list(losses.per_block_latent_dim_var.cpu().tolist()),
    }


def checkpoint_metrics(
    *,
    model: ParallelDiagonalModel,
    aux_module: HierarchicalPredictionAux | None,
    val_inputs: Tensor,
    val_targets: Tensor,
    batch_size: int,
    step: int,
    wall_seconds: float,
) -> dict[str, float | int | list[float]]:
    metrics = evaluate_hierarchical_prediction(
        model=model,
        aux_module=aux_module,
        inputs=val_inputs,
        targets=val_targets,
        batch_size=batch_size,
    )
    return {
        "step": step,
        "val_loss": round(float(metrics["loss"]), 6),
        "val_accuracy": round(float(metrics["accuracy"]), 6),
        "probe_loss": round(float(metrics["probe_loss"]), 6),
        "hier_loss": round(float(metrics["hier_loss"]), 6),
        "per_block_probe_losses": round_list(list(metrics["per_block_probe_losses"])),
        "per_pair_hier_losses": round_list(list(metrics["per_pair_hier_losses"])),
        "per_block_latent_dim_var": round_list(list(metrics["per_block_latent_dim_var"])),
        "wall_seconds": round(wall_seconds, 6),
    }


def make_environment_report(device: torch.device) -> dict[str, object]:
    git_status_short = current_git_status_short()
    return {
        "git_sha": current_git_sha(),
        "git_working_tree_clean": git_status_short == [],
        "git_status_short": git_status_short,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
    }


def build_schedule(
    *,
    dataset_size: int,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[Tensor]:
    return fixed_step_indices(
        dataset_size,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )


def train_condition(
    *,
    seed: int,
    spec: ConditionSpec,
    args: argparse.Namespace,
    device: torch.device,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    log_path: Path,
) -> dict[str, object]:
    set_seed(seed)
    model = build_model(device=device)
    aux_module = build_aux_module(device=device, spec=spec)
    parameter_count = count_parameters(model) + (count_parameters(aux_module) if aux_module is not None else 0)

    batch_schedule = build_schedule(
        dataset_size=train_inputs.shape[0],
        steps=args.training_steps,
        batch_size=args.batch_size,
        seed=seed,
        device=device,
    )
    checkpoints = [
        checkpoint_metrics(
            model=model,
            aux_module=aux_module,
            val_inputs=val_inputs,
            val_targets=val_targets,
            batch_size=args.eval_batch_size,
            step=0,
            wall_seconds=0.0,
        )
    ]
    append_log(
        log_path,
        {
            "stage": "condition_started",
            "seed": seed,
            "condition": spec.key,
            "label": spec.label,
            "parameter_count": parameter_count,
            "initial_checkpoint": checkpoints[-1],
        },
    )

    started_at = perf_counter()
    if aux_module is None:
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
        trainer.capture(warmup_batches)
        last_batch_metrics = {
            "task_loss": round(trainer.static_loss.item(), 6),
            "probe_loss": 0.0,
            "hier_loss": 0.0,
            "total_loss": round(trainer.static_loss.item(), 6),
            "per_block_probe_losses": [0.0] * NUM_BLOCKS,
            "per_pair_hier_losses": [0.0] * (NUM_BLOCKS - 1),
            "per_block_latent_dim_var": [0.0] * NUM_BLOCKS,
        }
        for zero_based_index, indices in enumerate(batch_schedule[WARMUP_STEPS:], start=WARMUP_STEPS):
            batch_input, batch_target = materialize_batch(train_inputs, train_targets, indices)
            loss = trainer.step(batch_input, batch_target)
            step = zero_based_index + 1
            last_batch_metrics["task_loss"] = round(loss.item(), 6)
            last_batch_metrics["total_loss"] = round(loss.item(), 6)
            if step % args.eval_interval != 0 and step != args.training_steps:
                continue
            trainer.synchronize()
            checkpoint = checkpoint_metrics(
                model=model,
                aux_module=aux_module,
                val_inputs=val_inputs,
                val_targets=val_targets,
                batch_size=args.eval_batch_size,
                step=step,
                wall_seconds=perf_counter() - started_at,
            )
            checkpoints.append(checkpoint)
            append_log(log_path, {"stage": "checkpoint", "seed": seed, "condition": spec.key, **checkpoint})
        trainer.synchronize()
        del trainer
    else:
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(aux_module.parameters()),
            lr=args.learning_rate,
            capturable=True,
        )
        trainer = HierarchicalPredictionGraphTrainer(
            model=model,
            aux_module=aux_module,
            optimizer=optimizer,
            batch_size=args.batch_size,
            seq_len=CONTEXT_SIZE,
            device=device,
            lambda_probe=spec.lambda_probe,
            lambda_hier=spec.lambda_hier,
            aux_warmup_steps=AUX_WARMUP_STEPS,
        )
        warmup_batches = [
            (*materialize_batch(train_inputs, train_targets, indices), step)
            for step, indices in enumerate(batch_schedule[:WARMUP_STEPS], start=1)
        ]
        trainer.capture(warmup_batches)
        snapshot = trainer.snapshot()
        last_batch_metrics = batch_metrics(snapshot)
        for zero_based_index, indices in enumerate(batch_schedule[WARMUP_STEPS:], start=WARMUP_STEPS):
            step = zero_based_index + 1
            batch_input, batch_target = materialize_batch(train_inputs, train_targets, indices)
            snapshot = trainer.step(batch_input=batch_input, batch_target=batch_target, step=step)
            last_batch_metrics = batch_metrics(snapshot)
            if step % args.eval_interval != 0 and step != args.training_steps:
                continue
            trainer.synchronize()
            checkpoint = checkpoint_metrics(
                model=model,
                aux_module=aux_module,
                val_inputs=val_inputs,
                val_targets=val_targets,
                batch_size=args.eval_batch_size,
                step=step,
                wall_seconds=perf_counter() - started_at,
            )
            checkpoints.append(checkpoint)
            append_log(log_path, {"stage": "checkpoint", "seed": seed, "condition": spec.key, **checkpoint})
        trainer.synchronize()
        del trainer

    wall_seconds = perf_counter() - started_at
    result = {
        "condition": spec.key,
        "label": spec.label,
        "include_probe": spec.include_probe,
        "include_hierarchical": spec.include_hierarchical,
        "lambda_probe": spec.lambda_probe,
        "lambda_hier": spec.lambda_hier,
        "parameter_count": parameter_count,
        "checkpoints": checkpoints,
        "best_checkpoint": min(checkpoints, key=lambda checkpoint: float(checkpoint["val_loss"])),
        "final_checkpoint": checkpoints[-1],
        "final_training_metrics": last_batch_metrics,
        "wall_seconds": round(wall_seconds, 6),
        "model_mix_coefficients": model.mix_coefficients(),
    }
    append_log(
        log_path,
        {
            "stage": "condition_finished",
            "seed": seed,
            "condition": spec.key,
            "final_checkpoint": result["final_checkpoint"],
            "final_training_metrics": result["final_training_metrics"],
            "wall_seconds": result["wall_seconds"],
        },
    )

    del optimizer
    del aux_module
    del model
    del batch_schedule
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_full(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.training_steps < WARMUP_STEPS:
        raise ValueError(f"training_steps must be at least {WARMUP_STEPS}.")
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    resolved_specs = specs()
    for condition in args.conditions:
        if condition not in resolved_specs:
            raise ValueError(f"Unknown condition {condition!r}.")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text("", encoding="utf-8")

    device = torch.device("cuda")
    set_seed(args.seeds[0])
    (train_inputs, train_targets), (val_inputs, val_targets), vocab_size = load_dataset(context_size=CONTEXT_SIZE)
    if vocab_size > VOCAB_SIZE:
        raise ValueError(f"Loaded vocab_size {vocab_size} exceeds configured {VOCAB_SIZE}.")

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    val_inputs = val_inputs.to(device)
    val_targets = val_targets.to(device)

    append_log(
        args.log_path,
        {
            "stage": "run_started",
            "mode": args.mode,
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
            "conditions": args.conditions,
            "context_size": CONTEXT_SIZE,
            "vocab_size": vocab_size,
            "d_model": D_MODEL,
            "feedforward_dim": FEEDFORWARD_DIM,
            "num_blocks": NUM_BLOCKS,
            "rates": list(RATES),
            "d_z": D_Z,
            "lambda_probe": LAMBDA_PROBE,
            "lambda_hier": LAMBDA_HIER,
            "aux_warmup_steps": AUX_WARMUP_STEPS,
        },
    )

    overall_started_at = perf_counter()
    seed_results: list[dict[str, object]] = []
    for seed in args.seeds:
        condition_results = [
            train_condition(
                seed=seed,
                spec=resolved_specs[condition],
                args=args,
                device=device,
                train_inputs=train_inputs,
                train_targets=train_targets,
                val_inputs=val_inputs,
                val_targets=val_targets,
                log_path=args.log_path,
            )
            for condition in args.conditions
        ]
        seed_results.append({"seed": seed, "conditions": condition_results})

    averages = {}
    for condition in args.conditions:
        per_seed_results = [
            next(result for result in seed_result["conditions"] if result["condition"] == condition)
            for seed_result in seed_results
        ]
        averages[condition] = {
            "final_val_loss": mean_rounded([float(result["final_checkpoint"]["val_loss"]) for result in per_seed_results]),
            "final_val_accuracy": mean_rounded(
                [float(result["final_checkpoint"]["val_accuracy"]) for result in per_seed_results]
            ),
            "best_val_loss": mean_rounded([float(result["best_checkpoint"]["val_loss"]) for result in per_seed_results]),
            "probe_loss": mean_rounded([float(result["final_checkpoint"]["probe_loss"]) for result in per_seed_results]),
            "hier_loss": mean_rounded([float(result["final_checkpoint"]["hier_loss"]) for result in per_seed_results]),
            "wall_seconds": mean_rounded([float(result["wall_seconds"]) for result in per_seed_results]),
        }

    report = {
        "config": {
            "mode": args.mode,
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
            "conditions": args.conditions,
            "warmup_steps": WARMUP_STEPS,
            "context_size": CONTEXT_SIZE,
            "dataset_vocab_size": vocab_size,
            "model_vocab_size": VOCAB_SIZE,
            "d_model": D_MODEL,
            "feedforward_dim": FEEDFORWARD_DIM,
            "num_blocks": NUM_BLOCKS,
            "rates": list(RATES),
            "d_z": D_Z,
            "lambda_probe": LAMBDA_PROBE,
            "lambda_hier": LAMBDA_HIER,
            "aux_warmup_steps": AUX_WARMUP_STEPS,
        },
        "environment": make_environment_report(device),
        "timing": {"overall_wall_seconds": round(perf_counter() - overall_started_at, 6)},
        "conditions": {condition: asdict(resolved_specs[condition]) for condition in args.conditions},
        "seed_results": seed_results,
        "averages": averages,
    }
    write_json(args.report_path, report)
    append_log(args.log_path, {"stage": "done", "report_path": str(args.report_path), "averages": averages})
    return 0


def run_sanity(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text("", encoding="utf-8")

    device = torch.device("cuda")
    spec = specs()[Condition.HIERARCHICAL.value]
    set_seed(args.seeds[0])
    (train_inputs, train_targets), (val_inputs, val_targets), vocab_size = load_dataset(context_size=CONTEXT_SIZE)
    if vocab_size > VOCAB_SIZE:
        raise ValueError(f"Loaded vocab_size {vocab_size} exceeds configured {VOCAB_SIZE}.")

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    val_inputs = val_inputs.to(device)
    val_targets = val_targets.to(device)

    model = build_model(device=device)
    aux_module = build_aux_module(device=device, spec=spec)
    if aux_module is None:
        raise RuntimeError("Hierarchical sanity check requires aux module.")
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(aux_module.parameters()),
        lr=args.learning_rate,
        capturable=True,
    )
    trainer = HierarchicalPredictionGraphTrainer(
        model=model,
        aux_module=aux_module,
        optimizer=optimizer,
        batch_size=args.batch_size,
        seq_len=CONTEXT_SIZE,
        device=device,
        lambda_probe=spec.lambda_probe,
        lambda_hier=spec.lambda_hier,
        aux_warmup_steps=AUX_WARMUP_STEPS,
    )

    schedule = build_schedule(
        dataset_size=train_inputs.shape[0],
        steps=SANITY_STEPS,
        batch_size=args.batch_size,
        seed=args.seeds[0],
        device=device,
    )
    warmup_batches = [
        (*materialize_batch(train_inputs, train_targets, indices), step)
        for step, indices in enumerate(schedule[:WARMUP_STEPS], start=1)
    ]
    trainer.capture(warmup_batches)

    first_snapshot = trainer.snapshot()
    last_snapshot = first_snapshot
    sample_trace = [{"step": 0, **batch_metrics(first_snapshot)}]
    for zero_based_index, indices in enumerate(schedule[WARMUP_STEPS:], start=WARMUP_STEPS):
        step = zero_based_index + 1
        batch_input, batch_target = materialize_batch(train_inputs, train_targets, indices)
        snapshot = trainer.step(batch_input=batch_input, batch_target=batch_target, step=step)
        if step in {1, 10, 25, 50, 100}:
            sample_trace.append({"step": step, **batch_metrics(snapshot)})
        last_snapshot = snapshot

    trainer.synchronize()
    val_metrics = checkpoint_metrics(
        model=model,
        aux_module=aux_module,
        val_inputs=val_inputs,
        val_targets=val_targets,
        batch_size=args.eval_batch_size,
        step=SANITY_STEPS,
        wall_seconds=0.0,
    )
    checks = {
        "loss_decreased": last_snapshot.total_loss.item() < first_snapshot.total_loss.item(),
        "probe_loss_decreased": last_snapshot.probe_loss.item() < first_snapshot.probe_loss.item(),
        "hier_loss_decreased": last_snapshot.hier_loss.item() < first_snapshot.hier_loss.item(),
        "all_finite": bool(
            torch.isfinite(first_snapshot.total_loss)
            and torch.isfinite(last_snapshot.total_loss)
            and torch.isfinite(last_snapshot.per_block_latent_dim_var).all()
        ),
        "latent_variance_healthy": bool((last_snapshot.per_block_latent_dim_var > 1e-4).all().item()),
    }
    report = {
        "mode": "sanity",
        "condition": asdict(spec),
        "config": {
            "steps": SANITY_STEPS,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seeds[0],
            "context_size": CONTEXT_SIZE,
            "rates": list(RATES),
            "d_model": D_MODEL,
            "feedforward_dim": FEEDFORWARD_DIM,
            "d_z": D_Z,
        },
        "environment": make_environment_report(device),
        "first_batch": batch_metrics(first_snapshot),
        "last_batch": batch_metrics(last_snapshot),
        "sample_trace": sample_trace,
        "val_metrics": val_metrics,
        "checks": checks,
    }
    write_json(args.report_path, report)
    append_log(args.log_path, {"stage": "sanity_finished", "checks": checks, "report_path": str(args.report_path)})

    if not all(checks.values()):
        raise RuntimeError(f"Sanity check failed: {checks}")
    return 0


def run_overfit(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text("", encoding="utf-8")

    device = torch.device("cuda")
    spec = specs()[Condition.HIERARCHICAL.value]
    set_seed(args.seeds[0])
    (train_inputs, train_targets), (_, _), vocab_size = load_dataset(context_size=CONTEXT_SIZE)
    if vocab_size > VOCAB_SIZE:
        raise ValueError(f"Loaded vocab_size {vocab_size} exceeds configured {VOCAB_SIZE}.")

    batch_inputs = train_inputs[:OVERFIT_BATCH_SIZE].to(device)
    batch_targets = train_targets[:OVERFIT_BATCH_SIZE].to(device)
    model = build_model(device=device)
    aux_module = build_aux_module(device=device, spec=spec)
    if aux_module is None:
        raise RuntimeError("Hierarchical overfit check requires aux module.")
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(aux_module.parameters()), lr=args.learning_rate)

    trace: list[dict[str, float | int | list[float]]] = []
    final_losses = None
    for step in range(OVERFIT_STEPS + 1):
        logits, state = model.forward_with_state(batch_inputs)
        warmup_scale = torch.tensor(min(step / AUX_WARMUP_STEPS, 1.0), device=device, dtype=torch.float32)
        losses = aux_module.compute_losses(
            state=state,
            tokens=batch_inputs,
            final_targets=batch_targets,
            final_logits=logits,
            lambda_probe=spec.lambda_probe,
            lambda_hier=spec.lambda_hier,
            warmup_scale=warmup_scale,
        )
        accuracy = (logits.argmax(dim=1) == batch_targets).float().mean().item()
        if step % 50 == 0 or step == OVERFIT_STEPS:
            trace.append(
                {
                    "step": step,
                    **batch_metrics(losses),
                    "accuracy": round(accuracy, 6),
                }
            )
        if step == OVERFIT_STEPS:
            final_losses = losses
            break
        optimizer.zero_grad(set_to_none=True)
        losses.total_loss.backward()
        optimizer.step()

    if final_losses is None:
        raise RuntimeError("Overfit check did not complete.")
    final_accuracy = trace[-1]["accuracy"]
    report = {
        "mode": "overfit",
        "condition": asdict(spec),
        "config": {
            "steps": OVERFIT_STEPS,
            "batch_size": OVERFIT_BATCH_SIZE,
            "learning_rate": args.learning_rate,
            "seed": args.seeds[0],
        },
        "environment": make_environment_report(device),
        "trace": trace,
        "checks": {
            "memorized": bool(float(final_accuracy) >= 0.98 and final_losses.total_loss.item() < 0.2),
            "all_finite": bool(
                torch.isfinite(final_losses.total_loss)
                and torch.isfinite(final_losses.per_block_latent_dim_var).all()
            ),
        },
    }
    write_json(args.report_path, report)
    append_log(args.log_path, {"stage": "overfit_finished", "checks": report["checks"], "report_path": str(args.report_path)})
    if not all(report["checks"].values()):
        raise RuntimeError(f"Overfit check failed: {report['checks']}")
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "sanity":
        return run_sanity(args)
    if args.mode == "overfit":
        return run_overfit(args)
    return run_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
