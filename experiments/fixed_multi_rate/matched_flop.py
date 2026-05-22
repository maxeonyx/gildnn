from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import load_dataset, resolve_device, set_seed
from core.model import count_parameters
from core.training import evaluate_model, fixed_step_indices

from .run import (
    RunConfig,
    build_model,
    current_git_sha,
    current_git_status_short,
    replace_config,
    write_json,
)


MULTI_RATES = (1, 2, 4, 8)
CALIBRATION_CANDIDATE_D_MODELS = (108, 112, 116, 120)
SHALLOW_RATES = (1, 1, 1)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "experiments" / "fixed_multi_rate" / "artifacts" / "matched_flop",
    )
    parser.add_argument("--training-steps", type=int, default=2_000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--calibration-warmup-steps", type=int, default=50)
    parser.add_argument("--calibration-timed-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    return parser.parse_args()


def model_config(
    base_config: RunConfig,
    *,
    d_model: int,
    rates: tuple[int, ...],
) -> RunConfig:
    return replace_config(
        base_config,
        d_model=d_model,
        feedforward_dim=d_model * 2,
        num_blocks=len(rates),
        control_rates=rates,
        multi_rates=rates,
    )


def build_seeded_model(
    *,
    vocab_size: int,
    config: RunConfig,
    rates: tuple[int, ...],
    device: torch.device,
    seed: int,
) -> nn.Module:
    set_seed(seed)
    return build_model(
        vocab_size=vocab_size,
        config=config,
        rates=rates,
        device=device,
        diagonal_enabled=False,
    )


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_training_step_ms(
    *,
    model: nn.Module,
    train_inputs: Tensor,
    train_targets: Tensor,
    config: RunConfig,
    warmup_steps: int,
    timed_steps: int,
    schedule_seed: int,
) -> float:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_steps = warmup_steps + timed_steps
    batch_schedule = fixed_step_indices(
        train_inputs.shape[0],
        steps=total_steps,
        batch_size=config.batch_size,
        seed=schedule_seed,
        device=train_inputs.device,
    )

    for batch_indices in batch_schedule[:warmup_steps]:
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch_inputs), batch_targets)
        loss.backward()
        optimizer.step()

    synchronize_if_needed(train_inputs.device)
    started_at = time.perf_counter()
    for batch_indices in batch_schedule[warmup_steps:]:
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch_inputs), batch_targets)
        loss.backward()
        optimizer.step()
    synchronize_if_needed(train_inputs.device)
    total_ms = (time.perf_counter() - started_at) * 1000.0
    return total_ms / timed_steps


def run_calibration(
    *,
    base_config: RunConfig,
    vocab_size: int,
    device: torch.device,
    train_inputs: Tensor,
    train_targets: Tensor,
    warmup_steps: int,
    timed_steps: int,
    seed: int,
) -> dict[str, object]:
    results: list[dict[str, object]] = []

    multi_config = model_config(base_config, d_model=128, rates=MULTI_RATES)
    multi_model = build_seeded_model(
        vocab_size=vocab_size,
        config=multi_config,
        rates=MULTI_RATES,
        device=device,
        seed=seed,
    )
    multi_ms = measure_training_step_ms(
        model=multi_model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        config=multi_config,
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
        schedule_seed=seed,
    )
    results.append(
        {
            "label": "multi_rate",
            "num_blocks": len(MULTI_RATES),
            "d_model": 128,
            "rates": list(MULTI_RATES),
            "parameter_count": count_parameters(multi_model),
            "training_step_ms": round(multi_ms, 6),
        }
    )
    del multi_model

    candidate_results: list[dict[str, object]] = []
    for d_model in CALIBRATION_CANDIDATE_D_MODELS:
        rates = (1, 1, 1, 1)
        candidate_config = model_config(base_config, d_model=d_model, rates=rates)
        candidate_model = build_seeded_model(
            vocab_size=vocab_size,
            config=candidate_config,
            rates=rates,
            device=device,
            seed=seed,
        )
        candidate_ms = measure_training_step_ms(
            model=candidate_model,
            train_inputs=train_inputs,
            train_targets=train_targets,
            config=candidate_config,
            warmup_steps=warmup_steps,
            timed_steps=timed_steps,
            schedule_seed=seed,
        )
        candidate_result = {
            "label": f"all_rate_1_d{d_model}",
            "num_blocks": 4,
            "d_model": d_model,
            "rates": [1, 1, 1, 1],
            "parameter_count": count_parameters(candidate_model),
            "training_step_ms": round(candidate_ms, 6),
        }
        candidate_results.append(candidate_result)
        results.append(candidate_result)
        del candidate_model

    selected = min(
        candidate_results,
        key=lambda result: (
            abs(result["training_step_ms"] - round(multi_ms, 6)),
            abs(result["d_model"] - 128),
        ),
    )
    return {
        "measurement": {
            "warmup_steps": warmup_steps,
            "timed_steps": timed_steps,
            "batch_size": base_config.batch_size,
        },
        "results": results,
        "selected_match": {
            **selected,
            "abs_training_step_ms_delta_vs_multi": round(abs(selected["training_step_ms"] - round(multi_ms, 6)), 6),
            "signed_training_step_ms_delta_vs_multi": round(selected["training_step_ms"] - round(multi_ms, 6), 6),
        },
    }


def evaluate_checkpoint(model: nn.Module, val_inputs: Tensor, val_targets: Tensor, *, batch_size: int) -> dict[str, float]:
    metrics = evaluate_model(model, val_inputs, val_targets, batch_size=batch_size)
    return {
        "val_loss": round(metrics["loss"], 6),
        "val_accuracy": round(metrics["accuracy"], 6),
    }


def train_single_model(
    *,
    label: str,
    vocab_size: int,
    config: RunConfig,
    rates: tuple[int, ...],
    device: torch.device,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    training_schedule: list[Tensor],
    seed: int,
) -> dict[str, object]:
    model = build_seeded_model(
        vocab_size=vocab_size,
        config=config,
        rates=rates,
        device=device,
        seed=seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoints = [
        {
            "step": 0,
            **evaluate_checkpoint(model, val_inputs, val_targets, batch_size=config.eval_batch_size),
        }
    ]
    print(json.dumps({"stage": "train", "model": label, "checkpoint": checkpoints[-1]}), flush=True)

    synchronize_if_needed(device)
    wall_started_at = time.perf_counter()
    update_ms_total = 0.0
    segment_started_at = time.perf_counter()

    for step, batch_indices in enumerate(training_schedule, start=1):
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch_inputs), batch_targets)
        loss.backward()
        optimizer.step()

        if step % config.eval_interval == 0 or step == config.training_steps:
            synchronize_if_needed(device)
            update_ms_total += (time.perf_counter() - segment_started_at) * 1000.0
            checkpoint = {
                "step": step,
                **evaluate_checkpoint(model, val_inputs, val_targets, batch_size=config.eval_batch_size),
            }
            checkpoints.append(checkpoint)
            print(json.dumps({"stage": "train", "model": label, "checkpoint": checkpoint}), flush=True)
            segment_started_at = time.perf_counter()

    synchronize_if_needed(device)
    wall_ms_total = (time.perf_counter() - wall_started_at) * 1000.0
    best_checkpoint = min(checkpoints, key=lambda checkpoint: checkpoint["val_loss"])
    result = {
        "label": label,
        "num_blocks": len(rates),
        "d_model": config.d_model,
        "feedforward_dim": config.feedforward_dim,
        "rates": list(rates),
        "parameter_count": count_parameters(model),
        "checkpoints": checkpoints,
        "best_checkpoint": best_checkpoint,
        "final_checkpoint": checkpoints[-1],
        "final_timing": {
            "training_steps": config.training_steps,
            "eval_interval": config.eval_interval,
            "average_update_ms_per_step": round(update_ms_total / config.training_steps, 6),
            "training_wall_ms": round(wall_ms_total, 6),
            "training_wall_seconds": round(wall_ms_total / 1000.0, 6),
        },
    }
    del optimizer
    del model
    return result


def report_delta(multi_result: dict[str, object], matched_result: dict[str, object]) -> dict[str, object]:
    multi_final = multi_result["final_checkpoint"]
    matched_final = matched_result["final_checkpoint"]
    multi_best = multi_result["best_checkpoint"]
    matched_best = matched_result["best_checkpoint"]
    multi_timing = multi_result["final_timing"]
    matched_timing = matched_result["final_timing"]
    return {
        "final_val_loss_delta": round(multi_final["val_loss"] - matched_final["val_loss"], 6),
        "final_val_accuracy_delta": round(multi_final["val_accuracy"] - matched_final["val_accuracy"], 6),
        "best_val_loss_delta": round(multi_best["val_loss"] - matched_best["val_loss"], 6),
        "average_update_ms_per_step_delta": round(
            multi_timing["average_update_ms_per_step"] - matched_timing["average_update_ms_per_step"],
            6,
        ),
        "parameter_count_delta": int(multi_result["parameter_count"] - matched_result["parameter_count"]),
    }


def main() -> None:
    args = parse_args()
    base_config = replace_config(
        RunConfig(),
        training_steps=args.training_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        batch_size=args.batch_size if args.batch_size is not None else RunConfig.batch_size,
        eval_batch_size=args.eval_batch_size if args.eval_batch_size is not None else RunConfig.eval_batch_size,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(base_config.seed)
    device = resolve_device(args.device)
    train_data, val_data, vocab_size = load_dataset(context_size=base_config.context_size)
    train_inputs, train_targets = train_data
    val_inputs, val_targets = val_data
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    val_inputs = val_inputs.to(device)
    val_targets = val_targets.to(device)

    calibration = run_calibration(
        base_config=base_config,
        vocab_size=vocab_size,
        device=device,
        train_inputs=train_inputs,
        train_targets=train_targets,
        warmup_steps=args.calibration_warmup_steps,
        timed_steps=args.calibration_timed_steps,
        seed=base_config.seed,
    )
    selected_match = calibration["selected_match"]
    matched_rates = (1, 1, 1, 1)
    matched_config = model_config(base_config, d_model=selected_match["d_model"], rates=matched_rates)
    multi_config = model_config(base_config, d_model=128, rates=MULTI_RATES)
    shallow_config = model_config(base_config, d_model=128, rates=SHALLOW_RATES)
    training_schedule = fixed_step_indices(
        train_inputs.shape[0],
        steps=base_config.training_steps,
        batch_size=base_config.batch_size,
        seed=base_config.seed,
        device=device,
    )

    models = {
        "multi_rate_4x128_1248": train_single_model(
            label="multi_rate_4x128_1248",
            vocab_size=vocab_size,
            config=multi_config,
            rates=MULTI_RATES,
            device=device,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            training_schedule=training_schedule,
            seed=base_config.seed,
        ),
        f"matched_all_rate_1_4x{selected_match['d_model']}": train_single_model(
            label=f"matched_all_rate_1_4x{selected_match['d_model']}",
            vocab_size=vocab_size,
            config=matched_config,
            rates=matched_rates,
            device=device,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            training_schedule=training_schedule,
            seed=base_config.seed,
        ),
        "all_rate_1_3x128": train_single_model(
            label="all_rate_1_3x128",
            vocab_size=vocab_size,
            config=shallow_config,
            rates=SHALLOW_RATES,
            device=device,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            training_schedule=training_schedule,
            seed=base_config.seed,
        ),
    }

    matched_key = f"matched_all_rate_1_4x{selected_match['d_model']}"
    git_status_short = current_git_status_short()
    report = {
        "config": {
            "training_steps": base_config.training_steps,
            "eval_interval": base_config.eval_interval,
            "batch_size": base_config.batch_size,
            "eval_batch_size": base_config.eval_batch_size,
            "learning_rate": base_config.learning_rate,
            "context_size": base_config.context_size,
            "temporal_window": base_config.temporal_window,
            "num_heads": base_config.num_heads,
            "seed": base_config.seed,
        },
        "environment": {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": git_status_short == [],
            "git_status_short": git_status_short,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "calibration": calibration,
        "selected_match": selected_match,
        "parameter_counts": {key: value["parameter_count"] for key, value in models.items()},
        "models": models,
        "delta_multi_vs_matched": report_delta(models["multi_rate_4x128_1248"], models[matched_key]),
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps({"stage": "done", "report": str(output_dir / 'report.json')}), flush=True)


if __name__ == "__main__":
    main()
