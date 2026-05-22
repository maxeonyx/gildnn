from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys

import torch

from core.fixed_window_char import load_dataset, resolve_device, set_seed
from .run import (
    RunConfig,
    build_model,
    compare_diagonal_paths,
    current_git_sha,
    current_git_status_short,
    final_timing_summary,
    overfit_one_batch,
    replace_config,
    run_diagonal_smoke_check,
    summarize_history,
    train_comparison,
    write_json,
)


DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_RATES = (1, 2, 4, 8)


def parse_seed_tuple(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(seeds) == 0:
        raise ValueError("At least one seed is required.")
    return seeds


def parse_rate_tuple(value: str) -> tuple[int, ...]:
    rates = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(rates) == 0:
        raise ValueError("At least one rate is required.")
    return rates


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--training-steps", type=int, default=2_000)
    parser.add_argument("--overfit-steps", type=int, default=2_000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--checkpoint-timing-warmup-passes", type=int, default=20)
    parser.add_argument("--checkpoint-timing-passes", type=int, default=100)
    parser.add_argument("--final-timing-warmup-passes", type=int, default=100)
    parser.add_argument("--final-timing-passes", type=int, default=500)
    parser.add_argument("--control-rates", default=",".join(str(rate) for rate in DEFAULT_RATES))
    parser.add_argument("--multi-rates", default=",".join(str(rate) for rate in DEFAULT_RATES))
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def round_float(value: float) -> float:
    return round(value, 6)


def summarize_series(values: list[float]) -> dict[str, float]:
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": round_float(statistics.mean(values)),
        "std": round_float(std),
        "min": round_float(min(values)),
        "max": round_float(max(values)),
    }


def run_seed(
    *,
    seed: int,
    base_config: RunConfig,
    vocab_size: int,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
) -> dict[str, object]:
    config = replace_config(base_config, seed=seed)
    set_seed(seed)
    print(json.dumps({"seed_start": {"seed": seed, "config": asdict(config)}}, indent=2), flush=True)

    overfit_model = build_model(
        vocab_size=vocab_size,
        config=config,
        rates=config.multi_rates,
        device=device,
        diagonal_enabled=config.multi_diagonal,
    )
    smoke_check = run_diagonal_smoke_check(overfit_model, train_inputs)
    diagonal_effect = compare_diagonal_paths(overfit_model, train_inputs)
    overfit_result = overfit_one_batch(
        overfit_model,
        train_inputs[: config.batch_size],
        train_targets[: config.batch_size],
        steps=config.overfit_steps,
        learning_rate=config.learning_rate,
        loss_threshold=config.overfit_loss_threshold,
    )

    control_model = build_model(
        vocab_size=vocab_size,
        config=config,
        rates=config.control_rates,
        device=device,
        diagonal_enabled=config.control_diagonal,
    )
    multi_model = build_model(
        vocab_size=vocab_size,
        config=config,
        rates=config.multi_rates,
        device=device,
        diagonal_enabled=config.multi_diagonal,
    )
    multi_model.load_state_dict(control_model.state_dict())
    history = train_comparison(
        control_model=control_model,
        multi_model=multi_model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        config=config,
    )
    final_timing = final_timing_summary(
        control_model=control_model,
        multi_model=multi_model,
        timing_batch=train_inputs[: config.batch_size],
        config=config,
    )
    summary = summarize_history(history)
    seed_report = {
        "seed": seed,
        "config": asdict(config),
        "model": {
            "control": {
                "rates": list(config.control_rates),
                "diagonal_enabled": config.control_diagonal,
                "mix_coefficients": control_model.mix_coefficients(),
            },
            "experiment": {
                "rates": list(config.multi_rates),
                "diagonal_enabled": config.multi_diagonal,
                "mix_coefficients": multi_model.mix_coefficients(),
            },
        },
        "overfit": overfit_result,
        "smoke_check": smoke_check,
        "diagonal_effect": diagonal_effect,
        "comparison_history": history,
        "summary": summary,
        "final_timing": final_timing,
    }
    print(
        json.dumps(
            {
                "seed_complete": {
                    "seed": seed,
                    "final": summary["final"],
                    "final_timing": final_timing,
                }
            },
            indent=2,
        ),
        flush=True,
    )

    del overfit_model
    del control_model
    del multi_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return seed_report


def aggregate_checkpoints(seed_runs: list[dict[str, object]]) -> list[dict[str, object]]:
    reference_history = seed_runs[0]["comparison_history"]
    aggregates: list[dict[str, object]] = []
    for checkpoint_index, reference_checkpoint in enumerate(reference_history):
        step = reference_checkpoint["step"]
        checkpoints = [seed_run["comparison_history"][checkpoint_index] for seed_run in seed_runs]
        if any(checkpoint["step"] != step for checkpoint in checkpoints):
            raise ValueError("Seed histories use different checkpoint steps.")
        val_loss_deltas = [checkpoint["comparison"]["val_loss_delta"] for checkpoint in checkpoints]
        speedups = [checkpoint["comparison"]["speedup_percent"] for checkpoint in checkpoints]
        control_losses = [checkpoint["control"]["val_loss"] for checkpoint in checkpoints]
        experiment_losses = [checkpoint["experiment"]["val_loss"] for checkpoint in checkpoints]
        aggregates.append(
            {
                "step": step,
                "comparison": {
                    "val_loss_delta": summarize_series(val_loss_deltas),
                    "speedup_percent": summarize_series(speedups),
                },
                "control": {
                    "val_loss": summarize_series(control_losses),
                },
                "experiment": {
                    "val_loss": summarize_series(experiment_losses),
                },
            }
        )
    return aggregates


def aggregate_seed_runs(seed_runs: list[dict[str, object]]) -> dict[str, object]:
    final_records = [seed_run["summary"]["final"] for seed_run in seed_runs]
    final_val_loss_deltas = [record["comparison"]["val_loss_delta"] for record in final_records]
    final_speedups = [record["comparison"]["speedup_percent"] for record in final_records]
    final_control_losses = [record["control"]["val_loss"] for record in final_records]
    final_experiment_losses = [record["experiment"]["val_loss"] for record in final_records]
    return {
        "seed_count": len(seed_runs),
        "final": {
            "comparison": {
                "val_loss_delta": summarize_series(final_val_loss_deltas),
                "speedup_percent": summarize_series(final_speedups),
            },
            "control": {
                "val_loss": summarize_series(final_control_losses),
            },
            "experiment": {
                "val_loss": summarize_series(final_experiment_losses),
            },
        },
        "checkpoints": aggregate_checkpoints(seed_runs),
    }


def main() -> None:
    args = parse_args()
    seeds = parse_seed_tuple(args.seeds)
    control_rates = parse_rate_tuple(args.control_rates)
    multi_rates = parse_rate_tuple(args.multi_rates)
    if control_rates != multi_rates:
        raise ValueError("This run expects the control and experiment to share the same multi-rate schedule.")

    base_config = replace_config(
        RunConfig(),
        training_steps=args.training_steps,
        overfit_steps=args.overfit_steps,
        eval_interval=args.eval_interval,
        checkpoint_timing_warmup_passes=args.checkpoint_timing_warmup_passes,
        checkpoint_timing_passes=args.checkpoint_timing_passes,
        final_timing_warmup_passes=args.final_timing_warmup_passes,
        final_timing_passes=args.final_timing_passes,
        num_blocks=len(multi_rates),
        control_rates=control_rates,
        multi_rates=multi_rates,
        control_diagonal=False,
        multi_diagonal=True,
        seed=seeds[0],
    )
    output_dir = args.output_dir or (
        args.repo_root / "experiments" / "fixed_multi_rate" / "artifacts" / "diagonal_1248_multi_seed"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    train_data, val_data, vocab_size = load_dataset(context_size=base_config.context_size)
    train_inputs, train_targets = train_data
    val_inputs, val_targets = val_data
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    val_inputs = val_inputs.to(device)
    val_targets = val_targets.to(device)

    seed_runs = [
        run_seed(
            seed=seed,
            base_config=base_config,
            vocab_size=vocab_size,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            device=device,
        )
        for seed in seeds
    ]

    git_status_short = current_git_status_short()
    report = {
        "comparison": "multi_rate_1248_no_diagonal_vs_diagonal",
        "seeds": list(seeds),
        "config": asdict(base_config),
        "environment": {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": git_status_short == [],
            "git_status_short": git_status_short,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "seed_runs": seed_runs,
        "aggregate": aggregate_seed_runs(seed_runs),
    }
    write_json(output_dir / "report.json", report)
    print("PASS fixed_multi_rate multi_seed_diagonal")
    print(f"Wrote {output_dir / 'report.json'}")
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
