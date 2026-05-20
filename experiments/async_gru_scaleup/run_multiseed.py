from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from base_experiments.transformer import build_fixed_length_split
from core.fixed_window_char import resolve_device, set_seed

from .char_model import TrainableConfig, count_parameters, make_async_stale_variant, make_sync_variant, with_overrides
from .run_screen_train import fixed_step_indices, make_model, train_variant, write_json


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1337])
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def current_git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def current_git_status_short() -> list[str]:
    result = subprocess.run(["git", "status", "--short"], check=True, capture_output=True, text=True)
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    if args.max_steps % args.eval_every != 0:
        raise ValueError("max_steps must be divisible by eval_every so the last step is evaluated.")

    config = with_overrides(
        TrainableConfig(),
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
    )
    text_file = args.text_file or (args.repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt")
    output_dir = args.output_dir or (args.repo_root / "experiments" / "async_gru_scaleup" / "artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    raw_text = text_file.read_text(encoding="utf-8")

    per_seed_results: list[dict[str, object]] = []
    best_val_gaps: list[float] = []
    parameter_count: int | None = None
    vocab_size: int | None = None

    for seed in args.seeds:
        seed_config = with_overrides(config, seed=seed)
        set_seed(seed)
        split = build_fixed_length_split(raw_text, config=seed_config)
        train_inputs = split.train_inputs.to(device)
        train_targets = split.train_targets.to(device)
        val_inputs = split.val_inputs.to(device)
        val_targets = split.val_targets.to(device)
        vocab_size = split.train_dataset.vocab_size

        sync_variant = make_sync_variant(seed_config)
        async_variant = make_async_stale_variant(seed_config)
        sync_model = make_model(vocab_size=vocab_size, config=seed_config, variant=sync_variant, device=device)
        async_model = make_model(vocab_size=vocab_size, config=seed_config, variant=async_variant, device=device)
        async_model.load_state_dict(sync_model.state_dict())
        parameter_count = count_parameters(sync_model)

        step_indices = fixed_step_indices(
            train_inputs.shape[0],
            steps=args.max_steps,
            batch_size=seed_config.batch_size,
            seed=seed,
            device=device,
        )

        sync_result = train_variant(
            model=sync_model,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            train_dataset=split.train_dataset,
            config=seed_config,
            step_indices=step_indices,
            eval_every=args.eval_every,
            early_stop_patience=args.early_stop_patience,
        )
        async_result = train_variant(
            model=async_model,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            train_dataset=split.train_dataset,
            config=seed_config,
            step_indices=step_indices,
            eval_every=args.eval_every,
            early_stop_patience=args.early_stop_patience,
        )

        gap = round(async_result["best_eval"]["val_loss"] - sync_result["best_eval"]["val_loss"], 6)
        best_val_gaps.append(gap)
        per_seed_results.append(
            {
                "seed": seed,
                "sync": {
                    "best_val_loss": sync_result["best_eval"]["val_loss"],
                    "best_step": sync_result["best_eval"]["step"],
                    "final_val_loss": sync_result["final_eval"]["val_loss"],
                    "completed_steps": sync_result["completed_steps"],
                    "stop_reason": sync_result["stop_reason"],
                },
                "async": {
                    "best_val_loss": async_result["best_eval"]["val_loss"],
                    "best_step": async_result["best_eval"]["step"],
                    "final_val_loss": async_result["final_eval"]["val_loss"],
                    "completed_steps": async_result["completed_steps"],
                    "stop_reason": async_result["stop_reason"],
                },
                "best_val_gap_async_minus_sync": gap,
            }
        )

    if parameter_count is None or vocab_size is None:
        raise RuntimeError("No seeds were run.")

    mean_gap = round(statistics.mean(best_val_gaps), 6)
    std_gap = round(statistics.stdev(best_val_gaps), 6) if len(best_val_gaps) > 1 else 0.0
    report = {
        "config": asdict(config),
        "training_config": {
            "max_steps": args.max_steps,
            "eval_every": args.eval_every,
            "early_stop_patience": args.early_stop_patience,
            "seeds": args.seeds,
        },
        "environment": {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": current_git_status_short() == [],
            "git_status_short": current_git_status_short(),
            "python_version": sys.version,
            "device": str(device),
        },
        "corpus_summary": {
            "source_file": str(text_file),
            "source_total_characters": len(raw_text),
            "source_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "train_characters": config.train_characters,
            "val_characters": config.val_characters,
            "vocab_size": vocab_size,
        },
        "model": {
            "parameter_count": parameter_count,
            "d_model": config.d_model,
            "num_modules": config.num_modules,
            "num_ticks": config.num_ticks,
            "read_lags": {
                "synchronous_control": list(make_sync_variant(config).read_lags),
                "async_stale_reads": list(make_async_stale_variant(config).read_lags),
            },
        },
        "per_seed": per_seed_results,
        "aggregate": {
            "best_val_gaps_async_minus_sync": best_val_gaps,
            "mean_gap": mean_gap,
            "std_gap": std_gap,
        },
    }
    output_path = output_dir / "multiseed_report.json"
    write_json(output_path, report)
    print("PASS async_gru_scaleup multiseed")
    print(f"Wrote {output_path}")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
