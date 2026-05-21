from __future__ import annotations

import argparse
from dataclasses import asdict
from math import sqrt
from pathlib import Path
import statistics
import sys

import torch

from base_experiments.transformer import build_fixed_length_split
from core.fixed_window_char import resolve_device, set_seed

from .char_model import TrainableConfig, VariantSpec, count_parameters, make_async_stale_variant, make_sync_variant, with_overrides
from .data import corpus_summary, current_git_sha, current_git_status_short, preprocess_corpus_text, resolve_text_file, write_json
from .run_corpus_train import fixed_step_indices, make_model, train_variant


DEFAULT_SEEDS = (42, 123, 456, 789, 1337)
T_CRITICAL_95_BY_DF = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--corpus", choices=["tinyshakespeare", "enwik8"], default="tinyshakespeare")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=repo_root / "experiments" / "async_gru_corpus" / "artifacts" / "multiseed_corpus_report.json",
    )
    parser.add_argument("--train-characters", type=int, default=900_000)
    parser.add_argument("--val-characters", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.0015)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--allow-vocab-growth", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def compute_confidence_interval_95(values: list[float]) -> dict[str, float] | None:
    if len(values) < 2:
        return None
    df = len(values) - 1
    t_critical = T_CRITICAL_95_BY_DF.get(df, 1.96)
    mean = statistics.fmean(values)
    stddev = statistics.stdev(values)
    margin = t_critical * stddev / sqrt(len(values))
    return {
        "level": 0.95,
        "critical_value": round(t_critical, 6),
        "lower": round(mean - margin, 6),
        "upper": round(mean + margin, 6),
        "margin": round(margin, 6),
    }


def aggregate_seed_gaps(seed_results: list[dict[str, object]]) -> dict[str, object]:
    completed_seed_results = [seed_result for seed_result in seed_results if "analysis" in seed_result]
    if not completed_seed_results:
        raise ValueError("aggregate_seed_gaps requires at least one completed seed result.")
    gaps = [float(seed_result["analysis"]["best_val_gap_async_minus_sync"]) for seed_result in completed_seed_results]
    mean_gap = statistics.fmean(gaps)
    stddev_gap = statistics.stdev(gaps) if len(gaps) >= 2 else 0.0
    positive_count = sum(gap > 0 for gap in gaps)
    negative_count = sum(gap < 0 for gap in gaps)
    zero_count = sum(gap == 0 for gap in gaps)
    return {
        "seed_count": len(completed_seed_results),
        "per_seed_best_val_gap_async_minus_sync": [round(gap, 6) for gap in gaps],
        "mean_best_val_gap_async_minus_sync": round(mean_gap, 6),
        "stddev_best_val_gap_async_minus_sync": round(stddev_gap, 6),
        "positive_gap_count": positive_count,
        "negative_gap_count": negative_count,
        "zero_gap_count": zero_count,
        "all_gaps_positive": positive_count == len(gaps),
        "confidence_interval_95": compute_confidence_interval_95(gaps),
    }


def seed_report(
    *,
    seed: int,
    sync_variant: VariantSpec,
    async_variant: VariantSpec,
    sync_result: dict[str, object],
    async_result: dict[str, object],
    max_steps: int,
) -> dict[str, object]:
    return {
        "seed": seed,
        "training": {
            sync_variant.label: sync_result,
            async_variant.label: async_result,
        },
        "analysis": {
            "best_val_gap_async_minus_sync": round(
                float(async_result["best_eval"]["val_loss"]) - float(sync_result["best_eval"]["val_loss"]),
                6,
            ),
            "best_step": {
                sync_variant.label: int(sync_result["best_eval"]["step"]),
                async_variant.label: int(async_result["best_eval"]["step"]),
            },
            "completed_max_steps": max_steps,
        },
    }


def build_report(
    *,
    args: argparse.Namespace,
    config: TrainableConfig,
    device: torch.device,
    text_file: Path,
    raw_text: str,
    preprocessing: dict[str, object],
    split,
    sync_variant: VariantSpec,
    async_variant: VariantSpec,
    parameter_count: int | None,
    seed_results: list[dict[str, object]],
) -> dict[str, object]:
    report = {
        "config": asdict(config),
        "training_config": {
            "seeds": list(args.seeds),
            "max_steps": args.max_steps,
            "eval_every": args.eval_every,
        },
        "environment": {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": current_git_status_short() == [],
            "git_status_short": current_git_status_short(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "corpus_summary": corpus_summary(
            corpus=args.corpus,
            text_file=text_file,
            raw_text=raw_text,
            train_characters=len(split.train_text),
            val_characters=len(split.val_text),
            vocab_size=split.train_dataset.vocab_size,
            preprocessing=preprocessing,
        ),
        "model": {
            "parameter_count": parameter_count,
            "d_model": config.d_model,
            "num_modules": config.num_modules,
            "num_ticks": config.num_ticks,
            "read_lags": {
                sync_variant.label: list(sync_variant.read_lags),
                async_variant.label: list(async_variant.read_lags),
            },
        },
        "seed_results": seed_results,
    }
    if any("analysis" in seed_result for seed_result in seed_results):
        report["analysis"] = aggregate_seed_gaps(seed_results)
    return report


def main() -> None:
    args = parse_args()
    if args.max_steps % args.eval_every != 0:
        raise ValueError("max_steps must be divisible by eval_every so the last step is evaluated.")

    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_output_path = output_path.with_suffix(".partial.json")

    config = with_overrides(
        TrainableConfig(),
        train_characters=args.train_characters,
        val_characters=args.val_characters,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
    )
    device = resolve_device(args.device)
    text_file = resolve_text_file(
        repo_root=args.repo_root,
        corpus=args.corpus,
        text_file=args.text_file,
        download_missing=args.download_missing,
    )
    source_text = text_file.read_text(encoding="utf-8")
    raw_text, preprocessing = preprocess_corpus_text(
        repo_root=args.repo_root,
        corpus=args.corpus,
        text_file=text_file,
        raw_text=source_text,
        allow_vocab_growth=args.allow_vocab_growth,
    )
    split = build_fixed_length_split(raw_text, config=config)
    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)
    vocab_size = split.train_dataset.vocab_size

    sync_variant = make_sync_variant(config)
    async_variant = make_async_stale_variant(config)
    seed_results: list[dict[str, object]] = []
    parameter_count: int | None = None

    def write_progress() -> None:
        write_json(
            partial_output_path,
            build_report(
                args=args,
                config=config,
                device=device,
                text_file=text_file,
                raw_text=raw_text,
                preprocessing=preprocessing,
                split=split,
                sync_variant=sync_variant,
                async_variant=async_variant,
                parameter_count=parameter_count,
                seed_results=seed_results,
            ),
        )

    for seed in args.seeds:
        seed_config = with_overrides(config, seed=seed)
        set_seed(seed)
        sync_model = make_model(vocab_size=vocab_size, config=seed_config, variant=sync_variant, device=device)
        async_model = make_model(vocab_size=vocab_size, config=seed_config, variant=async_variant, device=device)
        async_model.load_state_dict(sync_model.state_dict())
        if parameter_count is None:
            parameter_count = count_parameters(sync_model)

        step_indices = fixed_step_indices(
            train_inputs.shape[0],
            steps=args.max_steps,
            batch_size=seed_config.batch_size,
            seed=seed,
            device=device,
        )

        sync_result: dict[str, object] | None = None
        async_result: dict[str, object] | None = None

        def update_sync_progress(result: dict[str, object]) -> None:
            nonlocal sync_result
            sync_result = result
            current_seed_results = seed_results + [
                {
                    "seed": seed,
                    "training": {sync_variant.label: sync_result},
                }
            ]
            write_json(
                partial_output_path,
                build_report(
                    args=args,
                    config=config,
                    device=device,
                    text_file=text_file,
                    raw_text=raw_text,
                    preprocessing=preprocessing,
                    split=split,
                    sync_variant=sync_variant,
                    async_variant=async_variant,
                    parameter_count=parameter_count,
                    seed_results=current_seed_results,
                ),
            )

        def update_async_progress(result: dict[str, object]) -> None:
            nonlocal async_result
            async_result = result
            if sync_result is None:
                raise RuntimeError("Async progress reported before sync result existed.")
            current_seed_results = seed_results + [
                {
                    "seed": seed,
                    "training": {
                        sync_variant.label: sync_result,
                        async_variant.label: async_result,
                    },
                }
            ]
            write_json(
                partial_output_path,
                build_report(
                    args=args,
                    config=config,
                    device=device,
                    text_file=text_file,
                    raw_text=raw_text,
                    preprocessing=preprocessing,
                    split=split,
                    sync_variant=sync_variant,
                    async_variant=async_variant,
                    parameter_count=parameter_count,
                    seed_results=current_seed_results,
                ),
            )

        sync_result = train_variant(
            model=sync_model,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            config=seed_config,
            step_indices=step_indices,
            eval_every=args.eval_every,
            early_stop_patience=None,
            on_evaluation=update_sync_progress,
        )
        async_result = train_variant(
            model=async_model,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            config=seed_config,
            step_indices=step_indices,
            eval_every=args.eval_every,
            early_stop_patience=None,
            on_evaluation=update_async_progress,
        )
        seed_results.append(
            seed_report(
                seed=seed,
                sync_variant=sync_variant,
                async_variant=async_variant,
                sync_result=sync_result,
                async_result=async_result,
                max_steps=args.max_steps,
            )
        )
        write_progress()

    report = build_report(
        args=args,
        config=config,
        device=device,
        text_file=text_file,
        raw_text=raw_text,
        preprocessing=preprocessing,
        split=split,
        sync_variant=sync_variant,
        async_variant=async_variant,
        parameter_count=parameter_count,
        seed_results=seed_results,
    )
    write_json(output_path, report)
    partial_output_path.unlink(missing_ok=True)
    print("PASS async_gru_corpus multiseed training")
    print(f"Wrote {output_path}")
    print(report["analysis"])


if __name__ == "__main__":
    main()
