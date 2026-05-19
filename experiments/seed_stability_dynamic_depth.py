from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.fixed_window_char import resolve_device, set_seed
from experiments.analyze_dynamic_depth import correlation_ratio, is_word_char
from experiments.pytorch_char_dynamic_depth import (
    current_git_sha,
    current_git_status_short,
    write_json,
)
from experiments.pytorch_char_dynamic_depth_improved import (
    RunConfig,
    build_large_corpus,
    build_threshold_rows,
    choose_recommended_point,
    collect_evaluation_details,
    make_quantile_candidates,
    pareto_frontier,
    train_model,
)


DEFAULT_SEEDS = [42, 123, 456, 789, 1337]
MAX_ATTEMPTS_PER_SEED = 3


def parse_args() -> argparse.Namespace:
    default_text_file = REPO_ROOT / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    default_output_file = (
        REPO_ROOT
        / "research"
        / "questions"
        / "dynamic-depth"
        / "artifacts"
        / "improved"
        / "seed_stability"
        / "summary.json"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", type=Path, default=default_text_file)
    parser.add_argument("--output-file", type=Path, default=default_output_file)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    return parser.parse_args()


def build_config(seed: int) -> RunConfig:
    return RunConfig(
        context_size=32,
        hidden_dim=128,
        max_depth=8,
        train_batch_size=256,
        train_steps=3000,
        train_learning_rate=0.003,
        seed=seed,
    )


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_position_in_word_eta(*, corpus, used_depths: list[int], context_size: int) -> dict[str, object]:
    val_text = corpus.val_text
    target_positions = [context_size + index for index in range(len(used_depths))]
    target_chars = [val_text[position] for position in target_positions]
    previous_chars = [val_text[position - 1] for position in target_positions]
    next_chars = [val_text[position + 1] if position + 1 < len(val_text) else None for position in target_positions]

    categories = []
    for char, previous_char, next_char in zip(target_chars, previous_chars, next_chars, strict=True):
        if not is_word_char(char):
            categories.append("non_word")
        elif previous_char == " ":
            categories.append("first_after_space")
        elif next_char is None or not is_word_char(next_char):
            categories.append("end_before_boundary")
        elif is_word_char(previous_char):
            categories.append("middle_of_word")
        else:
            categories.append("other_word_start")

    depths_as_float = [float(depth) for depth in used_depths]
    rows = []
    for category in sorted(set(categories), key=lambda value: (-statistics.fmean(depth for depth, label in zip(depths_as_float, categories, strict=True) if label == value), value)):
        category_depths = [depth for depth, label in zip(depths_as_float, categories, strict=True) if label == category]
        rows.append(
            {
                "category": category,
                "mean_depth": statistics.fmean(category_depths),
                "count": len(category_depths),
            }
        )

    return {
        "effect_metric": "correlation_ratio_eta",
        "effect_size": correlation_ratio(categories, depths_as_float),
        "categories": rows,
    }


def summarize_metric(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty metric list.")
    if len(values) == 1:
        variance = 0.0
        stdev = 0.0
    else:
        variance = statistics.pvariance(values)
        stdev = statistics.pstdev(values)
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "variance": variance,
        "stdev": stdev,
    }


def select_used_depths(predicted_losses: torch.Tensor, *, threshold: float) -> list[int]:
    used_depths = []
    for row in predicted_losses:
        chosen_depth = None
        for depth_index in range(int(row.shape[0])):
            if float(row[depth_index].item()) <= threshold:
                chosen_depth = depth_index + 1
                break
        used_depths.append(chosen_depth if chosen_depth is not None else int(row.shape[0]))
    return used_depths


def run_seed(*, seed: int, device: torch.device, text_file: Path, output_root: Path) -> dict[str, object]:
    config = build_config(seed)
    seed_dir = output_root / f"seed_{seed}"
    seed_summary_path = seed_dir / "seed_summary.json"
    if seed_summary_path.exists():
        return read_json(seed_summary_path)

    corpus = build_large_corpus(config, text_file=text_file)
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS_PER_SEED + 1):
        if seed_dir.exists():
            shutil.rmtree(seed_dir)
        seed_dir.mkdir(parents=True, exist_ok=True)
        try:
            model, training_summary = train_model(corpus=corpus, config=config, device=device, output_dir=seed_dir)
            break
        except RuntimeError as error:
            last_error = error
            if attempt == MAX_ATTEMPTS_PER_SEED:
                raise RuntimeError(f"Seed {seed} failed after {MAX_ATTEMPTS_PER_SEED} attempts.") from error
    else:
        raise RuntimeError(f"Seed {seed} failed without raising a tracked exception: {last_error}")

    train_inputs = corpus.train_inputs.to(device)
    train_targets = corpus.train_targets.to(device)
    val_inputs = corpus.val_inputs.to(device)
    val_targets = corpus.val_targets.to(device)

    train_details = collect_evaluation_details(
        model,
        train_inputs,
        train_targets,
        max_depth=config.max_depth,
        batch_size=config.depth_eval_batch_size,
    )
    val_details = collect_evaluation_details(
        model,
        val_inputs,
        val_targets,
        max_depth=config.max_depth,
        batch_size=config.depth_eval_batch_size,
    )

    threshold_candidates = make_quantile_candidates(
        train_details.predicted_losses,
        quantiles=config.threshold_quantiles,
    )
    threshold_frontier, _threshold_rows = pareto_frontier(
        build_threshold_rows(
            train_details=train_details,
            val_details=val_details,
            candidates=threshold_candidates,
        )
    )
    threshold_recommended = choose_recommended_point(
        threshold_frontier,
        tolerance_fraction=config.recommendation_loss_tolerance,
    )
    threshold = float(threshold_recommended["threshold"])
    used_depths = select_used_depths(val_details.predicted_losses, threshold=threshold)
    position_in_word = compute_position_in_word_eta(
        corpus=corpus,
        used_depths=used_depths,
        context_size=config.context_size,
    )

    fixed_depth_8_val_loss = float(val_details.actual_losses[:, config.max_depth - 1].mean().item())
    recommended_val_loss = float(threshold_recommended["val_loss"])
    recommended_mean_depth = float(threshold_recommended["val_avg_depth"])
    position_in_word_eta = position_in_word["effect_size"]
    if position_in_word_eta is None:
        raise RuntimeError(f"Position-in-word eta was undefined for seed {seed}.")

    seed_summary = {
        "seed": seed,
        "attempts": attempt,
        "training_runtime_seconds": float(training_summary["runtime_seconds"]),
        "fixed_depth_8_val_loss": fixed_depth_8_val_loss,
        "recommended_threshold": threshold,
        "recommended_point": {
            "val_loss": recommended_val_loss,
            "mean_depth": recommended_mean_depth,
        },
        "position_in_word": {
            "effect_metric": "correlation_ratio_eta",
            "effect_size": float(position_in_word_eta),
            "categories": position_in_word["categories"],
        },
    }
    write_json(seed_dir / "seed_summary.json", seed_summary)
    return seed_summary


def print_table(runs: list[dict[str, object]]) -> None:
    headers = [
        "seed",
        "fixed8_val_loss",
        "rec_threshold",
        "rec_val_loss",
        "rec_mean_depth",
        "position_eta",
        "runtime_s",
    ]
    rows = [headers]
    for run in runs:
        rows.append(
            [
                str(run["seed"]),
                f"{float(run['fixed_depth_8_val_loss']):.6f}",
                f"{float(run['recommended_threshold']):.6f}",
                f"{float(run['recommended_point']['val_loss']):.6f}",
                f"{float(run['recommended_point']['mean_depth']):.4f}",
                f"{float(run['position_in_word']['effect_size']):.4f}",
                f"{float(run['training_runtime_seconds']):.1f}",
            ]
        )

    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    for row_index, row in enumerate(rows):
        print("  ".join(value.rjust(widths[index]) for index, value in enumerate(row)))
        if row_index == 0:
            print("  ".join("-" * width for width in widths))


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(0)

    output_file: Path = args.output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    runs = [run_seed(seed=seed, device=device, text_file=args.text_file, output_root=output_file.parent) for seed in args.seeds]
    summary = {
        "config": asdict(build_config(int(args.seeds[0]))),
        "seeds": [int(seed) for seed in args.seeds],
        "environment": {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": current_git_status_short() == [],
            "git_status_short": current_git_status_short(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_is_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "runs": runs,
        "variance_summary": {
            "fixed_depth_8_val_loss": summarize_metric([float(run["fixed_depth_8_val_loss"]) for run in runs]),
            "recommended_val_loss": summarize_metric([float(run["recommended_point"]["val_loss"]) for run in runs]),
            "recommended_mean_depth": summarize_metric([float(run["recommended_point"]["mean_depth"]) for run in runs]),
            "position_in_word_eta": summarize_metric([float(run["position_in_word"]["effect_size"]) for run in runs]),
            "recommended_threshold": summarize_metric([float(run["recommended_threshold"]) for run in runs]),
        },
    }
    write_json(output_file, summary)
    print_table(runs)


if __name__ == "__main__":
    main()
