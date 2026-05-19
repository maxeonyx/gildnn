from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from experiments.pytorch_char_dynamic_depth import write_json
from experiments.pytorch_char_dynamic_depth_improved import (
    RunConfig,
    build_large_corpus,
    collect_evaluation_details,
    select_dynamic_depth,
    train_model,
)
from core.fixed_window_char import resolve_device, set_seed


PUNCTUATION = {".", ",", ":", ";", "!", "?"}
LOCAL_ENTROPY_WINDOW_RADIUS = 64


@dataclass(frozen=True)
class NumericFeatureSummary:
    effect_size: float | None
    effect_metric: str
    mean_depth_by_bucket: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = (
        repo_root / "research" / "questions" / "dynamic-depth" / "artifacts" / "improved" / "depth_analysis"
    )
    default_sweep_summary = (
        repo_root / "research" / "questions" / "dynamic-depth" / "artifacts" / "improved" / "large_100k" / "sweep_summary.json"
    )
    default_text_file = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--sweep-summary", type=Path, default=default_sweep_summary)
    parser.add_argument("--text-file", type=Path, default=default_text_file)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--local-entropy-radius", type=int, default=LOCAL_ENTROPY_WINDOW_RADIUS)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def format_char(char: str) -> str:
    if char == "\n":
        return "\\n"
    if char == " ":
        return "<space>"
    if char == "\t":
        return "\\t"
    return char


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean of an empty sequence.")
    return sum(values) / len(values)


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("Sequences must have the same length.")
    if len(xs) < 2:
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    numerator = sum(x * y for x, y in zip(centered_x, centered_y, strict=True))
    denominator = math.sqrt(sum(x * x for x in centered_x) * sum(y * y for y in centered_y))
    if denominator == 0.0:
        return None
    return numerator / denominator


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[indexed[position][0]] = rank
        start = end
    return ranks


def spearman_correlation(xs: list[float], ys: list[float]) -> float | None:
    return pearson_correlation(average_ranks(xs), average_ranks(ys))


def correlation_ratio(categories: list[str], values: list[float]) -> float | None:
    if len(categories) != len(values):
        raise ValueError("Sequences must have the same length.")
    if len(values) < 2:
        return None
    grouped: dict[str, list[float]] = defaultdict(list)
    for category, value in zip(categories, values, strict=True):
        grouped[category].append(value)
    if len(grouped) < 2:
        return None
    overall_mean = mean(values)
    ss_between = sum(len(group) * (mean(group) - overall_mean) ** 2 for group in grouped.values())
    ss_total = sum((value - overall_mean) ** 2 for value in values)
    if ss_total == 0.0:
        return None
    return math.sqrt(ss_between / ss_total)


def quantile_edges(values: list[float], quantiles: int) -> list[float]:
    if quantiles < 1:
        raise ValueError("quantiles must be positive.")
    sorted_values = sorted(values)
    edges: list[float] = []
    for bucket_index in range(1, quantiles):
        position = round((len(sorted_values) - 1) * bucket_index / quantiles)
        edges.append(sorted_values[position])
    deduped = []
    for edge in edges:
        if not deduped or edge > deduped[-1]:
            deduped.append(edge)
    return deduped


def quantile_bucket_rows(
    values: list[float],
    depths: list[float],
    *,
    quantiles: int,
    value_name: str,
) -> list[dict[str, object]]:
    edges = quantile_edges(values, quantiles)
    buckets: list[dict[str, object]] = [
        {
            "label": f"Q{index + 1}",
            "values": [],
            "depths": [],
        }
        for index in range(len(edges) + 1)
    ]
    for value, depth in zip(values, depths, strict=True):
        bucket_index = 0
        while bucket_index < len(edges) and value > edges[bucket_index]:
            bucket_index += 1
        buckets[bucket_index]["values"].append(value)
        buckets[bucket_index]["depths"].append(depth)
    rows = []
    for bucket in buckets:
        bucket_values = bucket["values"]
        bucket_depths = bucket["depths"]
        if not bucket_values:
            continue
        rows.append(
            {
                "bucket": bucket["label"],
                f"mean_{value_name}": mean(bucket_values),
                "mean_depth": mean(bucket_depths),
                "count": len(bucket_depths),
                f"min_{value_name}": min(bucket_values),
                f"max_{value_name}": max(bucket_values),
            }
        )
    return rows


def shannon_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def compute_local_entropies(target_chars: list[str], *, radius: int) -> list[float]:
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    entropies: list[float] = []
    for index in range(len(target_chars)):
        start = max(0, index - radius)
        stop = min(len(target_chars), index + radius + 1)
        entropies.append(shannon_entropy(Counter(target_chars[start:stop])))
    return entropies


def is_word_char(char: str) -> bool:
    return char.isalpha() or char == "'"


def scan_words(text: str) -> list[tuple[int, int, str]]:
    words = []
    start: int | None = None
    for index, char in enumerate(text):
        if is_word_char(char):
            if start is None:
                start = index
            continue
        if start is not None:
            words.append((start, index, text[start:index]))
            start = None
    if start is not None:
        words.append((start, len(text), text[start:]))
    return words


def build_word_lookup(text: str) -> list[str | None]:
    lookup: list[str | None] = [None] * len(text)
    for start, stop, word in scan_words(text):
        normalized = word.lower()
        for index in range(start, stop):
            lookup[index] = normalized
    return lookup


def summarize_character_identity(chars: list[str], depths: list[int]) -> dict[str, object]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for char, depth in zip(chars, depths, strict=True):
        grouped[char].append(depth)
    by_character = [
        {
            "char": format_char(char),
            "mean_depth": mean([float(depth) for depth in grouped[char]]),
            "count": len(grouped[char]),
        }
        for char in grouped
    ]
    ranked_desc = sorted(by_character, key=lambda row: (-float(row["mean_depth"]), -int(row["count"]), str(row["char"])))
    ranked_asc = sorted(by_character, key=lambda row: (float(row["mean_depth"]), -int(row["count"]), str(row["char"])))
    return {
        "effect_metric": "correlation_ratio_eta",
        "effect_size": correlation_ratio([format_char(char) for char in chars], [float(depth) for depth in depths]),
        "by_character": ranked_desc,
        "top_characters": ranked_desc[:10],
        "bottom_characters": ranked_asc[:10],
    }


def summarize_numeric_feature(
    values: list[float],
    depths: list[int],
    *,
    value_name: str,
    quantiles: int = 5,
) -> NumericFeatureSummary:
    return NumericFeatureSummary(
        effect_metric="spearman_rho",
        effect_size=spearman_correlation(values, [float(depth) for depth in depths]),
        mean_depth_by_bucket=quantile_bucket_rows(values, [float(depth) for depth in depths], quantiles=quantiles, value_name=value_name),
    )


def build_markdown(summary: dict[str, object]) -> str:
    reproduction = summary["reproduction"]
    effect_ranking = summary["effect_ranking"]
    character_identity = summary["character_identity"]
    bigram_novelty = summary["bigram_novelty"]
    position_in_word = summary["position_in_word"]
    local_entropy = summary["local_entropy"]
    after_punctuation = summary["after_punctuation"]
    word_frequency = summary["word_frequency"]

    def signed(value: float) -> str:
        return f"{value:+.4f}"

    lines = [
        "# Dynamic depth feature analysis",
        "",
        "## Reproduction",
        "",
        f"- Threshold: {reproduction['threshold']:.5f}",
        f"- Validation tokens analyzed: {reproduction['validation_token_count']}",
        f"- Reproduced mean depth: {reproduction['mean_depth']:.4f}",
        f"- Reproduced validation loss at used depth: {reproduction['task_loss']:.6f}",
        f"- Reference mean depth from saved sweep: {reproduction['reference_mean_depth']:.4f}",
        f"- Reference validation loss from saved sweep: {reproduction['reference_task_loss']:.6f}",
        "",
        "## Effect sizes",
        "",
        "| Rank | Feature | Metric | Abs effect size | Signed effect | Notes |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for index, row in enumerate(effect_ranking, start=1):
        lines.append(
            f"| {index} | {row['feature']} | {row['effect_metric']} | {row['effect_size']:.4f} | {signed(row['signed_effect_size'])} | {row['notes']} |"
        )

    lines.extend(
        [
            "",
            "## Character identity",
            "",
            f"- Effect size (`summary.json.character_identity.effect_size`): {character_identity['effect_size']:.4f}",
            "- Highest-mean-depth characters (`summary.json.character_identity.top_characters`):",
            "",
            "| Character | Mean depth | Count |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in character_identity["top_characters"]:
        lines.append(f"| {row['char']} | {row['mean_depth']:.4f} | {row['count']} |")
    lines.extend(
        [
            "",
            "- Lowest-mean-depth characters (`summary.json.character_identity.bottom_characters`):",
            "",
            "| Character | Mean depth | Count |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in character_identity["bottom_characters"]:
        lines.append(f"| {row['char']} | {row['mean_depth']:.4f} | {row['count']} |")

    lines.extend(
        [
            "",
            "## Bigram novelty",
            "",
            f"- Spearman rho (`summary.json.bigram_novelty.effect_size`): {bigram_novelty['effect_size']:.4f}",
            "- Mean depth by novelty quintile (`summary.json.bigram_novelty.mean_depth_by_bucket`):",
            "",
            "| Bucket | Mean novelty | Mean depth | Count |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in bigram_novelty["mean_depth_by_bucket"]:
        lines.append(
            f"| {row['bucket']} | {row['mean_novelty']:.4f} | {row['mean_depth']:.4f} | {row['count']} |"
        )

    lines.extend(
        [
            "",
            "## Position in word",
            "",
            f"- Effect size (`summary.json.position_in_word.effect_size`): {position_in_word['effect_size']:.4f}",
            "",
            "| Category | Mean depth | Count |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in position_in_word["categories"]:
        lines.append(f"| {row['category']} | {row['mean_depth']:.4f} | {row['count']} |")

    lines.extend(
        [
            "",
            "## Local entropy",
            "",
            f"- Spearman rho (`summary.json.local_entropy.effect_size`): {local_entropy['effect_size']:.4f}",
            f"- Sliding radius: {local_entropy['window_radius']}",
            "",
            "| Bucket | Mean entropy (bits) | Mean depth | Count |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in local_entropy["mean_depth_by_bucket"]:
        lines.append(
            f"| {row['bucket']} | {row['mean_entropy_bits']:.4f} | {row['mean_depth']:.4f} | {row['count']} |"
        )

    lines.extend(
        [
            "",
            "## After punctuation",
            "",
            f"- Point-biserial r (`summary.json.after_punctuation.effect_size`): {after_punctuation['effect_size']:.4f}",
            f"- Mean depth after punctuation: {after_punctuation['after_punctuation_mean_depth']:.4f}",
            f"- Mean depth otherwise: {after_punctuation['other_mean_depth']:.4f}",
            f"- Mean depth difference: {after_punctuation['mean_depth_difference']:.4f}",
            "",
            "## Word frequency",
            "",
            f"- Spearman rho (`summary.json.word_frequency.effect_size`): {word_frequency['effect_size']:.4f}",
            f"- Word-character tokens analyzed: {word_frequency['word_character_count']}",
            "",
            "| Bucket | Mean rarity | Mean depth | Count |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in word_frequency["mean_depth_by_bucket"]:
        lines.append(
            f"| {row['bucket']} | {row['mean_rarity']:.4f} | {row['mean_depth']:.4f} | {row['count']} |"
        )

    lines.extend(
        [
            "",
            "## Direct findings",
            "",
            f"- Strongest effect by the script's ranking is **{effect_ranking[0]['feature']}** at {effect_ranking[0]['effect_size']:.4f} (`summary.json.effect_ranking[0]`).",
            f"- Weakest requested effect is **{effect_ranking[-1]['feature']}** at {effect_ranking[-1]['effect_size']:.4f} (`summary.json.effect_ranking[-1]`).",
            f"- Bigram novelty is negative in this reproduction: depth moves from {bigram_novelty['mean_depth_by_bucket'][0]['mean_depth']:.4f} in the lowest-novelty bucket to {bigram_novelty['mean_depth_by_bucket'][-1]['mean_depth']:.4f} in the highest-novelty bucket (`summary.json.bigram_novelty.mean_depth_by_bucket`).",
            f"- Local entropy is also negative here: depth moves from {local_entropy['mean_depth_by_bucket'][0]['mean_depth']:.4f} in the lowest-entropy bucket to {local_entropy['mean_depth_by_bucket'][-1]['mean_depth']:.4f} in the highest-entropy bucket (`summary.json.local_entropy.mean_depth_by_bucket`).",
            f"- Immediate post-punctuation positions are shallower in this reproduction: {after_punctuation['after_punctuation_mean_depth']:.4f} after punctuation vs {after_punctuation['other_mean_depth']:.4f} elsewhere (`summary.json.after_punctuation`).",
            f"- Rare-word membership is directional but modest: rho = {word_frequency['effect_size']:.4f}, with mean depth {word_frequency['mean_depth_by_bucket'][0]['mean_depth']:.4f} in the most frequent bucket and {word_frequency['mean_depth_by_bucket'][-1]['mean_depth']:.4f} in the rarest bucket (`summary.json.word_frequency`).",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_summary = read_json(args.sweep_summary)
    threshold_summary = sweep_summary["threshold_sweep"]["recommended"]
    threshold = float(threshold_summary["threshold"])

    config = RunConfig(
        context_size=32,
        hidden_dim=128,
        max_depth=8,
        train_batch_size=256,
        train_steps=3000,
        train_learning_rate=0.003,
        seed=args.seed,
    )
    corpus = build_large_corpus(config, text_file=args.text_file)

    reproduction_output_dir = output_dir / "reproduced_training.ignore"
    model, training_summary = train_model(
        corpus=corpus,
        config=config,
        device=device,
        output_dir=reproduction_output_dir,
    )

    val_inputs = corpus.val_inputs.to(device)
    val_targets = corpus.val_targets.to(device)
    details = collect_evaluation_details(
        model,
        val_inputs,
        val_targets,
        max_depth=config.max_depth,
        batch_size=config.depth_eval_batch_size,
    )

    used_depths = [
        select_dynamic_depth(details.predicted_losses[index], threshold=threshold, improvement_epsilon=None)
        for index in range(details.predicted_losses.shape[0])
    ]
    row_indices = torch.arange(len(used_depths), dtype=torch.long)
    selected_depth_indices = torch.tensor([depth - 1 for depth in used_depths], dtype=torch.long)
    selected_losses = details.actual_losses[row_indices, selected_depth_indices]

    val_text = corpus.val_text
    target_positions = [config.context_size + index for index in range(len(used_depths))]
    target_chars = [val_text[position] for position in target_positions]
    previous_chars = [val_text[position - 1] for position in target_positions]
    next_chars = [val_text[position + 1] if position + 1 < len(val_text) else None for position in target_positions]

    character_identity = summarize_character_identity(target_chars, used_depths)

    train_bigram_counts: Counter[str] = Counter(
        corpus.train_text[position - 2 : position] for position in range(2, len(corpus.train_text))
    )
    bigram_counts = [train_bigram_counts.get(val_text[position - 2 : position], 0) for position in target_positions]
    bigram_novelties = [math.log10(len(corpus.train_text)) - math.log10(count + 1) for count in bigram_counts]
    bigram_novelty_summary = summarize_numeric_feature(bigram_novelties, used_depths, value_name="novelty")

    position_categories = []
    for char, previous_char, next_char in zip(target_chars, previous_chars, next_chars, strict=True):
        if not is_word_char(char):
            position_categories.append("non_word")
        elif previous_char == " ":
            position_categories.append("first_after_space")
        elif next_char is None or not is_word_char(next_char):
            position_categories.append("end_before_boundary")
        elif is_word_char(previous_char):
            position_categories.append("middle_of_word")
        else:
            position_categories.append("other_word_start")
    position_in_word = {
        "effect_metric": "correlation_ratio_eta",
        "effect_size": correlation_ratio(position_categories, [float(depth) for depth in used_depths]),
        "categories": [
            {
                "category": category,
                "mean_depth": mean([float(depth) for depth, label in zip(used_depths, position_categories, strict=True) if label == category]),
                "count": sum(1 for label in position_categories if label == category),
            }
            for category in sorted(set(position_categories), key=lambda value: (
                -mean([float(depth) for depth, label in zip(used_depths, position_categories, strict=True) if label == value]),
                value,
            ))
        ],
    }

    local_entropies = compute_local_entropies(target_chars, radius=args.local_entropy_radius)
    local_entropy_summary = summarize_numeric_feature(local_entropies, used_depths, value_name="entropy_bits")

    after_punctuation_flags = [1.0 if previous_char in PUNCTUATION else 0.0 for previous_char in previous_chars]
    after_depths = [float(depth) for depth, flag in zip(used_depths, after_punctuation_flags, strict=True) if flag == 1.0]
    other_depths = [float(depth) for depth, flag in zip(used_depths, after_punctuation_flags, strict=True) if flag == 0.0]
    after_punctuation = {
        "effect_metric": "point_biserial_r",
        "effect_size": pearson_correlation(after_punctuation_flags, [float(depth) for depth in used_depths]),
        "after_punctuation_mean_depth": mean(after_depths),
        "other_mean_depth": mean(other_depths),
        "mean_depth_difference": mean(after_depths) - mean(other_depths),
        "after_punctuation_count": len(after_depths),
        "other_count": len(other_depths),
    }

    train_word_counts = Counter(word.lower() for _start, _stop, word in scan_words(corpus.train_text))
    val_word_lookup = build_word_lookup(corpus.val_text)
    word_rarities = []
    word_depths = []
    for position, depth in zip(target_positions, used_depths, strict=True):
        word = val_word_lookup[position]
        if word is None:
            continue
        word_rarities.append(math.log10(len(corpus.train_text)) - math.log10(train_word_counts.get(word, 0) + 1))
        word_depths.append(depth)
    word_frequency_summary = summarize_numeric_feature(word_rarities, word_depths, value_name="rarity")

    effect_ranking = [
        {
            "feature": "character_identity",
            "effect_metric": character_identity["effect_metric"],
            "effect_size": abs(float(character_identity["effect_size"])),
            "signed_effect_size": float(character_identity["effect_size"]),
            "notes": "depth spread across target characters",
        },
        {
            "feature": "bigram_novelty",
            "effect_metric": bigram_novelty_summary.effect_metric,
            "effect_size": abs(float(bigram_novelty_summary.effect_size)),
            "signed_effect_size": float(bigram_novelty_summary.effect_size),
            "notes": "higher novelty is shallower in this reproduction",
        },
        {
            "feature": "position_in_word",
            "effect_metric": position_in_word["effect_metric"],
            "effect_size": abs(float(position_in_word["effect_size"])),
            "signed_effect_size": float(position_in_word["effect_size"]),
            "notes": "word starts/ends vs middle/non-word",
        },
        {
            "feature": "local_entropy",
            "effect_metric": local_entropy_summary.effect_metric,
            "effect_size": abs(float(local_entropy_summary.effect_size)),
            "signed_effect_size": float(local_entropy_summary.effect_size),
            "notes": "nearby next-character entropy",
        },
        {
            "feature": "after_punctuation",
            "effect_metric": after_punctuation["effect_metric"],
            "effect_size": abs(float(after_punctuation["effect_size"])),
            "signed_effect_size": float(after_punctuation["effect_size"]),
            "notes": "immediate post-punctuation positions are shallower here",
        },
        {
            "feature": "word_frequency",
            "effect_metric": word_frequency_summary.effect_metric,
            "effect_size": abs(float(word_frequency_summary.effect_size)),
            "signed_effect_size": float(word_frequency_summary.effect_size),
            "notes": "rarer words get slightly deeper compute",
        },
    ]
    effect_ranking.sort(key=lambda row: (-row["effect_size"], row["feature"]))

    summary = {
        "config": {
            **asdict(config),
            "device": str(device),
            "local_entropy_radius": args.local_entropy_radius,
        },
        "reproduction": {
            "threshold": threshold,
            "validation_token_count": len(used_depths),
            "mean_depth": mean([float(depth) for depth in used_depths]),
            "task_loss": float(selected_losses.mean().item()),
            "reference_mean_depth": float(threshold_summary["val_avg_depth"]),
            "reference_task_loss": float(threshold_summary["val_loss"]),
            "depth_histogram": dict(sorted(Counter(used_depths).items())),
            "training_runtime_seconds": float(training_summary["runtime_seconds"]),
        },
        "effect_ranking": effect_ranking,
        "character_identity": character_identity,
        "bigram_novelty": {
            "effect_metric": bigram_novelty_summary.effect_metric,
            "effect_size": bigram_novelty_summary.effect_size,
            "mean_depth_by_bucket": bigram_novelty_summary.mean_depth_by_bucket,
            "top_novel_bigrams": [
                {
                    "bigram": val_text[position - 2 : position].replace("\n", "\\n"),
                    "count_in_training": count,
                    "novelty": novelty,
                    "mean_depth": depth,
                }
                for position, count, novelty, depth in sorted(
                    zip(target_positions, bigram_counts, bigram_novelties, used_depths, strict=True),
                    key=lambda row: (-row[2], -row[3], row[0]),
                )[:10]
            ],
        },
        "position_in_word": position_in_word,
        "local_entropy": {
            "effect_metric": local_entropy_summary.effect_metric,
            "effect_size": local_entropy_summary.effect_size,
            "window_radius": args.local_entropy_radius,
            "mean_depth_by_bucket": local_entropy_summary.mean_depth_by_bucket,
        },
        "after_punctuation": after_punctuation,
        "word_frequency": {
            "effect_metric": word_frequency_summary.effect_metric,
            "effect_size": word_frequency_summary.effect_size,
            "word_character_count": len(word_depths),
            "mean_depth_by_bucket": word_frequency_summary.mean_depth_by_bucket,
            "rarest_words": [
                {
                    "word": word,
                    "count_in_training": count,
                }
                for word, count in sorted(train_word_counts.items(), key=lambda item: (item[1], item[0]))[:10]
            ],
        },
    }

    write_json(output_dir / "summary.json", summary)
    (output_dir / "analysis.md").write_text(build_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
