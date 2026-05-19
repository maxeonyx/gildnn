from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from core.fixed_window_char import FixedWindowCharDataset, resolve_device, set_seed
from core.tiny_char_transformer import count_parameters
from experiments.pytorch_char_dynamic_depth import (
    DynamicDepthCharModel,
    analyze_dynamic_depth,
    compute_loss_bundle,
    current_git_sha,
    current_git_status_short,
    train_dynamic_depth_model,
    write_json,
)
from experiments.pytorch_char_shakespeare_comparison import (
    SHAKESPEARE_SNIPPET,
    build_train_val_split,
    select_prompts,
)


@dataclass(frozen=True)
class RunConfig:
    context_size: int
    hidden_dim: int
    max_depth: int
    train_batch_size: int
    train_steps: int
    train_learning_rate: float
    embedding_dim: int = 32
    loss_pred_weight: float = 0.1
    overfit_batch_size: int = 128
    overfit_steps: int = 1500
    overfit_learning_rate: float = 0.01
    sample_length: int = 120
    seed: int = 7
    depth_eval_batch_size: int = 512
    val_analysis_chars: int = 320
    threshold_quantiles: int = 15
    epsilon_quantiles: int = 15
    recommendation_loss_tolerance: float = 0.01


@dataclass(frozen=True)
class CorpusBundle:
    corpus_name: str
    source_file: str | None
    source_text: str
    train_text: str
    val_text: str
    dataset: FixedWindowCharDataset
    train_inputs: Tensor
    train_targets: Tensor
    val_inputs: Tensor
    val_targets: Tensor
    prompts: list[str]
    corpus_summary: dict[str, object]


@dataclass(frozen=True)
class EvaluationDetails:
    actual_losses: Tensor
    predicted_losses: Tensor
    correctness: Tensor
    predictions: Tensor


def encode_windows(
    text: str,
    *,
    context_size: int,
    stoi: dict[str, int],
) -> tuple[Tensor, Tensor]:
    if len(text) <= context_size:
        raise ValueError("Text split must be longer than the context size.")
    encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    inputs = []
    targets = []
    for start in range(len(encoded) - context_size):
        stop = start + context_size
        inputs.append(encoded[start:stop])
        targets.append(encoded[stop])
    return torch.stack(inputs), torch.stack(targets)


def build_small_corpus(config: RunConfig) -> CorpusBundle:
    dataset = FixedWindowCharDataset(SHAKESPEARE_SNIPPET, context_size=config.context_size)
    split = build_train_val_split(
        SHAKESPEARE_SNIPPET,
        context_size=config.context_size,
        stoi=dataset.stoi,
        train_fraction=0.8,
    )
    prompts = select_prompts(split.train_text, split.val_text, context_size=config.context_size)
    return CorpusBundle(
        corpus_name="small_snippet",
        source_file=None,
        source_text=SHAKESPEARE_SNIPPET,
        train_text=split.train_text,
        val_text=split.val_text,
        dataset=dataset,
        train_inputs=split.train_inputs,
        train_targets=split.train_targets,
        val_inputs=split.val_inputs,
        val_targets=split.val_targets,
        prompts=prompts,
        corpus_summary={
            "corpus_name": "hardcoded_shakespeare_excerpt",
            "source_file": None,
            "source_sha256": hashlib.sha256(SHAKESPEARE_SNIPPET.encode("utf-8")).hexdigest(),
            "source_total_characters": len(SHAKESPEARE_SNIPPET),
            "train_characters": len(split.train_text),
            "val_characters": len(split.val_text),
            "train_windows": int(split.train_inputs.shape[0]),
            "val_windows": int(split.val_inputs.shape[0]),
            "vocab_size": dataset.vocab_size,
            "split": "train_fraction=0.8",
            "prompts": [prompt.replace("\n", "\\n") for prompt in prompts],
        },
    )


def build_large_corpus(config: RunConfig, *, text_file: Path) -> CorpusBundle:
    source_text = text_file.read_text(encoding="utf-8")
    train_characters = 100_000
    val_characters = 20_000
    required_characters = train_characters + val_characters
    if len(source_text) < required_characters:
        raise ValueError(f"Need at least {required_characters} characters, got {len(source_text)}.")

    train_text = source_text[:train_characters]
    val_text = source_text[train_characters : train_characters + val_characters]
    dataset = FixedWindowCharDataset(train_text, context_size=config.context_size)
    missing_val_chars = sorted(set(val_text) - set(train_text))
    if missing_val_chars:
        raise ValueError(
            f"Validation text contains characters absent from training text: {missing_val_chars}"
        )
    val_inputs, val_targets = encode_windows(val_text, context_size=config.context_size, stoi=dataset.stoi)
    prompts = select_prompts(train_text, val_text, context_size=config.context_size)
    used_text = train_text + val_text
    return CorpusBundle(
        corpus_name="large_100k",
        source_file=str(text_file),
        source_text=used_text,
        train_text=train_text,
        val_text=val_text,
        dataset=dataset,
        train_inputs=dataset.inputs,
        train_targets=dataset.targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        prompts=prompts,
        corpus_summary={
            "corpus_name": "tinyshakespeare_100k_train_20k_val",
            "source_file": str(text_file),
            "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "source_total_characters": len(source_text),
            "used_sha256": hashlib.sha256(used_text.encode("utf-8")).hexdigest(),
            "used_total_characters": len(used_text),
            "train_characters": len(train_text),
            "val_characters": len(val_text),
            "train_windows": int(dataset.inputs.shape[0]),
            "val_windows": int(val_inputs.shape[0]),
            "vocab_size": dataset.vocab_size,
            "split": "first_100k_train_next_20k_val",
            "prompts": [prompt.replace("\n", "\\n") for prompt in prompts],
        },
    )


def select_dynamic_depth(
    predicted_losses: Tensor,
    *,
    threshold: float | None,
    improvement_epsilon: float | None,
) -> int:
    depth_count = int(predicted_losses.shape[0])
    for depth_index in range(depth_count):
        current_loss = float(predicted_losses[depth_index].item())
        if threshold is not None and current_loss <= threshold:
            return depth_index + 1
        if improvement_epsilon is not None and depth_index + 1 < depth_count:
            next_loss = float(predicted_losses[depth_index + 1].item())
            improvement = current_loss - next_loss
            if improvement <= improvement_epsilon:
                return depth_index + 1
    return depth_count


def collect_evaluation_details(
    model: DynamicDepthCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    max_depth: int,
    batch_size: int,
) -> EvaluationDetails:
    predicted_losses: list[Tensor] = []
    actual_losses: list[Tensor] = []
    correctness: list[Tensor] = []
    predictions: list[Tensor] = []

    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            stop = min(start + batch_size, inputs.shape[0])
            batch_inputs = inputs[start:stop]
            batch_targets = targets[start:stop]
            logits, batch_predicted_losses = model.forward_depths(batch_inputs, max_depth=max_depth)
            final_logits = logits[:, -1, :, :]
            flat_logits = final_logits.reshape(-1, model.vocab_size)
            repeated_targets = batch_targets.unsqueeze(1).expand(-1, max_depth).reshape(-1)
            batch_actual_losses = F.cross_entropy(flat_logits, repeated_targets, reduction="none").view(-1, max_depth)
            batch_predictions = final_logits.argmax(dim=2)

            predicted_losses.append(batch_predicted_losses[:, -1, :].cpu())
            actual_losses.append(batch_actual_losses.cpu())
            predictions.append(batch_predictions.cpu())
            correctness.append(batch_predictions.eq(batch_targets.unsqueeze(1)).cpu())

    return EvaluationDetails(
        actual_losses=torch.cat(actual_losses, dim=0),
        predicted_losses=torch.cat(predicted_losses, dim=0),
        correctness=torch.cat(correctness, dim=0),
        predictions=torch.cat(predictions, dim=0),
    )


def summarize_fixed_depth(details: EvaluationDetails, *, depth: int) -> dict[str, object]:
    selected_losses = details.actual_losses[:, depth - 1]
    selected_correctness = details.correctness[:, depth - 1].float()
    return {
        "task_loss": float(selected_losses.mean().item()),
        "accuracy": float(selected_correctness.mean().item()),
        "loss_prediction_loss": float((details.predicted_losses - details.actual_losses).square().mean().item()),
        "mean_used_depth": float(depth),
        "depth_histogram": {str(depth): int(details.actual_losses.shape[0])},
        "per_depth_task_loss": [float(value) for value in details.actual_losses.mean(dim=0).tolist()],
        "per_depth_accuracy": [float(value) for value in details.correctness.float().mean(dim=0).tolist()],
    }


def summarize_dynamic_depth(
    details: EvaluationDetails,
    *,
    threshold: float | None,
    improvement_epsilon: float | None,
) -> dict[str, object]:
    used_depths = torch.tensor(
        [
            select_dynamic_depth(
                details.predicted_losses[index],
                threshold=threshold,
                improvement_epsilon=improvement_epsilon,
            )
            for index in range(details.predicted_losses.shape[0])
        ],
        dtype=torch.long,
    )
    row_indices = torch.arange(used_depths.shape[0], dtype=torch.long)
    selected_losses = details.actual_losses[row_indices, used_depths - 1]
    selected_correctness = details.correctness[row_indices, used_depths - 1].float()
    max_depth = int(details.actual_losses.shape[1])
    depth_histogram = {
        str(depth): int((used_depths == depth).sum().item())
        for depth in range(1, max_depth + 1)
        if int((used_depths == depth).sum().item()) > 0
    }
    return {
        "task_loss": float(selected_losses.mean().item()),
        "accuracy": float(selected_correctness.mean().item()),
        "loss_prediction_loss": float((details.predicted_losses - details.actual_losses).square().mean().item()),
        "mean_used_depth": float(used_depths.float().mean().item()),
        "depth_histogram": depth_histogram,
        "per_depth_task_loss": [float(value) for value in details.actual_losses.mean(dim=0).tolist()],
        "per_depth_accuracy": [float(value) for value in details.correctness.float().mean(dim=0).tolist()],
    }


def pearson_correlation(predicted: Tensor, actual: Tensor) -> float | None:
    predicted_centered = predicted - predicted.mean()
    actual_centered = actual - actual.mean()
    denominator = torch.sqrt(predicted_centered.square().sum() * actual_centered.square().sum())
    denominator_value = float(denominator.item())
    if denominator_value == 0.0:
        return None
    return float((predicted_centered * actual_centered).sum().item() / denominator_value)


def calibration_summary(details: EvaluationDetails) -> dict[str, object]:
    error = details.predicted_losses - details.actual_losses
    flat_predicted = details.predicted_losses.reshape(-1).float()
    flat_actual = details.actual_losses.reshape(-1).float()
    per_depth_correlation = []
    for depth_index in range(details.actual_losses.shape[1]):
        depth_corr = pearson_correlation(
            details.predicted_losses[:, depth_index].float(),
            details.actual_losses[:, depth_index].float(),
        )
        per_depth_correlation.append(depth_corr)
    return {
        "mse": float(error.square().mean().item()),
        "mae": float(error.abs().mean().item()),
        "overall_correlation": pearson_correlation(flat_predicted, flat_actual),
        "per_depth_mse": [float(value) for value in error.square().mean(dim=0).tolist()],
        "per_depth_mae": [float(value) for value in error.abs().mean(dim=0).tolist()],
        "per_depth_correlation": per_depth_correlation,
    }


def make_quantile_candidates(values: Tensor, *, quantiles: int) -> list[float]:
    flat_values = values.reshape(-1).float()
    grid = torch.linspace(0.0, 1.0, steps=quantiles)
    candidates = [float(flat_values.min().item()) - 1e-6]
    candidates.extend(float(value) for value in torch.quantile(flat_values, grid).tolist())
    candidates.append(float(flat_values.max().item()) + 1e-6)
    rounded = sorted({round(candidate, 6) for candidate in candidates})
    return rounded


def build_threshold_rows(
    *,
    train_details: EvaluationDetails,
    val_details: EvaluationDetails,
    candidates: list[float],
) -> list[dict[str, object]]:
    rows = []
    for threshold in candidates:
        train_metrics = summarize_dynamic_depth(train_details, threshold=threshold, improvement_epsilon=None)
        val_metrics = summarize_dynamic_depth(val_details, threshold=threshold, improvement_epsilon=None)
        rows.append(
            {
                "method": "threshold",
                "threshold": threshold,
                "train_loss": train_metrics["task_loss"],
                "train_avg_depth": train_metrics["mean_used_depth"],
                "val_loss": val_metrics["task_loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_avg_depth": val_metrics["mean_used_depth"],
                "val_depth_histogram": val_metrics["depth_histogram"],
            }
        )
    return rows


def build_epsilon_rows(
    *,
    train_details: EvaluationDetails,
    val_details: EvaluationDetails,
    candidates: list[float],
) -> list[dict[str, object]]:
    rows = []
    for epsilon in candidates:
        train_metrics = summarize_dynamic_depth(train_details, threshold=None, improvement_epsilon=epsilon)
        val_metrics = summarize_dynamic_depth(val_details, threshold=None, improvement_epsilon=epsilon)
        rows.append(
            {
                "method": "relative_improvement",
                "improvement_epsilon": epsilon,
                "train_loss": train_metrics["task_loss"],
                "train_avg_depth": train_metrics["mean_used_depth"],
                "val_loss": val_metrics["task_loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_avg_depth": val_metrics["mean_used_depth"],
                "val_depth_histogram": val_metrics["depth_histogram"],
            }
        )
    return rows


def pareto_frontier(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ranked = sorted(rows, key=lambda row: (float(row["val_avg_depth"]), float(row["val_loss"])))
    frontier: list[dict[str, object]] = []
    best_loss = math.inf
    annotated_rows: list[dict[str, object]] = []
    frontier_keys: set[tuple[str, float]] = set()

    for row in ranked:
        row_key_value = row.get("threshold", row.get("improvement_epsilon"))
        row_key = (str(row["method"]), float(row_key_value))
        if float(row["val_loss"]) < best_loss - 1e-9:
            frontier.append(row)
            frontier_keys.add(row_key)
            best_loss = float(row["val_loss"])

    for row in rows:
        row_key_value = row.get("threshold", row.get("improvement_epsilon"))
        row_key = (str(row["method"]), float(row_key_value))
        annotated_rows.append({**row, "is_pareto": row_key in frontier_keys})
    return frontier, annotated_rows


def choose_recommended_point(
    frontier: list[dict[str, object]],
    *,
    tolerance_fraction: float,
) -> dict[str, object]:
    best_loss = min(float(row["val_loss"]) for row in frontier)
    allowed_loss = best_loss * (1.0 + tolerance_fraction)
    candidates = [row for row in frontier if float(row["val_loss"]) <= allowed_loss]
    ranked_candidates = sorted(
        candidates,
        key=lambda row: (float(row["val_avg_depth"]), float(row["val_loss"])),
    )
    return ranked_candidates[0]


def write_tsv(path: Path, rows: list[dict[str, object]], *, value_key: str) -> None:
    header = [
        value_key,
        "train_loss",
        "train_avg_depth",
        "val_loss",
        "val_accuracy",
        "val_avg_depth",
        "is_pareto",
        "is_recommended",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    f"{float(row[value_key]):.6f}",
                    f"{float(row['train_loss']):.6f}",
                    f"{float(row['train_avg_depth']):.6f}",
                    f"{float(row['val_loss']):.6f}",
                    f"{float(row['val_accuracy']):.6f}",
                    f"{float(row['val_avg_depth']):.6f}",
                    str(bool(row.get("is_pareto", False))).lower(),
                    str(bool(row.get("is_recommended", False))).lower(),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_combined_threshold_table(path: Path, results: list[dict[str, object]]) -> None:
    lines = ["corpus\tthreshold\tval_loss\tavg_depth\tis_pareto\tis_recommended"]
    for result in results:
        for row in result["threshold_rows"]:
            lines.append(
                "\t".join(
                    [
                        str(result["corpus_name"]),
                        f"{float(row['threshold']):.6f}",
                        f"{float(row['val_loss']):.6f}",
                        f"{float(row['val_avg_depth']):.6f}",
                        str(bool(row.get("is_pareto", False))).lower(),
                        str(bool(row.get("is_recommended", False))).lower(),
                    ]
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_loss_bundle_batched(
    model: DynamicDepthCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    active_depth: int,
    loss_pred_weight: float,
    batch_size: int,
) -> dict[str, object]:
    total_examples = 0
    total_loss_sum = 0.0
    task_loss_sum = 0.0
    loss_prediction_loss_sum = 0.0
    accuracy_sum = 0.0
    per_depth_task_loss_sum = torch.zeros(active_depth, dtype=torch.float64)
    per_depth_accuracy_sum = torch.zeros(active_depth, dtype=torch.float64)

    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            stop = min(start + batch_size, inputs.shape[0])
            batch_inputs = inputs[start:stop]
            batch_targets = targets[start:stop]
            bundle = compute_loss_bundle(
                model,
                batch_inputs,
                batch_targets,
                active_depth=active_depth,
                loss_pred_weight=loss_pred_weight,
            )
            example_count = batch_targets.shape[0]
            total_examples += example_count
            total_loss_sum += float(bundle.total_loss.item()) * example_count
            task_loss_sum += float(bundle.task_loss.item()) * example_count
            loss_prediction_loss_sum += float(bundle.loss_prediction_loss.item()) * example_count
            accuracy_sum += float(bundle.accuracy) * example_count
            per_depth_task_loss_sum += torch.tensor(bundle.per_depth_task_loss, dtype=torch.float64) * example_count
            per_depth_accuracy_sum += torch.tensor(bundle.per_depth_accuracy, dtype=torch.float64) * example_count

    return {
        "total_loss": total_loss_sum / total_examples,
        "task_loss": task_loss_sum / total_examples,
        "loss_prediction_loss": loss_prediction_loss_sum / total_examples,
        "accuracy": accuracy_sum / total_examples,
        "per_depth_task_loss": [float(value) for value in (per_depth_task_loss_sum / total_examples).tolist()],
        "per_depth_accuracy": [float(value) for value in (per_depth_accuracy_sum / total_examples).tolist()],
    }


def train_dynamic_depth_model_batched(
    model: DynamicDepthCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
    steps: int,
    learning_rate: float,
    active_depth: int,
    loss_pred_weight: float,
    trace_batch_size: int,
    trace_eval_examples: int,
    trace_interval: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    sample_count = inputs.shape[0]
    trace: list[dict[str, object]] = []
    trace_count = min(trace_eval_examples, sample_count)
    trace_inputs = inputs[:trace_count]
    trace_targets = targets[:trace_count]

    for step in range(steps + 1):
        if step % trace_interval == 0 or step == steps:
            metrics = evaluate_loss_bundle_batched(
                model,
                trace_inputs,
                trace_targets,
                active_depth=active_depth,
                loss_pred_weight=loss_pred_weight,
                batch_size=trace_batch_size,
            )
            trace.append(
                {
                    "step": step,
                    "total_loss": round(float(metrics["total_loss"]), 6),
                    "task_loss": round(float(metrics["task_loss"]), 6),
                    "loss_prediction_loss": round(float(metrics["loss_prediction_loss"]), 6),
                    "accuracy": round(float(metrics["accuracy"]), 6),
                    "per_depth_task_loss": [round(float(value), 6) for value in metrics["per_depth_task_loss"]],
                    "per_depth_accuracy": [round(float(value), 6) for value in metrics["per_depth_accuracy"]],
                    "evaluated_examples": trace_count,
                }
            )

        if step == steps:
            break

        batch_indices = torch.randint(0, sample_count, (batch_size,), device=inputs.device)
        batch_inputs = inputs[batch_indices]
        batch_targets = targets[batch_indices]
        train_bundle = compute_loss_bundle(
            model,
            batch_inputs,
            batch_targets,
            active_depth=active_depth,
            loss_pred_weight=loss_pred_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        train_bundle.total_loss.backward()
        optimizer.step()

    final_metrics = evaluate_loss_bundle_batched(
        model,
        inputs,
        targets,
        active_depth=active_depth,
        loss_pred_weight=loss_pred_weight,
        batch_size=trace_batch_size,
    )
    return trace, final_metrics


def train_model(
    *,
    corpus: CorpusBundle,
    config: RunConfig,
    device: torch.device,
    output_dir: Path,
) -> tuple[DynamicDepthCharModel, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    def build_model() -> DynamicDepthCharModel:
        return DynamicDepthCharModel(
            vocab_size=corpus.dataset.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            max_depth=config.max_depth,
        )

    overfit_model = build_model().to(device)
    overfit_inputs = corpus.train_inputs[: config.overfit_batch_size].to(device)
    overfit_targets = corpus.train_targets[: config.overfit_batch_size].to(device)
    overfit_trace, overfit_bundle = train_dynamic_depth_model(
        overfit_model,
        overfit_inputs,
        overfit_targets,
        batch_size=config.overfit_batch_size,
        steps=config.overfit_steps,
        learning_rate=config.overfit_learning_rate,
        active_depth=config.max_depth,
        loss_pred_weight=config.loss_pred_weight,
    )
    overfit_reached = overfit_bundle.accuracy == 1.0 and overfit_bundle.task_loss.item() < 1e-2
    if not overfit_reached:
        raise RuntimeError(f"{corpus.corpus_name} failed the one-batch overfit check.")

    model = build_model().to(device)
    train_inputs = corpus.train_inputs.to(device)
    train_targets = corpus.train_targets.to(device)
    started_at = time.perf_counter()
    trace_interval = 50 if config.train_steps <= 2200 else 100
    trace_eval_examples = min(4096, int(corpus.train_inputs.shape[0]))
    train_trace, final_train_metrics = train_dynamic_depth_model_batched(
        model,
        train_inputs,
        train_targets,
        batch_size=config.train_batch_size,
        steps=config.train_steps,
        learning_rate=config.train_learning_rate,
        active_depth=config.max_depth,
        loss_pred_weight=config.loss_pred_weight,
        trace_batch_size=config.depth_eval_batch_size,
        trace_eval_examples=trace_eval_examples,
        trace_interval=trace_interval,
    )
    runtime_seconds = time.perf_counter() - started_at

    training_summary = {
        "parameter_count": count_parameters(model),
        "runtime_seconds": runtime_seconds,
        "overfit_reached": overfit_reached,
        "overfit_trace": overfit_trace,
        "final_train_bundle": {
            "total_loss": float(final_train_metrics["total_loss"]),
            "task_loss": float(final_train_metrics["task_loss"]),
            "loss_prediction_loss": float(final_train_metrics["loss_prediction_loss"]),
            "accuracy": float(final_train_metrics["accuracy"]),
            "per_depth_task_loss": [float(value) for value in final_train_metrics["per_depth_task_loss"]],
            "per_depth_accuracy": [float(value) for value in final_train_metrics["per_depth_accuracy"]],
        },
        "train_trace": train_trace,
    }
    write_json(output_dir / "training_summary.json", training_summary)
    return model, training_summary


def run_corpus_experiment(
    *,
    corpus: CorpusBundle,
    config: RunConfig,
    device: torch.device,
    output_root: Path,
) -> dict[str, object]:
    corpus_dir = output_root / corpus.corpus_name
    corpus_dir.mkdir(parents=True, exist_ok=True)
    write_json(corpus_dir / "config.json", asdict(config))
    write_json(corpus_dir / "corpus_summary.json", corpus.corpus_summary)

    model, training_summary = train_model(corpus=corpus, config=config, device=device, output_dir=corpus_dir)
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

    train_calibration = calibration_summary(train_details)
    val_calibration = calibration_summary(val_details)
    threshold_candidates = make_quantile_candidates(
        train_details.predicted_losses,
        quantiles=config.threshold_quantiles,
    )
    improvement_candidates = make_quantile_candidates(
        train_details.predicted_losses[:, :-1] - train_details.predicted_losses[:, 1:],
        quantiles=config.epsilon_quantiles,
    )

    threshold_frontier, threshold_rows = pareto_frontier(
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
    threshold_rows = [
        {
            **row,
            "is_recommended": float(row["threshold"]) == float(threshold_recommended["threshold"]),
        }
        for row in threshold_rows
    ]

    improvement_frontier, improvement_rows = pareto_frontier(
        build_epsilon_rows(
            train_details=train_details,
            val_details=val_details,
            candidates=improvement_candidates,
        )
    )
    improvement_recommended = choose_recommended_point(
        improvement_frontier,
        tolerance_fraction=config.recommendation_loss_tolerance,
    )
    improvement_rows = [
        {
            **row,
            "is_recommended": float(row["improvement_epsilon"]) == float(improvement_recommended["improvement_epsilon"]),
        }
        for row in improvement_rows
    ]

    write_tsv(corpus_dir / "threshold_sweep.tsv", threshold_rows, value_key="threshold")
    write_tsv(corpus_dir / "relative_improvement_sweep.tsv", improvement_rows, value_key="improvement_epsilon")

    write_json(
        corpus_dir / "sweep_summary.json",
        {
            "training": training_summary,
            "fixed_depth_8_train": summarize_fixed_depth(train_details, depth=config.max_depth),
            "fixed_depth_8_val": summarize_fixed_depth(val_details, depth=config.max_depth),
            "calibration": {
                "train": train_calibration,
                "val": val_calibration,
            },
            "threshold_sweep": {
                "rows": threshold_rows,
                "pareto_frontier": threshold_frontier,
                "recommended": threshold_recommended,
            },
            "relative_improvement_sweep": {
                "rows": improvement_rows,
                "pareto_frontier": improvement_frontier,
                "recommended": improvement_recommended,
            },
        },
    )

    analysis_dir = corpus_dir / "recommended_threshold_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    annotation = analyze_dynamic_depth(
        model=model,
        dataset=corpus.dataset,
        val_text=corpus.val_text,
        val_inputs=val_inputs,
        val_targets=val_targets,
        output_dir=analysis_dir,
        analysis_chars=config.val_analysis_chars,
        halting_threshold=float(threshold_recommended["threshold"]),
        improvement_epsilon=float("-inf"),
    )

    return {
        "corpus_name": corpus.corpus_name,
        "config": asdict(config),
        "corpus_summary": corpus.corpus_summary,
        "training": training_summary,
        "calibration": {
            "train": train_calibration,
            "val": val_calibration,
        },
        "threshold_rows": threshold_rows,
        "threshold_frontier": threshold_frontier,
        "threshold_recommended": threshold_recommended,
        "relative_improvement_rows": improvement_rows,
        "relative_improvement_frontier": improvement_frontier,
        "relative_improvement_recommended": improvement_recommended,
        "recommended_threshold_annotation": annotation,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_text_file = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    default_output_dir = (
        repo_root / "research" / "questions" / "dynamic-depth" / "artifacts" / "improved"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--text-file", type=Path, default=default_text_file)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--small-train-steps", type=int, default=2000)
    parser.add_argument("--large-train-steps", type=int, default=3000)
    args = parser.parse_args()

    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    git_status_short = current_git_status_short()
    write_json(
        output_dir / "environment.json",
        {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": git_status_short == [],
            "git_status_short": git_status_short,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_is_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
    )

    small_config = RunConfig(
        context_size=10,
        hidden_dim=128,
        max_depth=8,
        train_batch_size=128,
        train_steps=args.small_train_steps,
        train_learning_rate=0.005,
        seed=args.seed,
    )
    large_config = RunConfig(
        context_size=32,
        hidden_dim=128,
        max_depth=8,
        train_batch_size=256,
        train_steps=args.large_train_steps,
        train_learning_rate=0.003,
        seed=args.seed,
    )

    small_result = run_corpus_experiment(
        corpus=build_small_corpus(small_config),
        config=small_config,
        device=device,
        output_root=output_dir,
    )
    large_result = run_corpus_experiment(
        corpus=build_large_corpus(large_config, text_file=args.text_file),
        config=large_config,
        device=device,
        output_root=output_dir,
    )

    write_combined_threshold_table(output_dir / "threshold_pareto_table.tsv", [small_result, large_result])
    write_json(
        output_dir / "comparison_summary.json",
        {
            "small_corpus": small_result,
            "large_corpus": large_result,
            "calibration_delta": {
                "val_mse_small": small_result["calibration"]["val"]["mse"],
                "val_mse_large": large_result["calibration"]["val"]["mse"],
                "val_mae_small": small_result["calibration"]["val"]["mae"],
                "val_mae_large": large_result["calibration"]["val"]["mae"],
                "val_correlation_small": small_result["calibration"]["val"]["overall_correlation"],
                "val_correlation_large": large_result["calibration"]["val"]["overall_correlation"],
            },
        },
    )


if __name__ == "__main__":
    main()
