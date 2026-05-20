from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from core.fixed_window_char import resolve_device, set_seed
from core.tiny_char_transformer import count_parameters
from experiments.pytorch_char_dynamic_depth import (
    DynamicDepthCharModel,
    analyze_dynamic_depth,
    current_git_sha,
    current_git_status_short,
    write_json,
)
from experiments.pytorch_char_dynamic_depth_improved import (
    build_large_corpus,
    build_threshold_rows,
    calibration_summary,
    choose_recommended_point,
    collect_evaluation_details,
    make_quantile_candidates,
    pareto_frontier,
    summarize_fixed_depth,
)


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 32
    embedding_dim: int = 32
    hidden_dim: int = 128
    max_depth: int = 8
    loss_pred_weight: float = 0.1
    self_prediction_weight: float = 0.5
    overfit_batch_size: int = 128
    overfit_steps: int = 1500
    overfit_learning_rate: float = 0.01
    train_batch_size: int = 256
    train_steps: int = 2000
    train_learning_rate: float = 0.003
    depth_eval_batch_size: int = 512
    threshold_quantiles: int = 15
    recommendation_loss_tolerance: float = 0.01
    seed: int = 7


@dataclass(frozen=True)
class VariantSpec:
    name: str
    self_prediction_weight: float


@dataclass(frozen=True)
class LossBundle:
    total_loss: Tensor
    task_loss: Tensor
    loss_prediction_loss: Tensor
    self_prediction_loss: Tensor
    accuracy: float
    per_depth_task_loss: list[float]
    per_depth_accuracy: list[float]
    per_depth_self_prediction_loss: list[float]


def compute_self_prediction_loss(final_logits_by_depth: Tensor) -> tuple[Tensor, list[float]]:
    active_depth = int(final_logits_by_depth.shape[1])
    if active_depth <= 1:
        zero = final_logits_by_depth.new_zeros(())
        return zero, []

    target_distribution = F.softmax(final_logits_by_depth[:, -1, :].detach(), dim=-1)
    shallow_log_probs = F.log_softmax(final_logits_by_depth[:, :-1, :], dim=-1)
    target_distribution = target_distribution.unsqueeze(1).expand_as(shallow_log_probs)
    per_example_per_depth = F.kl_div(
        shallow_log_probs,
        target_distribution,
        reduction="none",
    ).sum(dim=-1)
    return per_example_per_depth.mean(), [
        float(value) for value in per_example_per_depth.mean(dim=0).tolist()
    ]


def compute_loss_bundle(
    model: DynamicDepthCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    active_depth: int,
    loss_pred_weight: float,
    self_prediction_weight: float,
) -> LossBundle:
    logits, predicted_losses = model.forward_depths(inputs, max_depth=active_depth)
    final_logits_by_depth = logits[:, -1, :, :]
    per_example_losses = []
    per_depth_losses = []
    per_depth_accuracies = []

    for depth_index in range(active_depth):
        depth_logits = final_logits_by_depth[:, depth_index, :]
        depth_loss_per_example = F.cross_entropy(depth_logits, targets, reduction="none")
        per_example_losses.append(depth_loss_per_example)
        per_depth_losses.append(depth_loss_per_example.mean())
        per_depth_accuracies.append(
            (depth_logits.argmax(dim=1) == targets).float().mean().item()
        )

    actual_losses = torch.stack(per_example_losses, dim=1)
    predicted_final_losses = predicted_losses[:, -1, :]
    task_loss = torch.stack(per_depth_losses).mean()
    loss_prediction_loss = F.mse_loss(predicted_final_losses, actual_losses.detach())
    self_prediction_loss, per_depth_self_prediction_loss = compute_self_prediction_loss(final_logits_by_depth)
    total_loss = (
        task_loss
        + loss_pred_weight * loss_prediction_loss
        + self_prediction_weight * self_prediction_loss
    )
    return LossBundle(
        total_loss=total_loss,
        task_loss=task_loss,
        loss_prediction_loss=loss_prediction_loss,
        self_prediction_loss=self_prediction_loss,
        accuracy=per_depth_accuracies[-1],
        per_depth_task_loss=[float(loss.item()) for loss in per_depth_losses],
        per_depth_accuracy=per_depth_accuracies,
        per_depth_self_prediction_loss=per_depth_self_prediction_loss,
    )


def verify_detached_target(
    model: DynamicDepthCharModel,
    inputs: Tensor,
    *,
    active_depth: int,
) -> dict[str, float | bool | None]:
    if active_depth <= 1:
        return {
            "checked": False,
            "shallow_grad_norm": None,
            "final_grad_norm": None,
        }

    logits, _predicted_losses = model.forward_depths(inputs, max_depth=active_depth)
    final_logits_by_depth = logits[:, -1, :, :]
    shallow_logits = final_logits_by_depth[:, 0, :]
    deepest_logits = final_logits_by_depth[:, -1, :]
    target_distribution = F.softmax(deepest_logits.detach(), dim=-1)
    self_prediction_loss = F.kl_div(
        F.log_softmax(shallow_logits, dim=-1),
        target_distribution,
        reduction="batchmean",
    )
    shallow_grad, deepest_grad = torch.autograd.grad(
        self_prediction_loss,
        (shallow_logits, deepest_logits),
        allow_unused=True,
    )
    return {
        "checked": True,
        "shallow_grad_norm": float(shallow_grad.norm().item()) if shallow_grad is not None else None,
        "final_grad_norm": float(deepest_grad.norm().item()) if deepest_grad is not None else None,
    }


def evaluate_loss_bundle_batched(
    model: DynamicDepthCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    active_depth: int,
    loss_pred_weight: float,
    self_prediction_weight: float,
    batch_size: int,
) -> dict[str, object]:
    total_examples = 0
    total_loss_sum = 0.0
    task_loss_sum = 0.0
    loss_prediction_loss_sum = 0.0
    self_prediction_loss_sum = 0.0
    accuracy_sum = 0.0
    per_depth_task_loss_sum = torch.zeros(active_depth, dtype=torch.float64)
    per_depth_accuracy_sum = torch.zeros(active_depth, dtype=torch.float64)
    per_depth_self_prediction_sum = torch.zeros(max(active_depth - 1, 0), dtype=torch.float64)

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
                self_prediction_weight=self_prediction_weight,
            )
            example_count = batch_targets.shape[0]
            total_examples += example_count
            total_loss_sum += float(bundle.total_loss.item()) * example_count
            task_loss_sum += float(bundle.task_loss.item()) * example_count
            loss_prediction_loss_sum += float(bundle.loss_prediction_loss.item()) * example_count
            self_prediction_loss_sum += float(bundle.self_prediction_loss.item()) * example_count
            accuracy_sum += float(bundle.accuracy) * example_count
            per_depth_task_loss_sum += torch.tensor(bundle.per_depth_task_loss, dtype=torch.float64) * example_count
            per_depth_accuracy_sum += torch.tensor(bundle.per_depth_accuracy, dtype=torch.float64) * example_count
            if active_depth > 1:
                per_depth_self_prediction_sum += (
                    torch.tensor(bundle.per_depth_self_prediction_loss, dtype=torch.float64) * example_count
                )

    return {
        "total_loss": total_loss_sum / total_examples,
        "task_loss": task_loss_sum / total_examples,
        "loss_prediction_loss": loss_prediction_loss_sum / total_examples,
        "self_prediction_loss": self_prediction_loss_sum / total_examples,
        "accuracy": accuracy_sum / total_examples,
        "per_depth_task_loss": [float(value) for value in (per_depth_task_loss_sum / total_examples).tolist()],
        "per_depth_accuracy": [float(value) for value in (per_depth_accuracy_sum / total_examples).tolist()],
        "per_depth_self_prediction_loss": [
            float(value) for value in (per_depth_self_prediction_sum / total_examples).tolist()
        ],
    }


def train_model_batched(
    model: DynamicDepthCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
    steps: int,
    learning_rate: float,
    active_depth: int,
    loss_pred_weight: float,
    self_prediction_weight: float,
    trace_batch_size: int,
    trace_eval_examples: int,
    trace_interval: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    sample_count = inputs.shape[0]
    trace = []
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
                self_prediction_weight=self_prediction_weight,
                batch_size=trace_batch_size,
            )
            trace.append(
                {
                    "step": step,
                    "total_loss": round(float(metrics["total_loss"]), 6),
                    "task_loss": round(float(metrics["task_loss"]), 6),
                    "loss_prediction_loss": round(float(metrics["loss_prediction_loss"]), 6),
                    "self_prediction_loss": round(float(metrics["self_prediction_loss"]), 6),
                    "accuracy": round(float(metrics["accuracy"]), 6),
                    "per_depth_task_loss": [round(float(value), 6) for value in metrics["per_depth_task_loss"]],
                    "per_depth_accuracy": [round(float(value), 6) for value in metrics["per_depth_accuracy"]],
                    "per_depth_self_prediction_loss": [
                        round(float(value), 6) for value in metrics["per_depth_self_prediction_loss"]
                    ],
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
            self_prediction_weight=self_prediction_weight,
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
        self_prediction_weight=self_prediction_weight,
        batch_size=trace_batch_size,
    )
    return trace, final_metrics


def add_compute_savings(rows: list[dict[str, object]], *, max_depth: int) -> list[dict[str, object]]:
    enriched = []
    for row in rows:
        avg_depth = float(row["val_avg_depth"])
        enriched.append({
            **row,
            "compute_savings": 1.0 - (avg_depth / max_depth),
        })
    return enriched


def summarize_dynamic_depth(details, *, threshold: float) -> dict[str, object]:
    used_depths = torch.tensor(
        [
            next(
                (
                    depth_index + 1
                    for depth_index, predicted_loss in enumerate(details.predicted_losses[index])
                    if float(predicted_loss) <= threshold
                ),
                details.predicted_losses.shape[1],
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
        "mean_used_depth": float(used_depths.float().mean().item()),
        "depth_histogram": depth_histogram,
    }


def train_variant(
    *,
    variant: VariantSpec,
    config: RunConfig,
    corpus,
    device: torch.device,
    output_root: Path,
) -> dict[str, object]:
    variant_dir = output_root / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)

    def build_model() -> DynamicDepthCharModel:
        return DynamicDepthCharModel(
            vocab_size=corpus.dataset.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            max_depth=config.max_depth,
        )

    train_inputs = corpus.train_inputs.to(device)
    train_targets = corpus.train_targets.to(device)
    val_inputs = corpus.val_inputs.to(device)
    val_targets = corpus.val_targets.to(device)
    overfit_inputs = train_inputs[: config.overfit_batch_size]
    overfit_targets = train_targets[: config.overfit_batch_size]

    set_seed(config.seed)
    detach_probe_model = build_model().to(device)
    detach_probe = verify_detached_target(detach_probe_model, overfit_inputs, active_depth=config.max_depth)

    set_seed(config.seed)
    overfit_model = build_model().to(device)
    overfit_trace, overfit_metrics = train_model_batched(
        overfit_model,
        overfit_inputs,
        overfit_targets,
        batch_size=config.overfit_batch_size,
        steps=config.overfit_steps,
        learning_rate=config.overfit_learning_rate,
        active_depth=config.max_depth,
        loss_pred_weight=config.loss_pred_weight,
        self_prediction_weight=variant.self_prediction_weight,
        trace_batch_size=config.overfit_batch_size,
        trace_eval_examples=config.overfit_batch_size,
        trace_interval=50,
    )
    overfit_reached = (
        float(overfit_metrics["accuracy"]) == 1.0 and float(overfit_metrics["task_loss"]) < 1e-2
    )
    if not overfit_reached:
        raise RuntimeError(f"{variant.name} failed the one-batch overfit check.")

    set_seed(config.seed + 1)
    model = build_model().to(device)
    started_at = time.perf_counter()
    train_trace, final_train_metrics = train_model_batched(
        model,
        train_inputs,
        train_targets,
        batch_size=config.train_batch_size,
        steps=config.train_steps,
        learning_rate=config.train_learning_rate,
        active_depth=config.max_depth,
        loss_pred_weight=config.loss_pred_weight,
        self_prediction_weight=variant.self_prediction_weight,
        trace_batch_size=config.depth_eval_batch_size,
        trace_eval_examples=min(4096, int(train_inputs.shape[0])),
        trace_interval=50,
    )
    runtime_seconds = time.perf_counter() - started_at

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
    threshold_frontier, threshold_rows = pareto_frontier(
        build_threshold_rows(
            train_details=train_details,
            val_details=val_details,
            candidates=threshold_candidates,
        )
    )
    threshold_frontier = add_compute_savings(threshold_frontier, max_depth=config.max_depth)
    threshold_rows = add_compute_savings(threshold_rows, max_depth=config.max_depth)
    threshold_recommended = choose_recommended_point(
        threshold_frontier,
        tolerance_fraction=config.recommendation_loss_tolerance,
    )

    fixed_depths = sorted({1, 2, 3, 4, config.max_depth})
    fixed_depth_val = {
        str(depth): summarize_fixed_depth(val_details, depth=depth)
        for depth in fixed_depths
    }
    fixed_depth_train = {
        str(depth): summarize_fixed_depth(train_details, depth=depth)
        for depth in fixed_depths
    }
    recommended_dynamic_val = summarize_dynamic_depth(
        val_details,
        threshold=float(threshold_recommended["threshold"]),
    )
    recommended_dynamic_train = summarize_dynamic_depth(
        train_details,
        threshold=float(threshold_recommended["threshold"]),
    )
    analysis_dir = variant_dir / "recommended_threshold_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    recommended_analysis = analyze_dynamic_depth(
        model=model,
        dataset=corpus.dataset,
        val_text=corpus.val_text,
        val_inputs=val_inputs,
        val_targets=val_targets,
        output_dir=analysis_dir,
        analysis_chars=320,
        halting_threshold=float(threshold_recommended["threshold"]),
        improvement_epsilon=float("-inf"),
    )

    summary = {
        "variant": variant.name,
        "self_prediction_weight": variant.self_prediction_weight,
        "parameter_count": count_parameters(model),
        "runtime_seconds": runtime_seconds,
        "detach_probe": detach_probe,
        "overfit": {
            "reached_memorization_bar": overfit_reached,
            "trace": overfit_trace,
            "final_metrics": final_train_or_overfit_metrics(overfit_metrics),
        },
        "train": {
            "trace": train_trace,
            "final_metrics": final_train_or_overfit_metrics(final_train_metrics),
        },
        "calibration": {
            "train": calibration_summary(train_details),
            "val": calibration_summary(val_details),
        },
        "fixed_depth_train": fixed_depth_train,
        "fixed_depth_val": fixed_depth_val,
        "threshold_frontier": threshold_frontier,
        "threshold_rows": threshold_rows,
        "threshold_recommended": threshold_recommended,
        "recommended_dynamic_train": recommended_dynamic_train,
        "recommended_dynamic_val": recommended_dynamic_val,
        "recommended_threshold_annotation": recommended_analysis,
    }
    write_json(variant_dir / "summary.json", summary)
    return summary


def final_train_or_overfit_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "total_loss": float(metrics["total_loss"]),
        "task_loss": float(metrics["task_loss"]),
        "loss_prediction_loss": float(metrics["loss_prediction_loss"]),
        "self_prediction_loss": float(metrics["self_prediction_loss"]),
        "accuracy": float(metrics["accuracy"]),
        "per_depth_task_loss": [float(value) for value in metrics["per_depth_task_loss"]],
        "per_depth_accuracy": [float(value) for value in metrics["per_depth_accuracy"]],
        "per_depth_self_prediction_loss": [
            float(value) for value in metrics["per_depth_self_prediction_loss"]
        ],
    }


def dominates(left: dict[str, object], right: dict[str, object]) -> bool:
    left_loss = float(left["val_loss"])
    right_loss = float(right["val_loss"])
    left_depth = float(left["val_avg_depth"])
    right_depth = float(right["val_avg_depth"])
    return (
        left_loss <= right_loss + 1e-9
        and left_depth <= right_depth + 1e-9
        and (left_loss < right_loss - 1e-9 or left_depth < right_depth - 1e-9)
    )


def frontier_dominance_summary(
    baseline_frontier: list[dict[str, object]],
    self_prediction_frontier: list[dict[str, object]],
) -> dict[str, bool]:
    return {
        "self_prediction_dominates_any_baseline_point": any(
            dominates(left, right)
            for left in self_prediction_frontier
            for right in baseline_frontier
        ),
        "baseline_dominates_any_self_prediction_point": any(
            dominates(left, right)
            for left in baseline_frontier
            for right in self_prediction_frontier
        ),
    }


def markdown_json(payload: object) -> str:
    return json.dumps(payload, indent=2)


def write_report(
    *,
    path: Path,
    comparison: dict[str, object],
    experiment_dir: Path,
) -> None:
    baseline = comparison["variants"]["baseline"]
    self_prediction = comparison["variants"]["self_prediction"]
    fixed_depths = [1, 2, 3, 4, comparison["config"]["max_depth"]]

    fixed_rows = []
    for depth in fixed_depths:
        baseline_loss = float(baseline["fixed_depth_val"][str(depth)]["task_loss"])
        self_prediction_loss = float(self_prediction["fixed_depth_val"][str(depth)]["task_loss"])
        fixed_rows.append(
            f"| {depth} | {baseline_loss:.4f} | {self_prediction_loss:.4f} | {self_prediction_loss - baseline_loss:+.4f} |"
        )

    frontier_rows = []
    for variant_name, variant_summary in comparison["variants"].items():
        for row in variant_summary["threshold_frontier"]:
            frontier_rows.append(
                f"| {variant_name} | {float(row['threshold']):.4f} | {float(row['val_loss']):.4f} | {float(row['val_avg_depth']):.2f} | {float(row['compute_savings']) * 100:.1f}% |"
            )

    readme = f"""# Self-prediction compute compression

## Question

Does adding a detached self-prediction KL loss make shallow recurrent steps approximate the model's own final-depth output distribution well enough to improve quality at lower depth or improve the dynamic halting quality/compute frontier?

Serves dictation [`../../dictations/2026-05-20-12.md`](../../dictations/2026-05-20-12.md) with the matched-control constraint from [`../../dictations/2026-05-20-11.md`](../../dictations/2026-05-20-11.md).

## Simplification

- Only logits, not internal latents
- Existing weight-shared GRU dynamic-depth scaffold
- Same TinyShakespeare `100K/20K`, char-level, `ctx=32`, `d_model` proxy `hidden_dim=128`, `max_depth=8`, same optimizer and seed frame for both variants
- Only changed variable: auxiliary shallow-to-final detached KL loss

## Overfit wiring check

Artifacts: [`comparison_summary.json`](../../../{experiment_dir.as_posix()}/comparison_summary.json)

| Variant | Detached target check | Shallow grad norm | Final-target grad norm | Initial aux KL | Final aux KL | Final task loss | Final accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | {str(bool(baseline['detach_probe']['checked']))} | {float(baseline['detach_probe']['shallow_grad_norm'] or 0.0):.6f} | {float(baseline['detach_probe']['final_grad_norm'] or 0.0):.6f} | {float(baseline['overfit']['trace'][0]['self_prediction_loss']):.6f} | {float(baseline['overfit']['trace'][-1]['self_prediction_loss']):.6f} | {float(baseline['overfit']['final_metrics']['task_loss']):.6f} | {float(baseline['overfit']['final_metrics']['accuracy']):.3f} |
| self_prediction | {str(bool(self_prediction['detach_probe']['checked']))} | {float(self_prediction['detach_probe']['shallow_grad_norm'] or 0.0):.6f} | {float(self_prediction['detach_probe']['final_grad_norm'] or 0.0):.6f} | {float(self_prediction['overfit']['trace'][0]['self_prediction_loss']):.6f} | {float(self_prediction['overfit']['trace'][-1]['self_prediction_loss']):.6f} | {float(self_prediction['overfit']['final_metrics']['task_loss']):.6f} | {float(self_prediction['overfit']['final_metrics']['accuracy']):.3f} |

## Fixed-depth validation comparison

| Depth | Baseline val loss | Self-pred val loss | Delta |
|---|---:|---:|---:|
{chr(10).join(fixed_rows)}

## Dynamic halting frontier comparison

| Variant | Threshold | Val loss | Avg depth | Compute savings |
|---|---:|---:|---:|---:|
{chr(10).join(frontier_rows)}

## Frontier dominance check

```json
{markdown_json(comparison['frontier_dominance'])}
```

## Recommended operating points

```json
{markdown_json({
    'baseline': baseline['threshold_recommended'],
    'self_prediction': self_prediction['threshold_recommended'],
})}
```

## Inline evidence

```json
{markdown_json({
    'baseline_fixed_depth_val': {depth: baseline['fixed_depth_val'][str(depth)]['task_loss'] for depth in fixed_depths},
    'self_prediction_fixed_depth_val': {depth: self_prediction['fixed_depth_val'][str(depth)]['task_loss'] for depth in fixed_depths},
    'baseline_recommended_dynamic_val': baseline['recommended_dynamic_val'],
    'self_prediction_recommended_dynamic_val': self_prediction['recommended_dynamic_val'],
})}
```

## Non-goals

- Does not test latent self-prediction yet
- Does not settle larger-scale training or seed stability
- Does not compare against non-GRU architectures

## Remaining prose work

- Interpret the result in Max-readable prose
- Explain why the frontier moved or failed to move
"""
    path.write_text(readme, encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_text_file = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    default_output_dir = repo_root / "experiments" / "self_prediction" / "artifacts" / "tiny_rung"
    default_report_path = (
        repo_root / "research" / "questions" / "self-prediction-compute-compression" / "README.md"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", type=Path, default=default_text_file)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--report-path", type=Path, default=default_report_path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--train-steps", type=int, default=RunConfig.train_steps)
    parser.add_argument("--overfit-steps", type=int, default=RunConfig.overfit_steps)
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument(
        "--self-prediction-weight",
        type=float,
        default=RunConfig.self_prediction_weight,
    )
    parser.add_argument(
        "--variant",
        choices=["baseline", "self_prediction", "both"],
        default="both",
    )
    args = parser.parse_args()

    config = RunConfig(
        train_steps=args.train_steps,
        overfit_steps=args.overfit_steps,
        seed=args.seed,
        self_prediction_weight=args.self_prediction_weight,
    )
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    write_json(
        args.output_dir / "environment.json",
        {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": current_git_status_short() == [],
            "git_status_short": current_git_status_short(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_is_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
    )
    write_json(args.output_dir / "config.json", asdict(config))

    corpus = build_large_corpus(config, text_file=args.text_file)
    write_json(args.output_dir / "corpus_summary.json", corpus.corpus_summary)

    variant_specs = {
        "baseline": VariantSpec(name="baseline", self_prediction_weight=0.0),
        "self_prediction": VariantSpec(
            name="self_prediction",
            self_prediction_weight=config.self_prediction_weight,
        ),
    }
    if args.variant == "both":
        selected_specs = [variant_specs["baseline"], variant_specs["self_prediction"]]
    else:
        selected_specs = [variant_specs[args.variant]]

    variants = {
        spec.name: train_variant(
            variant=spec,
            config=config,
            corpus=corpus,
            device=device,
            output_root=args.output_dir,
        )
        for spec in selected_specs
    }
    comparison = {
        "config": asdict(config),
        "variants": variants,
    }
    if "baseline" in variants and "self_prediction" in variants:
        comparison["frontier_dominance"] = frontier_dominance_summary(
            variants["baseline"]["threshold_frontier"],
            variants["self_prediction"]["threshold_frontier"],
        )
        write_report(path=args.report_path, comparison=comparison, experiment_dir=args.output_dir.relative_to(repo_root))
    write_json(args.output_dir / "comparison_summary.json", comparison)


if __name__ == "__main__":
    main()
