from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import FixedWindowCharDataset, resolve_device, set_seed
from core.tiny_char_transformer import count_parameters
from experiments.pytorch_char_shakespeare_comparison import (
    SHAKESPEARE_SNIPPET,
    build_train_val_split,
    select_prompts,
)


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 10
    train_fraction: float = 0.8
    embedding_dim: int = 32
    hidden_dim: int = 128
    max_depth: int = 8
    loss_pred_weight: float = 0.1
    overfit_batch_size: int = 128
    overfit_steps: int = 1500
    overfit_learning_rate: float = 0.01
    train_batch_size: int = 128
    train_steps: int = 2000
    train_learning_rate: float = 0.005
    sample_length: int = 120
    seed: int = 7
    depth_eval_batch_size: int = 512
    val_analysis_chars: int = 240
    dynamic_improvement_epsilon: float = 0.01


def current_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def current_git_status_short() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def format_float_token(value: float) -> str:
    return format(value, "g").replace("-", "neg").replace(".", "p")


class DynamicDepthCharModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        max_depth: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru_cell = nn.GRUCell(embedding_dim, hidden_dim)
        self.task_head = nn.Linear(hidden_dim, vocab_size)
        self.loss_head = nn.Linear(hidden_dim, 1)
        self.max_depth = max_depth

    def forward_depths(self, tokens: Tensor, *, max_depth: int | None = None) -> tuple[Tensor, Tensor]:
        depth_limit = max_depth or self.max_depth
        embedded = self.embedding(tokens)
        batch_size, context_size, embedding_dim = embedded.shape
        hidden = embedded.new_zeros(batch_size, self.gru_cell.hidden_size)
        logits_by_depth: list[Tensor] = []
        predicted_loss_by_depth: list[Tensor] = []

        for _position in range(context_size):
            token_embedding = embedded[:, _position, :]
            for _depth_index in range(depth_limit):
                hidden = self.gru_cell(token_embedding, hidden)
                logits_by_depth.append(self.task_head(hidden))
                predicted_loss_by_depth.append(self.loss_head(hidden).squeeze(1))

        logits = torch.stack(logits_by_depth, dim=1).view(batch_size, context_size, depth_limit, self.vocab_size)
        predicted_losses = torch.stack(predicted_loss_by_depth, dim=1).view(batch_size, context_size, depth_limit)
        return logits, predicted_losses

    def forward(self, tokens: Tensor) -> Tensor:
        logits, _predicted_losses = self.forward_depths(tokens, max_depth=1)
        return logits[:, -1, 0, :]


@dataclass(frozen=True)
class LossBundle:
    total_loss: Tensor
    task_loss: Tensor
    loss_prediction_loss: Tensor
    accuracy: float
    per_depth_task_loss: list[float]
    per_depth_accuracy: list[float]


def compute_loss_bundle(
    model: DynamicDepthCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    active_depth: int,
    loss_pred_weight: float,
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
        depth_accuracy = (depth_logits.argmax(dim=1) == targets).float().mean().item()
        per_depth_accuracies.append(depth_accuracy)

    actual_losses = torch.stack(per_example_losses, dim=1)
    predicted_final_losses = predicted_losses[:, -1, :]
    task_loss = torch.stack(per_depth_losses).mean()
    loss_prediction_loss = F.mse_loss(predicted_final_losses, actual_losses.detach())
    total_loss = task_loss + loss_pred_weight * loss_prediction_loss
    final_accuracy = per_depth_accuracies[-1]
    return LossBundle(
        total_loss=total_loss,
        task_loss=task_loss,
        loss_prediction_loss=loss_prediction_loss,
        accuracy=final_accuracy,
        per_depth_task_loss=[loss.item() for loss in per_depth_losses],
        per_depth_accuracy=per_depth_accuracies,
    )


def train_dynamic_depth_model(
    model: DynamicDepthCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
    steps: int,
    learning_rate: float,
    active_depth: int,
    loss_pred_weight: float,
) -> tuple[list[dict[str, object]], LossBundle]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    sample_count = inputs.shape[0]
    trace: list[dict[str, object]] = []

    for step in range(steps + 1):
        if step % 50 == 0 or step == steps:
            eval_bundle = compute_loss_bundle(
                model,
                inputs,
                targets,
                active_depth=active_depth,
                loss_pred_weight=loss_pred_weight,
            )
            trace.append(
                {
                    "step": step,
                    "total_loss": round(eval_bundle.total_loss.item(), 6),
                    "task_loss": round(eval_bundle.task_loss.item(), 6),
                    "loss_prediction_loss": round(eval_bundle.loss_prediction_loss.item(), 6),
                    "accuracy": round(eval_bundle.accuracy, 6),
                    "per_depth_task_loss": [round(value, 6) for value in eval_bundle.per_depth_task_loss],
                    "per_depth_accuracy": [round(value, 6) for value in eval_bundle.per_depth_accuracy],
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

    final_bundle = compute_loss_bundle(
        model,
        inputs,
        targets,
        active_depth=active_depth,
        loss_pred_weight=loss_pred_weight,
    )
    return trace, final_bundle


def generate_text_with_depth(
    model: DynamicDepthCharModel,
    dataset: FixedWindowCharDataset,
    prompt: str,
    *,
    length: int,
    device: torch.device,
    halting_threshold: float | None,
    improvement_epsilon: float | None,
    max_depth: int,
) -> str:
    if len(prompt) != dataset.context_size:
        raise ValueError(f"Prompt must be exactly {dataset.context_size} characters long.")

    window = dataset.encode(prompt)
    generated = prompt
    for _ in range(length):
        tokens = torch.tensor([window], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, predicted_losses = model.forward_depths(tokens, max_depth=max_depth)
            final_logits = logits[:, -1, :, :]
            final_predictions = predicted_losses[:, -1, :]
            chosen_depth = select_dynamic_depth(
                final_predictions[0],
                threshold=halting_threshold,
                improvement_epsilon=improvement_epsilon,
            )
            next_token = final_logits[:, chosen_depth - 1, :].argmax(dim=1).item()
        generated += dataset.decode([next_token])
        window = window[1:] + [next_token]
    return generated


def evaluate_model(
    model: DynamicDepthCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    active_depth: int,
    loss_pred_weight: float,
    inference_mode: str,
    halting_threshold: float | None = None,
    improvement_epsilon: float | None = None,
    batch_size: int = 512,
) -> dict[str, object]:
    if inference_mode == "fixed":
        bundle = compute_loss_bundle(
            model,
            inputs,
            targets,
            active_depth=active_depth,
            loss_pred_weight=loss_pred_weight,
        )
        return {
            "total_loss": bundle.total_loss.item(),
            "task_loss": bundle.task_loss.item(),
            "loss_prediction_loss": bundle.loss_prediction_loss.item(),
            "accuracy": bundle.accuracy,
            "per_depth_task_loss": bundle.per_depth_task_loss,
            "per_depth_accuracy": bundle.per_depth_accuracy,
            "mean_used_depth": float(active_depth),
            "depth_histogram": {str(active_depth): int(inputs.shape[0])},
        }

    total_losses: list[Tensor] = []
    total_loss_prediction_losses: list[Tensor] = []
    used_depths: list[Tensor] = []
    correct_predictions = 0
    total_predictions = 0
    max_depth = active_depth
    per_depth_loss_sums = torch.zeros(max_depth, device=inputs.device)
    per_depth_correct_sums = torch.zeros(max_depth, device=inputs.device)
    observed_examples = 0

    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            stop = min(start + batch_size, inputs.shape[0])
            batch_inputs = inputs[start:stop]
            batch_targets = targets[start:stop]
            logits, predicted_losses = model.forward_depths(batch_inputs, max_depth=max_depth)
            final_logits = logits[:, -1, :, :]
            final_predicted_losses = predicted_losses[:, -1, :]
            flat_logits = final_logits.reshape(-1, model.vocab_size)
            repeated_targets = batch_targets.unsqueeze(1).expand(-1, max_depth).reshape(-1)
            actual_losses = F.cross_entropy(flat_logits, repeated_targets, reduction="none").view(-1, max_depth)
            predicted_loss_mse = (final_predicted_losses - actual_losses.detach()).square().mean(dim=1)
            predictions = final_logits.argmax(dim=2)
            correctness = predictions.eq(batch_targets.unsqueeze(1))

            per_depth_loss_sums += actual_losses.sum(dim=0)
            per_depth_correct_sums += correctness.float().sum(dim=0)
            observed_examples += batch_inputs.shape[0]

            batch_used_depths = []
            batch_selected_losses = []
            batch_selected_correct = []
            for row_index in range(batch_inputs.shape[0]):
                depth = select_dynamic_depth(
                    final_predicted_losses[row_index],
                    threshold=halting_threshold,
                    improvement_epsilon=improvement_epsilon,
                )
                batch_used_depths.append(depth)
                batch_selected_losses.append(actual_losses[row_index, depth - 1])
                batch_selected_correct.append(correctness[row_index, depth - 1])

            used_depth_tensor = torch.tensor(batch_used_depths, device=inputs.device)
            used_depths.append(used_depth_tensor)
            total_losses.append(torch.stack(batch_selected_losses))
            total_loss_prediction_losses.append(predicted_loss_mse)
            correct_predictions += torch.stack(batch_selected_correct).float().sum().item()
            total_predictions += batch_inputs.shape[0]

    used_depth_vector = torch.cat(used_depths)
    selected_loss_vector = torch.cat(total_losses)
    selected_loss_prediction_vector = torch.cat(total_loss_prediction_losses)
    depth_histogram = {
        str(depth): int((used_depth_vector == depth).sum().item())
        for depth in range(1, max_depth + 1)
        if int((used_depth_vector == depth).sum().item()) > 0
    }
    return {
        "total_loss": selected_loss_vector.mean().item(),
        "task_loss": selected_loss_vector.mean().item(),
        "loss_prediction_loss": selected_loss_prediction_vector.mean().item(),
        "accuracy": correct_predictions / total_predictions,
        "per_depth_task_loss": (per_depth_loss_sums / observed_examples).tolist(),
        "per_depth_accuracy": (per_depth_correct_sums / observed_examples).tolist(),
        "mean_used_depth": used_depth_vector.float().mean().item(),
        "depth_histogram": depth_histogram,
    }


def select_dynamic_depth(
    predicted_losses: Tensor,
    *,
    threshold: float | None,
    improvement_epsilon: float | None,
) -> int:
    depth_count = int(predicted_losses.shape[0])
    for depth_index in range(depth_count):
        current_loss = predicted_losses[depth_index].item()
        if threshold is not None and current_loss <= threshold:
            return depth_index + 1
        if improvement_epsilon is not None and depth_index + 1 < depth_count:
            next_loss = predicted_losses[depth_index + 1].item()
            improvement = current_loss - next_loss
            if improvement <= improvement_epsilon:
                return depth_index + 1
    return depth_count


def save_predictions(
    path: Path,
    *,
    dataset: FixedWindowCharDataset,
    inputs: Tensor,
    targets: Tensor,
    predictions: Tensor,
) -> None:
    rows = ["context | target | prediction"]
    for input_tokens, target_token, predicted_token in zip(
        inputs.tolist(), targets.tolist(), predictions.tolist(), strict=True
    ):
        rows.append(
            " | ".join(
                [
                    dataset.decode(input_tokens).replace("\n", "\\n"),
                    dataset.decode([target_token]).replace("\n", "\\n"),
                    dataset.decode([predicted_token]).replace("\n", "\\n"),
                ]
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def save_generation_samples(
    path: Path,
    *,
    model: DynamicDepthCharModel,
    dataset: FixedWindowCharDataset,
    prompts: list[str],
    sample_length: int,
    device: torch.device,
    halting_threshold: float | None,
    improvement_epsilon: float | None,
    max_depth: int,
) -> dict[str, str]:
    with torch.no_grad():
        samples = {
            prompt.replace("\n", "\\n"): generate_text_with_depth(
                model,
                dataset,
                prompt,
                length=sample_length,
                device=device,
                halting_threshold=halting_threshold,
                improvement_epsilon=improvement_epsilon,
                max_depth=max_depth,
            )
            for prompt in prompts
        }
    write_json(path, samples)
    return samples


def analyze_dynamic_depth(
    *,
    model: DynamicDepthCharModel,
    dataset: FixedWindowCharDataset,
    val_text: str,
    val_inputs: Tensor,
    val_targets: Tensor,
    output_dir: Path,
    analysis_chars: int,
    halting_threshold: float,
    improvement_epsilon: float,
) -> dict[str, object]:
    count = min(analysis_chars, val_inputs.shape[0])
    text_chars = []
    annotated = []
    rows = []
    depth_histogram = {depth: 0 for depth in range(1, model.max_depth + 1)}
    hard_examples = []

    with torch.no_grad():
        logits, predicted_losses = model.forward_depths(val_inputs[:count], max_depth=model.max_depth)
        final_logits = logits[:, -1, :, :]
        final_predicted_losses = predicted_losses[:, -1, :]
        flat_logits = final_logits.reshape(-1, model.vocab_size)
        repeated_targets = val_targets[:count].unsqueeze(1).expand(-1, model.max_depth).reshape(-1)
        actual_losses = F.cross_entropy(flat_logits, repeated_targets, reduction="none").view(count, model.max_depth)
        predictions = final_logits.argmax(dim=2)

        for index in range(count):
            used_depth = select_dynamic_depth(
                final_predicted_losses[index],
                threshold=halting_threshold,
                improvement_epsilon=improvement_epsilon,
            )
            depth_histogram[used_depth] += 1
            target_char = dataset.decode([val_targets[index].item()])
            predicted_char = dataset.decode([predictions[index, used_depth - 1].item()])
            context = dataset.decode(val_inputs[index].tolist()).replace("\n", "\\n")
            display_char = val_text[index + dataset.context_size].replace("\n", "⏎")
            text_chars.append(display_char)
            annotated.append(f"{display_char}{used_depth}")
            row = {
                "index": index,
                "context": context,
                "target": target_char.replace("\n", "\\n"),
                "prediction": predicted_char.replace("\n", "\\n"),
                "used_depth": used_depth,
                "predicted_losses": [round(value, 6) for value in final_predicted_losses[index].tolist()],
                "actual_losses": [round(value, 6) for value in actual_losses[index].tolist()],
                "correct_at_used_depth": bool(predictions[index, used_depth - 1].item() == val_targets[index].item()),
            }
            rows.append(row)
            hard_examples.append(
                {
                    "char": display_char,
                    "used_depth": used_depth,
                    "actual_loss_at_used_depth": actual_losses[index, used_depth - 1].item(),
                    "row": row,
                }
            )

    hard_examples.sort(
        key=lambda item: (item["used_depth"], item["actual_loss_at_used_depth"]),
        reverse=True,
    )
    top_hard = [
        {
            "char": item["char"],
            "used_depth": item["used_depth"],
            "actual_loss_at_used_depth": round(item["actual_loss_at_used_depth"], 6),
            "context": item["row"]["context"],
            "target": item["row"]["target"],
            "prediction": item["row"]["prediction"],
        }
        for item in hard_examples[:20]
    ]
    annotated_text = "".join(annotated)
    (output_dir / "depth_annotation.txt").write_text(annotated_text + "\n", encoding="utf-8")
    write_json(
        output_dir / "depth_analysis.json",
        {
            "threshold": halting_threshold,
            "improvement_epsilon": improvement_epsilon,
            "depth_histogram": {str(key): value for key, value in depth_histogram.items()},
            "top_hard_examples": top_hard,
            "rows": rows,
        },
    )
    return {
        "annotated_text": annotated_text,
        "depth_histogram": {str(key): value for key, value in depth_histogram.items()},
        "top_hard_examples": top_hard,
    }


def write_readme(
    path: Path,
    *,
    summary: dict[str, object],
    metrics_path: str,
    analysis_path: str,
    annotation_path: str,
) -> None:
    fixed_1 = summary["runs"]["fixed_depth_1"]
    fixed_8 = summary["runs"]["fixed_depth_8"]
    dynamic_run = summary["runs"]["dynamic_depth"]
    readme = f"""# Dynamic depth on Shakespeare

## Question

Does a weight-shared recurrent model trained with losses at every depth learn a useful loss-prediction signal, and at inference does that signal allocate different amounts of compute to different characters?

## Setup

- Corpus and split follow `experiments/pytorch_char_shakespeare_comparison.py`
- Model: shared `GRUCell`, task head, loss-prediction head
- Training depth: 8
- Comparators: fixed depth 1, fixed depth 8, dynamic depth halting at inference

## Main result

- Fixed depth 1 val loss: {fixed_1['val_metrics']['task_loss']:.4f}
- Fixed depth 8 val loss: {fixed_8['val_metrics']['task_loss']:.4f}
- Dynamic val loss: {dynamic_run['val_metrics']['task_loss']:.4f}
- Dynamic mean used depth: {dynamic_run['val_metrics']['mean_used_depth']:.3f}

## Depth allocation

Dynamic inference used this histogram: `{dynamic_run['val_metrics']['depth_histogram']}`

Hardest examples saved in `{analysis_path}`. Annotated text saved in `{annotation_path}`.

## Artifacts

- Metrics summary: `{metrics_path}`
- Dynamic depth analysis: `{analysis_path}`
- Annotated validation text: `{annotation_path}`
"""
    path.write_text(readme, encoding="utf-8")


def run_variant(
    *,
    model_name: str,
    output_dir: Path,
    dataset: FixedWindowCharDataset,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    prompts: list[str],
    config: RunConfig,
    device: torch.device,
    training_depth: int,
    inference_mode: str,
    halting_threshold: float | None,
    improvement_epsilon: float | None,
) -> tuple[dict[str, object], DynamicDepthCharModel]:
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    def build_model() -> DynamicDepthCharModel:
        return DynamicDepthCharModel(
            vocab_size=dataset.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            max_depth=config.max_depth,
        )

    overfit_model = build_model().to(device)
    write_json(
        model_dir / "model_summary.json",
        {
            "parameter_count": count_parameters(overfit_model),
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "max_depth": config.max_depth,
            "training_depth": training_depth,
            "inference_mode": inference_mode,
        },
    )

    overfit_inputs = train_inputs[: config.overfit_batch_size]
    overfit_targets = train_targets[: config.overfit_batch_size]
    overfit_trace, overfit_bundle = train_dynamic_depth_model(
        overfit_model,
        overfit_inputs,
        overfit_targets,
        batch_size=config.overfit_batch_size,
        steps=config.overfit_steps,
        learning_rate=config.overfit_learning_rate,
        active_depth=training_depth,
        loss_pred_weight=config.loss_pred_weight,
    )
    overfit_logits, _overfit_predicted_losses = overfit_model.forward_depths(overfit_inputs, max_depth=training_depth)
    overfit_predictions = overfit_logits[:, -1, training_depth - 1, :].argmax(dim=1)
    overfit_reached = overfit_bundle.accuracy == 1.0 and overfit_bundle.task_loss.item() < 1e-2
    write_json(
        model_dir / "overfit_metrics.json",
        {
            "final_total_loss": overfit_bundle.total_loss.item(),
            "final_task_loss": overfit_bundle.task_loss.item(),
            "final_loss_prediction_loss": overfit_bundle.loss_prediction_loss.item(),
            "final_accuracy": overfit_bundle.accuracy,
            "reached_memorization_bar": overfit_reached,
            "trace": overfit_trace,
        },
    )
    save_predictions(
        model_dir / "overfit_predictions.txt",
        dataset=dataset,
        inputs=overfit_inputs.cpu(),
        targets=overfit_targets.cpu(),
        predictions=overfit_predictions.cpu(),
    )
    if not overfit_reached:
        raise RuntimeError(f"{model_name} failed the one-batch overfit check.")

    train_model = build_model().to(device)
    started_at = time.perf_counter()
    train_trace, _train_bundle = train_dynamic_depth_model(
        train_model,
        train_inputs,
        train_targets,
        batch_size=config.train_batch_size,
        steps=config.train_steps,
        learning_rate=config.train_learning_rate,
        active_depth=training_depth,
        loss_pred_weight=config.loss_pred_weight,
    )
    runtime_seconds = time.perf_counter() - started_at

    train_metrics = evaluate_model(
        train_model,
        train_inputs,
        train_targets,
        active_depth=training_depth,
        loss_pred_weight=config.loss_pred_weight,
        inference_mode=inference_mode,
        halting_threshold=halting_threshold,
        improvement_epsilon=improvement_epsilon,
        batch_size=config.depth_eval_batch_size,
    )
    val_metrics = evaluate_model(
        train_model,
        val_inputs,
        val_targets,
        active_depth=training_depth,
        loss_pred_weight=config.loss_pred_weight,
        inference_mode=inference_mode,
        halting_threshold=halting_threshold,
        improvement_epsilon=improvement_epsilon,
        batch_size=config.depth_eval_batch_size,
    )
    with torch.no_grad():
        val_logits, val_predicted_losses = train_model.forward_depths(
            val_inputs[:32],
            max_depth=training_depth,
        )
        final_predicted = val_predicted_losses[:, -1, :]
        if inference_mode == "dynamic":
            chosen_depths = [
                select_dynamic_depth(
                    final_predicted[index],
                    threshold=halting_threshold,
                    improvement_epsilon=improvement_epsilon,
                )
                for index in range(val_inputs[:32].shape[0])
            ]
            val_predictions = torch.stack(
                [
                    val_logits[index, -1, depth - 1, :].argmax(dim=0)
                    for index, depth in enumerate(chosen_depths)
                ]
            )
        else:
            val_predictions = val_logits[:, -1, training_depth - 1, :].argmax(dim=1)
    save_predictions(
        model_dir / "validation_predictions.txt",
        dataset=dataset,
        inputs=val_inputs[:32].cpu(),
        targets=val_targets[:32].cpu(),
        predictions=val_predictions.cpu(),
    )
    samples = save_generation_samples(
        model_dir / "samples.json",
        model=train_model,
        dataset=dataset,
        prompts=prompts,
        sample_length=config.sample_length,
        device=device,
        halting_threshold=halting_threshold if inference_mode == "dynamic" else None,
        improvement_epsilon=improvement_epsilon if inference_mode == "dynamic" else None,
        max_depth=training_depth,
    )
    write_json(
        model_dir / "train_val_metrics.json",
        {
            "runtime_seconds": runtime_seconds,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_trace": train_trace,
            "halting_threshold": halting_threshold,
            "improvement_epsilon": improvement_epsilon,
        },
    )
    return {
        "model_name": model_name,
        "runtime_seconds": runtime_seconds,
        "overfit_reached": overfit_reached,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "samples": samples,
        "halting_threshold": halting_threshold,
        "improvement_epsilon": improvement_epsilon,
    }, train_model


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_question_dir = repo_root / "research" / "questions" / "dynamic-depth"
    default_output_dir = default_question_dir / "artifacts"

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--overfit-steps", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = RunConfig(
        train_steps=args.train_steps or RunConfig.train_steps,
        overfit_steps=args.overfit_steps or RunConfig.overfit_steps,
        seed=args.seed or RunConfig.seed,
    )
    set_seed(config.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    default_question_dir.mkdir(parents=True, exist_ok=True)

    corpus_text = SHAKESPEARE_SNIPPET
    dataset = FixedWindowCharDataset(corpus_text, context_size=config.context_size)
    split = build_train_val_split(
        corpus_text,
        context_size=config.context_size,
        stoi=dataset.stoi,
        train_fraction=config.train_fraction,
    )
    prompts = select_prompts(split.train_text, split.val_text, context_size=config.context_size)
    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)

    git_status_short = current_git_status_short()
    write_json(output_dir / "config.json", asdict(config))
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
    write_json(
        output_dir / "corpus_summary.json",
        {
            "corpus_name": "hardcoded_shakespeare_excerpt",
            "corpus_sha256": hashlib.sha256(corpus_text.encode("utf-8")).hexdigest(),
            "total_characters": len(corpus_text),
            "vocab_size": dataset.vocab_size,
            "split_index": split.split_index,
            "train_characters": len(split.train_text),
            "val_characters": len(split.val_text),
            "train_windows": int(split.train_inputs.shape[0]),
            "val_windows": int(split.val_inputs.shape[0]),
            "prompts": [prompt.replace("\n", "\\n") for prompt in prompts],
        },
    )

    fixed_depth_1, _fixed_depth_1_model = run_variant(
        model_name="fixed_depth_1",
        output_dir=output_dir,
        dataset=dataset,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        prompts=prompts,
        config=config,
        device=device,
        training_depth=1,
        inference_mode="fixed",
        halting_threshold=None,
        improvement_epsilon=None,
    )
    fixed_depth_8, _fixed_depth_8_model = run_variant(
        model_name="fixed_depth_8",
        output_dir=output_dir,
        dataset=dataset,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        prompts=prompts,
        config=config,
        device=device,
        training_depth=config.max_depth,
        inference_mode="fixed",
        halting_threshold=None,
        improvement_epsilon=None,
    )

    halting_threshold = fixed_depth_8["train_metrics"]["task_loss"]
    dynamic_depth, dynamic_model = run_variant(
        model_name="dynamic_depth",
        output_dir=output_dir,
        dataset=dataset,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        prompts=prompts,
        config=config,
        device=device,
        training_depth=config.max_depth,
        inference_mode="dynamic",
        halting_threshold=halting_threshold,
        improvement_epsilon=config.dynamic_improvement_epsilon,
    )

    analysis = analyze_dynamic_depth(
        model=dynamic_model,
        dataset=dataset,
        val_text=split.val_text,
        val_inputs=val_inputs,
        val_targets=val_targets,
        output_dir=output_dir,
        analysis_chars=config.val_analysis_chars,
        halting_threshold=halting_threshold,
        improvement_epsilon=config.dynamic_improvement_epsilon,
    )

    summary_model = DynamicDepthCharModel(
        vocab_size=dataset.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        max_depth=config.max_depth,
    ).to(device)
    summary = {
        "parameter_count": count_parameters(summary_model),
        "runs": {
            "fixed_depth_1": fixed_depth_1,
            "fixed_depth_8": fixed_depth_8,
            "dynamic_depth": dynamic_depth,
        },
        "dynamic_depth_analysis": analysis,
        "halting_threshold": halting_threshold,
        "improvement_epsilon": config.dynamic_improvement_epsilon,
    }
    write_json(output_dir / "comparison_summary.json", summary)
    write_readme(
        default_question_dir / "README.md",
        summary=summary,
        metrics_path="artifacts/comparison_summary.json",
        analysis_path="artifacts/depth_analysis.json",
        annotation_path="artifacts/depth_annotation.txt",
    )


if __name__ == "__main__":
    main()
