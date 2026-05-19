from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import (
    FixedWindowCharDataset,
    generate_text,
    render_predictions,
    resolve_device,
    set_seed,
)
from core.tiny_char_transformer import count_parameters
from experiments.pytorch_char_predictive_chain import (
    align_embedding_target_to_message_dim,
    full_tokens,
    node_labels,
)
from experiments.pytorch_char_shakespeare_comparison import (
    SHAKESPEARE_SNIPPET,
    build_train_val_split,
    current_git_sha,
    current_git_status_short,
    select_prompts,
    write_json,
)


AggregationMode = Literal["single_parent", "uniform_mean", "attention"]

SKIP2_PREDECESSORS: list[list[int]] = [
    [],
    [0],
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
]
HISTORICAL_CHAIN_VAL_LOSS = 2.593513250350952
PARAMETER_MIN = 33_000
PARAMETER_MAX = 37_000
NONTRIVIAL_ATTENTION_DELTA = 0.05


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 5
    train_fraction: float = 0.8
    embedding_dim: int = 24
    message_dim: int = 24
    hidden_dim: int = 19
    attention_dim: int = 24
    auxiliary_weight: float = 1.0
    detach_messages: bool = True
    overfit_batch_size: int = 32
    overfit_steps: int = 3000
    overfit_learning_rate: float = 0.02
    train_batch_size: int = 128
    train_steps: int = 1200
    train_learning_rate: float = 0.01
    gradient_clip_norm: float = 1.0
    sample_length: int = 120
    rollout_examples: int = 3
    seed: int = 7


@dataclass(frozen=True)
class LossBreakdown:
    total_loss: Tensor
    task_loss: Tensor
    auxiliary_total: Tensor
    auxiliary_by_node: dict[str, Tensor]
    logits: Tensor
    accuracy: float


@dataclass(frozen=True)
class ModelRollout:
    embeddings: Tensor
    hidden_states: list[Tensor]
    messages: list[Tensor]
    incoming_inputs: list[Tensor]
    attention_weights: list[Tensor | None]


class PredictiveGraphCharModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        predecessors: list[list[int]],
        aggregation_mode: AggregationMode,
        embedding_dim: int,
        hidden_dim: int,
        message_dim: int,
        attention_dim: int,
        detach_messages: bool,
    ) -> None:
        super().__init__()
        if not predecessors:
            raise ValueError("predecessors must not be empty")
        if predecessors[0] != []:
            raise ValueError("node 0 must not have predecessors")
        if aggregation_mode == "single_parent" and any(
            len(node_predecessors) > 1 for node_predecessors in predecessors
        ):
            raise ValueError("single_parent mode requires at most one predecessor per node")

        self.predecessors = [list(node_predecessors) for node_predecessors in predecessors]
        self.num_nodes = len(predecessors)
        self.labels = node_labels(self.num_nodes)
        self.aggregation_mode = aggregation_mode
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.attention_dim = attention_dim
        self.detach_messages = detach_messages
        self.multi_input_nodes = [
            node_index
            for node_index, node_predecessors in enumerate(self.predecessors)
            if len(node_predecessors) > 1
        ]

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.nodes = nn.ModuleList(
            [
                nn.GRUCell(
                    embedding_dim if node_index == 0 else message_dim,
                    hidden_dim,
                )
                for node_index in range(self.num_nodes)
            ]
        )
        self.message_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, message_dim) for _ in range(self.num_nodes)]
        )
        self.attention_queries = nn.ModuleDict(
            {
                str(node_index): nn.Linear(hidden_dim, attention_dim)
                for node_index in self.multi_input_nodes
            }
        )
        self.attention_keys = nn.ModuleDict(
            {
                str(node_index): nn.Linear(message_dim, attention_dim)
                for node_index in self.multi_input_nodes
            }
        )
        self.task_head = nn.Sequential(
            nn.Linear(hidden_dim * self.num_nodes, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def _bounded_embedding(self, tokens: Tensor) -> Tensor:
        return torch.tanh(self.embedding(tokens))

    def _predictive_message(self, projection: nn.Linear, hidden: Tensor) -> Tensor:
        return torch.tanh(projection(hidden))

    def _aggregate_predecessors(
        self,
        *,
        node_index: int,
        previous_hidden: Tensor,
        previous_messages: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        predecessor_indices = self.predecessors[node_index]
        if len(predecessor_indices) == 1:
            aggregate = previous_messages[predecessor_indices[0]]
            weights = torch.ones(
                aggregate.shape[0],
                1,
                device=aggregate.device,
                dtype=aggregate.dtype,
            )
            return aggregate, weights

        predecessor_messages = torch.stack(
            [previous_messages[predecessor_index] for predecessor_index in predecessor_indices],
            dim=1,
        )
        if self.aggregation_mode == "uniform_mean":
            weights = torch.full(
                (predecessor_messages.shape[0], predecessor_messages.shape[1]),
                1.0 / predecessor_messages.shape[1],
                device=predecessor_messages.device,
                dtype=predecessor_messages.dtype,
            )
            return predecessor_messages.mean(dim=1), weights

        query = self.attention_queries[str(node_index)](previous_hidden)
        keys = self.attention_keys[str(node_index)](predecessor_messages)
        scores = torch.einsum("bd,bpd->bp", query, keys) / math.sqrt(self.attention_dim)
        weights = torch.softmax(scores, dim=1)
        aggregate = torch.sum(predecessor_messages * weights.unsqueeze(-1), dim=1)
        return aggregate, weights

    def rollout(self, tokens: Tensor) -> ModelRollout:
        batch_size, sequence_length = tokens.shape
        embeddings = self._bounded_embedding(tokens)
        hidden_states = [
            torch.zeros(batch_size, self.hidden_dim, device=tokens.device)
            for _ in range(self.num_nodes)
        ]
        previous_messages = [
            torch.zeros(batch_size, self.message_dim, device=tokens.device)
            for _ in range(self.num_nodes)
        ]

        hidden_steps = [[] for _ in range(self.num_nodes)]
        message_steps = [[] for _ in range(self.num_nodes)]
        incoming_steps = [[] for _ in range(self.num_nodes)]
        attention_steps = [[] for _ in range(self.num_nodes)]

        for step in range(sequence_length):
            current_messages: list[Tensor] = []
            for node_index, (node, message_head) in enumerate(
                zip(self.nodes, self.message_heads, strict=True)
            ):
                if node_index == 0:
                    node_input = embeddings[:, step, :]
                    node_attention = None
                else:
                    node_input, node_attention = self._aggregate_predecessors(
                        node_index=node_index,
                        previous_hidden=hidden_states[node_index],
                        previous_messages=previous_messages,
                    )
                hidden_states[node_index] = node(node_input, hidden_states[node_index])
                message = self._predictive_message(message_head, hidden_states[node_index])
                hidden_steps[node_index].append(hidden_states[node_index])
                message_steps[node_index].append(message)
                incoming_steps[node_index].append(node_input)
                attention_steps[node_index].append(node_attention)
                current_messages.append(message)

            previous_messages = [
                message.detach() if self.detach_messages else message
                for message in current_messages
            ]

        stacked_attention_weights: list[Tensor | None] = []
        for node_index, node_attention_steps in enumerate(attention_steps):
            if node_index == 0:
                stacked_attention_weights.append(None)
                continue
            stacked_attention_weights.append(torch.stack(node_attention_steps, dim=1))

        return ModelRollout(
            embeddings=embeddings,
            hidden_states=[torch.stack(steps, dim=1) for steps in hidden_steps],
            messages=[torch.stack(steps, dim=1) for steps in message_steps],
            incoming_inputs=[torch.stack(steps, dim=1) for steps in incoming_steps],
            attention_weights=stacked_attention_weights,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        rollout = self.rollout(tokens)
        task_step = tokens.shape[1] - 1
        final_state = torch.cat(
            [hidden_state[:, task_step, :] for hidden_state in rollout.hidden_states],
            dim=-1,
        )
        return self.task_head(final_state)


def compute_losses(
    model: PredictiveGraphCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    auxiliary_weight: float,
) -> LossBreakdown:
    sequence = full_tokens(inputs, targets)
    rollout = model.rollout(sequence)
    task_step = inputs.shape[1] - 1
    task_state = torch.cat(
        [hidden_state[:, task_step, :] for hidden_state in rollout.hidden_states],
        dim=-1,
    )
    logits = model.task_head(task_state)

    auxiliary_by_node: dict[str, Tensor] = {}
    for node_index, label in enumerate(model.labels):
        target_inputs = rollout.incoming_inputs[node_index][:, 1:, :].detach()
        if target_inputs.shape[-1] != model.message_dim:
            target_inputs = align_embedding_target_to_message_dim(
                target_inputs,
                message_dim=model.message_dim,
            )
        auxiliary_by_node[label] = F.mse_loss(
            rollout.messages[node_index][:, :-1, :],
            target_inputs,
        )

    auxiliary_total = torch.stack(tuple(auxiliary_by_node.values())).sum()
    task_loss = F.cross_entropy(logits, targets)
    total_loss = task_loss + auxiliary_weight * auxiliary_total
    accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    return LossBreakdown(
        total_loss=total_loss,
        task_loss=task_loss,
        auxiliary_total=auxiliary_total,
        auxiliary_by_node=auxiliary_by_node,
        logits=logits,
        accuracy=accuracy,
    )


def rounded_metric_row(step: int, losses: LossBreakdown) -> dict[str, float | int]:
    row: dict[str, float | int | dict[str, float]] = {
        "step": step,
        "total_loss": round(losses.total_loss.item(), 6),
        "task_loss": round(losses.task_loss.item(), 6),
        "auxiliary_total": round(losses.auxiliary_total.item(), 6),
        "accuracy": round(losses.accuracy, 6),
    }
    auxiliary_by_node = {
        label: round(value.item(), 6)
        for label, value in losses.auxiliary_by_node.items()
    }
    row["auxiliary_by_node"] = auxiliary_by_node
    for label, value in auxiliary_by_node.items():
        row[f"auxiliary_{label.lower()}"] = value
    return row


def ensure_finite(losses: LossBreakdown, *, step: int, model_name: str) -> None:
    if not torch.isfinite(losses.total_loss):
        raise RuntimeError(f"{model_name} produced non-finite total loss at step {step}")
    if not torch.isfinite(losses.task_loss):
        raise RuntimeError(f"{model_name} produced non-finite task loss at step {step}")
    if not torch.isfinite(losses.auxiliary_total):
        raise RuntimeError(f"{model_name} produced non-finite auxiliary loss at step {step}")


def train_on_fixed_batch(
    model: PredictiveGraphCharModel,
    batch_inputs: Tensor,
    batch_targets: Tensor,
    *,
    steps: int,
    learning_rate: float,
    auxiliary_weight: float,
    gradient_clip_norm: float,
    model_name: str,
) -> tuple[list[dict[str, float | int]], LossBreakdown]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trace: list[dict[str, float | int]] = []

    for step in range(steps + 1):
        losses = compute_losses(
            model,
            batch_inputs,
            batch_targets,
            auxiliary_weight=auxiliary_weight,
        )
        ensure_finite(losses, step=step, model_name=model_name)
        if step % 50 == 0 or step == steps:
            trace.append(rounded_metric_row(step, losses))

        if losses.accuracy == 1.0 and losses.task_loss.item() < 1e-3:
            break

        optimizer.zero_grad(set_to_none=True)
        losses.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

    final_losses = compute_losses(
        model,
        batch_inputs,
        batch_targets,
        auxiliary_weight=auxiliary_weight,
    )
    ensure_finite(final_losses, step=step, model_name=model_name)
    if not trace or trace[-1]["step"] != step:
        trace.append(rounded_metric_row(step, final_losses))
    return trace, final_losses


def train_on_dataset(
    model: PredictiveGraphCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
    steps: int,
    learning_rate: float,
    auxiliary_weight: float,
    gradient_clip_norm: float,
    model_name: str,
) -> tuple[list[dict[str, float | int]], LossBreakdown]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sample_count = inputs.shape[0]
    trace: list[dict[str, float | int]] = []

    for step in range(steps + 1):
        if step % 50 == 0 or step == steps:
            full_losses = compute_losses(
                model,
                inputs,
                targets,
                auxiliary_weight=auxiliary_weight,
            )
            ensure_finite(full_losses, step=step, model_name=model_name)
            trace.append(rounded_metric_row(step, full_losses))

        if step == steps:
            break

        batch_indices = torch.randint(0, sample_count, (batch_size,), device=inputs.device)
        batch_inputs = inputs[batch_indices]
        batch_targets = targets[batch_indices]
        losses = compute_losses(
            model,
            batch_inputs,
            batch_targets,
            auxiliary_weight=auxiliary_weight,
        )
        ensure_finite(losses, step=step, model_name=model_name)
        optimizer.zero_grad(set_to_none=True)
        losses.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

    final_losses = compute_losses(
        model,
        inputs,
        targets,
        auxiliary_weight=auxiliary_weight,
    )
    ensure_finite(final_losses, step=steps, model_name=model_name)
    return trace, final_losses


def evaluate_model(
    model: PredictiveGraphCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    auxiliary_weight: float,
) -> tuple[LossBreakdown, ModelRollout]:
    with torch.no_grad():
        sequence = full_tokens(inputs, targets)
        rollout = model.rollout(sequence)
        task_step = inputs.shape[1] - 1
        task_state = torch.cat(
            [hidden_state[:, task_step, :] for hidden_state in rollout.hidden_states],
            dim=-1,
        )
        logits = model.task_head(task_state)
        auxiliary_by_node: dict[str, Tensor] = {}
        for node_index, label in enumerate(model.labels):
            target_inputs = rollout.incoming_inputs[node_index][:, 1:, :].detach()
            if target_inputs.shape[-1] != model.message_dim:
                target_inputs = align_embedding_target_to_message_dim(
                    target_inputs,
                    message_dim=model.message_dim,
                )
            auxiliary_by_node[label] = F.mse_loss(
                rollout.messages[node_index][:, :-1, :],
                target_inputs,
            )
        auxiliary_total = torch.stack(tuple(auxiliary_by_node.values())).sum()
        task_loss = F.cross_entropy(logits, targets)
        total_loss = task_loss + auxiliary_weight * auxiliary_total
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
        losses = LossBreakdown(
            total_loss=total_loss,
            task_loss=task_loss,
            auxiliary_total=auxiliary_total,
            auxiliary_by_node=auxiliary_by_node,
            logits=logits,
            accuracy=accuracy,
        )
    return losses, rollout


def _round_vector(values: Tensor) -> list[float]:
    return [round(value, 4) for value in values.detach().cpu().tolist()]


def save_prediction_table(
    path: Path,
    *,
    dataset: FixedWindowCharDataset,
    inputs: Tensor,
    targets: Tensor,
    predictions: Tensor,
) -> None:
    path.write_text(
        render_predictions(
            dataset,
            inputs.cpu(),
            targets.cpu(),
            predictions.cpu(),
        ),
        encoding="utf-8",
    )


def rollout_examples(
    *,
    model: PredictiveGraphCharModel,
    dataset: FixedWindowCharDataset,
    inputs: Tensor,
    targets: Tensor,
    limit: int,
) -> dict[str, object]:
    sample_inputs = inputs[:limit]
    sample_targets = targets[:limit]
    losses, rollout = evaluate_model(
        model,
        sample_inputs,
        sample_targets,
        auxiliary_weight=0.0,
    )
    predictions = losses.logits.argmax(dim=1)
    labels = model.labels

    examples = []
    for example_index in range(sample_inputs.shape[0]):
        steps = []
        sequence = full_tokens(sample_inputs, sample_targets)
        for step_index in range(sequence.shape[1]):
            token = dataset.decode([sequence[example_index, step_index].item()]).replace(
                "\n", "\\n"
            )
            next_token = None
            if step_index + 1 < sequence.shape[1]:
                next_token = dataset.decode(
                    [sequence[example_index, step_index + 1].item()]
                ).replace("\n", "\\n")
            step_payload = {
                "step": step_index,
                "token": token,
                "next_token_target": next_token,
                "incoming_inputs": {
                    label: _round_vector(rollout.incoming_inputs[node_index][example_index, step_index])
                    for node_index, label in enumerate(labels)
                },
                "messages": {
                    label: _round_vector(rollout.messages[node_index][example_index, step_index])
                    for node_index, label in enumerate(labels)
                },
                "attention_weights": {},
            }
            for node_index, label in enumerate(labels[1:], start=1):
                predecessor_labels = [labels[index] for index in model.predecessors[node_index]]
                attention = rollout.attention_weights[node_index]
                step_payload["attention_weights"][label] = {
                    predecessor_label: round(weight, 4)
                    for predecessor_label, weight in zip(
                        predecessor_labels,
                        attention[example_index, step_index].detach().cpu().tolist(),
                        strict=True,
                    )
                }
            steps.append(step_payload)

        examples.append(
            {
                "context": dataset.decode(sample_inputs[example_index].tolist()).replace(
                    "\n", "\\n"
                ),
                "target": dataset.decode([sample_targets[example_index].item()]).replace(
                    "\n", "\\n"
                ),
                "prediction": dataset.decode([predictions[example_index].item()]).replace(
                    "\n", "\\n"
                ),
                "steps": steps,
            }
        )

    return {"examples": examples}


def write_metrics(
    path: Path,
    *,
    trace: list[dict[str, float | int]],
    losses: LossBreakdown,
    reached_memorization_bar: bool | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    auxiliary_by_node = {
        label: value.item() for label, value in losses.auxiliary_by_node.items()
    }
    payload: dict[str, object] = {
        "final_total_loss": losses.total_loss.item(),
        "final_task_loss": losses.task_loss.item(),
        "final_auxiliary_total": losses.auxiliary_total.item(),
        "final_auxiliary_by_node": auxiliary_by_node,
        "final_accuracy": losses.accuracy,
        "trace": trace,
    }
    if reached_memorization_bar is not None:
        payload["reached_memorization_bar"] = reached_memorization_bar
    if extra is not None:
        payload.update(extra)
    write_json(path, payload)


def write_auxiliary_position_summary(
    path: Path,
    *,
    overfit_losses: LossBreakdown,
    train_losses: LossBreakdown,
    val_losses: LossBreakdown,
) -> None:
    labels = list(overfit_losses.auxiliary_by_node.keys())
    stages = {
        "overfit": overfit_losses,
        "train": train_losses,
        "val": val_losses,
    }
    payload: dict[str, object] = {"node_labels": labels}
    for stage_name, losses in stages.items():
        values = {label: losses.auxiliary_by_node[label].item() for label in labels}
        payload[f"{stage_name}_final_auxiliary_by_node"] = values
        payload[f"{stage_name}_nodes_sorted_by_auxiliary"] = sorted(
            values,
            key=values.__getitem__,
            reverse=True,
        )
    write_json(path, payload)


def attention_summary_for_rollout(
    model: PredictiveGraphCharModel,
    rollout: ModelRollout,
) -> dict[str, object]:
    labels = model.labels
    summary: dict[str, object] = {}
    for node_index in range(1, model.num_nodes):
        predecessor_labels = [labels[index] for index in model.predecessors[node_index]]
        weights = rollout.attention_weights[node_index]
        if weights is None:
            continue
        flat_weights = weights.detach().reshape(-1, weights.shape[-1]).float()
        mean_weights = flat_weights.mean(dim=0)
        std_weights = flat_weights.std(dim=0, unbiased=False)
        entropy = -(flat_weights * flat_weights.clamp_min(1e-9).log()).sum(dim=1)
        uniform_value = 1.0 / flat_weights.shape[-1]
        max_abs_deviation = (flat_weights - uniform_value).abs().max().item()
        summary[labels[node_index]] = {
            "predecessors": predecessor_labels,
            "mean_weight_by_predecessor": {
                predecessor_label: round(value, 6)
                for predecessor_label, value in zip(
                    predecessor_labels,
                    mean_weights.cpu().tolist(),
                    strict=True,
                )
            },
            "std_weight_by_predecessor": {
                predecessor_label: round(value, 6)
                for predecessor_label, value in zip(
                    predecessor_labels,
                    std_weights.cpu().tolist(),
                    strict=True,
                )
            },
            "entropy_mean": round(entropy.mean().item(), 6),
            "entropy_std": round(entropy.std(unbiased=False).item(), 6),
            "uniform_weight": round(uniform_value, 6),
            "max_abs_deviation_from_uniform": round(max_abs_deviation, 6),
            "shows_nontrivial_attention": bool(max_abs_deviation >= NONTRIVIAL_ATTENTION_DELTA),
        }
    return summary


def save_attention_weight_summary(
    path: Path,
    *,
    model: PredictiveGraphCharModel,
    overfit_rollout: ModelRollout,
    train_rollout: ModelRollout,
    val_rollout: ModelRollout,
) -> dict[str, object]:
    payload = {
        "aggregation_mode": model.aggregation_mode,
        "overfit": attention_summary_for_rollout(model, overfit_rollout),
        "train": attention_summary_for_rollout(model, train_rollout),
        "val": attention_summary_for_rollout(model, val_rollout),
    }
    write_json(path, payload)
    return payload


def save_samples(
    path: Path,
    *,
    model: PredictiveGraphCharModel,
    dataset: FixedWindowCharDataset,
    prompts: list[str],
    sample_length: int,
    device: torch.device,
) -> dict[str, str]:
    with torch.no_grad():
        samples = {
            prompt.replace("\n", "\\n"): generate_text(
                model,
                dataset,
                prompt,
                length=sample_length,
                device=device,
            )
            for prompt in prompts
        }
    write_json(path, samples)
    return samples


def build_model_summary(
    *,
    model: PredictiveGraphCharModel,
    dataset: FixedWindowCharDataset,
    config: RunConfig,
    parameter_count: int,
) -> dict[str, object]:
    return {
        "model_family": "predictive_graph",
        "parameter_count": parameter_count,
        "num_nodes": model.num_nodes,
        "node_labels": model.labels,
        "predecessors": {
            model.labels[node_index]: [model.labels[parent] for parent in node_predecessors]
            for node_index, node_predecessors in enumerate(model.predecessors)
        },
        "aggregation_mode": model.aggregation_mode,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "message_dim": config.message_dim,
        "attention_dim": config.attention_dim,
        "attention_projection_nodes": [model.labels[index] for index in model.multi_input_nodes],
        "attention_active_nodes": (
            [model.labels[index] for index in model.multi_input_nodes]
            if model.aggregation_mode == "attention"
            else []
        ),
        "message_input_detach": config.detach_messages,
        "message_target_detach": True,
        "auxiliary_weight": config.auxiliary_weight,
        "context_size": config.context_size,
        "vocab_size": dataset.vocab_size,
    }


def save_graph_run(
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
    aggregation_mode: AggregationMode,
    device: torch.device,
) -> dict[str, object]:
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    def build_model() -> PredictiveGraphCharModel:
        return PredictiveGraphCharModel(
            vocab_size=dataset.vocab_size,
            predecessors=SKIP2_PREDECESSORS,
            aggregation_mode=aggregation_mode,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            message_dim=config.message_dim,
            attention_dim=config.attention_dim,
            detach_messages=config.detach_messages,
        )

    parameter_count = count_parameters(build_model())
    if parameter_count < PARAMETER_MIN or parameter_count > PARAMETER_MAX:
        raise RuntimeError(
            f"{model_name} parameter_count={parameter_count} outside {PARAMETER_MIN}-{PARAMETER_MAX}"
        )

    write_json(
        model_dir / "model_summary.json",
        build_model_summary(
            model=build_model(),
            dataset=dataset,
            config=config,
            parameter_count=parameter_count,
        ),
    )

    overfit_model = build_model().to(device)
    overfit_inputs = train_inputs[: config.overfit_batch_size]
    overfit_targets = train_targets[: config.overfit_batch_size]
    overfit_trace, overfit_losses = train_on_fixed_batch(
        overfit_model,
        overfit_inputs,
        overfit_targets,
        steps=config.overfit_steps,
        learning_rate=config.overfit_learning_rate,
        auxiliary_weight=config.auxiliary_weight,
        gradient_clip_norm=config.gradient_clip_norm,
        model_name=model_name,
    )
    overfit_reached = (
        overfit_losses.accuracy == 1.0 and overfit_losses.task_loss.item() < 1e-3
    )
    if not overfit_reached:
        raise RuntimeError(f"{model_name} failed the one-batch overfit check")

    overfit_eval_losses, overfit_rollout = evaluate_model(
        overfit_model,
        overfit_inputs,
        overfit_targets,
        auxiliary_weight=config.auxiliary_weight,
    )
    overfit_predictions = overfit_eval_losses.logits.argmax(dim=1)
    overfit_attention_summary = attention_summary_for_rollout(overfit_model, overfit_rollout)
    attention_nontrivial = any(
        node_summary["shows_nontrivial_attention"]
        for node_summary in overfit_attention_summary.values()
    )
    if aggregation_mode == "attention" and not attention_nontrivial:
        raise RuntimeError(f"{model_name} overfit run kept attention near-uniform")

    write_metrics(
        model_dir / "overfit_metrics.json",
        trace=overfit_trace,
        losses=overfit_eval_losses,
        reached_memorization_bar=overfit_reached,
        extra={
            "attention_nontrivial": attention_nontrivial,
            "attention_nontrivial_threshold": NONTRIVIAL_ATTENTION_DELTA,
        },
    )
    save_prediction_table(
        model_dir / "overfit_predictions.txt",
        dataset=dataset,
        inputs=overfit_inputs,
        targets=overfit_targets,
        predictions=overfit_predictions,
    )
    write_json(
        model_dir / "overfit_rollout_examples.json",
        rollout_examples(
            model=overfit_model,
            dataset=dataset,
            inputs=overfit_inputs,
            targets=overfit_targets,
            limit=config.rollout_examples,
        ),
    )

    train_model = build_model().to(device)
    started_at = time.perf_counter()
    train_trace, _train_losses = train_on_dataset(
        train_model,
        train_inputs,
        train_targets,
        batch_size=config.train_batch_size,
        steps=config.train_steps,
        learning_rate=config.train_learning_rate,
        auxiliary_weight=config.auxiliary_weight,
        gradient_clip_norm=config.gradient_clip_norm,
        model_name=model_name,
    )
    runtime_seconds = time.perf_counter() - started_at

    train_losses, train_rollout = evaluate_model(
        train_model,
        train_inputs,
        train_targets,
        auxiliary_weight=config.auxiliary_weight,
    )
    val_losses, val_rollout = evaluate_model(
        train_model,
        val_inputs,
        val_targets,
        auxiliary_weight=config.auxiliary_weight,
    )
    val_predictions = val_losses.logits[:32].argmax(dim=1)
    save_prediction_table(
        model_dir / "validation_predictions.txt",
        dataset=dataset,
        inputs=val_inputs[:32],
        targets=val_targets[:32],
        predictions=val_predictions,
    )
    write_json(
        model_dir / "validation_rollout_examples.json",
        rollout_examples(
            model=train_model,
            dataset=dataset,
            inputs=val_inputs,
            targets=val_targets,
            limit=config.rollout_examples,
        ),
    )
    samples = save_samples(
        model_dir / "samples.json",
        model=train_model,
        dataset=dataset,
        prompts=prompts,
        sample_length=config.sample_length,
        device=device,
    )
    attention_summary = save_attention_weight_summary(
        model_dir / "attention_weight_summary.json",
        model=train_model,
        overfit_rollout=overfit_rollout,
        train_rollout=train_rollout,
        val_rollout=val_rollout,
    )
    write_auxiliary_position_summary(
        model_dir / "auxiliary_position_summary.json",
        overfit_losses=overfit_eval_losses,
        train_losses=train_losses,
        val_losses=val_losses,
    )
    write_json(
        model_dir / "train_val_metrics.json",
        {
            "parameter_count": parameter_count,
            "runtime_seconds": runtime_seconds,
            "train_metrics": {
                "task_loss": train_losses.task_loss.item(),
                "total_loss": train_losses.total_loss.item(),
                "auxiliary_total": train_losses.auxiliary_total.item(),
                "accuracy": train_losses.accuracy,
            },
            "val_metrics": {
                "task_loss": val_losses.task_loss.item(),
                "total_loss": val_losses.total_loss.item(),
                "auxiliary_total": val_losses.auxiliary_total.item(),
                "accuracy": val_losses.accuracy,
            },
            "train_trace": train_trace,
            "attention_nontrivial_overfit": attention_nontrivial,
            "attention_nodes_nontrivial_overfit": sorted(
                [
                    node_label
                    for node_label, node_summary in overfit_attention_summary.items()
                    if node_summary["shows_nontrivial_attention"]
                ]
            ),
        },
    )
    return {
        "model_name": model_name,
        "aggregation_mode": aggregation_mode,
        "parameter_count": parameter_count,
        "overfit_reached": overfit_reached,
        "attention_nontrivial_overfit": attention_nontrivial,
        "runtime_seconds": runtime_seconds,
        "train_loss": train_losses.task_loss.item(),
        "train_accuracy": train_losses.accuracy,
        "val_loss": val_losses.task_loss.item(),
        "val_accuracy": val_losses.accuracy,
        "train_total_loss": train_losses.total_loss.item(),
        "val_total_loss": val_losses.total_loss.item(),
        "train_auxiliary_total": train_losses.auxiliary_total.item(),
        "val_auxiliary_total": val_losses.auxiliary_total.item(),
        "samples": samples,
        "attention_weight_summary": attention_summary,
    }


def print_summary(*, results: list[dict[str, object]]) -> None:
    ordered_models = sorted(results, key=lambda row: row["val_loss"])
    summary = {
        "historical_chain": {
            "model_name": "predictive_chain_aux_1p0_detach",
            "val_loss": HISTORICAL_CHAIN_VAL_LOSS,
            "parameter_count": 13662,
        },
        "models": [
            {
                "model_name": row["model_name"],
                "aggregation_mode": row["aggregation_mode"],
                "parameter_count": row["parameter_count"],
                "overfit_reached": row["overfit_reached"],
                "attention_nontrivial_overfit": row["attention_nontrivial_overfit"],
                "val_loss": round(float(row["val_loss"]), 6),
                "val_accuracy": round(float(row["val_accuracy"]), 6),
                "delta_vs_historical_chain": round(
                    float(row["val_loss"]) - HISTORICAL_CHAIN_VAL_LOSS,
                    6,
                ),
            }
            for row in ordered_models
        ],
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = (
        repo_root
        / "research"
        / "questions"
        / "predictive-chain"
        / "artifacts"
        / "shakespeare_graph_attention"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--overfit-steps", type=int)
    parser.add_argument("--train-steps", type=int)
    args = parser.parse_args()

    config = RunConfig()
    if args.hidden_dim is not None:
        config = replace(config, hidden_dim=args.hidden_dim)
    if args.overfit_steps is not None:
        config = replace(config, overfit_steps=args.overfit_steps)
    if args.train_steps is not None:
        config = replace(config, train_steps=args.train_steps)

    set_seed(config.seed)
    device = resolve_device(args.device)
    dataset = FixedWindowCharDataset(SHAKESPEARE_SNIPPET, context_size=config.context_size)
    split = build_train_val_split(
        SHAKESPEARE_SNIPPET,
        context_size=config.context_size,
        stoi=dataset.stoi,
        train_fraction=config.train_fraction,
    )
    prompts = select_prompts(
        split.train_text,
        split.val_text,
        context_size=config.context_size,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

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
            "cuda_device_name": torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None,
        },
    )
    write_json(
        output_dir / "corpus_summary.json",
        {
            "corpus_name": "hardcoded_shakespeare_excerpt",
            "corpus_sha256": hashlib.sha256(SHAKESPEARE_SNIPPET.encode("utf-8")).hexdigest(),
            "total_characters": len(SHAKESPEARE_SNIPPET),
            "vocab_size": dataset.vocab_size,
            "split_index": split.split_index,
            "train_characters": len(split.train_text),
            "val_characters": len(split.val_text),
            "train_windows": int(split.train_inputs.shape[0]),
            "val_windows": int(split.val_inputs.shape[0]),
            "prompts": [prompt.replace("\n", "\\n") for prompt in prompts],
        },
    )

    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)

    variant_specs = [
        ("graph_skip2_uniform_aux_1_detach", "uniform_mean"),
        ("graph_skip2_attention_aux_1_detach", "attention"),
    ]
    results = []
    for model_name, aggregation_mode in variant_specs:
        results.append(
            save_graph_run(
                model_name=model_name,
                output_dir=output_dir,
                dataset=dataset,
                train_inputs=train_inputs,
                train_targets=train_targets,
                val_inputs=val_inputs,
                val_targets=val_targets,
                prompts=prompts,
                config=config,
                aggregation_mode=aggregation_mode,
                device=device,
            )
        )

    comparison_summary = {
        "historical_chain": {
            "model_name": "predictive_chain_aux_1p0_detach",
            "parameter_count": 13662,
            "val_loss": HISTORICAL_CHAIN_VAL_LOSS,
        },
        "models": sorted(results, key=lambda row: row["val_loss"]),
        "best_by_val_loss": min(results, key=lambda row: row["val_loss"]),
        "best_by_val_accuracy": max(results, key=lambda row: row["val_accuracy"]),
    }
    write_json(output_dir / "comparison_summary.json", comparison_summary)
    print_summary(results=results)


if __name__ == "__main__":
    main()
