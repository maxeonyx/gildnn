from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from base_experiments.rnn import (
    build_fixed_length_split,
    current_git_sha,
    current_git_status_short,
    write_json,
)
from core.fixed_window_char import FixedWindowCharDataset, generate_text, resolve_device, set_seed
from core.tiny_char_transformer import count_parameters
from experiments.pytorch_char_dynamic_depth_improved import (
    build_epsilon_rows,
    build_threshold_rows,
    calibration_summary,
    choose_recommended_point,
    collect_evaluation_details,
    make_quantile_candidates,
    pareto_frontier,
    summarize_dynamic_depth,
    summarize_fixed_depth,
    write_tsv,
)


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 32
    train_characters: int = 100_000
    val_characters: int = 20_000
    batch_size: int = 256
    epochs: int = 13
    learning_rate: float = 0.003
    gradient_clip_norm: float = 1.0
    eval_batch_size: int = 512
    num_nodes: int = 3
    embedding_dim: int = 56
    hidden_dim: int = 144
    message_dim: int = 56
    max_depth: int = 8
    auxiliary_weight: float = 0.01
    loss_pred_weight: float = 0.1
    detach_messages: bool = True
    sample_length: int = 320
    seed: int = 42
    check_batch_size: int = 8
    memorization_batch_size: int = 32
    memorization_steps: int = 400
    memorization_learning_rate: float = 0.02
    threshold_quantiles: int = 15
    epsilon_quantiles: int = 15
    recommendation_loss_tolerance: float = 0.01


@dataclass(frozen=True)
class LossBundle:
    total_loss: Tensor
    task_loss: Tensor
    loss_prediction_loss: Tensor
    auxiliary_total: Tensor
    auxiliary_by_node: dict[str, Tensor]
    logits_by_depth: Tensor
    predicted_losses: Tensor
    final_accuracy: float
    per_depth_task_loss: list[float]
    per_depth_accuracy: list[float]


@dataclass(frozen=True)
class ModelRollout:
    embeddings: Tensor
    hidden_states: list[Tensor]
    messages: list[Tensor]
    logits: Tensor
    predicted_losses: Tensor


def replace_config(config: RunConfig, **changes: object) -> RunConfig:
    payload = asdict(config)
    payload.update(changes)
    return RunConfig(**payload)


def node_label(index: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(letters):
        return letters[index]
    return f"node_{index + 1}"


def node_labels(count: int) -> list[str]:
    return [node_label(index) for index in range(count)]


def align_embedding_target_to_message_dim(
    target_embeddings: Tensor,
    *,
    message_dim: int,
) -> Tensor:
    embedding_dim = target_embeddings.shape[-1]
    if embedding_dim == message_dim:
        return target_embeddings
    if embedding_dim > message_dim:
        return target_embeddings[..., :message_dim]

    padding_shape = (*target_embeddings.shape[:-1], message_dim - embedding_dim)
    padding = torch.zeros(
        padding_shape,
        device=target_embeddings.device,
        dtype=target_embeddings.dtype,
    )
    return torch.cat((target_embeddings, padding), dim=-1)


class DynamicDepthPredictiveChainModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_nodes: int,
        embedding_dim: int,
        hidden_dim: int,
        message_dim: int,
        max_depth: int,
        detach_messages: bool,
    ) -> None:
        super().__init__()
        if num_nodes < 1:
            raise ValueError(f"num_nodes must be at least 1, got {num_nodes}")

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.nodes = nn.ModuleList(
            [
                nn.RNNCell(
                    input_size=embedding_dim if node_index == 0 else message_dim,
                    hidden_size=hidden_dim,
                    nonlinearity="tanh",
                )
                for node_index in range(num_nodes)
            ]
        )
        self.message_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, message_dim) for _ in range(num_nodes)]
        )
        self.task_head = nn.Sequential(
            nn.Linear(hidden_dim * num_nodes, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, vocab_size),
        )
        self.loss_head = nn.Linear(hidden_dim * num_nodes, 1)

        self.num_nodes = num_nodes
        self.labels = node_labels(num_nodes)
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.max_depth = max_depth
        self.detach_messages = detach_messages

    def _bounded_embedding(self, tokens: Tensor) -> Tensor:
        return torch.tanh(self.embedding(tokens))

    def _predictive_message(self, projection: nn.Linear, hidden: Tensor) -> Tensor:
        return torch.tanh(projection(hidden))

    def rollout(self, tokens: Tensor, *, max_depth: int | None = None) -> ModelRollout:
        depth_limit = max_depth or self.max_depth
        batch_size, sequence_length = tokens.shape
        embeddings = self._bounded_embedding(tokens)

        hidden_states = [
            torch.zeros(batch_size, self.hidden_dim, device=tokens.device)
            for _ in range(self.num_nodes)
        ]
        previous_messages = [
            torch.zeros(batch_size, self.message_dim, device=tokens.device)
            for _ in range(self.num_nodes - 1)
        ]

        hidden_steps = [[] for _ in range(self.num_nodes)]
        message_steps = [[] for _ in range(self.num_nodes)]
        logits_steps = []
        predicted_loss_steps = []

        for step in range(sequence_length):
            depth_hidden_states = hidden_states
            depth_previous_messages = previous_messages
            step_hidden_by_node = [[] for _ in range(self.num_nodes)]
            step_message_by_node = [[] for _ in range(self.num_nodes)]
            step_logits = []
            step_predicted_losses = []

            for _depth_index in range(depth_limit):
                current_hidden_states = []
                current_messages = []
                for node_index, (node, message_head) in enumerate(
                    zip(self.nodes, self.message_heads, strict=True)
                ):
                    node_input = (
                        embeddings[:, step, :]
                        if node_index == 0
                        else depth_previous_messages[node_index - 1]
                    )
                    next_hidden = node(node_input, depth_hidden_states[node_index])
                    message = self._predictive_message(message_head, next_hidden)
                    current_hidden_states.append(next_hidden)
                    current_messages.append(message)
                    step_hidden_by_node[node_index].append(next_hidden)
                    step_message_by_node[node_index].append(message)

                chain_state = torch.cat(current_hidden_states, dim=-1)
                step_logits.append(self.task_head(chain_state))
                step_predicted_losses.append(self.loss_head(chain_state).squeeze(1))
                depth_hidden_states = current_hidden_states
                depth_previous_messages = [
                    message.detach() if self.detach_messages else message
                    for message in current_messages[:-1]
                ]

            hidden_states = depth_hidden_states
            previous_messages = depth_previous_messages
            for node_index in range(self.num_nodes):
                hidden_steps[node_index].append(torch.stack(step_hidden_by_node[node_index], dim=1))
                message_steps[node_index].append(torch.stack(step_message_by_node[node_index], dim=1))
            logits_steps.append(torch.stack(step_logits, dim=1))
            predicted_loss_steps.append(torch.stack(step_predicted_losses, dim=1))

        return ModelRollout(
            embeddings=embeddings,
            hidden_states=[torch.stack(steps, dim=1) for steps in hidden_steps],
            messages=[torch.stack(steps, dim=1) for steps in message_steps],
            logits=torch.stack(logits_steps, dim=1),
            predicted_losses=torch.stack(predicted_loss_steps, dim=1),
        )

    def forward_depths(self, tokens: Tensor, *, max_depth: int | None = None) -> tuple[Tensor, Tensor]:
        rollout = self.rollout(tokens, max_depth=max_depth)
        return rollout.logits, rollout.predicted_losses

    def forward(self, tokens: Tensor) -> Tensor:
        logits, _predicted_losses = self.forward_depths(tokens, max_depth=self.max_depth)
        return logits[:, -1, -1, :]


def make_model(
    config: RunConfig,
    *,
    vocab_size: int,
    device: torch.device,
) -> DynamicDepthPredictiveChainModel:
    return DynamicDepthPredictiveChainModel(
        vocab_size=vocab_size,
        num_nodes=config.num_nodes,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        message_dim=config.message_dim,
        max_depth=config.max_depth,
        detach_messages=config.detach_messages,
    ).to(device)


def compute_loss_bundle(
    model: DynamicDepthPredictiveChainModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    active_depth: int,
    auxiliary_weight: float,
    loss_pred_weight: float,
) -> LossBundle:
    rollout = model.rollout(inputs, max_depth=active_depth)
    final_logits_by_depth = rollout.logits[:, -1, :, :]
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
    predicted_final_losses = rollout.predicted_losses[:, -1, :]
    task_loss = torch.stack(per_depth_losses).mean()
    loss_prediction_loss = F.mse_loss(predicted_final_losses, actual_losses.detach())

    auxiliary_by_node: dict[str, Tensor] = {}
    next_embeddings = align_embedding_target_to_message_dim(
        rollout.embeddings[:, 1:, :].detach(),
        message_dim=model.message_dim,
    )
    next_embeddings = next_embeddings.unsqueeze(2).expand(-1, -1, active_depth, -1)
    auxiliary_by_node[model.labels[0]] = F.mse_loss(
        rollout.messages[0][:, :-1, :, :],
        next_embeddings,
    )
    for node_index in range(1, model.num_nodes):
        auxiliary_by_node[model.labels[node_index]] = F.mse_loss(
            rollout.messages[node_index],
            rollout.messages[node_index - 1].detach(),
        )

    auxiliary_total = torch.stack(tuple(auxiliary_by_node.values())).sum()
    total_loss = task_loss + loss_pred_weight * loss_prediction_loss + auxiliary_weight * auxiliary_total
    return LossBundle(
        total_loss=total_loss,
        task_loss=task_loss,
        loss_prediction_loss=loss_prediction_loss,
        auxiliary_total=auxiliary_total,
        auxiliary_by_node=auxiliary_by_node,
        logits_by_depth=final_logits_by_depth,
        predicted_losses=predicted_final_losses,
        final_accuracy=per_depth_accuracies[-1],
        per_depth_task_loss=[loss.item() for loss in per_depth_losses],
        per_depth_accuracy=per_depth_accuracies,
    )


def batched_pairs(inputs: Tensor, targets: Tensor, *, batch_size: int):
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        yield inputs[start:stop], targets[start:stop]


def evaluate_training_objective(
    model: DynamicDepthPredictiveChainModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    active_depth: int,
    auxiliary_weight: float,
    loss_pred_weight: float,
    batch_size: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_examples = 0
    total_total_loss = 0.0
    total_task_loss = 0.0
    total_loss_prediction = 0.0
    total_auxiliary = 0.0
    total_correct = 0.0

    with torch.no_grad():
        for batch_inputs, batch_targets in batched_pairs(inputs, targets, batch_size=batch_size):
            bundle = compute_loss_bundle(
                model,
                batch_inputs,
                batch_targets,
                active_depth=active_depth,
                auxiliary_weight=auxiliary_weight,
                loss_pred_weight=loss_pred_weight,
            )
            batch_examples = batch_targets.shape[0]
            total_examples += batch_examples
            total_total_loss += bundle.total_loss.item() * batch_examples
            total_task_loss += bundle.task_loss.item() * batch_examples
            total_loss_prediction += bundle.loss_prediction_loss.item() * batch_examples
            total_auxiliary += bundle.auxiliary_total.item() * batch_examples
            total_correct += bundle.final_accuracy * batch_examples

    if was_training:
        model.train()
    return {
        "total_loss": total_total_loss / total_examples,
        "task_loss": total_task_loss / total_examples,
        "loss_prediction_loss": total_loss_prediction / total_examples,
        "auxiliary_total": total_auxiliary / total_examples,
        "accuracy": total_correct / total_examples,
    }


def evaluate_fixed_depth(
    model: DynamicDepthPredictiveChainModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    depth: int,
    batch_size: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_examples = 0
    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for batch_inputs, batch_targets in batched_pairs(inputs, targets, batch_size=batch_size):
            logits, _predicted_losses = model.forward_depths(batch_inputs, max_depth=depth)
            depth_logits = logits[:, -1, depth - 1, :]
            total_loss += F.cross_entropy(depth_logits, batch_targets, reduction="sum").item()
            total_correct += (depth_logits.argmax(dim=1) == batch_targets).sum().item()
            total_examples += batch_targets.shape[0]

    if was_training:
        model.train()
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_one_epoch(
    model: DynamicDepthPredictiveChainModel,
    optimizer: torch.optim.Optimizer,
    train_inputs: Tensor,
    train_targets: Tensor,
    *,
    batch_size: int,
    active_depth: int,
    auxiliary_weight: float,
    loss_pred_weight: float,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    permutation = torch.randperm(train_inputs.shape[0], device=train_inputs.device)
    total_examples = 0
    total_total_loss = 0.0
    total_task_loss = 0.0
    total_loss_prediction = 0.0
    total_auxiliary = 0.0
    total_correct = 0.0

    for start in range(0, permutation.shape[0], batch_size):
        batch_indices = permutation[start : start + batch_size]
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]
        bundle = compute_loss_bundle(
            model,
            batch_inputs,
            batch_targets,
            active_depth=active_depth,
            auxiliary_weight=auxiliary_weight,
            loss_pred_weight=loss_pred_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        bundle.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        batch_examples = batch_targets.shape[0]
        total_examples += batch_examples
        total_total_loss += bundle.total_loss.item() * batch_examples
        total_task_loss += bundle.task_loss.item() * batch_examples
        total_loss_prediction += bundle.loss_prediction_loss.item() * batch_examples
        total_auxiliary += bundle.auxiliary_total.item() * batch_examples
        total_correct += bundle.final_accuracy * batch_examples

    return {
        "total_loss": total_total_loss / total_examples,
        "task_loss": total_task_loss / total_examples,
        "loss_prediction_loss": total_loss_prediction / total_examples,
        "auxiliary_total": total_auxiliary / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_model(
    model: DynamicDepthPredictiveChainModel,
    *,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    config: RunConfig,
) -> tuple[list[dict[str, float | int]], dict[str, float], float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []
    started_at = time.perf_counter()

    initial_train_objective = evaluate_training_objective(
        model,
        train_inputs,
        train_targets,
        active_depth=config.max_depth,
        auxiliary_weight=config.auxiliary_weight,
        loss_pred_weight=config.loss_pred_weight,
        batch_size=config.eval_batch_size,
    )
    initial_val_objective = evaluate_training_objective(
        model,
        val_inputs,
        val_targets,
        active_depth=config.max_depth,
        auxiliary_weight=config.auxiliary_weight,
        loss_pred_weight=config.loss_pred_weight,
        batch_size=config.eval_batch_size,
    )
    initial_val_fixed = evaluate_fixed_depth(
        model,
        val_inputs,
        val_targets,
        depth=config.max_depth,
        batch_size=config.eval_batch_size,
    )
    history.append(
        {
            "epoch": 0,
            "train_total_loss": round(initial_train_objective["total_loss"], 6),
            "train_task_loss": round(initial_train_objective["task_loss"], 6),
            "train_loss_prediction_loss": round(initial_train_objective["loss_prediction_loss"], 6),
            "train_auxiliary_total": round(initial_train_objective["auxiliary_total"], 6),
            "train_accuracy": round(initial_train_objective["accuracy"], 6),
            "val_total_loss": round(initial_val_objective["total_loss"], 6),
            "val_task_loss": round(initial_val_objective["task_loss"], 6),
            "val_loss_prediction_loss": round(initial_val_objective["loss_prediction_loss"], 6),
            "val_auxiliary_total": round(initial_val_objective["auxiliary_total"], 6),
            "val_accuracy": round(initial_val_objective["accuracy"], 6),
            "val_fixed_depth_loss": round(initial_val_fixed["loss"], 6),
            "val_fixed_depth_accuracy": round(initial_val_fixed["accuracy"], 6),
        }
    )
    print(
        f"epoch=0 train_fixed_depth_loss={initial_train_objective['accuracy']:.4f} "
        f"val_fixed_depth_loss={initial_val_fixed['loss']:.4f}"
    )

    for epoch in range(1, config.epochs + 1):
        train_objective = train_one_epoch(
            model,
            optimizer,
            train_inputs,
            train_targets,
            batch_size=config.batch_size,
            active_depth=config.max_depth,
            auxiliary_weight=config.auxiliary_weight,
            loss_pred_weight=config.loss_pred_weight,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        val_objective = evaluate_training_objective(
            model,
            val_inputs,
            val_targets,
            active_depth=config.max_depth,
            auxiliary_weight=config.auxiliary_weight,
            loss_pred_weight=config.loss_pred_weight,
            batch_size=config.eval_batch_size,
        )
        val_fixed = evaluate_fixed_depth(
            model,
            val_inputs,
            val_targets,
            depth=config.max_depth,
            batch_size=config.eval_batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": round(train_objective["total_loss"], 6),
                "train_task_loss": round(train_objective["task_loss"], 6),
                "train_loss_prediction_loss": round(train_objective["loss_prediction_loss"], 6),
                "train_auxiliary_total": round(train_objective["auxiliary_total"], 6),
                "train_accuracy": round(train_objective["accuracy"], 6),
                "val_total_loss": round(val_objective["total_loss"], 6),
                "val_task_loss": round(val_objective["task_loss"], 6),
                "val_loss_prediction_loss": round(val_objective["loss_prediction_loss"], 6),
                "val_auxiliary_total": round(val_objective["auxiliary_total"], 6),
                "val_accuracy": round(val_objective["accuracy"], 6),
                "val_fixed_depth_loss": round(val_fixed["loss"], 6),
                "val_fixed_depth_accuracy": round(val_fixed["accuracy"], 6),
            }
        )
        print(
            f"epoch={epoch} train_task_loss={train_objective['task_loss']:.4f} "
            f"val_fixed_depth_loss={val_fixed['loss']:.4f}"
        )

    runtime_seconds = time.perf_counter() - started_at
    final_val_fixed = evaluate_fixed_depth(
        model,
        val_inputs,
        val_targets,
        depth=config.max_depth,
        batch_size=config.eval_batch_size,
    )
    return history, final_val_fixed, runtime_seconds


def run_correctness_checks(
    split,
    *,
    config: RunConfig,
    device: torch.device,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    sample_text = split.train_text[: config.context_size]
    round_trip = split.train_dataset.decode(split.train_dataset.encode(sample_text))
    if round_trip != sample_text:
        raise RuntimeError("Dataset encode/decode round-trip failed.")
    checks.append(
        {
            "name": "dataset_round_trip",
            "ok": True,
            "sample": sample_text.replace("\n", "\\n"),
        }
    )

    set_seed(config.seed)
    shape_model = make_model(config, vocab_size=split.train_dataset.vocab_size, device=device)
    batch_inputs = split.train_inputs[: config.check_batch_size].to(device)
    batch_targets = split.train_targets[: config.check_batch_size].to(device)
    logits, predicted_losses = shape_model.forward_depths(batch_inputs, max_depth=config.max_depth)
    expected_logits_shape = [config.check_batch_size, config.context_size, config.max_depth, split.train_dataset.vocab_size]
    if list(logits.shape) != expected_logits_shape:
        raise RuntimeError(
            f"Forward logits shape mismatch: expected {expected_logits_shape}, got {list(logits.shape)}."
        )
    expected_loss_shape = [config.check_batch_size, config.context_size, config.max_depth]
    if list(predicted_losses.shape) != expected_loss_shape:
        raise RuntimeError(
            f"Predicted loss shape mismatch: expected {expected_loss_shape}, got {list(predicted_losses.shape)}."
        )
    checks.append(
        {
            "name": "forward_shapes",
            "ok": True,
            "logits_shape": expected_logits_shape,
            "predicted_loss_shape": expected_loss_shape,
        }
    )

    tiny_bundle = compute_loss_bundle(
        shape_model,
        batch_inputs,
        batch_targets,
        active_depth=config.max_depth,
        auxiliary_weight=config.auxiliary_weight,
        loss_pred_weight=config.loss_pred_weight,
    )
    expected_low = math.log(split.train_dataset.vocab_size) - 1.0
    expected_high = math.log(split.train_dataset.vocab_size) + 1.0
    if not expected_low <= tiny_bundle.task_loss.item() <= expected_high:
        raise RuntimeError(
            f"Known tiny-batch loss {tiny_bundle.task_loss.item():.6f} fell outside expected range "
            f"[{expected_low:.6f}, {expected_high:.6f}]."
        )
    checks.append(
        {
            "name": "known_tiny_batch_loss",
            "ok": True,
            "loss": tiny_bundle.task_loss.item(),
            "expected_range": [expected_low, expected_high],
        }
    )

    set_seed(config.seed)
    memorization_model = make_model(config, vocab_size=split.train_dataset.vocab_size, device=device)
    memorize_inputs = split.train_inputs[: config.memorization_batch_size].to(device)
    memorize_targets = split.train_targets[: config.memorization_batch_size].to(device)
    optimizer = torch.optim.AdamW(
        memorization_model.parameters(),
        lr=config.memorization_learning_rate,
    )
    trace: list[dict[str, float | int]] = []
    initial_bundle: LossBundle | None = None
    final_bundle: LossBundle | None = None
    for step in range(config.memorization_steps + 1):
        bundle = compute_loss_bundle(
            memorization_model,
            memorize_inputs,
            memorize_targets,
            active_depth=config.max_depth,
            auxiliary_weight=config.auxiliary_weight,
            loss_pred_weight=config.loss_pred_weight,
        )
        if initial_bundle is None:
            initial_bundle = bundle
        if step in {0, config.memorization_steps} or step % 20 == 0:
            trace.append(
                {
                    "step": step,
                    "total_loss": round(bundle.total_loss.item(), 6),
                    "task_loss": round(bundle.task_loss.item(), 6),
                    "loss_prediction_loss": round(bundle.loss_prediction_loss.item(), 6),
                    "auxiliary_total": round(bundle.auxiliary_total.item(), 6),
                    "accuracy": round(bundle.final_accuracy, 6),
                }
            )
        final_bundle = bundle
        if step == config.memorization_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        bundle.total_loss.backward()
        optimizer.step()
    if final_bundle is None:
        raise RuntimeError("Memorization loop produced no final bundle.")
    if initial_bundle is None:
        raise RuntimeError("Memorization loop produced no initial bundle.")
    loss_ratio = final_bundle.per_depth_task_loss[-1] / initial_bundle.per_depth_task_loss[-1]
    if not (final_bundle.per_depth_task_loss[-1] < 0.2 and final_bundle.final_accuracy >= 0.93 and loss_ratio < 0.1):
        raise RuntimeError(
            "One-batch memorization check failed: expected final-depth task loss < 0.2, "
            "accuracy >= 0.93, and >10x loss reduction, "
            f"got task_loss={final_bundle.per_depth_task_loss[-1]:.6f}, "
            f"accuracy={final_bundle.final_accuracy:.6f}, loss_ratio={loss_ratio:.6f}."
        )
    checks.append(
        {
            "name": "one_batch_memorization",
            "ok": True,
            "initial_final_depth_task_loss": initial_bundle.per_depth_task_loss[-1],
            "final_total_loss": final_bundle.total_loss.item(),
            "final_task_loss": final_bundle.task_loss.item(),
            "final_final_depth_task_loss": final_bundle.per_depth_task_loss[-1],
            "final_loss_prediction_loss": final_bundle.loss_prediction_loss.item(),
            "final_auxiliary_total": final_bundle.auxiliary_total.item(),
            "final_accuracy": final_bundle.final_accuracy,
            "loss_ratio": loss_ratio,
            "trace": trace,
        }
    )
    return checks


def make_default_paths(repo_root: Path, mode: str) -> tuple[Path, Path]:
    text_file = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    base_output_dir = repo_root / "experiments" / "chain_dynamic_depth" / "artifacts" / "predictive_chain_dynamic_depth_standardized"
    if mode == "full":
        return text_file, base_output_dir
    return text_file, base_output_dir / mode


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["checks-only", "tiny", "full"], default="full")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--train-characters", type=int)
    parser.add_argument("--val-characters", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--sample-length", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--message-dim", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--auxiliary-weight", type=float)
    parser.add_argument("--loss-pred-weight", type=float)
    parser.add_argument(
        "--detach-messages",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--memorization-steps", type=int)
    parser.add_argument("--memorization-learning-rate", type=float)
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def resolved_config(args: argparse.Namespace) -> RunConfig:
    config = RunConfig()
    if args.mode == "tiny":
        config = replace_config(
            config,
            train_characters=4_096,
            val_characters=1_024,
            epochs=2,
            sample_length=160,
            memorization_steps=1_200,
        )
    if args.mode == "checks-only":
        config = replace_config(
            config,
            train_characters=2_048,
            val_characters=512,
            epochs=0,
            sample_length=0,
            memorization_steps=800,
        )

    overrides = {
        "train_characters": args.train_characters,
        "val_characters": args.val_characters,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "sample_length": args.sample_length,
        "seed": args.seed,
        "num_nodes": args.num_nodes,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "message_dim": args.message_dim,
        "max_depth": args.max_depth,
        "auxiliary_weight": args.auxiliary_weight,
        "loss_pred_weight": args.loss_pred_weight,
        "detach_messages": args.detach_messages,
        "memorization_steps": args.memorization_steps,
        "memorization_learning_rate": args.memorization_learning_rate,
    }
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    if clean_overrides:
        config = replace_config(config, **clean_overrides)
    return config


def main() -> None:
    args = parse_args()
    config = resolved_config(args)
    default_text_file, default_output_dir = make_default_paths(args.repo_root, args.mode)
    text_file = args.text_file or default_text_file
    output_dir = args.output_dir or default_output_dir

    set_seed(config.seed)
    device = resolve_device(args.device)
    raw_text = text_file.read_text(encoding="utf-8")
    split = build_fixed_length_split(raw_text, config=config)
    output_dir.mkdir(parents=True, exist_ok=True)

    git_status_short = current_git_status_short()
    parameter_count = count_parameters(
        make_model(config, vocab_size=split.train_dataset.vocab_size, device=device)
    )

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
    used_text = split.train_text + split.val_text
    write_json(
        output_dir / "corpus_summary.json",
        {
            "source_file": str(text_file),
            "source_total_characters": len(raw_text),
            "source_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "used_total_characters": len(used_text),
            "used_sha256": hashlib.sha256(used_text.encode("utf-8")).hexdigest(),
            "train_characters": len(split.train_text),
            "val_characters": len(split.val_text),
            "val_start": split.val_start,
            "val_stop": split.val_start + len(split.val_text),
            "train_windows": int(split.train_inputs.shape[0]),
            "val_windows": int(split.val_inputs.shape[0]),
            "vocab_size": split.train_dataset.vocab_size,
        },
    )
    write_json(
        output_dir / "model_summary.json",
        {
            "model_family": "predictive_chain_dynamic_depth_standardized",
            "parameter_count": parameter_count,
            "context_size": config.context_size,
            "num_nodes": config.num_nodes,
            "node_labels": node_labels(config.num_nodes),
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "message_dim": config.message_dim,
            "max_depth": config.max_depth,
            "recurrent_cell": "RNNCell(tanh)",
            "task_head_input": "concat(all_final_hidden_states_per_depth)",
            "loss_head_target": "next_char_cross_entropy_at_each_depth",
            "message_target_detach": True,
            "message_input_detach": config.detach_messages,
            "delta_vs_stage1": "adds shared-depth refinement and loss-prediction halting head on top of the standardized predictive chain",
        },
    )

    checks = run_correctness_checks(split, config=config, device=device)
    write_json(output_dir / "correctness_checks.json", checks)
    print("correctness checks passed")

    if args.mode == "checks-only":
        return

    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)

    set_seed(config.seed)
    model = make_model(config, vocab_size=split.train_dataset.vocab_size, device=device)
    history, final_val_fixed, runtime_seconds = train_model(
        model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        config=config,
    )

    train_details = collect_evaluation_details(
        model,
        train_inputs,
        train_targets,
        max_depth=config.max_depth,
        batch_size=config.eval_batch_size,
    )
    val_details = collect_evaluation_details(
        model,
        val_inputs,
        val_targets,
        max_depth=config.max_depth,
        batch_size=config.eval_batch_size,
    )

    threshold_frontier, threshold_rows = pareto_frontier(
        build_threshold_rows(
            train_details=train_details,
            val_details=val_details,
            candidates=make_quantile_candidates(
                train_details.predicted_losses,
                quantiles=config.threshold_quantiles,
            ),
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
            candidates=make_quantile_candidates(
                train_details.predicted_losses[:, :-1] - train_details.predicted_losses[:, 1:],
                quantiles=config.epsilon_quantiles,
            ),
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

    write_tsv(output_dir / "threshold_sweep.tsv", threshold_rows, value_key="threshold")
    write_tsv(output_dir / "relative_improvement_sweep.tsv", improvement_rows, value_key="improvement_epsilon")

    prompt = split.train_text[: config.context_size]
    sample = generate_text(
        model,
        split.train_dataset,
        prompt,
        length=config.sample_length,
        device=device,
    )
    write_json(output_dir / "training_history.json", history)
    best_epoch_record = min(history[1:] or history, key=lambda record: float(record["val_fixed_depth_loss"]))
    write_json(
        output_dir / "final_metrics.json",
        {
            "parameter_count": parameter_count,
            "runtime_seconds": runtime_seconds,
            "final_val_loss": final_val_fixed["loss"],
            "final_val_accuracy": final_val_fixed["accuracy"],
            "best_val_loss": float(best_epoch_record["val_fixed_depth_loss"]),
            "best_val_accuracy": float(best_epoch_record["val_fixed_depth_accuracy"]),
            "best_epoch": int(best_epoch_record["epoch"]),
            "prompt": prompt.replace("\n", "\\n"),
            "fixed_depth_max_val": summarize_fixed_depth(val_details, depth=config.max_depth),
            "threshold_recommended_val": summarize_dynamic_depth(
                val_details,
                threshold=float(threshold_recommended["threshold"]),
                improvement_epsilon=None,
            ),
            "relative_improvement_recommended_val": summarize_dynamic_depth(
                val_details,
                threshold=None,
                improvement_epsilon=float(improvement_recommended["improvement_epsilon"]),
            ),
        },
    )
    write_json(
        output_dir / "sweep_summary.json",
        {
            "fixed_depth_max_train": summarize_fixed_depth(train_details, depth=config.max_depth),
            "fixed_depth_max_val": summarize_fixed_depth(val_details, depth=config.max_depth),
            "calibration": {
                "train": calibration_summary(train_details),
                "val": calibration_summary(val_details),
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
    (output_dir / "sample.txt").write_text(sample, encoding="utf-8")


if __name__ == "__main__":
    main()
