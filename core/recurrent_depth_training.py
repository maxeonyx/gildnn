from __future__ import annotations

from contextlib import nullcontext
from time import perf_counter

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional as F

from core.recurrent_depth import RecurrentDepthLM


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def choose_depth_indices(
    predicted_gains: Float[Tensor, "batch depth"],
    *,
    epsilon: float,
    full_depth_index: int,
) -> Int[Tensor, "batch"]:
    early_halt_mask = predicted_gains < epsilon
    any_halt = early_halt_mask.any(dim=1)
    return torch.where(
        any_halt,
        early_halt_mask.float().argmax(dim=1),
        torch.full((predicted_gains.shape[0],), full_depth_index, dtype=torch.int64, device=predicted_gains.device),
    )


def compute_actual_gains(per_depth_losses: Float[Tensor, "batch depth"]) -> Float[Tensor, "batch depth_minus_one"]:
    return per_depth_losses[:, :-1] - per_depth_losses[:, -1:].expand(-1, per_depth_losses.shape[1] - 1)


def compute_per_depth_losses_and_predictions(
    model: RecurrentDepthLM,
    batch_inputs: Int[Tensor, "batch context"],
    batch_targets: Int[Tensor, "batch"],
) -> tuple[
    Float[Tensor, "batch depth"],
    Float[Tensor, "batch depth_minus_one"],
]:
    _, iteration_states = model.iteration_states(batch_inputs, collect_iteration_states=True)
    per_depth_losses: list[Tensor] = []
    halt_predictions: list[Tensor] = []
    for depth_index, iteration_hidden in enumerate(iteration_states, start=1):
        last_hidden = iteration_hidden[:, -1, :]
        logits = model.lm_logits_from_hidden(last_hidden)
        per_depth_losses.append(F.cross_entropy(logits.float(), batch_targets, reduction="none"))
        if depth_index < model.config.iterations:
            halt_predictions.append(model.predicted_gain_from_hidden(last_hidden, depth_index=depth_index))
    return torch.stack(per_depth_losses, dim=1), torch.stack(halt_predictions, dim=1)


@torch.inference_mode()
def evaluate_model(
    model: RecurrentDepthLM,
    *,
    val_inputs: Int[Tensor, "examples context"],
    val_targets: Int[Tensor, "examples"],
    eval_batch_size: int,
    halt_epsilon: float,
    device: torch.device,
) -> tuple[float, float, float, float]:
    del device
    started_at = perf_counter()
    model.eval()
    total_examples = 0
    total_val_loss = 0.0
    total_val_halt_loss = 0.0
    total_depth = 0.0
    full_depth_index = model.config.iterations - 1

    for start in range(0, val_targets.shape[0], eval_batch_size):
        stop = min(start + eval_batch_size, val_targets.shape[0])
        batch_inputs = val_inputs[start:stop]
        batch_targets = val_targets[start:stop]
        with autocast_context(val_inputs.device):
            per_depth_losses, halt_predictions = compute_per_depth_losses_and_predictions(model, batch_inputs, batch_targets)
        actual_gains = compute_actual_gains(per_depth_losses)
        depth_indices = choose_depth_indices(halt_predictions.float(), epsilon=halt_epsilon, full_depth_index=full_depth_index)
        total_examples += batch_targets.shape[0]
        total_val_loss += per_depth_losses[:, -1].sum().item()
        total_val_halt_loss += F.mse_loss(halt_predictions.float(), actual_gains.float(), reduction="sum").item()
        total_depth += (depth_indices + 1).float().sum().item()

    if total_examples == 0:
        raise RuntimeError("validation set was empty")
    halting_predictions = model.halting_depth_count * total_examples
    mean_halt_loss = 0.0 if halting_predictions == 0 else total_val_halt_loss / halting_predictions
    return (
        total_val_loss / total_examples,
        mean_halt_loss,
        total_depth / total_examples,
        perf_counter() - started_at,
    )


__all__ = [
    "autocast_context",
    "choose_depth_indices",
    "compute_actual_gains",
    "compute_per_depth_losses_and_predictions",
    "evaluate_model",
]
