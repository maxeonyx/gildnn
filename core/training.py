"""Shared training and evaluation utilities.

Functions here are used by multiple experiments. Each was duplicated
across 15+ experiment scripts before extraction.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def batched_pairs(
    inputs: Tensor, targets: Tensor, *, batch_size: int
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield (input_batch, target_batch) pairs from full tensors."""
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        yield inputs[start:stop], targets[start:stop]


def evaluate_model(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate cross-entropy loss and accuracy in batches.

    Handles eval/train mode switching. Returns dict with 'loss' and 'accuracy'.
    """
    was_training = model.training
    model.eval()
    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    with torch.inference_mode():
        for batch_inputs, batch_targets in batched_pairs(inputs, targets, batch_size=batch_size):
            logits = model(batch_inputs)
            batch_examples = batch_targets.shape[0]
            total_loss += F.cross_entropy(logits, batch_targets, reduction="sum").item()
            total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
            total_examples += batch_examples
    if was_training:
        model.train()
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def fixed_step_indices(
    dataset_size: int,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[Tensor]:
    """Create a reproducible random batch-index schedule.

    Returns a list of index tensors, one per training step. Each tensor
    has shape (batch_size,) with random indices into the dataset.
    Using a CPU generator ensures determinism regardless of target device.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return [
        torch.randint(0, dataset_size, (batch_size,), generator=generator).to(device)
        for _ in range(steps)
    ]
