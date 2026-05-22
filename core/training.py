"""Shared training and evaluation utilities.

Functions here are used by multiple experiments. Each was duplicated
across 15+ experiment scripts before extraction.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import torch
from jaxtyping import Float, Int
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


def current_git_sha() -> str:
    """Return the current HEAD commit SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def current_git_status_short() -> list[str]:
    """Return `git status --short` output as a list of non-empty lines."""
    result = subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def capturable_adamw(
    model: nn.Module,
    *,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
) -> torch.optim.AdamW:
    """Create an AdamW optimizer configured for CUDA Graph capture."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        capturable=True,
    )


class GraphTrainer:
    """CUDA Graph training wrapper for fixed-shape language-model batches."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        resolved_device = device or next(model.parameters()).device
        if resolved_device.type != "cuda":
            raise ValueError(f"GraphTrainer requires a CUDA device, got {resolved_device}.")

        self.model = model
        self.optimizer = optimizer
        self.device = resolved_device
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.capture_stream = torch.cuda.Stream(device=self.device)
        self.graph = torch.cuda.CUDAGraph()
        self.static_input: Int[Tensor, "batch context"] = torch.empty(
            (batch_size, seq_len),
            device=self.device,
            dtype=torch.long,
        )
        self.static_target: Int[Tensor, "batch"] = torch.empty(
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        self.static_loss: Float[Tensor, ""] = torch.zeros((), device=self.device)
        self.is_captured = False

        self._validate_optimizer()

    def _validate_optimizer(self) -> None:
        if not isinstance(self.optimizer, torch.optim.AdamW):
            raise TypeError(
                f"GraphTrainer requires torch.optim.AdamW, got {type(self.optimizer).__name__}."
            )
        for group_index, group in enumerate(self.optimizer.param_groups):
            if group.get("capturable") is not True:
                raise ValueError(
                    "GraphTrainer requires AdamW(capturable=True); "
                    f"param group {group_index} has capturable={group.get('capturable')!r}."
                )

    def _validate_batch(
        self,
        batch_input: Int[Tensor, "batch context"],
        batch_target: Int[Tensor, "batch"],
    ) -> None:
        expected_input_shape = (self.batch_size, self.seq_len)
        expected_target_shape = (self.batch_size,)
        if tuple(batch_input.shape) != expected_input_shape:
            raise ValueError(
                f"Expected input batch shape {expected_input_shape}, got {tuple(batch_input.shape)}."
            )
        if tuple(batch_target.shape) != expected_target_shape:
            raise ValueError(
                f"Expected target batch shape {expected_target_shape}, got {tuple(batch_target.shape)}."
            )
        if batch_input.dtype != torch.long:
            raise ValueError(f"Expected input dtype torch.long, got {batch_input.dtype}.")
        if batch_target.dtype != torch.long:
            raise ValueError(f"Expected target dtype torch.long, got {batch_target.dtype}.")

    def _copy_batch(
        self,
        batch_input: Int[Tensor, "batch context"],
        batch_target: Int[Tensor, "batch"],
    ) -> None:
        self._validate_batch(batch_input, batch_target)
        self.static_input.copy_(batch_input, non_blocking=True)
        self.static_target.copy_(batch_target, non_blocking=True)

    def _training_step(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(self.static_input)
        loss = F.cross_entropy(logits, self.static_target)
        loss.backward()
        self.optimizer.step()
        self.static_loss.copy_(loss.detach())

    def capture(
        self,
        warmup_batches: Sequence[
            tuple[Int[Tensor, "batch context"], Int[Tensor, "batch"]]
        ],
    ) -> Float[Tensor, ""]:
        if self.is_captured:
            raise RuntimeError("GraphTrainer.capture() may only be called once.")
        if len(warmup_batches) < 3:
            raise ValueError(f"Graph capture requires at least 3 warmup batches, got {len(warmup_batches)}.")

        self.model.train()
        current_stream = torch.cuda.current_stream(device=self.device)
        self.capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.capture_stream):
            for batch_input, batch_target in warmup_batches[:3]:
                self._copy_batch(batch_input, batch_target)
                self._training_step()
        current_stream.wait_stream(self.capture_stream)
        torch.cuda.synchronize(self.device)

        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self._training_step()

        current_stream.wait_stream(self.capture_stream)
        self.is_captured = True
        return self.static_loss.detach().clone()

    def step(
        self,
        batch_input: Int[Tensor, "batch context"],
        batch_target: Int[Tensor, "batch"],
    ) -> Float[Tensor, ""]:
        if not self.is_captured:
            raise RuntimeError("GraphTrainer.step() requires capture() first.")
        self._copy_batch(batch_input, batch_target)
        self.graph.replay()
        return self.static_loss.detach().clone()

    def run_steps(
        self,
        batches: Sequence[tuple[Int[Tensor, "batch context"], Int[Tensor, "batch"]]],
        *,
        on_step: Callable[[int, Float[Tensor, ""]], None] | None = None,
        start_step: int = 0,
    ) -> list[Float[Tensor, ""]]:
        losses: list[Float[Tensor, ""]] = []
        for offset, (batch_input, batch_target) in enumerate(batches):
            loss = self.step(batch_input, batch_target)
            step = start_step + offset
            if on_step is not None:
                on_step(step, loss)
            losses.append(loss)
        return losses

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)


def write_json(path: Path, payload: object) -> None:
    """Write a JSON file with 2-space indent and trailing newline."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
