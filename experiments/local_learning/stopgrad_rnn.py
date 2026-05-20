from __future__ import annotations

import argparse
import hashlib
import json
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
    count_parameters,
    current_git_sha,
    current_git_status_short,
    write_json,
)
from core.fixed_window_char import FixedWindowCharDataset, generate_text, resolve_device, set_seed


MODULE_DEPTH = 3
DEFAULT_HIDDEN_DIMS = {1: 172, 2: 115, 4: 80}


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
    embedding_dim: int = 64
    module_count: int = 1
    hidden_dim: int = 172
    auxiliary_weight: float = 0.03
    module_depth: int = MODULE_DEPTH
    sample_length: int = 320
    seed: int = 42
    check_batch_size: int = 8
    memorization_batch_size: int = 32
    memorization_steps: int = 300
    memorization_learning_rate: float = 0.01


@dataclass(frozen=True)
class Rollout:
    embeddings: Tensor
    module_inputs: list[Tensor]
    module_outputs: list[Tensor]
    local_predictions: list[Tensor]


@dataclass(frozen=True)
class LossBundle:
    total_loss: Tensor
    task_loss: Tensor
    local_loss_total: Tensor
    local_losses: list[Tensor]
    logits: Tensor
    accuracy: float


def replace_config(config: RunConfig, **changes: object) -> RunConfig:
    payload = asdict(config)
    payload.update(changes)
    return RunConfig(**payload)


def default_hidden_dim(module_count: int) -> int:
    if module_count not in DEFAULT_HIDDEN_DIMS:
        raise ValueError(
            f"No default hidden width for module_count={module_count}. "
            f"Known counts: {sorted(DEFAULT_HIDDEN_DIMS)}"
        )
    return DEFAULT_HIDDEN_DIMS[module_count]


def variant_label(module_count: int) -> str:
    return f"modules_{module_count}"


class LocalRecurrentModule(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self.layers = nn.ModuleList(
            [
                nn.RNNCell(
                    input_size=input_dim if layer_index == 0 else hidden_dim,
                    hidden_size=hidden_dim,
                    nonlinearity="tanh",
                )
                for layer_index in range(depth)
            ]
        )
        self.local_head = nn.Linear(hidden_dim, input_dim)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def initial_state(self, batch_size: int, *, device: torch.device) -> list[Tensor]:
        return [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in self.layers]

    def step(self, inputs: Tensor, state: list[Tensor]) -> tuple[Tensor, list[Tensor], Tensor]:
        next_state: list[Tensor] = []
        layer_input = inputs
        for layer, hidden in zip(self.layers, state, strict=True):
            layer_output = layer(layer_input, hidden)
            next_state.append(layer_output)
            layer_input = layer_output
        module_output = next_state[-1]
        local_prediction = self.local_head(module_output)
        return module_output, next_state, local_prediction


class LocalLearningRnn(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        module_count: int,
        module_depth: int,
    ) -> None:
        super().__init__()
        if module_count < 1:
            raise ValueError(f"module_count must be at least 1, got {module_count}")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.modules_stack = nn.ModuleList(
            [
                LocalRecurrentModule(
                    input_dim=embedding_dim if module_index == 0 else hidden_dim,
                    hidden_dim=hidden_dim,
                    depth=module_depth,
                )
                for module_index in range(module_count)
            ]
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.module_count = module_count
        self.module_depth = module_depth

    def embedded_inputs(self, tokens: Tensor) -> Tensor:
        return torch.tanh(self.embedding(tokens))

    def rollout(self, tokens: Tensor) -> Rollout:
        embeddings = self.embedded_inputs(tokens)
        batch_size, sequence_length, _ = embeddings.shape
        states = [
            module.initial_state(batch_size, device=tokens.device)
            for module in self.modules_stack
        ]
        module_input_steps = [[] for _ in range(self.module_count)]
        module_output_steps = [[] for _ in range(self.module_count)]
        local_prediction_steps = [[] for _ in range(self.module_count)]

        for step in range(sequence_length):
            current_input = embeddings[:, step, :]
            for module_index, module in enumerate(self.modules_stack):
                if module_index > 0:
                    current_input = current_input.detach()
                module_input_steps[module_index].append(current_input)
                module_output, states[module_index], local_prediction = module.step(
                    current_input,
                    states[module_index],
                )
                module_output_steps[module_index].append(module_output)
                local_prediction_steps[module_index].append(local_prediction)
                current_input = module_output

        return Rollout(
            embeddings=embeddings,
            module_inputs=[torch.stack(steps, dim=1) for steps in module_input_steps],
            module_outputs=[torch.stack(steps, dim=1) for steps in module_output_steps],
            local_predictions=[torch.stack(steps, dim=1) for steps in local_prediction_steps],
        )

    def forward(self, tokens: Tensor) -> Tensor:
        rollout = self.rollout(tokens)
        final_representation = rollout.module_outputs[-1][:, -1, :]
        return self.lm_head(final_representation)


def make_model(config: RunConfig, *, vocab_size: int, device: torch.device) -> LocalLearningRnn:
    return LocalLearningRnn(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        module_count=config.module_count,
        module_depth=config.module_depth,
    ).to(device)


def full_tokens(inputs: Tensor, targets: Tensor) -> Tensor:
    return torch.cat([inputs, targets.unsqueeze(1)], dim=1)


def compute_loss_bundle(
    model: LocalLearningRnn,
    inputs: Tensor,
    targets: Tensor,
    *,
    auxiliary_weight: float,
) -> LossBundle:
    sequence = full_tokens(inputs, targets)
    rollout = model.rollout(sequence)
    task_logits = model.lm_head(rollout.module_outputs[-1][:, inputs.shape[1] - 1, :])
    task_loss = F.cross_entropy(task_logits, targets)
    local_losses = [
        F.mse_loss(
            prediction[:, :-1, :],
            module_inputs[:, 1:, :].detach(),
        )
        for prediction, module_inputs in zip(
            rollout.local_predictions,
            rollout.module_inputs,
            strict=True,
        )
    ]
    local_loss_total = torch.stack(local_losses).sum()
    total_loss = task_loss + auxiliary_weight * local_loss_total
    accuracy = (task_logits.argmax(dim=1) == targets).float().mean().item()
    return LossBundle(
        total_loss=total_loss,
        task_loss=task_loss,
        local_loss_total=local_loss_total,
        local_losses=local_losses,
        logits=task_logits,
        accuracy=accuracy,
    )


def batched_pairs(inputs: Tensor, targets: Tensor, *, batch_size: int):
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        yield inputs[start:stop], targets[start:stop]


def module_grad_norm(module: LocalRecurrentModule) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        total += parameter.grad.detach().float().pow(2).sum().item()
    return total**0.5


def linear_grad_norm(layer: nn.Linear) -> float:
    total = 0.0
    for parameter in layer.parameters():
        if parameter.grad is None:
            continue
        total += parameter.grad.detach().float().pow(2).sum().item()
    return total**0.5


def gradient_snapshot(model: LocalLearningRnn) -> dict[str, object]:
    return {
        "modules": [
            {
                "module_index": module_index + 1,
                "grad_norm": module_grad_norm(module),
                "local_head_grad_norm": linear_grad_norm(module.local_head),
            }
            for module_index, module in enumerate(model.modules_stack)
        ],
        "lm_head_grad_norm": linear_grad_norm(model.lm_head),
    }


def assert_zero(value: float, *, label: str, tolerance: float = 1e-12) -> None:
    if abs(value) > tolerance:
        raise RuntimeError(f"Expected zero gradient at {label}, got {value:.12f}")


def assert_positive(value: float, *, label: str, threshold: float = 1e-9) -> None:
    if value <= threshold:
        raise RuntimeError(f"Expected positive gradient at {label}, got {value:.12f}")


def gradient_boundary_checks(
    split,
    *,
    config: RunConfig,
    device: torch.device,
) -> dict[str, object]:
    set_seed(config.seed)
    model = make_model(config, vocab_size=split.train_dataset.vocab_size, device=device)
    batch_inputs = split.train_inputs[: config.check_batch_size].to(device)
    batch_targets = split.train_targets[: config.check_batch_size].to(device)
    bundle = compute_loss_bundle(
        model,
        batch_inputs,
        batch_targets,
        auxiliary_weight=config.auxiliary_weight,
    )

    checks: dict[str, object] = {}
    if config.module_count == 1:
        model.zero_grad(set_to_none=True)
        bundle.local_losses[0].backward(retain_graph=True)
        local_snapshot = gradient_snapshot(model)
        assert_positive(local_snapshot["modules"][0]["grad_norm"], label="single_module_local_loss")
        assert_positive(
            local_snapshot["modules"][0]["local_head_grad_norm"],
            label="single_module_local_head_local_loss",
        )
        assert_zero(local_snapshot["lm_head_grad_norm"], label="single_module_local_loss_to_lm_head")
        checks["single_module_local_loss"] = local_snapshot

        model.zero_grad(set_to_none=True)
        bundle.task_loss.backward()
        task_snapshot = gradient_snapshot(model)
        assert_positive(task_snapshot["modules"][0]["grad_norm"], label="single_module_task_loss")
        assert_positive(task_snapshot["lm_head_grad_norm"], label="single_module_task_loss_to_lm_head")
        assert_zero(
            task_snapshot["modules"][0]["local_head_grad_norm"],
            label="single_module_task_loss_to_local_head",
        )
        checks["single_module_task_loss"] = task_snapshot
        return checks

    local_loss_checks = []
    for module_index in range(1, config.module_count):
        model.zero_grad(set_to_none=True)
        bundle.local_losses[module_index].backward(retain_graph=True)
        snapshot = gradient_snapshot(model)
        assert_zero(
            snapshot["modules"][module_index - 1]["grad_norm"],
            label=f"module_{module_index + 1}_local_loss_to_previous_module",
        )
        assert_positive(
            snapshot["modules"][module_index]["grad_norm"],
            label=f"module_{module_index + 1}_local_loss_to_own_module",
        )
        assert_positive(
            snapshot["modules"][module_index]["local_head_grad_norm"],
            label=f"module_{module_index + 1}_local_loss_to_own_head",
        )
        local_loss_checks.append(
            {
                "local_loss_module_index": module_index + 1,
                "snapshot": snapshot,
            }
        )
    checks["downstream_local_losses"] = local_loss_checks

    model.zero_grad(set_to_none=True)
    bundle.task_loss.backward()
    task_snapshot = gradient_snapshot(model)
    for module_index in range(config.module_count - 1):
        assert_zero(
            task_snapshot["modules"][module_index]["grad_norm"],
            label=f"task_loss_to_module_{module_index + 1}",
        )
        assert_zero(
            task_snapshot["modules"][module_index]["local_head_grad_norm"],
            label=f"task_loss_to_module_{module_index + 1}_local_head",
        )
    assert_positive(
        task_snapshot["modules"][-1]["grad_norm"],
        label="task_loss_to_top_module",
    )
    assert_zero(
        task_snapshot["modules"][-1]["local_head_grad_norm"],
        label="task_loss_to_top_module_local_head",
    )
    assert_positive(task_snapshot["lm_head_grad_norm"], label="task_loss_to_lm_head")
    checks["task_loss"] = task_snapshot
    return checks


def evaluate_model(
    model: LocalLearningRnn,
    inputs: Tensor,
    targets: Tensor,
    *,
    auxiliary_weight: float,
    batch_size: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_examples = 0
    total_total_loss = 0.0
    total_task_loss = 0.0
    total_local_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in batched_pairs(inputs, targets, batch_size=batch_size):
            bundle = compute_loss_bundle(
                model,
                batch_inputs,
                batch_targets,
                auxiliary_weight=auxiliary_weight,
            )
            batch_examples = batch_targets.shape[0]
            total_examples += batch_examples
            total_total_loss += bundle.total_loss.item() * batch_examples
            total_task_loss += bundle.task_loss.item() * batch_examples
            total_local_loss += bundle.local_loss_total.item() * batch_examples
            total_correct += (bundle.logits.argmax(dim=1) == batch_targets).sum().item()
    if was_training:
        model.train()
    return {
        "total_loss": total_total_loss / total_examples,
        "task_loss": total_task_loss / total_examples,
        "local_loss_total": total_local_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_one_epoch(
    model: LocalLearningRnn,
    optimizer: torch.optim.Optimizer,
    train_inputs: Tensor,
    train_targets: Tensor,
    *,
    batch_size: int,
    auxiliary_weight: float,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    permutation = torch.randperm(train_inputs.shape[0], device=train_inputs.device)
    total_examples = 0
    total_total_loss = 0.0
    total_task_loss = 0.0
    total_local_loss = 0.0
    total_correct = 0

    for start in range(0, permutation.shape[0], batch_size):
        batch_indices = permutation[start : start + batch_size]
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]
        bundle = compute_loss_bundle(
            model,
            batch_inputs,
            batch_targets,
            auxiliary_weight=auxiliary_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        bundle.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        batch_examples = batch_targets.shape[0]
        total_examples += batch_examples
        total_total_loss += bundle.total_loss.item() * batch_examples
        total_task_loss += bundle.task_loss.item() * batch_examples
        total_local_loss += bundle.local_loss_total.item() * batch_examples
        total_correct += (bundle.logits.argmax(dim=1) == batch_targets).sum().item()

    return {
        "total_loss": total_total_loss / total_examples,
        "task_loss": total_task_loss / total_examples,
        "local_loss_total": total_local_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def capture_sample(
    model: LocalLearningRnn,
    dataset: FixedWindowCharDataset,
    prompt: str,
    *,
    length: int,
    device: torch.device,
) -> str:
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        sample = generate_text(model, dataset, prompt, length=length, device=device)
    if was_training:
        model.train()
    return sample


def train_model(
    model: LocalLearningRnn,
    *,
    dataset: FixedWindowCharDataset,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    config: RunConfig,
    device: torch.device,
) -> tuple[list[dict[str, float | int]], dict[str, float], float, str]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []
    started_at = time.perf_counter()

    initial_train_metrics = evaluate_model(
        model,
        train_inputs,
        train_targets,
        auxiliary_weight=config.auxiliary_weight,
        batch_size=config.eval_batch_size,
    )
    initial_val_metrics = evaluate_model(
        model,
        val_inputs,
        val_targets,
        auxiliary_weight=config.auxiliary_weight,
        batch_size=config.eval_batch_size,
    )
    history.append(
        {
            "epoch": 0,
            "train_total_loss": round(initial_train_metrics["total_loss"], 6),
            "train_task_loss": round(initial_train_metrics["task_loss"], 6),
            "train_local_loss_total": round(initial_train_metrics["local_loss_total"], 6),
            "train_accuracy": round(initial_train_metrics["accuracy"], 6),
            "val_total_loss": round(initial_val_metrics["total_loss"], 6),
            "val_task_loss": round(initial_val_metrics["task_loss"], 6),
            "val_local_loss_total": round(initial_val_metrics["local_loss_total"], 6),
            "val_accuracy": round(initial_val_metrics["accuracy"], 6),
        }
    )
    print(
        f"epoch=0 train_task_loss={initial_train_metrics['task_loss']:.4f} "
        f"val_task_loss={initial_val_metrics['task_loss']:.4f}"
    )

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_inputs,
            train_targets,
            batch_size=config.batch_size,
            auxiliary_weight=config.auxiliary_weight,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        val_metrics = evaluate_model(
            model,
            val_inputs,
            val_targets,
            auxiliary_weight=config.auxiliary_weight,
            batch_size=config.eval_batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": round(train_metrics["total_loss"], 6),
                "train_task_loss": round(train_metrics["task_loss"], 6),
                "train_local_loss_total": round(train_metrics["local_loss_total"], 6),
                "train_accuracy": round(train_metrics["accuracy"], 6),
                "val_total_loss": round(val_metrics["total_loss"], 6),
                "val_task_loss": round(val_metrics["task_loss"], 6),
                "val_local_loss_total": round(val_metrics["local_loss_total"], 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
            }
        )
        print(
            f"epoch={epoch} train_task_loss={train_metrics['task_loss']:.4f} "
            f"val_task_loss={val_metrics['task_loss']:.4f}"
        )

    runtime_seconds = time.perf_counter() - started_at
    final_metrics = evaluate_model(
        model,
        val_inputs,
        val_targets,
        auxiliary_weight=config.auxiliary_weight,
        batch_size=config.eval_batch_size,
    )
    prompt = dataset.text[: config.context_size]
    sample = capture_sample(model, dataset, prompt, length=config.sample_length, device=device)
    return history, final_metrics, runtime_seconds, sample


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
    logits = shape_model(batch_inputs)
    expected_shape = [config.check_batch_size, split.train_dataset.vocab_size]
    actual_shape = list(logits.shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"Forward output shape mismatch: expected {expected_shape}, got {actual_shape}."
        )
    checks.append(
        {
            "name": "forward_shape",
            "ok": True,
            "expected_shape": expected_shape,
            "actual_shape": actual_shape,
        }
    )

    bundle = compute_loss_bundle(
        shape_model,
        batch_inputs,
        batch_targets,
        auxiliary_weight=config.auxiliary_weight,
    )
    tiny_loss = bundle.task_loss.item()
    expected_low = math.log(split.train_dataset.vocab_size) - 1.0
    expected_high = math.log(split.train_dataset.vocab_size) + 1.0
    if not expected_low <= tiny_loss <= expected_high:
        raise RuntimeError(
            f"Known tiny-batch task loss {tiny_loss:.6f} fell outside expected range "
            f"[{expected_low:.6f}, {expected_high:.6f}]."
        )
    checks.append(
        {
            "name": "known_tiny_batch_loss",
            "ok": True,
            "task_loss": tiny_loss,
            "expected_range": [expected_low, expected_high],
        }
    )

    rollout = shape_model.rollout(full_tokens(batch_inputs, batch_targets))
    target_shape_checks = []
    for module_index, (prediction, module_inputs) in enumerate(
        zip(rollout.local_predictions, rollout.module_inputs, strict=True)
    ):
        prediction_shape = list(prediction[:, :-1, :].shape)
        target_shape = list(module_inputs[:, 1:, :].shape)
        if prediction_shape != target_shape:
            raise RuntimeError(
                f"Local target shape mismatch in module {module_index + 1}: "
                f"prediction {prediction_shape} vs target {target_shape}."
            )
        target_shape_checks.append(
            {
                "module_index": module_index + 1,
                "prediction_shape": prediction_shape,
                "target_shape": target_shape,
            }
        )
    checks.append(
        {
            "name": "local_target_shapes",
            "ok": True,
            "modules": target_shape_checks,
        }
    )

    expected_total = bundle.task_loss.item() + config.auxiliary_weight * bundle.local_loss_total.item()
    if not math.isclose(bundle.total_loss.item(), expected_total, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError("Total loss does not match task + weighted local losses.")
    checks.append(
        {
            "name": "loss_path_wiring",
            "ok": True,
            "task_loss": bundle.task_loss.item(),
            "local_loss_total": bundle.local_loss_total.item(),
            "weighted_total_loss": bundle.total_loss.item(),
        }
    )

    checks.append(
        {
            "name": "gradient_boundaries",
            "ok": True,
            "evidence": gradient_boundary_checks(split, config=config, device=device),
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
    final_bundle: LossBundle | None = None
    for step in range(config.memorization_steps + 1):
        bundle = compute_loss_bundle(
            memorization_model,
            memorize_inputs,
            memorize_targets,
            auxiliary_weight=config.auxiliary_weight,
        )
        if step in {0, config.memorization_steps} or step % 20 == 0:
            trace.append(
                {
                    "step": step,
                    "total_loss": round(bundle.total_loss.item(), 6),
                    "task_loss": round(bundle.task_loss.item(), 6),
                    "local_loss_total": round(bundle.local_loss_total.item(), 6),
                    "accuracy": round(bundle.accuracy, 6),
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
    if not (final_bundle.task_loss.item() < 0.05 and final_bundle.accuracy == 1.0):
        raise RuntimeError(
            "One-batch memorization check failed: expected task loss < 0.05 and accuracy 1.0, "
            f"got task_loss={final_bundle.task_loss.item():.6f}, accuracy={final_bundle.accuracy:.6f}."
        )
    checks.append(
        {
            "name": "one_batch_memorization",
            "ok": True,
            "final_total_loss": final_bundle.total_loss.item(),
            "final_task_loss": final_bundle.task_loss.item(),
            "final_local_loss_total": final_bundle.local_loss_total.item(),
            "final_accuracy": final_bundle.accuracy,
            "trace": trace,
        }
    )
    return checks


def make_default_paths(repo_root: Path, config: RunConfig, mode: str) -> tuple[Path, Path]:
    text_file = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    output_dir = repo_root / "experiments" / "local_learning" / "artifacts" / variant_label(config.module_count)
    if mode != "full":
        output_dir = output_dir / mode
    return text_file, output_dir


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["checks-only", "tiny", "full"], default="full")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--module-count", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--train-characters", type=int)
    parser.add_argument("--val-characters", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--auxiliary-weight", type=float)
    parser.add_argument("--sample-length", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--memorization-steps", type=int)
    parser.add_argument("--memorization-learning-rate", type=float)
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def resolved_config(args: argparse.Namespace) -> RunConfig:
    config = replace_config(
        RunConfig(),
        module_count=args.module_count,
        hidden_dim=args.hidden_dim or default_hidden_dim(args.module_count),
    )
    if args.mode == "tiny":
        config = replace_config(
            config,
            train_characters=4_096,
            val_characters=1_024,
            batch_size=128,
            eval_batch_size=256,
            epochs=4,
            sample_length=200,
        )
    if args.mode == "checks-only":
        config = replace_config(
            config,
            train_characters=2_048,
            val_characters=512,
            batch_size=128,
            eval_batch_size=256,
            epochs=0,
            sample_length=0,
        )

    overrides = {
        "hidden_dim": args.hidden_dim,
        "train_characters": args.train_characters,
        "val_characters": args.val_characters,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "auxiliary_weight": args.auxiliary_weight,
        "sample_length": args.sample_length,
        "seed": args.seed,
        "memorization_steps": args.memorization_steps,
        "memorization_learning_rate": args.memorization_learning_rate,
    }
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    if clean_overrides:
        config = replace_config(config, **clean_overrides)
    return config


def write_metadata(
    output_dir: Path,
    *,
    config: RunConfig,
    split,
    model: LocalLearningRnn,
    text_file: Path,
    raw_text: str,
    device: torch.device,
) -> None:
    git_status_short = current_git_status_short()
    parameter_count = count_parameters(model)
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
            "model_family": "local_learning_stopgrad_rnn",
            "parameter_count": parameter_count,
            "context_size": config.context_size,
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "module_count": config.module_count,
            "module_depth": config.module_depth,
            "module_input_dims": [module.input_dim for module in model.modules_stack],
            "local_head_targets": [
                "next_step_bounded_embedding" if index == 0 else f"next_step_detached_module_{index}_output"
                for index in range(config.module_count)
            ],
            "inter_module_boundary": "detach(previous_module_output_before_next_module_consumes_it)",
            "task_head": "linear(top_module_last_step_hidden -> vocab)",
            "hidden_state_handling": "zero_init_per_window_persistent_within_window_and_within_module",
        },
    )


def main() -> None:
    args = parse_args()
    config = resolved_config(args)
    default_text_file, default_output_dir = make_default_paths(args.repo_root, config, args.mode)
    text_file = args.text_file or default_text_file
    output_dir = args.output_dir or default_output_dir

    set_seed(config.seed)
    device = resolve_device(args.device)
    raw_text = text_file.read_text(encoding="utf-8")
    split = build_fixed_length_split(raw_text, config=config)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(config, vocab_size=split.train_dataset.vocab_size, device=device)
    write_metadata(
        output_dir,
        config=config,
        split=split,
        model=model,
        text_file=text_file,
        raw_text=raw_text,
        device=device,
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
    history, final_metrics, runtime_seconds, sample = train_model(
        model,
        dataset=split.train_dataset,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        config=config,
        device=device,
    )

    write_json(output_dir / "training_history.json", history)
    best_epoch_record = min(history[1:] or history, key=lambda record: float(record["val_task_loss"]))
    write_json(
        output_dir / "final_metrics.json",
        {
            "parameter_count": count_parameters(model),
            "runtime_seconds": runtime_seconds,
            "final_val_loss": final_metrics["task_loss"],
            "final_val_accuracy": final_metrics["accuracy"],
            "final_val_total_loss": final_metrics["total_loss"],
            "final_val_local_loss_total": final_metrics["local_loss_total"],
            "best_val_loss": float(best_epoch_record["val_task_loss"]),
            "best_val_accuracy": float(best_epoch_record["val_accuracy"]),
            "best_epoch": int(best_epoch_record["epoch"]),
            "prompt": split.train_text[: config.context_size].replace("\n", "\\n"),
        },
    )
    (output_dir / "sample.txt").write_text(sample, encoding="utf-8")
    (output_dir / "stdout_summary.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "variant": variant_label(config.module_count),
                "runtime_seconds": runtime_seconds,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
