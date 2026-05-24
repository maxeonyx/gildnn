from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

def _load_repo_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name!r} from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


try:
    from core.fixed_window_char import load_dataset, set_seed
    from core.model import ParallelDiagonalForwardState, ParallelDiagonalModel, count_parameters
    from core.training import batched_pairs, fixed_step_indices
except ModuleNotFoundError:
    fixed_window_char = _load_repo_module("propagation_delay_fixed_window_char", "core/fixed_window_char.py")
    model_module = _load_repo_module("propagation_delay_model", "core/model.py")
    training_module = _load_repo_module("propagation_delay_training", "core/training.py")

    load_dataset = fixed_window_char.load_dataset
    set_seed = fixed_window_char.set_seed
    ParallelDiagonalForwardState = model_module.ParallelDiagonalForwardState
    ParallelDiagonalModel = model_module.ParallelDiagonalModel
    count_parameters = model_module.count_parameters
    batched_pairs = training_module.batched_pairs
    fixed_step_indices = training_module.fixed_step_indices


ConditionName = Literal["single_block", "full_backprop", "stop_gradient"]


@dataclass(frozen=True)
class ExperimentConfig:
    context_size: int = 32
    d_model: int = 72
    feedforward_dim: int = 288
    batch_size: int = 64
    eval_batch_size: int = 512
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    training_steps: int = 3_000
    eval_interval: int = 250
    train_characters: int = 100_000
    val_characters: int = 20_000


@dataclass(frozen=True)
class ConditionSpec:
    name: ConditionName
    num_blocks: int
    topology: str
    readout_mode: str
    token_injection: str
    internal_steps: int
    detach_lateral: bool
    local_loss_weight: float | None


@dataclass(frozen=True)
class LossSnapshot:
    task_loss: Float[Tensor, ""]
    total_loss: Float[Tensor, ""]
    accuracy: float
    local_loss: Float[Tensor, ""] | None


CONDITIONS: dict[ConditionName, ConditionSpec] = {
    "single_block": ConditionSpec(
        name="single_block",
        num_blocks=1,
        topology="upward",
        readout_mode="last",
        token_injection="block0",
        internal_steps=1,
        detach_lateral=False,
        local_loss_weight=None,
    ),
    "full_backprop": ConditionSpec(
        name="full_backprop",
        num_blocks=2,
        topology="top_down_to_first",
        readout_mode="first",
        token_injection="block0",
        internal_steps=1,
        detach_lateral=False,
        local_loss_weight=0.0,
    ),
    "stop_gradient": ConditionSpec(
        name="stop_gradient",
        num_blocks=2,
        topology="top_down_to_first",
        readout_mode="first",
        token_injection="block0",
        internal_steps=1,
        detach_lateral=True,
        local_loss_weight=1.0,
    ),
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=tuple(CONDITIONS.keys()), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=ExperimentConfig.training_steps)
    parser.add_argument("--eval-interval", type=int, default=ExperimentConfig.eval_interval)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
    )
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def resolve_device(*, no_cuda: bool) -> torch.device:
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def default_log_file(*, repo_root: Path, condition: ConditionName, seed: int, steps: int) -> Path:
    return (
        repo_root
        / "experiments"
        / "propagation-delay"
        / "artifacts.ignore"
        / f"{condition}-seed{seed}-steps{steps}.jsonl"
    )


def maybe_round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def jsonl_append(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def build_model_and_local_head(
    *,
    vocab_size: int,
    spec: ConditionSpec,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[ParallelDiagonalModel, nn.Linear | None]:
    model = ParallelDiagonalModel(
        vocab_size=vocab_size,
        context_size=config.context_size,
        d_model=config.d_model,
        feedforward_dim=config.feedforward_dim,
        num_blocks=spec.num_blocks,
        internal_steps=spec.internal_steps,
        readout_mode=spec.readout_mode,
        token_injection=spec.token_injection,
        topology=spec.topology,
        detach_lateral=spec.detach_lateral,
    ).to(device)
    local_head = None
    if spec.local_loss_weight is not None:
        local_head = nn.Linear(config.d_model, vocab_size).to(device)
    return model, local_head


def local_head_logits(
    local_head: nn.Linear | None,
    state: ParallelDiagonalForwardState,
) -> Float[Tensor, "batch vocab"] | None:
    if local_head is None:
        return None
    block_b_final_state = state.block_outputs[1][:, -1, :]
    return local_head(block_b_final_state)


def compute_loss_snapshot(
    *,
    logits: Float[Tensor, "batch vocab"],
    targets: Int[Tensor, "batch"],
    local_logits: Float[Tensor, "batch vocab"] | None,
    local_loss_weight: float | None,
) -> LossSnapshot:
    task_loss = F.cross_entropy(logits, targets)
    local_loss = F.cross_entropy(local_logits, targets) if local_logits is not None else None
    if local_loss is None or local_loss_weight is None:
        total_loss = task_loss
    else:
        total_loss = task_loss + (local_loss_weight * local_loss)
    accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    return LossSnapshot(
        task_loss=task_loss,
        total_loss=total_loss,
        accuracy=accuracy,
        local_loss=local_loss,
    )


def forward_with_optional_ablation(
    *,
    model: ParallelDiagonalModel,
    tokens: Int[Tensor, "batch context"],
    batch_shuffle_ablation: bool,
    shuffle_generator: torch.Generator | None,
) -> tuple[Float[Tensor, "batch vocab"], ParallelDiagonalForwardState]:
    if not batch_shuffle_ablation:
        return model.forward_with_state(tokens)
    if model.num_blocks != 2:
        raise ValueError("Batch-shuffle ablation requires a 2-block model.")
    if model.topology != "top_down_to_first":
        raise ValueError("Batch-shuffle ablation requires topology='top_down_to_first'.")
    if model.internal_steps != 1:
        raise ValueError("Batch-shuffle ablation requires internal_steps=1.")
    if model.temporal_window != 0:
        raise ValueError("Batch-shuffle ablation helper only supports temporal_window=0.")

    embeddings = model.embedded_tokens(tokens)
    batch_size = tokens.shape[0]
    previous_states = [
        torch.zeros(batch_size, model.d_model, device=tokens.device, dtype=embeddings.dtype)
        for _ in range(model.num_blocks)
    ]
    block_output_history: list[list[Tensor]] = [[] for _ in range(model.num_blocks)]

    for time_index in range(model.context_size):
        token_state = embeddings[:, time_index, :]
        seeded_states = list(previous_states)
        seeded_states[0] = model.token_mixes[0](previous_states[0], token_state)

        shuffled_indices = torch.randperm(batch_size, generator=shuffle_generator)
        shuffled_indices = shuffled_indices.to(device=tokens.device)
        lateral_to_block0 = model._maybe_detach_lateral(previous_states[1])[shuffled_indices]
        block0_input = 0.5 * (seeded_states[0] + lateral_to_block0)
        block0_delta = model.blocks[0](block0_input)
        next_block0 = model.block_mixes[0](block0_input, block0_delta)

        lateral_to_block1 = model._maybe_detach_lateral(previous_states[0])
        block1_input = 0.5 * (seeded_states[1] + lateral_to_block1)
        block1_delta = model.blocks[1](block1_input)
        next_block1 = model.block_mixes[1](block1_input, block1_delta)

        previous_states = [next_block0, next_block1]
        for block_index, state in enumerate(previous_states):
            block_output_history[block_index].append(state)

    logits = model.output(model._readout_state(previous_states))
    return logits, ParallelDiagonalForwardState(
        block_outputs=[torch.stack(history, dim=1) for history in block_output_history]
    )


@torch.inference_mode()
def evaluate_condition(
    *,
    model: ParallelDiagonalModel,
    local_head: nn.Linear | None,
    inputs: Int[Tensor, "examples context"],
    targets: Int[Tensor, "examples"],
    batch_size: int,
    local_loss_weight: float | None,
    batch_shuffle_ablation: bool = False,
    shuffle_seed: int | None = None,
) -> dict[str, float | None]:
    was_training_model = model.training
    was_training_head = local_head.training if local_head is not None else False
    model.eval()
    if local_head is not None:
        local_head.eval()

    generator = None
    if batch_shuffle_ablation:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0 if shuffle_seed is None else shuffle_seed)

    total_examples = 0
    total_task_loss = 0.0
    total_local_loss = 0.0
    total_correct = 0

    for batch_inputs, batch_targets in batched_pairs(inputs, targets, batch_size=batch_size):
        logits, state = forward_with_optional_ablation(
            model=model,
            tokens=batch_inputs,
            batch_shuffle_ablation=batch_shuffle_ablation,
            shuffle_generator=generator,
        )
        local_logits = local_head_logits(local_head, state)
        batch_examples = batch_targets.shape[0]
        total_examples += batch_examples
        total_task_loss += F.cross_entropy(logits, batch_targets, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
        if local_logits is not None:
            total_local_loss += F.cross_entropy(local_logits, batch_targets, reduction="sum").item()

    if was_training_model:
        model.train()
    if local_head is not None and was_training_head:
        local_head.train()

    task_loss = total_task_loss / total_examples
    local_loss = (total_local_loss / total_examples) if local_head is not None else None
    total_loss = task_loss if local_loss is None or local_loss_weight is None else task_loss + (local_loss_weight * local_loss)
    accuracy = total_correct / total_examples
    return {
        "task_loss": task_loss,
        "local_loss": local_loss,
        "total_loss": total_loss,
        "accuracy": accuracy,
    }


def main() -> int:
    args = parse_args()
    spec = CONDITIONS[args.condition]
    config = ExperimentConfig(training_steps=args.steps, eval_interval=args.eval_interval)
    if config.training_steps <= 0:
        raise ValueError(f"steps must be positive, got {config.training_steps}.")
    if config.eval_interval <= 0:
        raise ValueError(f"eval-interval must be positive, got {config.eval_interval}.")

    device = resolve_device(no_cuda=args.no_cuda)
    set_seed(args.seed)

    log_file = args.log_file or default_log_file(
        repo_root=args.repo_root,
        condition=spec.name,
        seed=args.seed,
        steps=config.training_steps,
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.unlink()

    (train_data, val_data, vocab_size) = load_dataset(
        context_size=config.context_size,
        train_characters=config.train_characters,
        val_characters=config.val_characters,
    )
    train_inputs, train_targets = train_data
    val_inputs, val_targets = val_data
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    val_inputs = val_inputs.to(device)
    val_targets = val_targets.to(device)

    model, local_head = build_model_and_local_head(
        vocab_size=vocab_size,
        spec=spec,
        config=config,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        chain(model.parameters(), [] if local_head is None else local_head.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    batch_schedule = fixed_step_indices(
        train_inputs.shape[0],
        steps=config.training_steps,
        batch_size=config.batch_size,
        seed=args.seed,
        device=train_inputs.device,
    )

    started_at = time.perf_counter()
    final_eval_metrics: dict[str, float | None] | None = None

    for step, batch_indices in enumerate(batch_schedule, start=1):
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]

        logits, state = model.forward_with_state(batch_inputs)
        local_logits = local_head_logits(local_head, state)
        snapshot = compute_loss_snapshot(
            logits=logits,
            targets=batch_targets,
            local_logits=local_logits,
            local_loss_weight=spec.local_loss_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        snapshot.total_loss.backward()
        optimizer.step()

        if step % config.eval_interval == 0 or step == config.training_steps:
            final_eval_metrics = evaluate_condition(
                model=model,
                local_head=local_head,
                inputs=val_inputs,
                targets=val_targets,
                batch_size=config.eval_batch_size,
                local_loss_weight=spec.local_loss_weight,
            )
            elapsed_sec = time.perf_counter() - started_at
            record = {
                "event": "eval",
                "step": step,
                "train_loss": round(snapshot.task_loss.item(), 6),
                "train_total_loss": round(snapshot.total_loss.item(), 6),
                "train_local_loss": maybe_round(snapshot.local_loss.item() if snapshot.local_loss is not None else None),
                "train_accuracy": round(snapshot.accuracy, 6),
                "val_loss": round(float(final_eval_metrics["task_loss"]), 6),
                "val_total_loss": round(float(final_eval_metrics["total_loss"]), 6),
                "val_local_loss": maybe_round(
                    None if final_eval_metrics["local_loss"] is None else float(final_eval_metrics["local_loss"])
                ),
                "val_accuracy": round(float(final_eval_metrics["accuracy"]), 6),
                "elapsed_sec": round(elapsed_sec, 3),
            }
            jsonl_append(log_file, record)
            print(json.dumps(record), flush=True)

    if final_eval_metrics is None:
        raise RuntimeError("Training finished without any evaluation record.")

    ablation_record: dict[str, object]
    if spec.num_blocks == 1:
        ablation_record = {
            "event": "ablation",
            "step": config.training_steps,
            "baseline_val_loss": round(float(final_eval_metrics["task_loss"]), 6),
            "ablated_val_loss": None,
            "ablation_effect_nats": None,
            "elapsed_sec": round(time.perf_counter() - started_at, 3),
            "note": "single_block has no block-B lateral stream to shuffle",
        }
    else:
        ablated_metrics = evaluate_condition(
            model=model,
            local_head=local_head,
            inputs=val_inputs,
            targets=val_targets,
            batch_size=config.eval_batch_size,
            local_loss_weight=spec.local_loss_weight,
            batch_shuffle_ablation=True,
            shuffle_seed=args.seed + 10_000,
        )
        baseline_val_loss = float(final_eval_metrics["task_loss"])
        ablated_val_loss = float(ablated_metrics["task_loss"])
        ablation_record = {
            "event": "ablation",
            "step": config.training_steps,
            "baseline_val_loss": round(baseline_val_loss, 6),
            "ablated_val_loss": round(ablated_val_loss, 6),
            "ablation_effect_nats": round(ablated_val_loss - baseline_val_loss, 6),
            "baseline_val_local_loss": maybe_round(
                None if final_eval_metrics["local_loss"] is None else float(final_eval_metrics["local_loss"])
            ),
            "ablated_val_local_loss": maybe_round(
                None if ablated_metrics["local_loss"] is None else float(ablated_metrics["local_loss"])
            ),
            "elapsed_sec": round(time.perf_counter() - started_at, 3),
        }
    jsonl_append(log_file, ablation_record)
    print(json.dumps(ablation_record), flush=True)

    parameter_count = count_parameters(model) + (0 if local_head is None else count_parameters(local_head))
    summary = {
        "condition": spec.name,
        "seed": args.seed,
        "steps": config.training_steps,
        "device": str(device),
        "parameter_count": parameter_count,
        "final_val_loss": round(float(final_eval_metrics["task_loss"]), 6),
        "final_val_accuracy": round(float(final_eval_metrics["accuracy"]), 6),
        "final_val_local_loss": maybe_round(
            None if final_eval_metrics["local_loss"] is None else float(final_eval_metrics["local_loss"])
        ),
        "ablation": ablation_record,
        "log_file": str(log_file),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
