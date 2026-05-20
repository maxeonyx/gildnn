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
    auxiliary_weight: float = 0.01
    detach_messages: bool = True
    sample_length: int = 320
    seed: int = 42
    check_batch_size: int = 8
    memorization_batch_size: int = 32
    memorization_steps: int = 200
    memorization_learning_rate: float = 0.02
    inspection_examples: int = 2


@dataclass(frozen=True)
class LossBundle:
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


class PredictiveChainStandardizedModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_nodes: int,
        embedding_dim: int,
        hidden_dim: int,
        message_dim: int,
        detach_messages: bool,
    ) -> None:
        super().__init__()
        if num_nodes < 1:
            raise ValueError(f"num_nodes must be at least 1, got {num_nodes}")

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

        self.num_nodes = num_nodes
        self.labels = node_labels(num_nodes)
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.detach_messages = detach_messages

    def _bounded_embedding(self, tokens: Tensor) -> Tensor:
        return torch.tanh(self.embedding(tokens))

    def _predictive_message(self, projection: nn.Linear, hidden: Tensor) -> Tensor:
        return torch.tanh(projection(hidden))

    def rollout(self, tokens: Tensor) -> ModelRollout:
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

        for step in range(sequence_length):
            current_messages = []
            for node_index, (node, message_head) in enumerate(
                zip(self.nodes, self.message_heads, strict=True)
            ):
                node_input = (
                    embeddings[:, step, :]
                    if node_index == 0
                    else previous_messages[node_index - 1]
                )
                hidden_states[node_index] = node(node_input, hidden_states[node_index])
                message = self._predictive_message(message_head, hidden_states[node_index])
                hidden_steps[node_index].append(hidden_states[node_index])
                message_steps[node_index].append(message)
                current_messages.append(message)

            previous_messages = [
                message.detach() if self.detach_messages else message
                for message in current_messages[:-1]
            ]

        return ModelRollout(
            embeddings=embeddings,
            hidden_states=[torch.stack(steps, dim=1) for steps in hidden_steps],
            messages=[torch.stack(steps, dim=1) for steps in message_steps],
        )

    def forward(self, tokens: Tensor) -> Tensor:
        rollout = self.rollout(tokens)
        final_state = torch.cat(
            [hidden_state[:, -1, :] for hidden_state in rollout.hidden_states],
            dim=-1,
        )
        return self.task_head(final_state)


def full_tokens(inputs: Tensor, targets: Tensor) -> Tensor:
    return torch.cat([inputs, targets.unsqueeze(1)], dim=1)


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


def make_model(
    config: RunConfig,
    *,
    vocab_size: int,
    device: torch.device,
) -> PredictiveChainStandardizedModel:
    return PredictiveChainStandardizedModel(
        vocab_size=vocab_size,
        num_nodes=config.num_nodes,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        message_dim=config.message_dim,
        detach_messages=config.detach_messages,
    ).to(device)


def compute_loss_bundle(
    model: PredictiveChainStandardizedModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    auxiliary_weight: float,
) -> LossBundle:
    sequence = full_tokens(inputs, targets)
    rollout = model.rollout(sequence)
    task_step = inputs.shape[1] - 1
    task_state = torch.cat(
        [hidden_state[:, task_step, :] for hidden_state in rollout.hidden_states],
        dim=-1,
    )
    logits = model.task_head(task_state)

    auxiliary_by_node: dict[str, Tensor] = {}
    next_embeddings = align_embedding_target_to_message_dim(
        rollout.embeddings[:, 1:, :].detach(),
        message_dim=model.message_dim,
    )
    auxiliary_by_node[model.labels[0]] = F.mse_loss(
        rollout.messages[0][:, :-1, :],
        next_embeddings,
    )
    for node_index in range(1, model.num_nodes):
        auxiliary_by_node[model.labels[node_index]] = F.mse_loss(
            rollout.messages[node_index],
            rollout.messages[node_index - 1].detach(),
        )

    auxiliary_total = torch.stack(tuple(auxiliary_by_node.values())).sum()
    task_loss = F.cross_entropy(logits, targets)
    total_loss = task_loss + auxiliary_weight * auxiliary_total
    accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    return LossBundle(
        total_loss=total_loss,
        task_loss=task_loss,
        auxiliary_total=auxiliary_total,
        auxiliary_by_node=auxiliary_by_node,
        logits=logits,
        accuracy=accuracy,
    )


def batched_pairs(inputs: Tensor, targets: Tensor, *, batch_size: int):
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        yield inputs[start:stop], targets[start:stop]


def evaluate_bundle(
    model: PredictiveChainStandardizedModel,
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
    total_auxiliary = 0.0
    total_correct = 0

    with torch.no_grad():
        for batch_inputs, batch_targets in batched_pairs(
            inputs,
            targets,
            batch_size=batch_size,
        ):
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
            total_auxiliary += bundle.auxiliary_total.item() * batch_examples
            total_correct += (bundle.logits.argmax(dim=1) == batch_targets).sum().item()

    if was_training:
        model.train()
    return {
        "total_loss": total_total_loss / total_examples,
        "task_loss": total_task_loss / total_examples,
        "auxiliary_total": total_auxiliary / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_one_epoch(
    model: PredictiveChainStandardizedModel,
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
    total_auxiliary = 0.0
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
        total_auxiliary += bundle.auxiliary_total.item() * batch_examples
        total_correct += (bundle.logits.argmax(dim=1) == batch_targets).sum().item()

    return {
        "total_loss": total_total_loss / total_examples,
        "task_loss": total_task_loss / total_examples,
        "auxiliary_total": total_auxiliary / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_model(
    model: PredictiveChainStandardizedModel,
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

    initial_train_metrics = evaluate_bundle(
        model,
        train_inputs,
        train_targets,
        auxiliary_weight=config.auxiliary_weight,
        batch_size=config.eval_batch_size,
    )
    initial_val_metrics = evaluate_bundle(
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
            "train_auxiliary_total": round(initial_train_metrics["auxiliary_total"], 6),
            "train_accuracy": round(initial_train_metrics["accuracy"], 6),
            "val_total_loss": round(initial_val_metrics["total_loss"], 6),
            "val_task_loss": round(initial_val_metrics["task_loss"], 6),
            "val_auxiliary_total": round(initial_val_metrics["auxiliary_total"], 6),
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
        val_metrics = evaluate_bundle(
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
                "train_auxiliary_total": round(train_metrics["auxiliary_total"], 6),
                "train_accuracy": round(train_metrics["accuracy"], 6),
                "val_total_loss": round(val_metrics["total_loss"], 6),
                "val_task_loss": round(val_metrics["task_loss"], 6),
                "val_auxiliary_total": round(val_metrics["auxiliary_total"], 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
            }
        )
        print(
            f"epoch={epoch} train_task_loss={train_metrics['task_loss']:.4f} "
            f"val_task_loss={val_metrics['task_loss']:.4f}"
        )

    runtime_seconds = time.perf_counter() - started_at
    final_metrics = evaluate_bundle(
        model,
        val_inputs,
        val_targets,
        auxiliary_weight=config.auxiliary_weight,
        batch_size=config.eval_batch_size,
    )
    return history, final_metrics, runtime_seconds


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

    tiny_bundle = compute_loss_bundle(
        shape_model,
        batch_inputs,
        batch_targets,
        auxiliary_weight=config.auxiliary_weight,
    )
    tiny_loss = tiny_bundle.task_loss.item()
    expected_low = math.log(split.train_dataset.vocab_size) - 1.0
    expected_high = math.log(split.train_dataset.vocab_size) + 1.0
    if not expected_low <= tiny_loss <= expected_high:
        raise RuntimeError(
            f"Known tiny-batch loss {tiny_loss:.6f} fell outside expected range "
            f"[{expected_low:.6f}, {expected_high:.6f}]."
        )
    checks.append(
        {
            "name": "known_tiny_batch_loss",
            "ok": True,
            "loss": tiny_loss,
            "expected_range": [expected_low, expected_high],
        }
    )

    set_seed(config.seed)
    memorization_model = make_model(
        config,
        vocab_size=split.train_dataset.vocab_size,
        device=device,
    )
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
                    "auxiliary_total": round(bundle.auxiliary_total.item(), 6),
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
            "final_auxiliary_total": final_bundle.auxiliary_total.item(),
            "final_accuracy": final_bundle.accuracy,
            "trace": trace,
        }
    )
    return checks


def round_metric_row(step: int, metrics: dict[str, float]) -> dict[str, float | int]:
    return {
        "step": step,
        "total_loss": round(metrics["total_loss"], 6),
        "task_loss": round(metrics["task_loss"], 6),
        "auxiliary_total": round(metrics["auxiliary_total"], 6),
        "accuracy": round(metrics["accuracy"], 6),
    }


def inspect_examples(
    *,
    model: PredictiveChainStandardizedModel,
    dataset: FixedWindowCharDataset,
    inputs: Tensor,
    targets: Tensor,
    limit: int,
) -> dict[str, object]:
    model_device = next(model.parameters()).device
    sample_inputs = inputs[:limit].to(model_device)
    sample_targets = targets[:limit].to(model_device)
    sequence = full_tokens(sample_inputs, sample_targets)
    rollout = model.rollout(sequence)
    logits = model(sample_inputs)
    predictions = logits.argmax(dim=1)

    examples = []
    for example_index in range(sample_inputs.shape[0]):
        steps = []
        for step_index in range(sequence.shape[1]):
            token = dataset.decode([sequence[example_index, step_index].item()]).replace("\n", "\\n")
            steps.append(
                {
                    "step": step_index,
                    "token": token,
                    "message_norms": {
                        label: round(message[example_index, step_index].norm().item(), 6)
                        for label, message in zip(model.labels, rollout.messages, strict=True)
                    },
                }
            )
        examples.append(
            {
                "context": dataset.decode(sample_inputs[example_index].detach().cpu().tolist()).replace("\n", "\\n"),
                "target": dataset.decode([sample_targets[example_index].item()]).replace("\n", "\\n"),
                "prediction": dataset.decode([predictions[example_index].item()]).replace("\n", "\\n"),
                "steps": steps,
            }
        )
    return {"examples": examples}


def make_default_paths(repo_root: Path, mode: str) -> tuple[Path, Path]:
    text_file = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    base_output_dir = repo_root / "experiments" / "chain_dynamic_depth" / "artifacts" / "predictive_chain_standardized"
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
    parser.add_argument("--auxiliary-weight", type=float)
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
        )
    if args.mode == "checks-only":
        config = replace_config(
            config,
            train_characters=2_048,
            val_characters=512,
            epochs=0,
            sample_length=0,
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
        "auxiliary_weight": args.auxiliary_weight,
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
            "model_family": "predictive_chain_standardized",
            "parameter_count": parameter_count,
            "context_size": config.context_size,
            "num_nodes": config.num_nodes,
            "node_labels": node_labels(config.num_nodes),
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "message_dim": config.message_dim,
            "recurrent_cell": "RNNCell(tanh)",
            "task_head_input": "concat(all_final_hidden_states)",
            "auxiliary_targets": {
                node_labels(config.num_nodes)[0]: "next_token_embedding",
                **{
                    node_labels(config.num_nodes)[index]: f"next_message_from_{node_labels(config.num_nodes)[index - 1]}"
                    for index in range(1, config.num_nodes)
                },
            },
            "message_target_detach": True,
            "message_input_detach": config.detach_messages,
            "simplification_note": "Stage-1 anchor keeps the old chain-family global readout so Stage 2 can change adaptive depth as the main variable.",
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
    history, final_val_metrics, runtime_seconds = train_model(
        model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        config=config,
    )

    prompt = split.train_text[: config.context_size]
    sample = generate_text(
        model,
        split.train_dataset,
        prompt,
        length=config.sample_length,
        device=device,
    )
    write_json(output_dir / "training_history.json", history)
    best_epoch_record = min(history[1:] or history, key=lambda record: float(record["val_task_loss"]))
    write_json(
        output_dir / "final_metrics.json",
        {
            "parameter_count": parameter_count,
            "runtime_seconds": runtime_seconds,
            "final_val_loss": final_val_metrics["task_loss"],
            "final_val_accuracy": final_val_metrics["accuracy"],
            "final_val_total_loss": final_val_metrics["total_loss"],
            "final_val_auxiliary_total": final_val_metrics["auxiliary_total"],
            "best_val_loss": float(best_epoch_record["val_task_loss"]),
            "best_val_accuracy": float(best_epoch_record["val_accuracy"]),
            "best_epoch": int(best_epoch_record["epoch"]),
            "prompt": prompt.replace("\n", "\\n"),
        },
    )
    write_json(
        output_dir / "inspection_examples.json",
        inspect_examples(
            model=model,
            dataset=split.train_dataset,
            inputs=val_inputs[: config.inspection_examples].cpu(),
            targets=val_targets[: config.inspection_examples].cpu(),
            limit=config.inspection_examples,
        ),
    )
    (output_dir / "sample.txt").write_text(sample, encoding="utf-8")


if __name__ == "__main__":
    main()
