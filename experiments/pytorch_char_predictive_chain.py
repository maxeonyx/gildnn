from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 5
    num_nodes: int = 3
    embedding_dim: int = 24
    hidden_dim: int = 96
    message_dim: int = 24
    auxiliary_weight: float = 0.001
    detach_messages: bool = False
    overfit_batch_size: int = 16
    overfit_steps: int = 12000
    overfit_learning_rate: float = 0.003
    tiny_batch_size: int = 64
    tiny_steps: int = 3000
    tiny_learning_rate: float = 0.01
    gradient_clip_norm: float = 1.0
    sample_length: int = 80
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


def node_label(index: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(letters):
        return letters[index]
    return f"node_{index + 1}"


def node_labels(count: int) -> list[str]:
    return [node_label(index) for index in range(count)]


class PredictiveChainCharModel(nn.Module):
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
                nn.GRUCell(
                    embedding_dim if node_index == 0 else message_dim,
                    hidden_dim,
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


def full_tokens(inputs: Tensor, targets: Tensor) -> Tensor:
    return torch.cat([inputs, targets.unsqueeze(1)], dim=1)


def compute_losses(
    model: PredictiveChainCharModel,
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
    next_embeddings = rollout.embeddings[:, 1:, :].detach()
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


def train_on_fixed_batch(
    model: PredictiveChainCharModel,
    batch_inputs: Tensor,
    batch_targets: Tensor,
    *,
    steps: int,
    learning_rate: float,
    auxiliary_weight: float,
    gradient_clip_norm: float,
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
        if step % 50 == 0 or step == steps:
            trace.append(rounded_metric_row(step, losses))

        if losses.accuracy == 1.0 and losses.total_loss.item() < 1e-3:
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
    if not trace or trace[-1]["step"] != step:
        trace.append(rounded_metric_row(step, final_losses))
    return trace, final_losses


def train_on_tiny_dataset(
    model: PredictiveChainCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
    steps: int,
    learning_rate: float,
    auxiliary_weight: float,
    gradient_clip_norm: float,
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
    return trace, final_losses


def write_metrics(
    path: Path,
    *,
    trace: list[dict[str, float | int]],
    losses: LossBreakdown,
    reached_memorization_bar: bool | None = None,
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
    for label, value in auxiliary_by_node.items():
        payload[f"final_auxiliary_{label.lower()}"] = value
    if reached_memorization_bar is not None:
        payload["reached_memorization_bar"] = reached_memorization_bar
    write_json(path, payload)


def write_auxiliary_traces(
    path: Path, trace: list[dict[str, float | int]]
) -> None:
    labels = list(trace[0]["auxiliary_by_node"].keys()) if trace else []
    payload: dict[str, object] = {
        "steps": [row["step"] for row in trace],
        "auxiliary_total": [row["auxiliary_total"] for row in trace],
        "auxiliary_by_node": {
            label: [row[f"auxiliary_{label.lower()}"] for row in trace] for label in labels
        },
    }
    for label in labels:
        payload[f"auxiliary_{label.lower()}"] = [
            row[f"auxiliary_{label.lower()}"] for row in trace
        ]
    write_json(path, payload)


def write_auxiliary_position_summary(
    path: Path,
    *,
    overfit_losses: LossBreakdown,
    tiny_losses: LossBreakdown,
) -> None:
    labels = list(overfit_losses.auxiliary_by_node.keys())
    overfit_values = {
        label: overfit_losses.auxiliary_by_node[label].item() for label in labels
    }
    tiny_values = {label: tiny_losses.auxiliary_by_node[label].item() for label in labels}
    write_json(
        path,
        {
            "node_labels": labels,
            "overfit_final_auxiliary_by_node": overfit_values,
            "tiny_final_auxiliary_by_node": tiny_values,
            "overfit_nodes_sorted_by_auxiliary": sorted(
                overfit_values,
                key=overfit_values.__getitem__,
                reverse=True,
            ),
            "tiny_nodes_sorted_by_auxiliary": sorted(
                tiny_values,
                key=tiny_values.__getitem__,
                reverse=True,
            ),
        },
    )


def _round_vector(values: Tensor) -> list[float]:
    return [round(value, 4) for value in values.detach().cpu().tolist()]


def rollout_examples(
    *,
    model: PredictiveChainCharModel,
    dataset: FixedWindowCharDataset,
    inputs: Tensor,
    targets: Tensor,
    limit: int,
) -> dict[str, object]:
    sample_inputs = inputs[:limit]
    sample_targets = targets[:limit]
    sequence = full_tokens(sample_inputs, sample_targets)
    rollout = model.rollout(sequence)
    logits = model(sample_inputs)
    predictions = logits.argmax(dim=1)

    examples = []
    labels = model.labels
    for example_index in range(sample_inputs.shape[0]):
        steps = []
        for step_index in range(sequence.shape[1]):
            token = dataset.decode([sequence[example_index, step_index].item()]).replace(
                "\n", "\\n"
            )
            next_token = None
            if step_index + 1 < sequence.shape[1]:
                next_token = dataset.decode(
                    [sequence[example_index, step_index + 1].item()]
                ).replace("\n", "\\n")

            steps.append(
                {
                    "step": step_index,
                    "token": token,
                    "next_token_target": next_token,
                    "messages": {
                        label: _round_vector(message[example_index, step_index])
                        for label, message in zip(labels, rollout.messages, strict=True)
                    },
                    "message_norms": {
                        label: round(message[example_index, step_index].norm().item(), 6)
                        for label, message in zip(labels, rollout.messages, strict=True)
                    },
                }
            )

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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_question_dir = repo_root / "research" / "questions" / "predictive-chain"
    default_config = RunConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text-file",
        type=Path,
        default=default_question_dir / "raw_text.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_question_dir / "artifacts",
    )
    parser.add_argument(
        "--auxiliary-weight",
        type=float,
        default=default_config.auxiliary_weight,
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=default_config.num_nodes,
    )
    parser.add_argument(
        "--detach-messages",
        action=argparse.BooleanOptionalAction,
        default=default_config.detach_messages,
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    config = RunConfig(
        num_nodes=args.num_nodes,
        auxiliary_weight=args.auxiliary_weight,
        detach_messages=args.detach_messages,
    )
    set_seed(config.seed)
    device = resolve_device(args.device)

    raw_text = args.text_file.read_text(encoding="utf-8")
    dataset = FixedWindowCharDataset(raw_text, context_size=config.context_size)
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
            "text_characters": len(raw_text),
            "vocab_size": dataset.vocab_size,
            "window_count": int(dataset.inputs.shape[0]),
        },
    )

    overfit_model = PredictiveChainCharModel(
        vocab_size=dataset.vocab_size,
        num_nodes=config.num_nodes,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        message_dim=config.message_dim,
        detach_messages=config.detach_messages,
    ).to(device)
    task_head_input = ", ".join(
        f"final_hidden_{label}" for label in overfit_model.labels
    )
    auxiliary_targets = {
        overfit_model.labels[0]: "next_token_embedding",
        **{
            label: f"next_message_from_{overfit_model.labels[index - 1]}"
            for index, label in enumerate(overfit_model.labels[1:], start=1)
        },
    }
    write_json(
        output_dir / "model_summary.json",
        {
            "model_family": "predictive_chain",
            "graph": "_to_".join(overfit_model.labels) + "_line",
            "node_count": config.num_nodes,
            "node_labels": overfit_model.labels,
            "recurrent_cell": "GRUCell",
            "parameter_count": count_parameters(overfit_model),
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "message_dim": config.message_dim,
            "context_size": config.context_size,
            "vocab_size": dataset.vocab_size,
            "task_head_input": f"concat({task_head_input})",
            "auxiliary_targets": auxiliary_targets,
            "message_target_detach": True,
            "message_input_detach": config.detach_messages,
        },
    )

    overfit_inputs = dataset.inputs[: config.overfit_batch_size].to(device)
    overfit_targets = dataset.targets[: config.overfit_batch_size].to(device)
    overfit_trace, overfit_losses = train_on_fixed_batch(
        overfit_model,
        overfit_inputs,
        overfit_targets,
        steps=config.overfit_steps,
        learning_rate=config.overfit_learning_rate,
        auxiliary_weight=config.auxiliary_weight,
        gradient_clip_norm=config.gradient_clip_norm,
    )
    overfit_predictions = overfit_model(overfit_inputs).argmax(dim=1)
    write_metrics(
        output_dir / "overfit_metrics.json",
        trace=overfit_trace,
        losses=overfit_losses,
        reached_memorization_bar=overfit_losses.accuracy == 1.0
        and overfit_losses.total_loss.item() < 1e-3,
    )
    write_auxiliary_traces(output_dir / "overfit_auxiliary_traces.json", overfit_trace)
    (output_dir / "overfit_predictions.txt").write_text(
        render_predictions(
            dataset,
            overfit_inputs.cpu(),
            overfit_targets.cpu(),
            overfit_predictions.cpu(),
        ),
        encoding="utf-8",
    )
    write_json(
        output_dir / "overfit_rollout_examples.json",
        rollout_examples(
            model=overfit_model,
            dataset=dataset,
            inputs=overfit_inputs,
            targets=overfit_targets,
            limit=config.rollout_examples,
        ),
    )

    tiny_model = PredictiveChainCharModel(
        vocab_size=dataset.vocab_size,
        num_nodes=config.num_nodes,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        message_dim=config.message_dim,
        detach_messages=config.detach_messages,
    ).to(device)
    tiny_inputs = dataset.inputs.to(device)
    tiny_targets = dataset.targets.to(device)
    tiny_trace, tiny_losses = train_on_tiny_dataset(
        tiny_model,
        tiny_inputs,
        tiny_targets,
        batch_size=config.tiny_batch_size,
        steps=config.tiny_steps,
        learning_rate=config.tiny_learning_rate,
        auxiliary_weight=config.auxiliary_weight,
        gradient_clip_norm=config.gradient_clip_norm,
    )
    write_metrics(output_dir / "tiny_run_metrics.json", trace=tiny_trace, losses=tiny_losses)
    write_auxiliary_traces(output_dir / "tiny_auxiliary_traces.json", tiny_trace)
    write_auxiliary_position_summary(
        output_dir / "auxiliary_position_summary.json",
        overfit_losses=overfit_losses,
        tiny_losses=tiny_losses,
    )

    prompts = ["hello", "small", " text"]
    samples = {
        prompt.replace("\n", "\\n"): generate_text(
            tiny_model,
            dataset,
            prompt,
            length=config.sample_length,
            device=device,
        )
        for prompt in prompts
    }
    write_json(output_dir / "tiny_samples.json", samples)
    write_json(
        output_dir / "tiny_rollout_examples.json",
        rollout_examples(
            model=tiny_model,
            dataset=dataset,
            inputs=tiny_inputs,
            targets=tiny_targets,
            limit=config.rollout_examples,
        ),
    )


if __name__ == "__main__":
    main()
