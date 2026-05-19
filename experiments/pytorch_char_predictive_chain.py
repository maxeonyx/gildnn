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
    embedding_dim: int = 24
    hidden_dim: int = 96
    message_dim: int = 24
    auxiliary_weight: float = 0.001
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
    auxiliary_a: Tensor
    auxiliary_b: Tensor
    auxiliary_c: Tensor
    logits: Tensor
    accuracy: float


class PredictiveChainCharModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        message_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.node_a = nn.GRUCell(embedding_dim, hidden_dim)
        self.node_b = nn.GRUCell(message_dim, hidden_dim)
        self.node_c = nn.GRUCell(message_dim, hidden_dim)

        self.message_a_head = nn.Linear(hidden_dim, message_dim)
        self.message_b_head = nn.Linear(hidden_dim, message_dim)
        self.message_c_head = nn.Linear(hidden_dim, message_dim)

        self.task_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, vocab_size),
        )

        self.hidden_dim = hidden_dim
        self.message_dim = message_dim

    def _bounded_embedding(self, tokens: Tensor) -> Tensor:
        return torch.tanh(self.embedding(tokens))

    def _predictive_message(self, projection: nn.Linear, hidden: Tensor) -> Tensor:
        return torch.tanh(projection(hidden))

    def rollout(self, tokens: Tensor) -> dict[str, Tensor]:
        batch_size, sequence_length = tokens.shape
        embeddings = self._bounded_embedding(tokens)

        hidden_a = torch.zeros(batch_size, self.hidden_dim, device=tokens.device)
        hidden_b = torch.zeros(batch_size, self.hidden_dim, device=tokens.device)
        hidden_c = torch.zeros(batch_size, self.hidden_dim, device=tokens.device)
        previous_a_message = torch.zeros(
            batch_size, self.message_dim, device=tokens.device
        )
        previous_b_message = torch.zeros(
            batch_size, self.message_dim, device=tokens.device
        )

        hidden_a_steps = []
        hidden_b_steps = []
        hidden_c_steps = []
        message_a_steps = []
        message_b_steps = []
        message_c_steps = []
        predicted_embedding_steps = []
        predicted_a_message_steps = []
        predicted_b_message_steps = []

        for step in range(sequence_length):
            hidden_a = self.node_a(embeddings[:, step, :], hidden_a)
            message_a = self._predictive_message(self.message_a_head, hidden_a)

            hidden_b = self.node_b(previous_a_message, hidden_b)
            message_b = self._predictive_message(self.message_b_head, hidden_b)

            hidden_c = self.node_c(previous_b_message, hidden_c)
            message_c = self._predictive_message(self.message_c_head, hidden_c)

            hidden_a_steps.append(hidden_a)
            hidden_b_steps.append(hidden_b)
            hidden_c_steps.append(hidden_c)
            message_a_steps.append(message_a)
            message_b_steps.append(message_b)
            message_c_steps.append(message_c)
            predicted_embedding_steps.append(message_a)
            predicted_a_message_steps.append(message_b)
            predicted_b_message_steps.append(message_c)

            previous_a_message = message_a
            previous_b_message = message_b

        return {
            "embeddings": embeddings,
            "hidden_a": torch.stack(hidden_a_steps, dim=1),
            "hidden_b": torch.stack(hidden_b_steps, dim=1),
            "hidden_c": torch.stack(hidden_c_steps, dim=1),
            "message_a": torch.stack(message_a_steps, dim=1),
            "message_b": torch.stack(message_b_steps, dim=1),
            "message_c": torch.stack(message_c_steps, dim=1),
            "predicted_embedding": torch.stack(predicted_embedding_steps, dim=1),
            "predicted_a_message": torch.stack(predicted_a_message_steps, dim=1),
            "predicted_b_message": torch.stack(predicted_b_message_steps, dim=1),
        }

    def forward(self, tokens: Tensor) -> Tensor:
        rollout = self.rollout(tokens)
        final_state = torch.cat(
            [
                rollout["hidden_a"][:, -1, :],
                rollout["hidden_b"][:, -1, :],
                rollout["hidden_c"][:, -1, :],
            ],
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
        [
            rollout["hidden_a"][:, task_step, :],
            rollout["hidden_b"][:, task_step, :],
            rollout["hidden_c"][:, task_step, :],
        ],
        dim=-1,
    )
    logits = model.task_head(task_state)

    next_embeddings = rollout["embeddings"][:, 1:, :].detach()
    current_a_messages = rollout["message_a"].detach()
    current_b_messages = rollout["message_b"].detach()

    auxiliary_a = F.mse_loss(rollout["predicted_embedding"][:, :-1, :], next_embeddings)
    auxiliary_b = F.mse_loss(
        rollout["predicted_a_message"], current_a_messages
    )
    auxiliary_c = F.mse_loss(
        rollout["predicted_b_message"], current_b_messages
    )
    auxiliary_total = auxiliary_a + auxiliary_b + auxiliary_c
    task_loss = F.cross_entropy(logits, targets)
    total_loss = task_loss + auxiliary_weight * auxiliary_total
    accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    return LossBreakdown(
        total_loss=total_loss,
        task_loss=task_loss,
        auxiliary_total=auxiliary_total,
        auxiliary_a=auxiliary_a,
        auxiliary_b=auxiliary_b,
        auxiliary_c=auxiliary_c,
        logits=logits,
        accuracy=accuracy,
    )


def rounded_metric_row(step: int, losses: LossBreakdown) -> dict[str, float | int]:
    return {
        "step": step,
        "total_loss": round(losses.total_loss.item(), 6),
        "task_loss": round(losses.task_loss.item(), 6),
        "auxiliary_total": round(losses.auxiliary_total.item(), 6),
        "auxiliary_a": round(losses.auxiliary_a.item(), 6),
        "auxiliary_b": round(losses.auxiliary_b.item(), 6),
        "auxiliary_c": round(losses.auxiliary_c.item(), 6),
        "accuracy": round(losses.accuracy, 6),
    }


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
    payload: dict[str, object] = {
        "final_total_loss": losses.total_loss.item(),
        "final_task_loss": losses.task_loss.item(),
        "final_auxiliary_total": losses.auxiliary_total.item(),
        "final_auxiliary_a": losses.auxiliary_a.item(),
        "final_auxiliary_b": losses.auxiliary_b.item(),
        "final_auxiliary_c": losses.auxiliary_c.item(),
        "final_accuracy": losses.accuracy,
        "trace": trace,
    }
    if reached_memorization_bar is not None:
        payload["reached_memorization_bar"] = reached_memorization_bar
    write_json(path, payload)


def write_auxiliary_traces(
    path: Path, trace: list[dict[str, float | int]]
) -> None:
    write_json(
        path,
        {
            "steps": [row["step"] for row in trace],
            "auxiliary_a": [row["auxiliary_a"] for row in trace],
            "auxiliary_b": [row["auxiliary_b"] for row in trace],
            "auxiliary_c": [row["auxiliary_c"] for row in trace],
            "auxiliary_total": [row["auxiliary_total"] for row in trace],
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
                    "message_a": _round_vector(rollout["message_a"][example_index, step_index]),
                    "message_b": _round_vector(rollout["message_b"][example_index, step_index]),
                    "message_c": _round_vector(rollout["message_c"][example_index, step_index]),
                    "predicted_next_token_embedding": _round_vector(
                        rollout["predicted_embedding"][example_index, step_index]
                    ),
                    "predicted_next_a_message": _round_vector(
                        rollout["predicted_a_message"][example_index, step_index]
                    ),
                    "predicted_next_b_message": _round_vector(
                        rollout["predicted_b_message"][example_index, step_index]
                    ),
                    "message_a_norm": round(
                        rollout["message_a"][example_index, step_index].norm().item(), 6
                    ),
                    "message_b_norm": round(
                        rollout["message_b"][example_index, step_index].norm().item(), 6
                    ),
                    "message_c_norm": round(
                        rollout["message_c"][example_index, step_index].norm().item(), 6
                    ),
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
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    config = RunConfig()
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
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        message_dim=config.message_dim,
    ).to(device)
    write_json(
        output_dir / "model_summary.json",
        {
            "model_family": "predictive_chain",
            "graph": "A_to_B_to_C_line",
            "node_count": 3,
            "recurrent_cell": "GRUCell",
            "parameter_count": count_parameters(overfit_model),
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "message_dim": config.message_dim,
            "context_size": config.context_size,
            "vocab_size": dataset.vocab_size,
            "task_head_input": "concat(final_hidden_a, final_hidden_b, final_hidden_c)",
            "auxiliary_targets": {
                "A": "next_token_embedding",
                "B": "next_message_from_A",
                "C": "next_message_from_B",
            },
            "message_target_detach": True,
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
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        message_dim=config.message_dim,
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
