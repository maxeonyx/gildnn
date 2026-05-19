from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import FixedWindowCharDataset, resolve_device, set_seed
from core.tiny_char_transformer import (
    PlainResidualCombine,
    TinyTransformerCharModel,
    count_parameters,
)
from experiments.pytorch_char_predictive_chain import (
    PredictiveChainCharModel,
    compute_losses,
)
from experiments.pytorch_char_rnn_baseline import TinyRnnCharModel
from experiments.pytorch_char_sanity_check import FeedForwardCharModel


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 32
    train_characters: int = 100_000
    val_characters: int = 20_000
    train_batch_size: int = 256
    train_steps: int = 5_000
    learning_rate: float = 0.003
    gradient_clip_norm: float = 1.0
    eval_batch_size: int = 512
    trace_eval_examples: int = 4_096
    trace_interval: int = 250
    seed: int = 42
    feedforward_embedding_dim: int = 24
    feedforward_hidden_dim: int = 224
    rnn_embedding_dim: int = 24
    rnn_hidden_dim: int = 160
    rnn_num_layers: int = 4
    transformer_d_model: int = 72
    transformer_feedforward_dim: int = 256
    transformer_num_heads: int = 4
    transformer_num_layers: int = 3
    predictive_num_nodes: int = 8
    predictive_embedding_dim: int = 24
    predictive_hidden_dim: int = 56
    predictive_message_dim: int = 56
    predictive_detach_messages: bool = True
    predictive_auxiliary_weight: float = 1.0


def replace_config(config: RunConfig, **changes: object) -> RunConfig:
    payload = asdict(config)
    payload.update(changes)
    return RunConfig(**payload)


@dataclass(frozen=True)
class DatasetSplit:
    dataset: FixedWindowCharDataset
    train_text: str
    val_text: str
    train_inputs: Tensor
    train_targets: Tensor
    val_inputs: Tensor
    val_targets: Tensor


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


def encode_windows(
    text: str,
    *,
    context_size: int,
    stoi: dict[str, int],
) -> tuple[Tensor, Tensor]:
    if len(text) <= context_size:
        raise ValueError("Text split must be longer than the context size.")
    encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    inputs = []
    targets = []
    for start in range(len(encoded) - context_size):
        stop = start + context_size
        inputs.append(encoded[start:stop])
        targets.append(encoded[stop])
    return torch.stack(inputs), torch.stack(targets)


def build_fixed_length_split(text: str, *, config: RunConfig) -> DatasetSplit:
    required_characters = config.train_characters + config.val_characters
    if len(text) < required_characters:
        raise ValueError(
            f"Need at least {required_characters} characters, got {len(text)}."
        )
    train_text = text[: config.train_characters]
    val_text = text[
        config.train_characters : config.train_characters + config.val_characters
    ]
    dataset = FixedWindowCharDataset(train_text, context_size=config.context_size)
    missing_val_chars = sorted(set(val_text) - set(train_text))
    if missing_val_chars:
        raise ValueError(
            f"Validation text contains characters absent from training text: {missing_val_chars}"
        )
    val_inputs, val_targets = encode_windows(
        val_text,
        context_size=config.context_size,
        stoi=dataset.stoi,
    )
    return DatasetSplit(
        dataset=dataset,
        train_text=train_text,
        val_text=val_text,
        train_inputs=dataset.inputs,
        train_targets=dataset.targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
    )


def _batched_pairs(
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
):
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        yield inputs[start:stop], targets[start:stop]


def evaluate_standard_model_batched(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in _batched_pairs(
            inputs,
            targets,
            batch_size=batch_size,
        ):
            logits = model(batch_inputs)
            total_loss += F.cross_entropy(
                logits,
                batch_targets,
                reduction="sum",
            ).item()
            total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
            total_examples += batch_targets.shape[0]
    if was_training:
        model.train()
    return {
        "task_loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def evaluate_predictive_chain_model_batched(
    model: PredictiveChainCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
    auxiliary_weight: float,
) -> dict[str, object]:
    was_training = model.training
    model.eval()
    total_examples = 0
    task_loss_sum = 0.0
    total_loss_sum = 0.0
    auxiliary_total_sum = 0.0
    correct_sum = 0.0
    auxiliary_by_node_sum = {label: 0.0 for label in model.labels}
    with torch.no_grad():
        for batch_inputs, batch_targets in _batched_pairs(
            inputs,
            targets,
            batch_size=batch_size,
        ):
            losses = compute_losses(
                model,
                batch_inputs,
                batch_targets,
                auxiliary_weight=auxiliary_weight,
            )
            batch_examples = batch_targets.shape[0]
            task_loss_sum += losses.task_loss.item() * batch_examples
            total_loss_sum += losses.total_loss.item() * batch_examples
            auxiliary_total_sum += losses.auxiliary_total.item() * batch_examples
            correct_sum += losses.accuracy * batch_examples
            for label, value in losses.auxiliary_by_node.items():
                auxiliary_by_node_sum[label] += value.item() * batch_examples
            total_examples += batch_examples
    if was_training:
        model.train()
    return {
        "task_loss": task_loss_sum / total_examples,
        "total_loss": total_loss_sum / total_examples,
        "auxiliary_total": auxiliary_total_sum / total_examples,
        "accuracy": correct_sum / total_examples,
        "auxiliary_by_node": {
            label: value / total_examples
            for label, value in auxiliary_by_node_sum.items()
        },
    }


def train_standard_model(
    model: nn.Module,
    *,
    train_inputs: Tensor,
    train_targets: Tensor,
    batch_size: int,
    steps: int,
    learning_rate: float,
    gradient_clip_norm: float,
    eval_batch_size: int,
    trace_eval_examples: int,
    trace_interval: int,
) -> tuple[list[dict[str, float | int]], dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    trace: list[dict[str, float | int]] = []
    trace_inputs = train_inputs[: min(trace_eval_examples, train_inputs.shape[0])]
    trace_targets = train_targets[: min(trace_eval_examples, train_targets.shape[0])]
    sample_count = train_inputs.shape[0]

    for step in range(steps + 1):
        if step % trace_interval == 0 or step == steps:
            metrics = evaluate_standard_model_batched(
                model,
                trace_inputs,
                trace_targets,
                batch_size=eval_batch_size,
            )
            trace.append(
                {
                    "step": step,
                    "task_loss": round(metrics["task_loss"], 6),
                    "accuracy": round(metrics["accuracy"], 6),
                    "evaluated_examples": int(trace_inputs.shape[0]),
                }
            )
        if step == steps:
            break
        batch_indices = torch.randint(0, sample_count, (batch_size,), device=train_inputs.device)
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

    final_metrics = evaluate_standard_model_batched(
        model,
        train_inputs,
        train_targets,
        batch_size=eval_batch_size,
    )
    return trace, final_metrics


def train_predictive_chain_model(
    model: PredictiveChainCharModel,
    *,
    train_inputs: Tensor,
    train_targets: Tensor,
    batch_size: int,
    steps: int,
    learning_rate: float,
    gradient_clip_norm: float,
    auxiliary_weight: float,
    eval_batch_size: int,
    trace_eval_examples: int,
    trace_interval: int,
) -> tuple[list[dict[str, float | int]], dict[str, object]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trace: list[dict[str, float | int]] = []
    trace_inputs = train_inputs[: min(trace_eval_examples, train_inputs.shape[0])]
    trace_targets = train_targets[: min(trace_eval_examples, train_targets.shape[0])]
    sample_count = train_inputs.shape[0]

    for step in range(steps + 1):
        if step % trace_interval == 0 or step == steps:
            metrics = evaluate_predictive_chain_model_batched(
                model,
                trace_inputs,
                trace_targets,
                batch_size=eval_batch_size,
                auxiliary_weight=auxiliary_weight,
            )
            trace.append(
                {
                    "step": step,
                    "task_loss": round(float(metrics["task_loss"]), 6),
                    "total_loss": round(float(metrics["total_loss"]), 6),
                    "auxiliary_total": round(float(metrics["auxiliary_total"]), 6),
                    "accuracy": round(float(metrics["accuracy"]), 6),
                    "evaluated_examples": int(trace_inputs.shape[0]),
                }
            )
        if step == steps:
            break
        batch_indices = torch.randint(0, sample_count, (batch_size,), device=train_inputs.device)
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]
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

    final_metrics = evaluate_predictive_chain_model_batched(
        model,
        train_inputs,
        train_targets,
        batch_size=eval_batch_size,
        auxiliary_weight=auxiliary_weight,
    )
    return trace, final_metrics


def render_results_table(results: list[dict[str, object]]) -> str:
    header = "model\tparams\ttrain_loss\tval_loss\tval_acc"
    rows = [header]
    for result in sorted(results, key=lambda row: float(row["val_loss"])):
        rows.append(
            "\t".join(
                [
                    str(result["model_name"]),
                    str(result["parameter_count"]),
                    f"{float(result['train_loss']):.4f}",
                    f"{float(result['val_loss']):.4f}",
                    f"{float(result['val_accuracy']):.4f}",
                ]
            )
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_text_file = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    default_output_dir = (
        repo_root
        / "research"
        / "questions"
        / "predictive-chain"
        / "artifacts"
        / "scale_up"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", type=Path, default=default_text_file)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--trace-interval", type=int)
    parser.add_argument("--trace-eval-examples", type=int)
    args = parser.parse_args()

    config = RunConfig()
    if args.train_steps is not None:
        config = replace_config(config, train_steps=args.train_steps)
    if args.trace_interval is not None:
        config = replace_config(config, trace_interval=args.trace_interval)
    if args.trace_eval_examples is not None:
        config = replace_config(config, trace_eval_examples=args.trace_eval_examples)
    set_seed(config.seed)
    device = resolve_device(args.device)

    raw_text = args.text_file.read_text(encoding="utf-8")
    split = build_fixed_length_split(raw_text, config=config)
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
    used_text = split.train_text + split.val_text
    write_json(
        output_dir / "corpus_summary.json",
        {
            "source_file": str(args.text_file),
            "source_total_characters": len(raw_text),
            "source_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "used_total_characters": len(used_text),
            "used_sha256": hashlib.sha256(used_text.encode("utf-8")).hexdigest(),
            "train_characters": len(split.train_text),
            "val_characters": len(split.val_text),
            "train_windows": int(split.train_inputs.shape[0]),
            "val_windows": int(split.val_inputs.shape[0]),
            "vocab_size": split.dataset.vocab_size,
        },
    )

    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)

    feedforward_model = FeedForwardCharModel(
        vocab_size=split.dataset.vocab_size,
        context_size=config.context_size,
        embedding_dim=config.feedforward_embedding_dim,
        hidden_dim=config.feedforward_hidden_dim,
    ).to(device)
    rnn_model = TinyRnnCharModel(
        vocab_size=split.dataset.vocab_size,
        embedding_dim=config.rnn_embedding_dim,
        hidden_dim=config.rnn_hidden_dim,
        num_layers=config.rnn_num_layers,
        nonlinearity="tanh",
    ).to(device)
    transformer_model = TinyTransformerCharModel(
        vocab_size=split.dataset.vocab_size,
        context_size=config.context_size,
        d_model=config.transformer_d_model,
        num_heads=config.transformer_num_heads,
        num_layers=config.transformer_num_layers,
        feedforward_dim=config.transformer_feedforward_dim,
        residual_factory=PlainResidualCombine,
    ).to(device)
    predictive_chain_model = PredictiveChainCharModel(
        vocab_size=split.dataset.vocab_size,
        num_nodes=config.predictive_num_nodes,
        embedding_dim=config.predictive_embedding_dim,
        hidden_dim=config.predictive_hidden_dim,
        message_dim=config.predictive_message_dim,
        detach_messages=config.predictive_detach_messages,
    ).to(device)

    parameter_counts = {
        "feedforward": count_parameters(feedforward_model),
        "rnn": count_parameters(rnn_model),
        "transformer": count_parameters(transformer_model),
        "predictive_chain": count_parameters(predictive_chain_model),
    }
    write_json(output_dir / "parameter_counts.json", parameter_counts)

    results: list[dict[str, object]] = []

    standard_runs = [
        (
            "feedforward",
            feedforward_model,
            {
                "embedding_dim": config.feedforward_embedding_dim,
                "hidden_dim": config.feedforward_hidden_dim,
            },
        ),
        (
            "rnn",
            rnn_model,
            {
                "embedding_dim": config.rnn_embedding_dim,
                "hidden_dim": config.rnn_hidden_dim,
                "num_layers": config.rnn_num_layers,
                "nonlinearity": "tanh",
            },
        ),
        (
            "transformer",
            transformer_model,
            {
                "d_model": config.transformer_d_model,
                "feedforward_dim": config.transformer_feedforward_dim,
                "num_heads": config.transformer_num_heads,
                "num_layers": config.transformer_num_layers,
            },
        ),
    ]

    for model_name, model, model_config in standard_runs:
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()
        train_trace, train_metrics = train_standard_model(
            model,
            train_inputs=train_inputs,
            train_targets=train_targets,
            batch_size=config.train_batch_size,
            steps=config.train_steps,
            learning_rate=config.learning_rate,
            gradient_clip_norm=config.gradient_clip_norm,
            eval_batch_size=config.eval_batch_size,
            trace_eval_examples=config.trace_eval_examples,
            trace_interval=config.trace_interval,
        )
        runtime_seconds = time.perf_counter() - started_at
        val_metrics = evaluate_standard_model_batched(
            model,
            val_inputs,
            val_targets,
            batch_size=config.eval_batch_size,
        )
        metrics_payload = {
            "model_name": model_name,
            "parameter_count": parameter_counts[model_name],
            "runtime_seconds": runtime_seconds,
            "optimizer": "AdamW",
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_trace": train_trace,
            "model_config": model_config,
        }
        write_json(model_dir / "metrics.json", metrics_payload)
        results.append(
            {
                "model_name": model_name,
                "parameter_count": parameter_counts[model_name],
                "runtime_seconds": runtime_seconds,
                "train_loss": train_metrics["task_loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["task_loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )

    predictive_dir = output_dir / "predictive_chain"
    predictive_dir.mkdir(parents=True, exist_ok=True)
    predictive_started_at = time.perf_counter()
    predictive_trace, predictive_train_metrics = train_predictive_chain_model(
        predictive_chain_model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        batch_size=config.train_batch_size,
        steps=config.train_steps,
        learning_rate=config.learning_rate,
        gradient_clip_norm=config.gradient_clip_norm,
        auxiliary_weight=config.predictive_auxiliary_weight,
        eval_batch_size=config.eval_batch_size,
        trace_eval_examples=config.trace_eval_examples,
        trace_interval=config.trace_interval,
    )
    predictive_runtime_seconds = time.perf_counter() - predictive_started_at
    predictive_val_metrics = evaluate_predictive_chain_model_batched(
        predictive_chain_model,
        val_inputs,
        val_targets,
        batch_size=config.eval_batch_size,
        auxiliary_weight=config.predictive_auxiliary_weight,
    )
    write_json(
        predictive_dir / "metrics.json",
        {
            "model_name": "predictive_chain",
            "parameter_count": parameter_counts["predictive_chain"],
            "runtime_seconds": predictive_runtime_seconds,
            "optimizer": "Adam",
            "train_metrics": predictive_train_metrics,
            "val_metrics": predictive_val_metrics,
            "train_trace": predictive_trace,
            "model_config": {
                "num_nodes": config.predictive_num_nodes,
                "embedding_dim": config.predictive_embedding_dim,
                "hidden_dim": config.predictive_hidden_dim,
                "message_dim": config.predictive_message_dim,
                "detach_messages": config.predictive_detach_messages,
                "auxiliary_weight": config.predictive_auxiliary_weight,
            },
        },
    )
    results.append(
        {
            "model_name": "predictive_chain",
            "parameter_count": parameter_counts["predictive_chain"],
            "runtime_seconds": predictive_runtime_seconds,
            "train_loss": float(predictive_train_metrics["task_loss"]),
            "train_accuracy": float(predictive_train_metrics["accuracy"]),
            "val_loss": float(predictive_val_metrics["task_loss"]),
            "val_accuracy": float(predictive_val_metrics["accuracy"]),
            "train_total_loss": float(predictive_train_metrics["total_loss"]),
            "val_total_loss": float(predictive_val_metrics["total_loss"]),
            "train_auxiliary_total": float(predictive_train_metrics["auxiliary_total"]),
            "val_auxiliary_total": float(predictive_val_metrics["auxiliary_total"]),
        }
    )

    comparison_summary = {
        "models": sorted(results, key=lambda row: float(row["val_loss"])),
        "best_by_val_loss": min(results, key=lambda row: float(row["val_loss"])),
        "best_by_val_accuracy": max(results, key=lambda row: float(row["val_accuracy"])),
    }
    write_json(output_dir / "comparison_summary.json", comparison_summary)
    table = render_results_table(results)
    (output_dir / "val_loss_table.txt").write_text(table, encoding="utf-8")
    sys.stdout.write(table)


if __name__ == "__main__":
    main()
