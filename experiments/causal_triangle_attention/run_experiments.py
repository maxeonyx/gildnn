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
from torch import Tensor
from torch.nn import functional as F

from base_experiments.transformer import build_fixed_length_split
from core.fixed_window_char import FixedWindowCharDataset, generate_text, resolve_device, set_seed

from .model import AttentionResidualCharModel, VariantSpec, count_parameters


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 32
    train_characters: int = 100_000
    val_characters: int = 20_000
    d_model: int = 72
    batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 0.003
    gradient_clip_norm: float = 1.0
    seed: int = 42
    stage4_batch_size: int = 32
    stage4_steps: int = 1500
    stage4_learning_rate: float = 0.01
    stage4_log_interval: int = 25
    sample_length: int = 320
    progression_sample_length: int = 200
    sample_checkpoints: tuple[int, ...] = (0, 1, 2, 4)
    tiny_train_characters: int = 4_096
    tiny_val_characters: int = 1_024
    tiny_epochs: int = 2
    full_epochs: int = 13


VARIANTS = {
    "external_control": VariantSpec(
        family="external_control",
        num_layers=3,
        num_heads=4,
        feedforward_dim=256,
    ),
    "depth_only": VariantSpec(
        family="depth_only",
        num_layers=3,
        num_heads=4,
        feedforward_dim=159,
    ),
    "causal_triangle": VariantSpec(
        family="causal_triangle",
        num_layers=3,
        num_heads=4,
        feedforward_dim=110,
    ),
}


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


def replace_config(config: RunConfig, **changes: object) -> RunConfig:
    payload = asdict(config)
    payload.update(changes)
    return RunConfig(**payload)


def make_model(
    *,
    vocab_size: int,
    config: RunConfig,
    variant: VariantSpec,
    device: torch.device,
) -> AttentionResidualCharModel:
    return AttentionResidualCharModel(
        vocab_size=vocab_size,
        context_size=config.context_size,
        d_model=config.d_model,
        variant=variant,
    ).to(device)


def batched_pairs(inputs: Tensor, targets: Tensor, *, batch_size: int):
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        yield inputs[start:stop], targets[start:stop]


def capture_sample(
    model: AttentionResidualCharModel,
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


def write_metadata(
    output_dir: Path,
    *,
    config: RunConfig,
    split,
    variant: VariantSpec,
    model: AttentionResidualCharModel,
    text_file: Path,
    raw_text: str,
    device: torch.device,
    stage: str,
) -> None:
    git_status_short = current_git_status_short()
    write_json(output_dir / "config.json", {**asdict(config), "stage": stage, "variant": variant.label})
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
            "variant": variant.label,
            "parameter_count": count_parameters(model),
            "context_size": config.context_size,
            "d_model": config.d_model,
            "num_layers": variant.num_layers,
            "num_heads": variant.num_heads,
            "feedforward_dim": variant.feedforward_dim,
        },
    )


def evaluate_model(
    model: AttentionResidualCharModel,
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
        for batch_inputs, batch_targets in batched_pairs(inputs, targets, batch_size=batch_size):
            logits = model(batch_inputs)
            total_loss += F.cross_entropy(logits, batch_targets, reduction="sum").item()
            total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
            total_examples += batch_targets.shape[0]
    if was_training:
        model.train()
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_one_epoch(
    model: AttentionResidualCharModel,
    optimizer: torch.optim.Optimizer,
    train_inputs: Tensor,
    train_targets: Tensor,
    *,
    batch_size: int,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    permutation = torch.randperm(train_inputs.shape[0], device=train_inputs.device)
    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    for start in range(0, permutation.shape[0], batch_size):
        batch_indices = permutation[start : start + batch_size]
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        batch_examples = batch_targets.shape[0]
        total_examples += batch_examples
        total_loss += loss.item() * batch_examples
        total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def overfit_one_batch(
    *,
    model: AttentionResidualCharModel,
    inputs: Tensor,
    targets: Tensor,
    config: RunConfig,
) -> tuple[list[dict[str, float | int]], dict[str, object]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.stage4_learning_rate)
    trace: list[dict[str, float | int]] = []
    final_loss = float("nan")
    final_accuracy = 0.0
    final_step = 0
    for step in range(config.stage4_steps + 1):
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
        if step == 0 or step == config.stage4_steps or step % config.stage4_log_interval == 0:
            trace.append(
                {
                    "step": step,
                    "loss": round(loss.item(), 6),
                    "accuracy": round(accuracy, 6),
                }
            )
        final_loss = loss.item()
        final_accuracy = accuracy
        final_step = step
        if accuracy == 1.0 and loss.item() < 0.02:
            break
        if step == config.stage4_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return trace, {
        "steps_run": final_step,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "memorized_batch": final_accuracy == 1.0 and final_loss < 0.02,
    }


def train_stage(
    *,
    model: AttentionResidualCharModel,
    dataset: FixedWindowCharDataset,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    prompt: str,
    config: RunConfig,
    device: torch.device,
    epochs: int,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int | str]], dict[str, float], float, str]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []
    progression_samples: list[dict[str, float | int | str]] = []
    sample_epochs = {epoch for epoch in config.sample_checkpoints if epoch <= epochs}
    started_at = time.perf_counter()

    initial_train_metrics = evaluate_model(model, train_inputs, train_targets, batch_size=config.eval_batch_size)
    initial_val_metrics = evaluate_model(model, val_inputs, val_targets, batch_size=config.eval_batch_size)
    history.append(
        {
            "epoch": 0,
            "train_loss": round(initial_train_metrics["loss"], 6),
            "train_accuracy": round(initial_train_metrics["accuracy"], 6),
            "val_loss": round(initial_val_metrics["loss"], 6),
            "val_accuracy": round(initial_val_metrics["accuracy"], 6),
        }
    )
    if 0 in sample_epochs:
        progression_samples.append(
            {
                "epoch": 0,
                "val_loss": round(initial_val_metrics["loss"], 6),
                "sample": capture_sample(
                    model,
                    dataset,
                    prompt,
                    length=config.progression_sample_length,
                    device=device,
                ),
            }
        )

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_inputs,
            train_targets,
            batch_size=config.batch_size,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        val_metrics = evaluate_model(model, val_inputs, val_targets, batch_size=config.eval_batch_size)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "train_accuracy": round(train_metrics["accuracy"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
            }
        )
        if epoch in sample_epochs:
            progression_samples.append(
                {
                    "epoch": epoch,
                    "val_loss": round(val_metrics["loss"], 6),
                    "sample": capture_sample(
                        model,
                        dataset,
                        prompt,
                        length=config.progression_sample_length,
                        device=device,
                    ),
                }
            )

    runtime_seconds = time.perf_counter() - started_at
    final_metrics = evaluate_model(model, val_inputs, val_targets, batch_size=config.eval_batch_size)
    sample = capture_sample(model, dataset, prompt, length=config.sample_length, device=device)
    return history, progression_samples, final_metrics, runtime_seconds, sample


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["overfit", "tiny", "full"], required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig()
    if args.stage == "tiny":
        config = replace_config(
            config,
            train_characters=config.tiny_train_characters,
            val_characters=config.tiny_val_characters,
            sample_length=160,
            progression_sample_length=120,
        )
    variant = VARIANTS[args.variant]
    text_file = args.text_file or (
        args.repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    )
    output_dir = args.output_dir or (
        args.repo_root / "experiments" / "causal_triangle_attention" / "artifacts" / args.stage / variant.label
    )

    set_seed(config.seed)
    device = resolve_device(args.device)
    raw_text = text_file.read_text(encoding="utf-8")
    split = build_fixed_length_split(raw_text, config=config)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(
        vocab_size=split.train_dataset.vocab_size,
        config=config,
        variant=variant,
        device=device,
    )
    write_metadata(
        output_dir,
        config=config,
        split=split,
        variant=variant,
        model=model,
        text_file=text_file,
        raw_text=raw_text,
        device=device,
        stage=args.stage,
    )

    if args.stage == "overfit":
        batch_inputs = split.train_inputs[: config.stage4_batch_size].to(device)
        batch_targets = split.train_targets[: config.stage4_batch_size].to(device)
        trace, final_metrics = overfit_one_batch(
            model=model,
            inputs=batch_inputs,
            targets=batch_targets,
            config=config,
        )
        write_json(output_dir / "overfit_trace.json", trace)
        write_json(output_dir / "final_metrics.json", final_metrics)
        print(
            f"{variant.label} overfit final_loss={final_metrics['final_loss']:.6f} "
            f"accuracy={final_metrics['final_accuracy']:.6f}"
        )
        return

    prompt = split.train_text[: config.context_size]
    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)
    epochs = config.tiny_epochs if args.stage == "tiny" else config.full_epochs
    history, progression_samples, final_metrics, runtime_seconds, sample = train_stage(
        model=model,
        dataset=split.train_dataset,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        prompt=prompt,
        config=config,
        device=device,
        epochs=epochs,
    )
    write_json(output_dir / "training_history.json", history)
    write_json(
        output_dir / "progression_samples.json",
        {
            "prompt": prompt.replace("\n", "\\n"),
            "sample_length": config.progression_sample_length,
            "checkpoints": progression_samples,
        },
    )
    best_epoch_record = min(history[1:] or history, key=lambda record: float(record["val_loss"]))
    write_json(
        output_dir / "final_metrics.json",
        {
            "parameter_count": count_parameters(model),
            "runtime_seconds": runtime_seconds,
            "final_val_loss": final_metrics["loss"],
            "final_val_accuracy": final_metrics["accuracy"],
            "best_val_loss": float(best_epoch_record["val_loss"]),
            "best_val_accuracy": float(best_epoch_record["val_accuracy"]),
            "best_epoch": int(best_epoch_record["epoch"]),
            "prompt": prompt.replace("\n", "\\n"),
        },
    )
    (output_dir / "sample.txt").write_text(sample, encoding="utf-8")
    print(
        f"{variant.label} {args.stage} best_val_loss={float(best_epoch_record['val_loss']):.6f} "
        f"runtime_seconds={runtime_seconds:.2f}"
    )


if __name__ == "__main__":
    main()
