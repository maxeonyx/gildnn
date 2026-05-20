from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from base_experiments.transformer import build_fixed_length_split
from core.fixed_window_char import generate_text, resolve_device, set_seed

from .char_model import ASYNC_STALE, SYNC_CONTROL, AsyncVolatileMemoryCharModel, TrainableConfig, count_parameters


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def current_git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def current_git_status_short() -> list[str]:
    result = subprocess.run(["git", "status", "--short"], check=True, capture_output=True, text=True)
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_model(*, vocab_size: int, config: TrainableConfig, variant, device: torch.device) -> AsyncVolatileMemoryCharModel:
    return AsyncVolatileMemoryCharModel(vocab_size=vocab_size, config=config, variant=variant).to(device)


def batched_pairs(inputs: Tensor, targets: Tensor, *, batch_size: int):
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        yield inputs[start:stop], targets[start:stop]


def evaluate_model(model: nn.Module, inputs: Tensor, targets: Tensor, *, batch_size: int) -> dict[str, float]:
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
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_inputs: Tensor,
    train_targets: Tensor,
    *,
    batch_size: int,
    gradient_clip_norm: float,
    permutation: Tensor,
) -> dict[str, float]:
    model.train()
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

        total_examples += batch_targets.shape[0]
        total_loss += loss.item() * batch_targets.shape[0]
        total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def fixed_epoch_permutation(size: int, *, epoch: int, seed: int, device: torch.device) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    permutation = torch.randperm(size, generator=generator)
    return permutation.to(device)


def train_variant(
    *,
    model: AsyncVolatileMemoryCharModel,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    train_dataset,
    config: TrainableConfig,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.train_epochs + 1):
        permutation = fixed_epoch_permutation(
            train_inputs.shape[0],
            epoch=epoch,
            seed=config.seed,
            device=train_inputs.device,
        )
        if train_inputs.device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_inputs,
            train_targets,
            batch_size=config.batch_size,
            gradient_clip_norm=config.gradient_clip_norm,
            permutation=permutation,
        )
        val_metrics = evaluate_model(
            model,
            val_inputs,
            val_targets,
            batch_size=config.eval_batch_size,
        )
        if train_inputs.device.type == "cuda":
            torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "train_accuracy": round(train_metrics["accuracy"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
                "epoch_seconds": round(epoch_seconds, 3),
            }
        )

    best_epoch = min(history, key=lambda row: row["val_loss"])
    prompt = "First Citizen:\nBefore we proceed"
    sample = generate_text(
        model,
        train_dataset,
        prompt,
        length=config.sample_length,
        device=train_inputs.device,
    )
    return {
        "history": history,
        "best_epoch": best_epoch,
        "final_epoch": history[-1],
        "sample_prompt": prompt,
        "sample_text": sample,
    }


def main() -> None:
    args = parse_args()
    config = TrainableConfig()
    text_file = args.text_file or (args.repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt")
    output_dir = args.output_dir or (args.repo_root / "experiments" / "async_volatile_memory" / "artifacts" / "stage5")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = resolve_device(args.device)
    raw_text = text_file.read_text(encoding="utf-8")
    split = build_fixed_length_split(raw_text, config=config)
    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)
    vocab_size = split.train_dataset.vocab_size

    sync_model = make_model(vocab_size=vocab_size, config=config, variant=SYNC_CONTROL, device=device)
    async_model = make_model(vocab_size=vocab_size, config=config, variant=ASYNC_STALE, device=device)
    async_model.load_state_dict(sync_model.state_dict())

    sync_result = train_variant(
        model=sync_model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        train_dataset=split.train_dataset,
        config=config,
    )
    async_result = train_variant(
        model=async_model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        train_dataset=split.train_dataset,
        config=config,
    )

    report = {
        "config": asdict(config),
        "environment": {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": current_git_status_short() == [],
            "git_status_short": current_git_status_short(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "corpus_summary": {
            "source_file": str(text_file),
            "source_total_characters": len(raw_text),
            "source_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "train_characters": len(split.train_text),
            "val_characters": len(split.val_text),
            "vocab_size": vocab_size,
        },
        "model": {
            "parameter_count": count_parameters(sync_model),
            "num_modules": config.num_modules,
            "num_ticks": config.num_ticks,
            "read_lags": {
                "synchronous_control": list(SYNC_CONTROL.read_lags),
                "async_stale_reads": list(ASYNC_STALE.read_lags),
            },
        },
        "training": {
            "synchronous_control": sync_result,
            "async_stale_reads": async_result,
        },
    }
    output_path = output_dir / "tiny_training_report.json"
    write_json(output_path, report)
    print("PASS async volatile memory stage5")
    print(f"Wrote {output_path}")
    print(
        json.dumps(
            {
                "synchronous_control": {
                    "best_epoch": sync_result["best_epoch"],
                    "final_epoch": sync_result["final_epoch"],
                },
                "async_stale_reads": {
                    "best_epoch": async_result["best_epoch"],
                    "final_epoch": async_result["final_epoch"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
