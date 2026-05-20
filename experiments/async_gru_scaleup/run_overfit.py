from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch.nn import functional as F

from base_experiments.transformer import build_fixed_length_split
from core.fixed_window_char import resolve_device, set_seed

from .char_model import (
    AsyncGRUCharModel,
    TrainableConfig,
    VariantSpec,
    count_parameters,
    make_async_stale_variant,
    make_sync_variant,
    with_overrides,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--memorization-batch-size", type=int)
    parser.add_argument("--memorization-steps", type=int)
    parser.add_argument("--memorization-learning-rate", type=float)
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


def make_model(*, vocab_size: int, config: TrainableConfig, variant: VariantSpec, device: torch.device) -> AsyncGRUCharModel:
    return AsyncGRUCharModel(vocab_size=vocab_size, config=config, variant=variant).to(device)


def overfit_one_batch(model, batch_inputs, batch_targets, *, config: TrainableConfig) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.memorization_learning_rate)
    trace: list[dict[str, float | int]] = []
    hit_step: int | None = None

    for step in range(config.memorization_steps + 1):
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)
        accuracy = (logits.argmax(dim=1) == batch_targets).float().mean().item()

        if step % 25 == 0 or step == config.memorization_steps:
            trace.append({"step": step, "loss": round(loss.item(), 6), "accuracy": round(accuracy, 6)})

        if hit_step is None and accuracy == 1.0 and loss.item() < 0.02:
            hit_step = step

        if step == config.memorization_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

    final_logits = model(batch_inputs)
    final_loss = F.cross_entropy(final_logits, batch_targets).item()
    final_accuracy = (final_logits.argmax(dim=1) == batch_targets).float().mean().item()
    return {
        "trace": trace,
        "final_loss": round(final_loss, 6),
        "final_accuracy": round(final_accuracy, 6),
        "memorized": final_accuracy == 1.0 and final_loss < 0.02,
        "hit_step": hit_step,
    }


def main() -> None:
    args = parse_args()
    config = with_overrides(
        TrainableConfig(),
        memorization_batch_size=args.memorization_batch_size,
        memorization_steps=args.memorization_steps,
        memorization_learning_rate=args.memorization_learning_rate,
    )
    text_file = args.text_file or (args.repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt")
    output_dir = args.output_dir or (args.repo_root / "experiments" / "async_gru_scaleup" / "artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = resolve_device(args.device)
    raw_text = text_file.read_text(encoding="utf-8")
    split = build_fixed_length_split(raw_text, config=config)
    batch_inputs = split.train_inputs[: config.memorization_batch_size].to(device)
    batch_targets = split.train_targets[: config.memorization_batch_size].to(device)
    vocab_size = split.train_dataset.vocab_size

    sync_variant = make_sync_variant(config)
    async_variant = make_async_stale_variant(config)
    sync_model = make_model(vocab_size=vocab_size, config=config, variant=sync_variant, device=device)
    async_model = make_model(vocab_size=vocab_size, config=config, variant=async_variant, device=device)
    async_model.load_state_dict(sync_model.state_dict())

    sync_result = overfit_one_batch(sync_model, batch_inputs, batch_targets, config=config)
    async_result = overfit_one_batch(async_model, batch_inputs, batch_targets, config=config)

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
            "d_model": config.d_model,
            "num_modules": config.num_modules,
            "num_ticks": config.num_ticks,
            "read_lags": {
                sync_variant.label: list(sync_variant.read_lags),
                async_variant.label: list(async_variant.read_lags),
            },
        },
        "overfit": {
            sync_variant.label: sync_result,
            async_variant.label: async_result,
        },
    }
    output_path = output_dir / "overfit_report.json"
    write_json(output_path, report)
    print("PASS async_gru_scaleup overfit")
    print(f"Wrote {output_path}")
    print(
        json.dumps(
            {
                sync_variant.label: {
                    "memorized": sync_result["memorized"],
                    "hit_step": sync_result["hit_step"],
                    "final_loss": sync_result["final_loss"],
                    "final_accuracy": sync_result["final_accuracy"],
                },
                async_variant.label: {
                    "memorized": async_result["memorized"],
                    "hit_step": async_result["hit_step"],
                    "final_loss": async_result["final_loss"],
                    "final_accuracy": async_result["final_accuracy"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
