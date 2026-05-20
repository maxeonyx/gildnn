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
from torch import Tensor
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
    parser.add_argument("--variant", choices=["sync", "async"], default="sync")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
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


def fixed_step_indices(size: int, *, steps: int, batch_size: int, seed: int, device: torch.device) -> list[Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return [
        torch.randint(0, size, (batch_size,), generator=generator).to(device)
        for _ in range(steps)
    ]


def make_model(*, vocab_size: int, config: TrainableConfig, variant: VariantSpec, device: torch.device) -> AsyncGRUCharModel:
    return AsyncGRUCharModel(vocab_size=vocab_size, config=config, variant=variant).to(device)


def run_probe(
    model: AsyncGRUCharModel,
    train_inputs: Tensor,
    train_targets: Tensor,
    *,
    batch_indices: list[Tensor],
    config: TrainableConfig,
    warmup_steps: int,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    grad_norms: list[float] = []
    timed_step_ms: list[float] = []

    model.train()
    for step_index, indices in enumerate(batch_indices):
        batch_inputs = train_inputs[indices]
        batch_targets = train_targets[indices]
        optimizer.zero_grad(set_to_none=True)
        if train_inputs.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(train_inputs.device)
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step_index}: {loss.item()}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm).item())
        optimizer.step()
        if train_inputs.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        loss_value = float(loss.item())
        if not torch.isfinite(torch.tensor(loss_value)):
            raise RuntimeError(f"Non-finite scalar loss at step {step_index}: {loss_value}")
        losses.append(loss_value)
        grad_norms.append(grad_norm)
        if step_index >= warmup_steps:
            timed_step_ms.append(elapsed_ms)

    mean_ms = sum(timed_step_ms) / len(timed_step_ms) if timed_step_ms else 0.0
    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(train_inputs.device))
        if train_inputs.device.type == "cuda"
        else None
    )
    return {
        "fit": True,
        "steps": len(batch_indices),
        "warmup_steps": warmup_steps,
        "first_loss": round(losses[0], 6),
        "last_loss": round(losses[-1], 6),
        "min_loss": round(min(losses), 6),
        "max_loss": round(max(losses), 6),
        "loss_trace": [round(value, 6) for value in losses],
        "grad_norm_trace": [round(value, 6) for value in grad_norms],
        "mean_ms_per_step": round(mean_ms, 3),
        "tokens_per_step": config.batch_size * config.context_size,
        "tokens_per_second": round((config.batch_size * config.context_size * 1000.0) / mean_ms, 1) if mean_ms else None,
        "peak_memory_bytes": peak_memory_bytes,
    }


def main() -> None:
    args = parse_args()
    config = with_overrides(
        TrainableConfig(),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    text_file = args.text_file or (args.repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt")
    output_dir = args.output_dir or (args.repo_root / "experiments" / "async_gru_scaleup" / "artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = resolve_device(args.device)
    raw_text = text_file.read_text(encoding="utf-8")
    split = build_fixed_length_split(raw_text, config=config)
    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    vocab_size = split.train_dataset.vocab_size

    variant = make_sync_variant(config) if args.variant == "sync" else make_async_stale_variant(config)
    model = make_model(vocab_size=vocab_size, config=config, variant=variant, device=device)
    batch_indices = fixed_step_indices(
        train_inputs.shape[0],
        steps=args.steps,
        batch_size=config.batch_size,
        seed=config.seed,
        device=device,
    )

    try:
        result = run_probe(
            model,
            train_inputs,
            train_targets,
            batch_indices=batch_indices,
            config=config,
            warmup_steps=args.warmup_steps,
        )
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        result = {
            "fit": False,
            "oom": True,
            "error": str(error),
        }

    report = {
        "config": asdict(config),
        "probe": {
            "variant": variant.label,
            "requested_steps": args.steps,
            "warmup_steps": args.warmup_steps,
        },
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
            "parameter_count": count_parameters(model),
            "d_model": config.d_model,
            "num_modules": config.num_modules,
            "num_ticks": config.num_ticks,
            "read_lags": list(variant.read_lags),
        },
        "result": result,
    }
    file_stem = f"calibration_{args.variant}_bs{config.batch_size}_lr{str(config.learning_rate).replace('.', '_')}"
    output_path = output_dir / f"{file_stem}.json"
    write_json(output_path, report)
    print(f"Wrote {output_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
