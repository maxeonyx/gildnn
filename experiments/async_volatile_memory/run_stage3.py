from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from base_experiments.transformer import build_fixed_length_split
from core.fixed_window_char import resolve_device, set_seed

from .char_model import ASYNC_STALE, ASYNC_ZERO, SYNC_CONTROL, AsyncVolatileMemoryCharModel, TrainableConfig, count_parameters


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


def timed_training_steps(model, batch_inputs, batch_targets, *, config: TrainableConfig) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    step_times_ms: list[float] = []
    losses: list[float] = []

    for step in range(config.benchmark_warmup_steps + config.benchmark_steps):
        if batch_inputs.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        if batch_inputs.device.type == "cuda":
            torch.cuda.synchronize()
        duration_ms = (time.perf_counter() - start) * 1000.0

        if step >= config.benchmark_warmup_steps:
            step_times_ms.append(duration_ms)
            losses.append(loss.item())

    return {
        "mean_step_ms": round(statistics.fmean(step_times_ms), 3),
        "median_step_ms": round(statistics.median(step_times_ms), 3),
        "min_step_ms": round(min(step_times_ms), 3),
        "max_step_ms": round(max(step_times_ms), 3),
        "final_loss": round(losses[-1], 6),
        "timed_steps": config.benchmark_steps,
        "warmup_steps": config.benchmark_warmup_steps,
    }


def profile_forward(model, batch_inputs, *, repeats: int = 12) -> dict[str, float]:
    model.eval()
    read_ms: list[float] = []
    module_ms: list[float] = []
    commit_ms: list[float] = []
    wall_ms: list[float] = []
    with torch.no_grad():
        for _ in range(repeats):
            _, profile = model.forward_with_profile(batch_inputs)
            read_ms.append(profile.read_visibility_ms)
            module_ms.append(profile.module_compute_ms)
            commit_ms.append(profile.commit_ms)
            wall_ms.append(profile.forward_wall_ms)

    mean_read = statistics.fmean(read_ms)
    mean_module = statistics.fmean(module_ms)
    mean_commit = statistics.fmean(commit_ms)
    mean_wall = statistics.fmean(wall_ms)
    accounted = mean_read + mean_module + mean_commit
    untracked = max(0.0, mean_wall - accounted)
    return {
        "mean_forward_wall_ms": round(mean_wall, 3),
        "read_visibility_ms": round(mean_read, 3),
        "module_compute_ms": round(mean_module, 3),
        "commit_ms": round(mean_commit, 3),
        "accounted_forward_ms": round(accounted, 3),
        "untracked_forward_ms": round(untracked, 3),
        "module_compute_fraction_of_accounted": round(mean_module / accounted, 6) if accounted else 0.0,
        "bookkeeping_fraction_of_accounted": round((mean_read + mean_commit) / accounted, 6) if accounted else 0.0,
        "untracked_fraction_of_wall": round(untracked / mean_wall, 6) if mean_wall else 0.0,
        "profile_repeats": repeats,
    }


def zero_staleness_check(control_model, batch_inputs, *, config: TrainableConfig, vocab_size: int, device: torch.device) -> dict[str, float]:
    zero_model = make_model(vocab_size=vocab_size, config=config, variant=ASYNC_ZERO, device=device)
    zero_model.load_state_dict(control_model.state_dict())
    control_model.eval()
    zero_model.eval()
    with torch.no_grad():
        diff = (control_model(batch_inputs) - zero_model(batch_inputs)).abs().max().item()
    if diff > 1e-7:
        raise RuntimeError(f"Zero-staleness check failed with max logits diff {diff}.")
    return {"max_logits_abs_diff": round(diff, 6)}


def fairness_decision(sync_profile: dict[str, float], async_profile: dict[str, float]) -> dict[str, object]:
    async_module = async_profile["module_compute_fraction_of_accounted"]
    async_untracked = async_profile["untracked_fraction_of_wall"]
    fair_vehicle = async_module >= 0.8 and async_untracked <= 0.25
    return {
        "fair_vehicle": fair_vehicle,
        "criteria": {
            "module_compute_fraction_of_accounted_at_least": 0.8,
            "untracked_fraction_of_wall_at_most": 0.25,
        },
        "async_measured": {
            "module_compute_fraction_of_accounted": async_module,
            "untracked_fraction_of_wall": async_untracked,
        },
        "reason": (
            "dense module compute dominates forward time and untracked/Python overhead stays secondary"
            if fair_vehicle
            else "forward time is still too dominated by untracked or bookkeeping overhead"
        ),
    }


def main() -> None:
    args = parse_args()
    config = TrainableConfig()
    text_file = args.text_file or (args.repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt")
    output_dir = args.output_dir or (args.repo_root / "experiments" / "async_volatile_memory" / "artifacts" / "stage3")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = resolve_device(args.device)
    raw_text = text_file.read_text(encoding="utf-8")
    split = build_fixed_length_split(raw_text, config=config)
    batch_inputs = split.train_inputs[: config.batch_size].to(device)
    batch_targets = split.train_targets[: config.batch_size].to(device)
    vocab_size = split.train_dataset.vocab_size

    sync_model = make_model(vocab_size=vocab_size, config=config, variant=SYNC_CONTROL, device=device)
    async_model = make_model(vocab_size=vocab_size, config=config, variant=ASYNC_STALE, device=device)
    async_model.load_state_dict(sync_model.state_dict())

    zero_check = zero_staleness_check(sync_model, batch_inputs, config=config, vocab_size=vocab_size, device=device)

    sync_timing = timed_training_steps(sync_model, batch_inputs, batch_targets, config=config)
    async_timing = timed_training_steps(async_model, batch_inputs, batch_targets, config=config)

    sync_profile = profile_forward(sync_model, batch_inputs)
    async_profile = profile_forward(async_model, batch_inputs)
    fairness = fairness_decision(sync_profile, async_profile)

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
        "checks": {
            "zero_staleness_equivalence": zero_check,
        },
        "timing": {
            "synchronous_control": sync_timing,
            "async_stale_reads": async_timing,
            "delta_mean_step_ms": round(async_timing["mean_step_ms"] - sync_timing["mean_step_ms"], 3),
            "delta_mean_step_fraction": round(async_timing["mean_step_ms"] / sync_timing["mean_step_ms"] - 1.0, 6),
        },
        "forward_profile": {
            "synchronous_control": sync_profile,
            "async_stale_reads": async_profile,
        },
        "fairness_assessment": fairness,
    }
    output_path = output_dir / "timing_report.json"
    write_json(output_path, report)
    print("PASS async volatile memory stage3")
    print(f"Wrote {output_path}")
    print(json.dumps(report["fairness_assessment"], indent=2))


if __name__ == "__main__":
    main()
