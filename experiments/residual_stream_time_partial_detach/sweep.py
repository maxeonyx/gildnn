"""Sweep detach_every_n values to map the quality-speed tradeoff.

Run: & .\.venv\Scripts\python.exe -m experiments.residual_stream_time_partial_detach.sweep
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from base_experiments.transformer import build_fixed_length_split
from core.fixed_window_char import FixedWindowCharDataset, set_seed

from .model import PartialDetachCharModel, PartialDetachConfig, count_parameters


REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_FILE = REPO_ROOT / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
ARTIFACTS_DIR = REPO_ROOT / "experiments" / "residual_stream_time_partial_detach" / "artifacts" / "sweep"

# Match the extended run config exactly.
CONTEXT_SIZE = 64
D_MODEL = 192
FEEDFORWARD_DIM = 768
NUM_HEADS = 4
TEMPORAL_WINDOW = 4
TRAIN_CHARS = 100_000
VAL_CHARS = 20_000
EPOCHS = 5
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 0.003
GRADIENT_CLIP = 1.0
SEED = 42

# N values to sweep. 1 = full stop-grad (known: 1.894), 0 = never detach.
# We test 2, 4, 8, 16 as intermediate points.
DETACH_VALUES = [2, 4, 8, 16]


class SplitHolder:
    """Minimal container matching build_fixed_length_split interface."""
    pass


def load_data(device: torch.device):
    """Load and prepare TinyShakespeare data."""
    raw_text = TEXT_FILE.read_text(encoding="utf-8")

    class FakeConfig:
        context_size = CONTEXT_SIZE
        train_characters = TRAIN_CHARS
        val_characters = VAL_CHARS

    split = build_fixed_length_split(raw_text, config=FakeConfig())
    return split, raw_text


def evaluate(model, inputs, targets, batch_size):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch_in = inputs[start:start + batch_size]
            batch_tgt = targets[start:start + batch_size]
            logits = model(batch_in)
            total_loss += F.cross_entropy(logits, batch_tgt, reduction="sum").item()
            total_correct += (logits.argmax(dim=1) == batch_tgt).sum().item()
            total_examples += batch_tgt.shape[0]
    model.train()
    return total_loss / total_examples, total_correct / total_examples


def train_one_epoch(model, optimizer, train_inputs, train_targets):
    model.train()
    perm = torch.randperm(train_inputs.shape[0], device=train_inputs.device)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for start in range(0, perm.shape[0], BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        batch_in = train_inputs[idx]
        batch_tgt = train_targets[idx]
        logits = model(batch_in)
        loss = F.cross_entropy(logits, batch_tgt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        n = batch_tgt.shape[0]
        total_loss += loss.item() * n
        total_correct += (logits.argmax(dim=1) == batch_tgt).sum().item()
        total_examples += n
    return total_loss / total_examples, total_correct / total_examples


def run_single(detach_n: int, split, device: torch.device) -> dict:
    """Run one full training with a specific detach_every_n value."""
    set_seed(SEED)
    config = PartialDetachConfig(
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        detach_every_n=detach_n,
        temporal_window=TEMPORAL_WINDOW,
        num_heads=NUM_HEADS,
    )
    model = PartialDetachCharModel(
        vocab_size=split.train_dataset.vocab_size,
        config=config,
    ).to(device)

    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history = []

    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, optimizer, train_inputs, train_targets)
        val_loss, val_acc = evaluate(model, val_inputs, val_targets, EVAL_BATCH_SIZE)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
        })
        print(f"  epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    runtime = time.perf_counter() - started
    final_val_loss, final_val_acc = evaluate(model, val_inputs, val_targets, EVAL_BATCH_SIZE)

    return {
        "detach_every_n": detach_n,
        "parameter_count": count_parameters(model),
        "runtime_seconds": round(runtime, 2),
        "final_val_loss": round(final_val_loss, 6),
        "final_val_accuracy": round(final_val_acc, 6),
        "history": history,
        "mix_coefficients": model.mix_coefficients(),
    }


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    split, raw_text = load_data(device)
    print(f"Data loaded: {split.train_inputs.shape[0]} train windows, {split.val_inputs.shape[0]} val windows")

    all_results = []
    for detach_n in DETACH_VALUES:
        print(f"\n=== Running detach_every_n={detach_n} ===")
        result = run_single(detach_n, split, device)
        all_results.append(result)
        # Save incrementally so partial results are available if interrupted.
        result_path = ARTIFACTS_DIR / f"detach_n{detach_n}.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"  => val_loss={result['final_val_loss']:.6f} runtime={result['runtime_seconds']:.1f}s")

    # Save summary.
    summary = {
        "sweep_config": {
            "context_size": CONTEXT_SIZE,
            "d_model": D_MODEL,
            "epochs": EPOCHS,
            "detach_values": DETACH_VALUES,
        },
        "known_endpoints": {
            "full_gradient_val_loss": 1.670509,
            "full_gradient_runtime": 606.0,
            "full_stopgrad_val_loss": 1.893501,
            "full_stopgrad_runtime": 257.0,
        },
        "results": [{k: v for k, v in r.items() if k != "history"} for r in all_results],
    }
    (ARTIFACTS_DIR / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== SWEEP COMPLETE ===")
    print(f"Full gradient: val_loss=1.670509, runtime=606s")
    print(f"Full stop-grad (N=1): val_loss=1.893501, runtime=257s")
    for r in all_results:
        print(f"N={r['detach_every_n']:3d}: val_loss={r['final_val_loss']:.6f}, runtime={r['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
