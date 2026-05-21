"""Window size ablation: k=4 (baseline), k=8, k=16.

Tests whether more temporal context improves quality.
Interesting because k=4 matched the partial-detach sweet spot — is that causal?

Run: & .\.venv\Scripts\python.exe -m experiments.residual_stream_time_partial_detach.window_ablation
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from base_experiments.transformer import build_fixed_length_split
from core.fixed_window_char import set_seed

from experiments.residual_stream_time_mixadd.model import (
    ResidualStreamTimeMixAddCharModel,
    ResidualStreamTimeMixAddConfig,
    count_parameters,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_FILE = REPO_ROOT / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
ARTIFACTS_DIR = REPO_ROOT / "experiments" / "residual_stream_time_partial_detach" / "artifacts" / "window_ablation"

# Same config as extended run, varying only temporal_window.
CONTEXT_SIZE = 64
D_MODEL = 192
FEEDFORWARD_DIM = 768
NUM_HEADS = 4
TRAIN_CHARS = 100_000
VAL_CHARS = 20_000
EPOCHS = 5
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 0.003
GRADIENT_CLIP = 1.0
SEED = 42

WINDOW_VALUES = [8, 16]  # k=4 is already known: val 1.670


def run_single(window_k: int, split, device: torch.device) -> dict:
    set_seed(SEED)
    config = ResidualStreamTimeMixAddConfig(
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        temporal_window=window_k,
        num_heads=NUM_HEADS,
    )
    model = ResidualStreamTimeMixAddCharModel(
        vocab_size=split.train_dataset.vocab_size,
        config=config,
    ).to(device)
    params = count_parameters(model)

    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history = []
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(train_inputs.shape[0], device=device)
        total_loss = 0.0
        total_n = 0
        for start in range(0, perm.shape[0], BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            logits = model(train_inputs[idx])
            loss = F.cross_entropy(logits, train_targets[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            n = idx.shape[0]
            total_loss += loss.item() * n
            total_n += n
        train_loss = total_loss / total_n

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_n = 0
        with torch.no_grad():
            for start in range(0, val_inputs.shape[0], EVAL_BATCH_SIZE):
                b_in = val_inputs[start:start + EVAL_BATCH_SIZE]
                b_tgt = val_targets[start:start + EVAL_BATCH_SIZE]
                logits = model(b_in)
                val_loss_sum += F.cross_entropy(logits, b_tgt, reduction="sum").item()
                val_correct += (logits.argmax(1) == b_tgt).sum().item()
                val_n += b_tgt.shape[0]
        val_loss = val_loss_sum / val_n
        val_acc = val_correct / val_n

        history.append({"epoch": epoch, "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6), "val_accuracy": round(val_acc, 6)})
        print(f"  epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.4f}", flush=True)

    runtime = time.perf_counter() - started
    return {
        "temporal_window": window_k,
        "parameter_count": params,
        "runtime_seconds": round(runtime, 2),
        "final_val_loss": history[-1]["val_loss"],
        "final_val_accuracy": history[-1]["val_accuracy"],
        "history": history,
    }


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    raw_text = TEXT_FILE.read_text(encoding="utf-8")

    class FakeConfig:
        context_size = CONTEXT_SIZE
        train_characters = TRAIN_CHARS
        val_characters = VAL_CHARS

    split = build_fixed_length_split(raw_text, config=FakeConfig())
    print(f"Data: {split.train_inputs.shape[0]} train, {split.val_inputs.shape[0]} val windows", flush=True)

    all_results = []
    for k in WINDOW_VALUES:
        print(f"\n=== Running temporal_window={k} ===", flush=True)
        result = run_single(k, split, device)
        all_results.append(result)
        (ARTIFACTS_DIR / f"window_k{k}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"  => val_loss={result['final_val_loss']:.6f} runtime={result['runtime_seconds']:.1f}s", flush=True)

    summary = {
        "baseline_k4": {"val_loss": 1.670509, "runtime": 606.0, "params": 479872},
        "results": all_results,
    }
    (ARTIFACTS_DIR / "window_ablation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== WINDOW ABLATION COMPLETE ===", flush=True)
    print(f"k=4 (known): val_loss=1.670509, runtime=606s", flush=True)
    for r in all_results:
        print(f"k={r['temporal_window']:2d}: val_loss={r['final_val_loss']:.6f}, runtime={r['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
