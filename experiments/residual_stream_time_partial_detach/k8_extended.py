r"""Extended k=8 run: 15 epochs at LR=0.001 to test convergence floor.

k=8 at LR=0.001 gave val 1.757 in 5 epochs (still declining).
k=4 at LR=0.003 gave val 1.670 in 5 epochs (still declining).
Question: does k=8 eventually beat k=4, given enough training at a stable LR?

Run: & .\.venv\Scripts\python.exe -m experiments.residual_stream_time_partial_detach.k8_extended
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from base_experiments.transformer import build_fixed_length_split
from core.fixed_window_char import set_seed

from core.model import (
    ResidualStreamTimeMixAddCharModel,
    ResidualStreamTimeMixAddConfig,
    count_parameters,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_FILE = REPO_ROOT / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
ARTIFACTS_DIR = REPO_ROOT / "experiments" / "residual_stream_time_partial_detach" / "artifacts" / "k8_extended"

CONTEXT_SIZE = 64
D_MODEL = 192
FEEDFORWARD_DIM = 768
NUM_HEADS = 4
TEMPORAL_WINDOW = 8
TRAIN_CHARS = 100_000
VAL_CHARS = 20_000
EPOCHS = 15
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 0.001
GRADIENT_CLIP = 1.0
SEED = 42


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
    print(f"Data: {split.train_inputs.shape[0]} train, {split.val_inputs.shape[0]} val", flush=True)

    set_seed(SEED)
    config = ResidualStreamTimeMixAddConfig(
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        temporal_window=TEMPORAL_WINDOW,
        num_heads=NUM_HEADS,
    )
    model = ResidualStreamTimeMixAddCharModel(
        vocab_size=split.train_dataset.vocab_size,
        config=config,
    ).to(device)
    params = count_parameters(model)
    print(f"Model: k={TEMPORAL_WINDOW}, d={D_MODEL}, params={params:,}, LR={LEARNING_RATE}, epochs={EPOCHS}", flush=True)

    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history = []
    best_val = float("inf")
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
        best_val = min(best_val, val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
        })
        print(f"  epoch {epoch:2d}: train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.4f} best={best_val:.4f}", flush=True)

        # Early stop on divergence.
        if train_loss > 100:
            print("  DIVERGED", flush=True)
            break

    runtime = time.perf_counter() - started
    result = {
        "temporal_window": TEMPORAL_WINDOW,
        "learning_rate": LEARNING_RATE,
        "epochs_run": len(history),
        "parameter_count": params,
        "runtime_seconds": round(runtime, 2),
        "best_val_loss": round(best_val, 6),
        "final_val_loss": history[-1]["val_loss"],
        "final_val_accuracy": history[-1]["val_accuracy"],
        "history": history,
        "comparison": {
            "k4_lr003_5ep": 1.670509,
            "k8_lr001_5ep": 1.757422,
            "transformer_186K": 1.632,
        },
    }
    (ARTIFACTS_DIR / "k8_extended_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nDONE: best_val={best_val:.6f} final_val={history[-1]['val_loss']:.6f} runtime={runtime:.1f}s", flush=True)
    print(f"Baseline (k=4, LR=0.003, 5ep): 1.670509", flush=True)
    print(f"Transformer: 1.632", flush=True)
    gap = best_val - 1.670509
    print(f"Gap to baseline: {gap:+.4f} nats", flush=True)


if __name__ == "__main__":
    main()
