"""Parameter-matched comparison: d_model=116 (185K params) vs transformer (186K params).

Answers: is the architecture fundamentally less parameter-efficient than transformers,
or does the d_model=192 result (1.670 at 479K) reflect excess capacity?

Run: & .\.venv\Scripts\python.exe -m experiments.residual_stream_time_partial_detach.param_matched
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
ARTIFACTS_DIR = REPO_ROOT / "experiments" / "residual_stream_time_partial_detach" / "artifacts" / "param_matched"

# Same training config as extended runs.
CONTEXT_SIZE = 64
D_MODEL = 116  # gives ~185K params, matching transformer's 186K
FEEDFORWARD_DIM = 464  # 4 * d_model
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
    print(f"Model: d_model={D_MODEL}, params={params:,}", flush=True)

    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history = []
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        perm = torch.randperm(train_inputs.shape[0], device=device)
        total_loss = 0.0
        total_correct = 0
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
            total_correct += (logits.argmax(1) == train_targets[idx]).sum().item()
            total_n += n
        train_loss = total_loss / total_n

        # Eval
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_n = 0
        with torch.no_grad():
            for start in range(0, val_inputs.shape[0], EVAL_BATCH_SIZE):
                batch_in = val_inputs[start:start + EVAL_BATCH_SIZE]
                batch_tgt = val_targets[start:start + EVAL_BATCH_SIZE]
                logits = model(batch_in)
                val_loss_sum += F.cross_entropy(logits, batch_tgt, reduction="sum").item()
                val_correct += (logits.argmax(1) == batch_tgt).sum().item()
                val_n += batch_tgt.shape[0]
        val_loss = val_loss_sum / val_n
        val_acc = val_correct / val_n

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
        })
        print(f"  epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.4f}", flush=True)

    runtime = time.perf_counter() - started
    result = {
        "d_model": D_MODEL,
        "parameter_count": params,
        "runtime_seconds": round(runtime, 2),
        "final_val_loss": history[-1]["val_loss"],
        "final_val_accuracy": history[-1]["val_accuracy"],
        "history": history,
        "comparison": {
            "transformer_186K": {"val_loss": 1.632, "params": 186_000},
            "mixadd_479K": {"val_loss": 1.670, "params": 479_000},
        },
    }
    (ARTIFACTS_DIR / "param_matched_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nDONE: val_loss={result['final_val_loss']:.6f} at {params:,} params, runtime={runtime:.1f}s", flush=True)
    print(f"Transformer anchor: 1.632 at 186K params", flush=True)
    print(f"Gap: +{result['final_val_loss'] - 1.632:.4f} nats at matched params", flush=True)


if __name__ == "__main__":
    main()
