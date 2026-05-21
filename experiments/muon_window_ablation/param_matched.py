"""Parameter-matched Muon test: d_model=116, k=8, Muon LR=0.01, 5 epochs.

Previous parameter-matched result (AdamW, k=4, d=116): val 1.717 at 184K params.
Transformer anchor: val 1.632 at 186K params.
Can Muon + k=8 close the 0.085 gap?

Run: & .\.venv\Scripts\python.exe -u -m experiments.muon_window_ablation.param_matched
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from muon import SingleDeviceMuonWithAuxAdam
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
ARTIFACTS_DIR = REPO_ROOT / "experiments" / "muon_window_ablation" / "artifacts"

CONTEXT_SIZE = 64
D_MODEL = 116  # Parameter-matched to transformer (186K)
FEEDFORWARD_DIM = 464  # 4x d_model
NUM_HEADS = 4
TEMPORAL_WINDOW = 8
TRAIN_CHARS = 100_000
VAL_CHARS = 20_000
EPOCHS = 5
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
GRADIENT_CLIP = 1.0
SEED = 42
MUON_LR = 0.01
ADAMW_LR = 3e-4
ADAMW_BETAS = (0.9, 0.95)

MUON_PARAMETER_NAMES = {
    "temporal_attention.query.weight",
    "temporal_attention.key.weight",
    "temporal_attention.value.weight",
    "temporal_attention.output.weight",
    "block.proj_in.weight",
    "block.proj_out.weight",
}


class FakeConfig:
    context_size = CONTEXT_SIZE
    train_characters = TRAIN_CHARS
    val_characters = VAL_CHARS


def build_optimizer(model):
    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if name in MUON_PARAMETER_NAMES:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return SingleDeviceMuonWithAuxAdam([
        {"params": muon_params, "lr": MUON_LR, "weight_decay": 0.0, "use_muon": True},
        {"params": adamw_params, "lr": ADAMW_LR, "betas": ADAMW_BETAS, "weight_decay": 0.0, "use_muon": False},
    ])


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    raw_text = TEXT_FILE.read_text(encoding="utf-8")
    split = build_fixed_length_split(raw_text, config=FakeConfig())
    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)
    vocab_size = split.train_dataset.vocab_size
    print(f"Data: {train_inputs.shape[0]} train, {val_inputs.shape[0]} val", flush=True)

    set_seed(SEED)
    config = ResidualStreamTimeMixAddConfig(
        context_size=CONTEXT_SIZE, d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM, temporal_window=TEMPORAL_WINDOW, num_heads=NUM_HEADS,
    )
    model = ResidualStreamTimeMixAddCharModel(vocab_size=vocab_size, config=config).to(device)
    optimizer = build_optimizer(model)
    params = count_parameters(model)
    print(f"Model: d={D_MODEL}, k={TEMPORAL_WINDOW}, params={params:,}, Muon LR={MUON_LR}", flush=True)

    history = []
    best_val = float("inf")
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(train_inputs.shape[0], device=device)
        total_loss, total_n = 0.0, 0
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
        val_loss_sum, val_correct, val_n = 0.0, 0, 0
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

        history.append({"epoch": epoch, "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6), "val_accuracy": round(val_acc, 6)})
        print(f"  epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.4f} best={best_val:.4f}", flush=True)

        if train_loss > 10:
            print("  DIVERGED", flush=True)
            break

    runtime = time.perf_counter() - started
    result = {
        "d_model": D_MODEL,
        "temporal_window": TEMPORAL_WINDOW,
        "parameter_count": params,
        "muon_lr": MUON_LR,
        "epochs": len(history),
        "best_val_loss": round(best_val, 6),
        "runtime_seconds": round(runtime, 1),
        "history": history,
        "comparison": {
            "transformer_186K": 1.632,
            "adamw_k4_d116": 1.717,
            "muon_k8_d192": 1.664,
        },
    }
    (ARTIFACTS_DIR / "muon_param_matched.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nDONE: params={params:,} best_val={best_val:.4f} runtime={runtime:.1f}s", flush=True)
    print(f"Transformer (186K): 1.632", flush=True)
    print(f"AdamW k=4 d=116: 1.717 (gap +0.085)", flush=True)
    print(f"This run gap: {best_val - 1.632:+.4f}", flush=True)


if __name__ == "__main__":
    main()
