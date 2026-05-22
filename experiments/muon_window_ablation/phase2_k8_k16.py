r"""Muon Phase 2 continuation: k=8 and k=16 at Muon LR=0.01, 5 epochs.

Phase 1 already found LR=0.01 is optimal for k=4.
k=4 at LR=0.01/5ep achieved val 1.670 — identical to AdamW baseline.
Now test: does Muon stabilize k=8 and k=16?

Run: & .\.venv\Scripts\python.exe -u -m experiments.muon_window_ablation.phase2_k8_k16
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
from core.model import (
    ResidualStreamTimeMixAddCharModel,
    ResidualStreamTimeMixAddConfig,
    count_parameters,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_FILE = REPO_ROOT / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
ARTIFACTS_DIR = REPO_ROOT / "experiments" / "muon_window_ablation" / "artifacts"

CONTEXT_SIZE = 64
D_MODEL = 192
FEEDFORWARD_DIM = 768
NUM_HEADS = 4
TRAIN_CHARS = 100_000
VAL_CHARS = 20_000
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
GRADIENT_CLIP = 1.0
SEED = 42
MUON_LR = 0.01
ADAMW_LR = 3e-4
ADAMW_BETAS = (0.9, 0.95)
EPOCHS = 5

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


def build_optimizer(model, muon_lr):
    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if name in MUON_PARAMETER_NAMES:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return SingleDeviceMuonWithAuxAdam([
        {"params": muon_params, "lr": muon_lr, "weight_decay": 0.0, "use_muon": True},
        {"params": adamw_params, "lr": ADAMW_LR, "betas": ADAMW_BETAS, "weight_decay": 0.0, "use_muon": False},
    ])


def train_run(*, temporal_window, device, train_inputs, train_targets, val_inputs, val_targets, vocab_size):
    set_seed(SEED)
    config = ResidualStreamTimeMixAddConfig(
        context_size=CONTEXT_SIZE, d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM, temporal_window=temporal_window, num_heads=NUM_HEADS,
    )
    model = ResidualStreamTimeMixAddCharModel(vocab_size=vocab_size, config=config).to(device)
    optimizer = build_optimizer(model, MUON_LR)
    params = count_parameters(model)
    history = []
    diverged = False
    started = time.perf_counter()

    print(f"  model params={params:,}", flush=True)

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

        history.append({"epoch": epoch, "train_loss": round(train_loss, 6), "val_loss": round(val_loss, 6), "val_accuracy": round(val_acc, 6)})
        print(f"  k={temporal_window} ep={epoch} train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.4f}", flush=True)

        if train_loss > 10:
            diverged = True
            print(f"  DIVERGED at epoch {epoch}", flush=True)
            break

    runtime = time.perf_counter() - started
    best_val = min(h["val_loss"] for h in history)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"k": temporal_window, "muon_lr": MUON_LR, "params": params, "epochs": len(history), "best_val": round(best_val, 6), "final_val": history[-1]["val_loss"], "diverged": diverged, "runtime": round(runtime, 1), "history": history}


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

    results = []
    for k in [8, 16]:
        print(f"\nStarting k={k} with Muon LR={MUON_LR}", flush=True)
        result = train_run(
            temporal_window=k, device=device,
            train_inputs=train_inputs, train_targets=train_targets,
            val_inputs=val_inputs, val_targets=val_targets, vocab_size=vocab_size,
        )
        results.append(result)

    # Also include the k=4 result we already have
    k4_result = {"k": 4, "muon_lr": MUON_LR, "best_val": 1.670571, "diverged": False, "note": "from prior run"}

    print("\n=== SUMMARY ===", flush=True)
    print(f"Muon LR: {MUON_LR}", flush=True)
    print(f"AdamW reference: k=4=1.670, k=8=DIVERGED, k=16=DIVERGED", flush=True)
    print(f"Muon k=4: best_val=1.6706 (from prior run)", flush=True)
    for r in results:
        status = "DIVERGED" if r["diverged"] else f"{r['best_val']:.4f}"
        print(f"Muon k={r['k']}: best_val={status}", flush=True)

    all_stable = all(not r["diverged"] for r in results)
    if all_stable:
        print("\nSUCCESS: Muon stabilizes k=8 and k=16!", flush=True)
        k8_val = next(r["best_val"] for r in results if r["k"] == 8)
        k16_val = next(r["best_val"] for r in results if r["k"] == 16)
        print(f"Quality ordering: k=16({k16_val:.4f}) vs k=8({k8_val:.4f}) vs k=4(1.6706)", flush=True)
    else:
        diverged_ks = [r["k"] for r in results if r["diverged"]]
        print(f"\nStill diverged at k={diverged_ks}", flush=True)

    payload = {"muon_lr": MUON_LR, "k4_reference": k4_result, "results": results}
    (ARTIFACTS_DIR / "muon_phase2_k8_k16.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nSaved to {ARTIFACTS_DIR / 'muon_phase2_k8_k16.json'}", flush=True)


if __name__ == "__main__":
    main()
