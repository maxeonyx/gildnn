"""Parameter-matched Muon run: k=8 with cosine LR annealing for 15 epochs.

This is the definitive parameter-matched comparison against the 186K-parameter
transformer baseline: Muon + cosine LR + k=8 temporal window.

Run: python -m experiments.muon_window_ablation.cosine_param_matched
"""

from __future__ import annotations

import json
import math
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
OUTPUT_FILE = ARTIFACTS_DIR / "muon_cosine_param_matched.json"

CONTEXT_SIZE = 64
D_MODEL = 116
FEEDFORWARD_DIM = 464
NUM_HEADS = 4
TRAIN_CHARS = 100_000
VAL_CHARS = 20_000
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
GRADIENT_CLIP = 1.0
SEED = 42
TEMPORAL_WINDOW = 8
MUON_LR = 0.01
MUON_MIN_LR = 0.001
ADAMW_LR = 3e-4
ADAMW_MIN_LR = 3e-5
ADAMW_BETAS = (0.9, 0.95)
EPOCHS = 15

TRANSFORMER_REFERENCE = 1.632
TRANSFORMER_REFERENCE_PARAMS = 186_000
PREVIOUS_PARAM_MATCHED_MUON_K8_REFERENCE = 1.694
PREVIOUS_PARAM_MATCHED_MUON_K8_LABEL = "Previous param-matched Muon k=8 (5ep flat LR)"
PREVIOUS_PARAM_MATCHED_ADAMW_K4_REFERENCE = 1.717
PREVIOUS_PARAM_MATCHED_ADAMW_K4_LABEL = "Previous param-matched AdamW k=4"

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


class ProportionalCosineAnnealingLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, *, T_max, eta_mins, last_epoch=-1):
        if len(eta_mins) != len(optimizer.param_groups):
            raise ValueError(
                f"eta_mins length {len(eta_mins)} does not match param groups {len(optimizer.param_groups)}"
            )
        self.eta_mins = list(eta_mins)
        super().__init__(optimizer, T_max=T_max, eta_min=0.0, last_epoch=last_epoch)

    def get_lr(self):
        if self._is_initial:
            return [group["lr"] for group in self.optimizer.param_groups]
        if self._step_count == 1 and self.last_epoch > 0:
            return [
                eta_min + (base_lr - eta_min) * (1 + math.cos(self.last_epoch * math.pi / self.T_max)) / 2
                for base_lr, eta_min in zip(self.base_lrs, self.eta_mins, strict=True)
            ]
        if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"] + (base_lr - eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for group, base_lr, eta_min in zip(
                    self.optimizer.param_groups, self.base_lrs, self.eta_mins, strict=True
                )
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group["lr"] - eta_min)
            + eta_min
            for group, eta_min in zip(self.optimizer.param_groups, self.eta_mins, strict=True)
        ]

    def _get_closed_form_lr(self):
        return [
            eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr, eta_min in zip(self.base_lrs, self.eta_mins, strict=True)
        ]


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


def train_run(*, device, train_inputs, train_targets, val_inputs, val_targets, vocab_size):
    set_seed(SEED)
    config = ResidualStreamTimeMixAddConfig(
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        temporal_window=TEMPORAL_WINDOW,
        num_heads=NUM_HEADS,
    )
    model = ResidualStreamTimeMixAddCharModel(vocab_size=vocab_size, config=config).to(device)
    optimizer = build_optimizer(model, MUON_LR)
    scheduler = ProportionalCosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_mins=[MUON_MIN_LR, ADAMW_MIN_LR],
    )
    params = count_parameters(model)
    history = []
    diverged = False
    started = time.perf_counter()

    print(f"  model params={params:,}", flush=True)
    print(
        f"  cosine schedule: muon {MUON_LR:.4g}->{MUON_MIN_LR:.4g}, adamw {ADAMW_LR:.4g}->{ADAMW_MIN_LR:.4g}",
        flush=True,
    )

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

        current_muon_lr = optimizer.param_groups[0]["lr"]
        current_adamw_lr = optimizer.param_groups[1]["lr"]
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "muon_lr": round(current_muon_lr, 8),
            "adamw_lr": round(current_adamw_lr, 8),
        })
        print(
            f"  k={TEMPORAL_WINDOW} ep={epoch} train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.4f} muon_lr={current_muon_lr:.6f} adamw_lr={current_adamw_lr:.6f}",
            flush=True,
        )

        if train_loss > 10:
            diverged = True
            print(f"  DIVERGED at epoch {epoch}", flush=True)
            break

        scheduler.step()

    runtime = time.perf_counter() - started
    best_val = min(h["val_loss"] for h in history)
    final_val = history[-1]["val_loss"]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "k": TEMPORAL_WINDOW,
        "muon_lr": MUON_LR,
        "muon_min_lr": MUON_MIN_LR,
        "adamw_lr": ADAMW_LR,
        "adamw_min_lr": ADAMW_MIN_LR,
        "params": params,
        "epochs": len(history),
        "best_val": round(best_val, 6),
        "final_val": final_val,
        "diverged": diverged,
        "runtime": round(runtime, 1),
        "history": history,
    }


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

    print(
        f"\nStarting param-matched k={TEMPORAL_WINDOW} with cosine Muon LR={MUON_LR} for {EPOCHS} epochs",
        flush=True,
    )
    result = train_run(
        device=device,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        vocab_size=vocab_size,
    )

    print("\n=== SUMMARY ===", flush=True)
    print(f"Muon cosine LR: {MUON_LR} -> {MUON_MIN_LR}", flush=True)
    status = "DIVERGED" if result["diverged"] else f"{result['best_val']:.4f}"
    print(f"Muon k=8 cosine, param-matched, 15ep: best_val={status}", flush=True)
    print(
        f"Transformer: {TRANSFORMER_REFERENCE:.3f} ({TRANSFORMER_REFERENCE_PARAMS // 1000}K params)",
        flush=True,
    )
    print(
        f"{PREVIOUS_PARAM_MATCHED_MUON_K8_LABEL}: {PREVIOUS_PARAM_MATCHED_MUON_K8_REFERENCE:.3f}",
        flush=True,
    )
    print(
        f"{PREVIOUS_PARAM_MATCHED_ADAMW_K4_LABEL}: {PREVIOUS_PARAM_MATCHED_ADAMW_K4_REFERENCE:.3f}",
        flush=True,
    )
    if not result["diverged"]:
        print(
            f"Delta vs transformer: {result['best_val'] - TRANSFORMER_REFERENCE:+.4f}",
            flush=True,
        )
        print(
            f"Delta vs previous param-matched Muon k=8: {result['best_val'] - PREVIOUS_PARAM_MATCHED_MUON_K8_REFERENCE:+.4f}",
            flush=True,
        )
        print(
            f"Delta vs previous param-matched AdamW k=4: {result['best_val'] - PREVIOUS_PARAM_MATCHED_ADAMW_K4_REFERENCE:+.4f}",
            flush=True,
        )

    payload = {
        "muon_lr": MUON_LR,
        "muon_min_lr": MUON_MIN_LR,
        "adamw_lr": ADAMW_LR,
        "adamw_min_lr": ADAMW_MIN_LR,
        "epochs": EPOCHS,
        "d_model": D_MODEL,
        "feedforward_dim": FEEDFORWARD_DIM,
        "num_heads": NUM_HEADS,
        "temporal_window": TEMPORAL_WINDOW,
        "references": {
            "transformer": {
                "val": TRANSFORMER_REFERENCE,
                "params": TRANSFORMER_REFERENCE_PARAMS,
            },
            "previous_param_matched_muon_k8_5ep_flat_lr": PREVIOUS_PARAM_MATCHED_MUON_K8_REFERENCE,
            "previous_param_matched_adamw_k4": PREVIOUS_PARAM_MATCHED_ADAMW_K4_REFERENCE,
        },
        "result": result,
    }
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nSaved to {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
