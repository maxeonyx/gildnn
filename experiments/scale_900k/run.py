from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from muon import SingleDeviceMuonWithAuxAdam
from torch import Tensor, nn
from torch.nn import functional as F

from base_experiments.transformer import build_fixed_length_split
from core.fixed_window_char import set_seed
from core.model import (
    ResidualStreamTimeMixAddCharModel,
    ResidualStreamTimeMixAddConfig,
    count_parameters,
)
from legacy_models.tiny_char_transformer import (
    PlainResidualCombine,
    TinyTransformerCharModel,
    count_parameters as tf_count_parameters,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_FILE = REPO_ROOT / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
ARTIFACTS_DIR = REPO_ROOT / "experiments" / "scale_900k" / "artifacts"
OUTPUT_FILE = ARTIFACTS_DIR / "results.json"

CONTEXT_SIZE = 64
TRAIN_CHARS = 900_000
VAL_CHARS = 100_000
EPOCHS = 5
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
GRADIENT_CLIP = 1.0
SEED = 42

RESIDUAL_D_MODEL = 116
RESIDUAL_FEEDFORWARD_DIM = 464
RESIDUAL_NUM_HEADS = 4
RESIDUAL_TEMPORAL_WINDOW = 8
MUON_LR = 0.003
MUON_MIN_LR = 0.0003
ADAMW_LR = 1e-4
ADAMW_MIN_LR = 1e-5
ADAMW_BETAS = (0.9, 0.95)

TRANSFORMER_D_MODEL = 72
TRANSFORMER_FEEDFORWARD_DIM = 256
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_NUM_LAYERS = 3
TRANSFORMER_LR = 0.001
TRANSFORMER_MIN_LR = 0.0001

MUON_PARAMETER_NAMES = {
    "temporal_attention.query.weight",
    "temporal_attention.key.weight",
    "temporal_attention.value.weight",
    "temporal_attention.output.weight",
    "block.proj_in.weight",
    "block.proj_out.weight",
}


@dataclass(frozen=True)
class SplitConfig:
    context_size: int = CONTEXT_SIZE
    train_characters: int = TRAIN_CHARS
    val_characters: int = VAL_CHARS


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


def current_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def batched_pairs(inputs: Tensor, targets: Tensor, *, batch_size: int):
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        yield inputs[start:stop], targets[start:stop]


def evaluate_model(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in batched_pairs(inputs, targets, batch_size=batch_size):
            logits = model(batch_inputs)
            total_loss += F.cross_entropy(logits, batch_targets, reduction="sum").item()
            total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
            total_examples += batch_targets.shape[0]
    if was_training:
        model.train()
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_inputs: Tensor,
    train_targets: Tensor,
    *,
    batch_size: int,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    permutation = torch.randperm(train_inputs.shape[0], device=train_inputs.device)
    total_examples = 0
    total_loss = 0.0
    total_correct = 0

    for start in range(0, permutation.shape[0], batch_size):
        batch_indices = permutation[start : start + batch_size]
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        batch_examples = batch_targets.shape[0]
        total_examples += batch_examples
        total_loss += loss.item() * batch_examples
        total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def round_metric(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def build_residual_optimizer(
    model: ResidualStreamTimeMixAddCharModel,
) -> SingleDeviceMuonWithAuxAdam:
    muon_params = []
    adamw_params = []
    found_muon_names: set[str] = set()

    for name, parameter in model.named_parameters():
        if name in MUON_PARAMETER_NAMES:
            if parameter.ndim != 2:
                raise ValueError(f"Expected Muon parameter {name} to be 2D, got shape {tuple(parameter.shape)}.")
            muon_params.append(parameter)
            found_muon_names.add(name)
        else:
            adamw_params.append(parameter)

    if found_muon_names != MUON_PARAMETER_NAMES:
        missing = sorted(MUON_PARAMETER_NAMES - found_muon_names)
        extra = sorted(found_muon_names - MUON_PARAMETER_NAMES)
        raise ValueError(f"Muon parameter partition mismatch. Missing={missing} extra={extra}")

    return SingleDeviceMuonWithAuxAdam(
        [
            {
                "params": muon_params,
                "lr": MUON_LR,
                "weight_decay": 0.0,
                "use_muon": True,
            },
            {
                "params": adamw_params,
                "lr": ADAMW_LR,
                "betas": ADAMW_BETAS,
                "weight_decay": 0.0,
                "use_muon": False,
            },
        ]
    )


def build_residual_model(vocab_size: int, *, device: torch.device) -> ResidualStreamTimeMixAddCharModel:
    config = ResidualStreamTimeMixAddConfig(
        context_size=CONTEXT_SIZE,
        d_model=RESIDUAL_D_MODEL,
        feedforward_dim=RESIDUAL_FEEDFORWARD_DIM,
        num_heads=RESIDUAL_NUM_HEADS,
        temporal_window=RESIDUAL_TEMPORAL_WINDOW,
    )
    return ResidualStreamTimeMixAddCharModel(vocab_size=vocab_size, config=config).to(device)


def build_transformer_model(vocab_size: int, *, device: torch.device) -> TinyTransformerCharModel:
    return TinyTransformerCharModel(
        vocab_size=vocab_size,
        context_size=CONTEXT_SIZE,
        d_model=TRANSFORMER_D_MODEL,
        num_heads=TRANSFORMER_NUM_HEADS,
        num_layers=TRANSFORMER_NUM_LAYERS,
        feedforward_dim=TRANSFORMER_FEEDFORWARD_DIM,
        residual_factory=PlainResidualCombine,
    ).to(device)


def train_model(
    *,
    model_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    parameter_count: int,
    learning_rate_fields: list[str],
) -> dict[str, object]:
    history: list[dict[str, float | int]] = []
    started_at = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_inputs,
            train_targets,
            batch_size=BATCH_SIZE,
            gradient_clip_norm=GRADIENT_CLIP,
        )
        val_metrics = evaluate_model(
            model,
            val_inputs,
            val_targets,
            batch_size=EVAL_BATCH_SIZE,
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": round_metric(train_metrics["loss"]),
            "train_accuracy": round_metric(train_metrics["accuracy"]),
            "val_loss": round_metric(val_metrics["loss"]),
            "val_accuracy": round_metric(val_metrics["accuracy"]),
        }
        for field_name, param_group in zip(learning_rate_fields, optimizer.param_groups, strict=True):
            record[field_name] = round_metric(param_group["lr"], digits=8)
        history.append(record)
        print(
            f"{model_name} epoch={epoch}/{EPOCHS} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} accuracy={val_metrics['accuracy']:.4f}",
            flush=True,
        )
        scheduler.step()

    runtime_seconds = time.perf_counter() - started_at
    best_record = min(history, key=lambda item: float(item["val_loss"]))
    final_record = history[-1]
    return {
        "parameter_count": parameter_count,
        "runtime_seconds": round_metric(runtime_seconds, digits=1),
        "best_epoch": int(best_record["epoch"]),
        "best_val_loss": float(best_record["val_loss"]),
        "best_val_accuracy": float(best_record["val_accuracy"]),
        "final_val_loss": float(final_record["val_loss"]),
        "final_val_accuracy": float(final_record["val_accuracy"]),
        "history": history,
    }


def main() -> None:
    import sys
    import time as _time
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    set_seed(SEED)

    raw_text = TEXT_FILE.read_text(encoding="utf-8")
    print(f"Text loaded: {len(raw_text)} chars. Building {TRAIN_CHARS}+{VAL_CHARS} split...", flush=True)
    _t0 = _time.perf_counter()
    split = build_fixed_length_split(raw_text, config=SplitConfig())
    print(f"Split built in {_time.perf_counter()-_t0:.1f}s: train={split.train_inputs.shape[0]} val={split.val_inputs.shape[0]}", flush=True)
    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)
    vocab_size = split.train_dataset.vocab_size

    residual_model = build_residual_model(vocab_size, device=device)
    residual_parameter_count = count_parameters(residual_model)
    residual_optimizer = build_residual_optimizer(residual_model)
    residual_scheduler = ProportionalCosineAnnealingLR(
        residual_optimizer,
        T_max=EPOCHS,
        eta_mins=[MUON_MIN_LR, ADAMW_MIN_LR],
    )
    residual_result = train_model(
        model_name="residual_stream_time_muon",
        model=residual_model,
        optimizer=residual_optimizer,
        scheduler=residual_scheduler,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        parameter_count=residual_parameter_count,
        learning_rate_fields=["muon_lr", "adamw_lr"],
    )

    del residual_model
    del residual_optimizer
    del residual_scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    set_seed(SEED)
    transformer_model = build_transformer_model(vocab_size, device=device)
    transformer_parameter_count = tf_count_parameters(transformer_model)
    transformer_optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=TRANSFORMER_LR)
    transformer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        transformer_optimizer,
        T_max=EPOCHS,
        eta_min=TRANSFORMER_MIN_LR,
    )
    transformer_result = train_model(
        model_name="transformer_control",
        model=transformer_model,
        optimizer=transformer_optimizer,
        scheduler=transformer_scheduler,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        parameter_count=transformer_parameter_count,
        learning_rate_fields=["learning_rate"],
    )

    gap = residual_result["best_val_loss"] - transformer_result["best_val_loss"]
    print("final_comparison", flush=True)
    print(
        f"residual_best_val_loss={float(residual_result['best_val_loss']):.6f} "
        f"transformer_best_val_loss={float(transformer_result['best_val_loss']):.6f} "
        f"gap_residual_minus_transformer={gap:.6f}",
        flush=True,
    )
    print(
        f"residual_params={residual_parameter_count} transformer_params={transformer_parameter_count}",
        flush=True,
    )

    results = {
        "git_sha": current_git_sha(),
        "seed": SEED,
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_is_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "dataset": {
            "text_file": str(TEXT_FILE),
            "context_size": CONTEXT_SIZE,
            "train_characters": TRAIN_CHARS,
            "val_characters": VAL_CHARS,
            "train_windows": int(split.train_inputs.shape[0]),
            "val_windows": int(split.val_inputs.shape[0]),
            "vocab_size": vocab_size,
            "val_start": int(split.val_start),
        },
        "models": {
            "residual_stream_time_muon": {
                "architecture": {
                    "d_model": RESIDUAL_D_MODEL,
                    "feedforward_dim": RESIDUAL_FEEDFORWARD_DIM,
                    "num_heads": RESIDUAL_NUM_HEADS,
                    "temporal_window": RESIDUAL_TEMPORAL_WINDOW,
                    "context_size": CONTEXT_SIZE,
                },
                "optimizer": {
                    "type": "SingleDeviceMuonWithAuxAdam",
                    "muon_lr": MUON_LR,
                    "muon_min_lr": MUON_MIN_LR,
                    "adamw_lr": ADAMW_LR,
                    "adamw_min_lr": ADAMW_MIN_LR,
                    "adamw_betas": list(ADAMW_BETAS),
                    "gradient_clip": GRADIENT_CLIP,
                    "muon_parameter_names": sorted(MUON_PARAMETER_NAMES),
                },
                **residual_result,
            },
            "transformer_control": {
                "architecture": {
                    "d_model": TRANSFORMER_D_MODEL,
                    "feedforward_dim": TRANSFORMER_FEEDFORWARD_DIM,
                    "num_heads": TRANSFORMER_NUM_HEADS,
                    "num_layers": TRANSFORMER_NUM_LAYERS,
                    "context_size": CONTEXT_SIZE,
                    "residual_combine": "PlainResidualCombine",
                },
                "optimizer": {
                    "type": "AdamW",
                    "learning_rate": TRANSFORMER_LR,
                    "min_learning_rate": TRANSFORMER_MIN_LR,
                    "gradient_clip": GRADIENT_CLIP,
                },
                **transformer_result,
            },
        },
        "comparison": {
            "best_val_loss_gap_residual_minus_transformer": round_metric(gap),
            "residual_best_val_loss": float(residual_result["best_val_loss"]),
            "transformer_best_val_loss": float(transformer_result["best_val_loss"]),
            "residual_parameter_count": residual_parameter_count,
            "transformer_parameter_count": transformer_parameter_count,
        },
    }
    OUTPUT_FILE.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
