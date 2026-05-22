from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from muon import SingleDeviceMuonWithAuxAdam
from torch import Tensor
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
RESULTS_FILE = ARTIFACTS_DIR / "muon_ablation_results.json"

CONTEXT_SIZE = 64
D_MODEL = 192
FEEDFORWARD_DIM = 768
NUM_HEADS = 4
TRAIN_CHARS = 100_000
VAL_CHARS = 20_000
EPOCHS = 5
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
GRADIENT_CLIP = 1.0
SEED = 42
MUON_LR = 0.02
ADAMW_LR = 3e-4
ADAMW_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.0
WINDOW_VALUES = [4, 8, 16]

MUON_PARAMETER_NAMES = {
    "temporal_attention.query.weight",
    "temporal_attention.key.weight",
    "temporal_attention.value.weight",
    "temporal_attention.output.weight",
    "block.proj_in.weight",
    "block.proj_out.weight",
}

ADAMW_REFERENCE = {
    4: 1.670,
    8: "DIVERGED",
    16: "DIVERGED",
}


@dataclass(frozen=True)
class SplitConfig:
    context_size: int
    train_characters: int
    val_characters: int


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def current_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return result.stdout.strip()


def batched_pairs(inputs: Tensor, targets: Tensor, *, batch_size: int):
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        yield inputs[start:stop], targets[start:stop]


def build_optimizer(model: ResidualStreamTimeMixAddCharModel) -> tuple[SingleDeviceMuonWithAuxAdam, list[str], list[str]]:
    muon_parameters = []
    adamw_parameters = []
    muon_names = []
    adamw_names = []

    for name, parameter in model.named_parameters():
        if name in MUON_PARAMETER_NAMES:
            if parameter.ndim < 2:
                raise ValueError(f"Muon parameter must be 2D+, got {name} with ndim={parameter.ndim}.")
            muon_parameters.append(parameter)
            muon_names.append(name)
        else:
            adamw_parameters.append(parameter)
            adamw_names.append(name)

    if set(muon_names) != MUON_PARAMETER_NAMES:
        raise ValueError(
            f"Muon parameter split mismatch. Expected {sorted(MUON_PARAMETER_NAMES)}, got {sorted(muon_names)}."
        )
    if not adamw_parameters:
        raise ValueError("AdamW parameter group is empty.")

    optimizer = SingleDeviceMuonWithAuxAdam(
        [
            {
                "params": muon_parameters,
                "lr": MUON_LR,
                "weight_decay": WEIGHT_DECAY,
                "use_muon": True,
            },
            {
                "params": adamw_parameters,
                "lr": ADAMW_LR,
                "betas": ADAMW_BETAS,
                "weight_decay": WEIGHT_DECAY,
                "use_muon": False,
            },
        ]
    )
    return optimizer, muon_names, adamw_names


def evaluate_model(
    model: ResidualStreamTimeMixAddCharModel,
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
            loss = F.cross_entropy(logits, batch_targets, reduction="sum")
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss.")
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
            total_examples += batch_targets.shape[0]

    if was_training:
        model.train()

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_one_epoch(
    model: ResidualStreamTimeMixAddCharModel,
    optimizer: SingleDeviceMuonWithAuxAdam,
    train_inputs: Tensor,
    train_targets: Tensor,
) -> dict[str, float]:
    model.train()
    permutation = torch.randperm(train_inputs.shape[0], device=train_inputs.device)
    total_examples = 0
    total_loss = 0.0

    for start in range(0, permutation.shape[0], BATCH_SIZE):
        batch_indices = permutation[start : start + BATCH_SIZE]
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

        batch_examples = batch_targets.shape[0]
        total_examples += batch_examples
        total_loss += loss.item() * batch_examples

    return {
        "loss": total_loss / total_examples,
    }


def run_single_window(
    window_size: int,
    *,
    vocab_size: int,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    device: torch.device,
) -> dict[str, object]:
    set_seed(SEED)
    config = ResidualStreamTimeMixAddConfig(
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        temporal_window=window_size,
        num_heads=NUM_HEADS,
    )
    model = ResidualStreamTimeMixAddCharModel(vocab_size=vocab_size, config=config).to(device)
    optimizer, muon_names, adamw_names = build_optimizer(model)
    parameter_count = count_parameters(model)
    history: list[dict[str, float | int]] = []
    diverged = False
    error_message: str | None = None

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()

    try:
        for epoch in range(1, EPOCHS + 1):
            train_metrics = train_one_epoch(model, optimizer, train_inputs, train_targets)
            val_metrics = evaluate_model(model, val_inputs, val_targets, batch_size=EVAL_BATCH_SIZE)
            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
            }
            history.append(epoch_record)
            print(
                f"k={window_size} epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_accuracy={val_metrics['accuracy']:.4f}",
                flush=True,
            )
    except FloatingPointError as error:
        diverged = True
        error_message = str(error)
        print(f"k={window_size} diverged: {error_message}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - started_at

    best_val_loss = min((float(record["val_loss"]) for record in history), default=None)
    result = {
        "temporal_window": window_size,
        "parameter_count": parameter_count,
        "diverged": diverged,
        "runtime_seconds": round(runtime_seconds, 2),
        "best_val_loss": None if best_val_loss is None else round(best_val_loss, 6),
        "history": history,
        "muon_parameter_names": muon_names,
        "adamw_parameter_names": adamw_names,
    }
    if error_message is not None:
        result["error"] = error_message

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    print("background_runnable=yes", flush=True)

    set_seed(SEED)
    raw_text = TEXT_FILE.read_text(encoding="utf-8")
    split = build_fixed_length_split(
        raw_text,
        config=SplitConfig(
            context_size=CONTEXT_SIZE,
            train_characters=TRAIN_CHARS,
            val_characters=VAL_CHARS,
        ),
    )
    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)

    print(
        f"train_windows={train_inputs.shape[0]} val_windows={val_inputs.shape[0]} vocab_size={split.train_dataset.vocab_size}",
        flush=True,
    )

    results = []
    payload = {
        "git_sha": current_git_sha(),
        "text_file": str(TEXT_FILE),
        "device": str(device),
        "hyperparameters": {
            "context_size": CONTEXT_SIZE,
            "d_model": D_MODEL,
            "feedforward_dim": FEEDFORWARD_DIM,
            "num_heads": NUM_HEADS,
            "train_characters": TRAIN_CHARS,
            "val_characters": VAL_CHARS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_clip": GRADIENT_CLIP,
            "seed": SEED,
            "muon_lr": MUON_LR,
            "adamw_lr": ADAMW_LR,
            "adamw_betas": list(ADAMW_BETAS),
            "weight_decay": WEIGHT_DECAY,
            "temporal_windows": WINDOW_VALUES,
        },
        "adamw_reference": ADAMW_REFERENCE,
        "results": results,
    }

    for window_size in WINDOW_VALUES:
        print(f"starting k={window_size}", flush=True)
        result = run_single_window(
            window_size,
            vocab_size=split.train_dataset.vocab_size,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            device=device,
        )
        results.append(result)
        write_json(RESULTS_FILE, payload)

    print("summary", flush=True)
    any_diverged = any(bool(result["diverged"]) for result in results)
    for result in results:
        window_size = int(result["temporal_window"])
        best_val_loss = result["best_val_loss"]
        best_display = "DIVERGED" if best_val_loss is None else f"{best_val_loss:.6f}"
        print(
            f"k={window_size} muon_best_val_loss={best_display} diverged={result['diverged']} adamw={ADAMW_REFERENCE[window_size]}",
            flush=True,
        )
    print(f"any_diverged={any_diverged}", flush=True)
    print(f"results_file={RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
