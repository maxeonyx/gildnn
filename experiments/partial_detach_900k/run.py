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


REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_FILE = REPO_ROOT / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
ARTIFACTS_DIR = REPO_ROOT / "experiments" / "partial_detach_900k" / "artifacts"
OUTPUT_FILE = ARTIFACTS_DIR / "results.json"

CONTEXT_SIZE = 64
TRAIN_CHARS = 900_000
VAL_CHARS = 100_000
EPOCHS = 5
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
GRADIENT_CLIP = 1.0
SEED = 42

D_MODEL = 116
FEEDFORWARD_DIM = 464
NUM_HEADS = 4
TEMPORAL_WINDOW = 8
DETACH_EVERY = 4

MUON_LR = 0.003
MUON_MIN_LR = 0.0003
ADAMW_LR = 1e-4
ADAMW_MIN_LR = 1e-5
ADAMW_BETAS = (0.9, 0.95)

REFERENCE_NO_DETACH_PARAMETER_COUNT = 185_088
REFERENCE_NO_DETACH_VAL_LOSS = 1.605575
REFERENCE_NO_DETACH_RUNTIME_SECONDS = 5649.7
REFERENCE_TRANSFORMER_PARAMETER_COUNT = 189_689
REFERENCE_TRANSFORMER_VAL_LOSS = 1.534788

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


@dataclass(frozen=True)
class PartialDetachConfig:
    context_size: int
    d_model: int
    feedforward_dim: int
    temporal_window: int = 4
    num_heads: int = 4
    token_mix_init: float = 0.5
    block_mix_init: float = 0.9
    time_mix_init: float = 0.9
    detach_every: int = 4


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


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def current_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def round_metric(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


class MixAdd(nn.Module):
    def __init__(self, *, init: float) -> None:
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.tensor(_logit(init), dtype=torch.float32))

    def coefficient(self) -> Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(self, stream: Tensor, delta: Tensor) -> Tensor:
        mix = self.coefficient().to(device=stream.device, dtype=stream.dtype)
        return (mix * stream) + ((1.0 - mix) * delta)


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError(f"Mix probability must be between 0 and 1, got {probability}.")
    return math.log(probability / (1.0 - probability))


class TemporalWindowAttention(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, query_source: Tensor, past_states: Tensor) -> Tensor:
        if past_states.shape[1] == 0:
            return torch.zeros_like(query_source)

        batch_size, _, d_model = past_states.shape
        query = self.query(query_source).reshape(batch_size, self.num_heads, self.head_dim)
        keys = self.key(past_states).reshape(batch_size, past_states.shape[1], self.num_heads, self.head_dim)
        values = self.value(past_states).reshape(batch_size, past_states.shape[1], self.num_heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        logits = torch.einsum("bhd,bhwd->bhw", query, keys) * (self.head_dim ** -0.5)
        weights = torch.softmax(logits, dim=-1)
        attended = torch.einsum("bhw,bhwd->bhd", weights, values).reshape(batch_size, d_model)
        return self.output(attended)


class ResidualFeedForwardBlock(nn.Module):
    def __init__(self, *, d_model: int, feedforward_dim: int) -> None:
        super().__init__()
        self.proj_in = nn.Linear(d_model, feedforward_dim)
        self.activation = nn.GELU()
        self.proj_out = nn.Linear(feedforward_dim, d_model)

    def forward(self, stream: Tensor) -> Tensor:
        return self.proj_out(self.activation(self.proj_in(stream)))


class PartialDetachResidualStreamTimeMixAddCharModel(nn.Module):
    def __init__(self, *, vocab_size: int, config: PartialDetachConfig) -> None:
        super().__init__()
        self.config = config
        self.context_size = config.context_size
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_size, config.d_model)
        self.mix_token = MixAdd(init=config.token_mix_init)
        self.mix_block = MixAdd(init=config.block_mix_init)
        self.mix_time = MixAdd(init=config.time_mix_init)
        self.temporal_attention = TemporalWindowAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
        )
        self.block = ResidualFeedForwardBlock(
            d_model=config.d_model,
            feedforward_dim=config.feedforward_dim,
        )
        self.output = nn.Linear(config.d_model, vocab_size)

    def embedded_tokens(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.shape[1]
        if sequence_length != self.context_size:
            raise ValueError(f"Expected context length {self.context_size}, got {sequence_length}.")
        positions = torch.arange(sequence_length, device=tokens.device)
        return self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)

    def forward(self, tokens: Tensor) -> Tensor:
        embeddings = self.embedded_tokens(tokens)
        batch_size = tokens.shape[0]
        stream = torch.zeros(batch_size, self.config.d_model, device=tokens.device, dtype=embeddings.dtype)
        history: list[Tensor] = []

        for time_index in range(self.context_size):
            if time_index > 0 and time_index % self.config.detach_every == 0:
                stream = stream.detach()

            block_input = self.mix_token(stream, embeddings[:, time_index, :])
            past_states = history[-self.config.temporal_window :]
            if past_states:
                stacked_past = torch.stack(past_states, dim=1)
            else:
                stacked_past = torch.empty(
                    batch_size,
                    0,
                    self.config.d_model,
                    device=tokens.device,
                    dtype=embeddings.dtype,
                )
            temporal_context = self.temporal_attention(block_input, stacked_past)
            block_delta = self.block(block_input)
            post_block = self.mix_block(block_input, block_delta)
            stream = self.mix_time(post_block, temporal_context)
            history.append(stream)

        return self.output(stream)


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
            loss = F.cross_entropy(logits, batch_targets, reduction="sum")
            if not torch.isfinite(loss):
                raise RuntimeError("Validation diverged: non-finite loss.")
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
        if not torch.isfinite(loss):
            raise RuntimeError(f"Training diverged: non-finite loss at batch start {start}.")

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


def build_optimizer(model: PartialDetachResidualStreamTimeMixAddCharModel) -> SingleDeviceMuonWithAuxAdam:
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


def build_model(vocab_size: int, *, device: torch.device) -> PartialDetachResidualStreamTimeMixAddCharModel:
    config = PartialDetachConfig(
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        num_heads=NUM_HEADS,
        temporal_window=TEMPORAL_WINDOW,
        detach_every=DETACH_EVERY,
    )
    return PartialDetachResidualStreamTimeMixAddCharModel(vocab_size=vocab_size, config=config).to(device)


def train_model(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    parameter_count: int,
) -> dict[str, object]:
    history: list[dict[str, float | int]] = []
    started_at = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_started_at = time.perf_counter()
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
        epoch_seconds = time.perf_counter() - epoch_started_at
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": round_metric(train_metrics["loss"]),
            "train_accuracy": round_metric(train_metrics["accuracy"]),
            "val_loss": round_metric(val_metrics["loss"]),
            "val_accuracy": round_metric(val_metrics["accuracy"]),
            "epoch_time_seconds": round_metric(epoch_seconds, digits=1),
            "muon_lr": round_metric(optimizer.param_groups[0]["lr"], digits=8),
            "adamw_lr": round_metric(optimizer.param_groups[1]["lr"], digits=8),
        }
        history.append(record)
        print(
            f"epoch={epoch}/{EPOCHS} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} epoch_time={epoch_seconds:.1f}s",
            flush=True,
        )
        scheduler.step()

    runtime_seconds = time.perf_counter() - started_at
    best_record = min(history, key=lambda item: float(item["val_loss"]))
    final_record = history[-1]
    return {
        "status": "completed",
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
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    set_seed(SEED)

    raw_text = TEXT_FILE.read_text(encoding="utf-8")
    print(f"Text loaded: {len(raw_text)} chars. Building {TRAIN_CHARS}+{VAL_CHARS} split...", flush=True)
    split_started_at = time.perf_counter()
    split = build_fixed_length_split(raw_text, config=SplitConfig())
    print(
        f"Split built in {time.perf_counter() - split_started_at:.1f}s: "
        f"train={split.train_inputs.shape[0]} val={split.val_inputs.shape[0]}",
        flush=True,
    )
    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)
    vocab_size = split.train_dataset.vocab_size

    model = build_model(vocab_size, device=device)
    parameter_count = count_parameters(model)
    optimizer = build_optimizer(model)
    scheduler = ProportionalCosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_mins=[MUON_MIN_LR, ADAMW_MIN_LR],
    )

    try:
        result = train_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            parameter_count=parameter_count,
        )
    except RuntimeError as error:
        runtime_seconds = round_metric(time.perf_counter() - split_started_at, digits=1)
        result = {
            "status": "diverged",
            "error": str(error),
            "parameter_count": parameter_count,
            "runtime_seconds": runtime_seconds,
            "history": [],
        }
        print(f"DIVERGED: {error}", flush=True)

    print("final_comparison", flush=True)
    print(
        f"this_run params={parameter_count} val_loss={result.get('final_val_loss', float('nan')):.6f} "
        f"total_time={result['runtime_seconds']:.1f}s",
        flush=True,
    )
    print(
        f"reference_no_detach params={REFERENCE_NO_DETACH_PARAMETER_COUNT} "
        f"val_loss={REFERENCE_NO_DETACH_VAL_LOSS:.6f} total_time={REFERENCE_NO_DETACH_RUNTIME_SECONDS:.1f}s",
        flush=True,
    )
    print(
        f"transformer params={REFERENCE_TRANSFORMER_PARAMETER_COUNT} "
        f"val_loss={REFERENCE_TRANSFORMER_VAL_LOSS:.6f}",
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
        "model": {
            "name": "partial_detach_residual_stream_time_muon",
            "architecture": {
                "d_model": D_MODEL,
                "feedforward_dim": FEEDFORWARD_DIM,
                "num_heads": NUM_HEADS,
                "temporal_window": TEMPORAL_WINDOW,
                "context_size": CONTEXT_SIZE,
                "detach_every": DETACH_EVERY,
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
            **result,
        },
        "comparison": {
            "this_run": {
                "parameter_count": parameter_count,
                "val_loss": result.get("final_val_loss"),
                "total_time_seconds": result["runtime_seconds"],
            },
            "reference_no_detach": {
                "parameter_count": REFERENCE_NO_DETACH_PARAMETER_COUNT,
                "val_loss": REFERENCE_NO_DETACH_VAL_LOSS,
                "total_time_seconds": REFERENCE_NO_DETACH_RUNTIME_SECONDS,
            },
            "transformer_control": {
                "parameter_count": REFERENCE_TRANSFORMER_PARAMETER_COUNT,
                "val_loss": REFERENCE_TRANSFORMER_VAL_LOSS,
            },
            "gap_vs_reference_no_detach": (
                None
                if result.get("final_val_loss") is None
                else round_metric(result["final_val_loss"] - REFERENCE_NO_DETACH_VAL_LOSS)
            ),
            "gap_vs_transformer": (
                None
                if result.get("final_val_loss") is None
                else round_metric(result["final_val_loss"] - REFERENCE_TRANSFORMER_VAL_LOSS)
            ),
            "speedup_vs_reference_no_detach": round_metric(
                REFERENCE_NO_DETACH_RUNTIME_SECONDS / result["runtime_seconds"],
                digits=3,
            ) if result["runtime_seconds"] > 0 else None,
        },
    }
    OUTPUT_FILE.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
