from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from muon import SingleDeviceMuonWithAuxAdam
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import set_seed
from experiments.image_patches_baseline.run import (
    ARTIFACTS_DIR as BASELINE_ARTIFACTS_DIR,
    BATCH_SIZE,
    CONTEXT_OPTIONS,
    EPOCHS,
    EVAL_BATCH_SIZE,
    GRID_SIZE,
    GRADIENT_CLIP,
    IMAGE_SIZE,
    MAX_CONTEXT_PATCHES,
    ORDER_NAMES,
    PATCH_COUNT,
    PATCH_DIM,
    PATCH_SIZE,
    REPO_ROOT,
    TRAIN_ORDER_PROBABILITIES,
    Batch,
    build_batch,
    build_patch_coordinates,
    current_git_sha,
    load_mnist_patches,
    psnr_from_mse,
    round_metric,
    timestamp,
)


EXPERIMENT_DIR = REPO_ROOT / "experiments" / "image_patches_probe"
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
OUTPUT_FILE = ARTIFACTS_DIR / "results.json"

SEED = 42
MUON_LR = 0.003
MUON_MIN_LR = 3e-4
ADAMW_LR = 3e-4
ADAMW_MIN_LR = 3e-5
ADAMW_BETAS = (0.9, 0.95)
WEIGHT_DECAY = 1e-4

D_MODEL = 112
FEEDFORWARD_DIM = 768
NUM_HEADS = 4
TEMPORAL_WINDOW = 8
TYPE_EMBED_DIM = 8

MUON_PARAMETER_NAMES = {
    "temporal_attention.query.weight",
    "temporal_attention.key.weight",
    "temporal_attention.value.weight",
    "temporal_attention.output.weight",
    "block.proj_in.weight",
    "block.proj_out.weight",
}


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError(f"Mix probability must be between 0 and 1, got {probability}.")
    return math.log(probability / (1.0 - probability))


class MixAdd(nn.Module):
    def __init__(self, *, init: float) -> None:
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.tensor(_logit(init), dtype=torch.float32))

    def coefficient_value(self) -> float:
        return torch.sigmoid(self.alpha_logit.detach()).item()

    def forward(self, stream: Tensor, delta: Tensor) -> Tensor:
        mix = torch.sigmoid(self.alpha_logit).to(device=stream.device, dtype=stream.dtype)
        return (mix * stream) + ((1.0 - mix) * delta)


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

    def forward(self, query_source: Tensor, past_states: Tensor, *, past_mask: Tensor) -> Tensor:
        if past_states.shape[1] == 0:
            return torch.zeros_like(query_source)

        batch_size, _, d_model = past_states.shape
        query = self.query(query_source).reshape(batch_size, self.num_heads, self.head_dim)
        keys = self.key(past_states).reshape(batch_size, past_states.shape[1], self.num_heads, self.head_dim)
        values = self.value(past_states).reshape(batch_size, past_states.shape[1], self.num_heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        logits = torch.einsum("bhd,bhwd->bhw", query, keys) * (self.head_dim ** -0.5)

        expanded_mask = past_mask[:, None, :]
        logits = logits.masked_fill(~expanded_mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * expanded_mask
        normalizer = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(normalizer > 0, weights / normalizer, torch.zeros_like(weights))

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


class ResidualStreamPatchPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(2, TYPE_EMBED_DIM)
        self.query_patch_token = nn.Parameter(torch.zeros(PATCH_DIM))
        self.input_projection = nn.Linear(PATCH_DIM + 2 + TYPE_EMBED_DIM, D_MODEL)
        self.mix_token = MixAdd(init=0.5)
        self.mix_block = MixAdd(init=0.9)
        self.mix_time = MixAdd(init=0.9)
        self.temporal_attention = TemporalWindowAttention(d_model=D_MODEL, num_heads=NUM_HEADS)
        self.block = ResidualFeedForwardBlock(d_model=D_MODEL, feedforward_dim=FEEDFORWARD_DIM)
        self.output_norm = nn.LayerNorm(D_MODEL)
        self.output_head = nn.Linear(D_MODEL, PATCH_DIM)

    def mix_coefficients(self) -> dict[str, float]:
        return {
            "token": self.mix_token.coefficient_value(),
            "block": self.mix_block.coefficient_value(),
            "time": self.mix_time.coefficient_value(),
        }

    def forward(
        self,
        patch_values: Tensor,
        coordinates: Tensor,
        token_types: Tensor,
        *,
        padding_mask: Tensor,
    ) -> Tensor:
        masked_patch_values = patch_values.clone()
        masked_patch_values[token_types == 1] = self.query_patch_token
        type_features = self.type_embedding(token_types)
        token_inputs = torch.cat((masked_patch_values, coordinates, type_features), dim=-1)
        embeddings = self.input_projection(token_inputs)

        batch_size, sequence_length, _ = embeddings.shape
        stream = torch.zeros(batch_size, D_MODEL, device=embeddings.device, dtype=embeddings.dtype)
        history_states: list[Tensor] = []
        history_valid: list[Tensor] = []

        for time_index in range(sequence_length):
            is_active = ~padding_mask[:, time_index]
            block_input = self.mix_token(stream, embeddings[:, time_index, :])

            if history_states:
                past_states = torch.stack(history_states[-TEMPORAL_WINDOW:], dim=1)
                past_mask = torch.stack(history_valid[-TEMPORAL_WINDOW:], dim=1)
            else:
                past_states = torch.empty(
                    batch_size,
                    0,
                    D_MODEL,
                    device=embeddings.device,
                    dtype=embeddings.dtype,
                )
                past_mask = torch.empty(batch_size, 0, device=embeddings.device, dtype=torch.bool)

            temporal_context = self.temporal_attention(block_input, past_states, past_mask=past_mask)
            block_delta = self.block(block_input)
            post_block = self.mix_block(block_input, block_delta)
            next_stream = self.mix_time(post_block, temporal_context)
            stream = torch.where(is_active[:, None], next_stream, stream)
            history_states.append(stream)
            history_valid.append(is_active)

        return self.output_head(self.output_norm(stream))


@dataclass(frozen=True)
class OptimizerGroups:
    optimizer: SingleDeviceMuonWithAuxAdam
    muon_names: list[str]
    adamw_names: list[str]


def build_optimizer(model: ResidualStreamPatchPredictor) -> OptimizerGroups:
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
    return OptimizerGroups(optimizer=optimizer, muon_names=muon_names, adamw_names=adamw_names)


def build_scheduler(optimizer: torch.optim.Optimizer, *, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    min_ratio = MUON_MIN_LR / MUON_LR

    def schedule(step: int) -> float:
        progress = min(step, total_steps) / total_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)


def evaluate_predictor(
    model: ResidualStreamPatchPredictor,
    patches: Tensor,
    coordinates: Tensor,
    *,
    batch_size: int,
    context_size: int,
    mean_patch: Tensor,
    eval_seed: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_examples = 0
    total_loss = 0.0
    zero_baseline_loss = 0.0
    mean_baseline_loss = 0.0

    set_seed(eval_seed)
    with torch.no_grad():
        for start in range(0, patches.shape[0], batch_size):
            stop = min(start + batch_size, patches.shape[0])
            image_indices = torch.arange(start, stop, device=patches.device)
            batch = build_batch(
                patches,
                coordinates,
                image_indices=image_indices,
                context_sizes=torch.full((stop - start,), context_size, device=patches.device, dtype=torch.long),
            )
            predictions = model(
                batch.patch_values,
                batch.coordinates,
                batch.token_types,
                padding_mask=batch.padding_mask,
            )
            batch_loss = F.mse_loss(predictions, batch.targets, reduction="sum")
            zero_loss = F.mse_loss(torch.zeros_like(batch.targets), batch.targets, reduction="sum")
            mean_predictions = mean_patch[None, :].expand_as(batch.targets)
            mean_loss = F.mse_loss(mean_predictions, batch.targets, reduction="sum")

            total_examples += batch.targets.shape[0]
            total_loss += batch_loss.item()
            zero_baseline_loss += zero_loss.item()
            mean_baseline_loss += mean_loss.item()

    if was_training:
        model.train()

    mse = total_loss / (total_examples * PATCH_DIM)
    zero_mse = zero_baseline_loss / (total_examples * PATCH_DIM)
    mean_mse = mean_baseline_loss / (total_examples * PATCH_DIM)
    return {
        "mse": round_metric(mse),
        "psnr": round_metric(psnr_from_mse(mse)),
        "zero_baseline_mse": round_metric(zero_mse),
        "zero_baseline_psnr": round_metric(psnr_from_mse(zero_mse)),
        "mean_patch_baseline_mse": round_metric(mean_mse),
        "mean_patch_baseline_psnr": round_metric(psnr_from_mse(mean_mse)),
    }


def train_one_epoch(
    model: ResidualStreamPatchPredictor,
    optimizer: SingleDeviceMuonWithAuxAdam,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_patches: Tensor,
    coordinates: Tensor,
    *,
    batch_size: int,
) -> float:
    model.train()
    steps_per_epoch = math.ceil(train_patches.shape[0] / batch_size)
    total_loss = 0.0
    total_examples = 0

    for _ in range(steps_per_epoch):
        image_indices = torch.randint(train_patches.shape[0], (batch_size,), device=train_patches.device)
        batch = build_batch(train_patches, coordinates, image_indices=image_indices)
        predictions = model(
            batch.patch_values,
            batch.coordinates,
            batch.token_types,
            padding_mask=batch.padding_mask,
        )
        loss = F.mse_loss(predictions, batch.targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        scheduler.step()

        total_examples += batch_size
        total_loss += loss.item() * batch_size

    return total_loss / total_examples


def build_results(
    *,
    parameter_count: int,
    runtime_seconds: float,
    history: list[dict[str, float | int]],
    eval_by_context: dict[str, dict[str, float]],
    train_patch_count: int,
    val_patch_count: int,
    device: torch.device,
    model: ResidualStreamPatchPredictor,
    muon_parameter_names: list[str],
    adamw_parameter_names: list[str],
) -> dict[str, object]:
    baseline_results = None
    baseline_results_path = BASELINE_ARTIFACTS_DIR / "results.json"
    if baseline_results_path.exists():
        baseline_results = json.loads(baseline_results_path.read_text(encoding="utf-8"))

    return {
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
            "name": "MNIST",
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH_SIZE,
            "grid_size": GRID_SIZE,
            "patch_count": PATCH_COUNT,
            "patch_dim": PATCH_DIM,
            "train_images": train_patch_count,
            "val_images": val_patch_count,
            "context_options": list(CONTEXT_OPTIONS),
            "order_probabilities": {name: probability for name, probability in zip(ORDER_NAMES, TRAIN_ORDER_PROBABILITIES, strict=True)},
        },
        "model": {
            "name": "residual_stream_patch_predictor",
            "parameter_count": parameter_count,
            "architecture": {
                "d_model": D_MODEL,
                "feedforward_dim": FEEDFORWARD_DIM,
                "num_heads": NUM_HEADS,
                "temporal_window": TEMPORAL_WINDOW,
                "type_embedding_dim": TYPE_EMBED_DIM,
                "max_context_patches": MAX_CONTEXT_PATCHES,
                "query_patch_representation": "learned_mask_token",
                "coordinates": "normalized_row_col_concat",
                "processing_order": "left_to_right_recurrent",
                "mix_coefficients": model.mix_coefficients(),
            },
            "optimizer": {
                "type": "SingleDeviceMuonWithAuxAdam",
                "muon_learning_rate": MUON_LR,
                "muon_min_learning_rate": MUON_MIN_LR,
                "adamw_learning_rate": ADAMW_LR,
                "adamw_min_learning_rate": ADAMW_MIN_LR,
                "adamw_betas": list(ADAMW_BETAS),
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip": GRADIENT_CLIP,
                "muon_parameter_names": muon_parameter_names,
                "adamw_parameter_names": adamw_parameter_names,
            },
            "runtime_seconds": round_metric(runtime_seconds, digits=2),
            "history": history,
            "evaluation": {
                "by_context_patches": eval_by_context,
            },
        },
        "baseline_reference": baseline_results,
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{timestamp()}] device={device}", flush=True)
    set_seed(SEED)

    print(f"[{timestamp()}] loading_mnist", flush=True)
    train_patches, val_patches = load_mnist_patches(device=device)
    coordinates = build_patch_coordinates(device=device)
    mean_patch = train_patches.mean(dim=(0, 1))
    print(
        f"[{timestamp()}] dataset_ready train_images={train_patches.shape[0]} val_images={val_patches.shape[0]} "
        f"patches_per_image={PATCH_COUNT}",
        flush=True,
    )

    model = ResidualStreamPatchPredictor().to(device)
    parameter_count = count_parameters(model)
    optimizer_groups = build_optimizer(model)
    steps_per_epoch = math.ceil(train_patches.shape[0] / BATCH_SIZE)
    scheduler = build_scheduler(optimizer_groups.optimizer, total_steps=EPOCHS * steps_per_epoch)

    history: list[dict[str, float | int]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_mse = train_one_epoch(
            model,
            optimizer_groups.optimizer,
            scheduler,
            train_patches,
            coordinates,
            batch_size=BATCH_SIZE,
        )
        val_metrics = evaluate_predictor(
            model,
            val_patches,
            coordinates,
            batch_size=EVAL_BATCH_SIZE,
            context_size=32,
            mean_patch=mean_patch,
            eval_seed=SEED + epoch,
        )
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "train_mse": round_metric(train_mse),
            "val_mse": val_metrics["mse"],
            "val_psnr": val_metrics["psnr"],
            "muon_learning_rate": round_metric(optimizer_groups.optimizer.param_groups[0]["lr"], digits=8),
            "adamw_learning_rate": round_metric(optimizer_groups.optimizer.param_groups[1]["lr"], digits=8),
        }
        history.append(epoch_record)
        print(
            f"[{timestamp()}] epoch={epoch}/{EPOCHS} train_mse={train_mse:.6f} "
            f"val_mse={val_metrics['mse']:.6f} val_psnr={val_metrics['psnr']:.3f} "
            f"muon_lr={optimizer_groups.optimizer.param_groups[0]['lr']:.6f} "
            f"adamw_lr={optimizer_groups.optimizer.param_groups[1]['lr']:.6f}",
            flush=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - started_at
    print(f"[{timestamp()}] training_complete runtime_seconds={runtime_seconds:.2f}", flush=True)

    eval_by_context: dict[str, dict[str, float]] = {}
    for offset, context_size in enumerate(CONTEXT_OPTIONS, start=1):
        metrics = evaluate_predictor(
            model,
            val_patches,
            coordinates,
            batch_size=EVAL_BATCH_SIZE,
            context_size=context_size,
            mean_patch=mean_patch,
            eval_seed=SEED + 100 + offset,
        )
        eval_by_context[str(context_size)] = metrics
        print(
            f"[{timestamp()}] eval context_patches={context_size} mse={metrics['mse']:.6f} psnr={metrics['psnr']:.3f}",
            flush=True,
        )

    results = build_results(
        parameter_count=parameter_count,
        runtime_seconds=runtime_seconds,
        history=history,
        eval_by_context=eval_by_context,
        train_patch_count=int(train_patches.shape[0]),
        val_patch_count=int(val_patches.shape[0]),
        device=device,
        model=model,
        muon_parameter_names=optimizer_groups.muon_names,
        adamw_parameter_names=optimizer_groups.adamw_names,
    )
    OUTPUT_FILE.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[{timestamp()}] wrote_results path={OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
