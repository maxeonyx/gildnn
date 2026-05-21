from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.datasets import MNIST

from core.fixed_window_char import set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "image_patches_baseline"
DATA_DIR = EXPERIMENT_DIR / "data.ignore"
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
OUTPUT_FILE = ARTIFACTS_DIR / "results.json"

SEED = 42
IMAGE_SIZE = 32
PATCH_SIZE = 4
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
PATCH_DIM = PATCH_SIZE * PATCH_SIZE
PATCH_COUNT = GRID_SIZE * GRID_SIZE
MAX_CONTEXT_PATCHES = 48
CONTEXT_OPTIONS = (8, 16, 32, 48)
ORDER_NAMES = ("random", "raster", "reverse_raster", "column_major")
TRAIN_ORDER_PROBABILITIES = (0.7, 0.1, 0.1, 0.1)

EPOCHS = 10
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0

D_MODEL = 96
NUM_HEADS = 4
NUM_LAYERS = 3
FEEDFORWARD_DIM = 192
TYPE_EMBED_DIM = 8
DROPOUT = 0.0


@dataclass(frozen=True)
class Batch:
    patch_values: Tensor
    coordinates: Tensor
    token_types: Tensor
    padding_mask: Tensor
    targets: Tensor
    context_sizes: Tensor
    order_ids: Tensor


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def current_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return result.stdout.strip()


def round_metric(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def psnr_from_mse(mse: float) -> float:
    return float(-10.0 * math.log10(max(mse, 1e-12)))


def build_patch_coordinates(*, device: torch.device) -> Tensor:
    rows = torch.arange(GRID_SIZE, device=device, dtype=torch.float32)
    cols = torch.arange(GRID_SIZE, device=device, dtype=torch.float32)
    grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack((grid_rows / (GRID_SIZE - 1), grid_cols / (GRID_SIZE - 1)), dim=-1).reshape(PATCH_COUNT, 2)


def load_mnist_patches(*, device: torch.device) -> tuple[Tensor, Tensor]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_dataset = MNIST(root=DATA_DIR, train=True, download=True)
    val_dataset = MNIST(root=DATA_DIR, train=False, download=True)

    def preprocess(data: Tensor) -> Tensor:
        images = data.to(dtype=torch.float32).div_(255.0).unsqueeze(1)
        padded = F.pad(images, (2, 2, 2, 2))
        patches = padded.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        return patches.contiguous().view(-1, GRID_SIZE, GRID_SIZE, PATCH_DIM).view(-1, PATCH_COUNT, PATCH_DIM)

    train_patches = preprocess(train_dataset.data).to(device)
    val_patches = preprocess(val_dataset.data).to(device)
    return train_patches, val_patches


class SetTransformerPatchPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(2, TYPE_EMBED_DIM)
        self.query_patch_token = nn.Parameter(torch.zeros(PATCH_DIM))
        self.input_projection = nn.Linear(PATCH_DIM + 2 + TYPE_EMBED_DIM, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=FEEDFORWARD_DIM,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.output_norm = nn.LayerNorm(D_MODEL)
        self.output_head = nn.Linear(D_MODEL, PATCH_DIM)

    def forward(
        self,
        patch_values: Tensor,
        coordinates: Tensor,
        token_types: Tensor,
        *,
        padding_mask: Tensor,
    ) -> Tensor:
        masked_patch_values = patch_values.clone()
        query_mask = token_types == 1
        masked_patch_values[query_mask] = self.query_patch_token
        type_features = self.type_embedding(token_types)
        hidden = self.input_projection(torch.cat((masked_patch_values, coordinates, type_features), dim=-1))
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        query_state = self.output_norm(encoded[:, -1, :])
        return self.output_head(query_state)


def sample_context_sizes(batch_size: int, *, device: torch.device) -> Tensor:
    option_indices = torch.randint(len(CONTEXT_OPTIONS), (batch_size,), device=device)
    return torch.tensor(CONTEXT_OPTIONS, device=device, dtype=torch.long)[option_indices]


def sample_order_ids(batch_size: int, *, device: torch.device) -> Tensor:
    thresholds = torch.tensor((0.7, 0.8, 0.9), device=device)
    selector = torch.rand(batch_size, device=device)
    return torch.bucketize(selector, thresholds)


def build_order_keys(context_indices: Tensor, order_ids: Tensor) -> Tensor:
    batch_size = context_indices.shape[0]
    raster = context_indices.to(torch.float32)
    reverse_raster = -context_indices.to(torch.float32)
    column_major = ((context_indices % GRID_SIZE) * GRID_SIZE + (context_indices // GRID_SIZE)).to(torch.float32)
    random_keys = torch.rand(batch_size, MAX_CONTEXT_PATCHES, device=context_indices.device)

    keys = random_keys
    keys = torch.where(order_ids[:, None] == 1, raster, keys)
    keys = torch.where(order_ids[:, None] == 2, reverse_raster, keys)
    keys = torch.where(order_ids[:, None] == 3, column_major, keys)
    return keys


def build_batch(
    patches: Tensor,
    coordinates: Tensor,
    *,
    image_indices: Tensor,
    context_sizes: Tensor | None = None,
    order_ids: Tensor | None = None,
) -> Batch:
    batch_size = image_indices.shape[0]
    device = patches.device
    if context_sizes is None:
        context_sizes = sample_context_sizes(batch_size, device=device)
    if order_ids is None:
        order_ids = sample_order_ids(batch_size, device=device)

    target_indices = torch.randint(PATCH_COUNT, (batch_size,), device=device)
    random_scores = torch.rand(batch_size, PATCH_COUNT, device=device)
    random_scores[torch.arange(batch_size, device=device), target_indices] = 2.0
    candidate_context = random_scores.argsort(dim=1)[:, :MAX_CONTEXT_PATCHES]

    order_keys = build_order_keys(candidate_context, order_ids)
    active_mask = torch.arange(MAX_CONTEXT_PATCHES, device=device)[None, :] < context_sizes[:, None]
    order_keys = torch.where(active_mask, order_keys, torch.full_like(order_keys, 1e9))
    order = order_keys.argsort(dim=1)
    ordered_context = candidate_context.gather(1, order)

    batch_patches = patches[image_indices]
    context_patches = batch_patches.gather(1, ordered_context[:, :, None].expand(-1, -1, PATCH_DIM))
    context_coords = coordinates[ordered_context]
    context_type_ids = torch.zeros(batch_size, MAX_CONTEXT_PATCHES, device=device, dtype=torch.long)

    query_coords = coordinates[target_indices]
    query_patches = batch_patches[torch.arange(batch_size, device=device), target_indices]
    query_patch_values = torch.zeros(batch_size, 1, PATCH_DIM, device=device, dtype=batch_patches.dtype)
    query_coordinates = query_coords[:, None, :]
    query_type_ids = torch.ones(batch_size, 1, device=device, dtype=torch.long)

    patch_values = torch.cat((context_patches, query_patch_values), dim=1)
    token_coordinates = torch.cat((context_coords, query_coordinates), dim=1)
    token_types = torch.cat((context_type_ids, query_type_ids), dim=1)

    padding_mask = torch.zeros(batch_size, MAX_CONTEXT_PATCHES + 1, device=device, dtype=torch.bool)
    padding_mask[:, :MAX_CONTEXT_PATCHES] = ~active_mask
    return Batch(
        patch_values=patch_values,
        coordinates=token_coordinates,
        token_types=token_types,
        padding_mask=padding_mask,
        targets=query_patches,
        context_sizes=context_sizes,
        order_ids=order_ids,
    )


def evaluate_predictor(
    model: SetTransformerPatchPredictor,
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
    model: SetTransformerPatchPredictor,
    optimizer: torch.optim.Optimizer,
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
    model: SetTransformerPatchPredictor,
    parameter_count: int,
    runtime_seconds: float,
    history: list[dict[str, float | int]],
    eval_by_context: dict[str, dict[str, float]],
    train_patch_count: int,
    val_patch_count: int,
    device: torch.device,
) -> dict[str, object]:
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
            "patch_count": PATCH_COUNT,
            "patch_dim": PATCH_DIM,
            "train_images": train_patch_count,
            "val_images": val_patch_count,
            "context_options": list(CONTEXT_OPTIONS),
            "order_probabilities": {
                name: probability for name, probability in zip(ORDER_NAMES, TRAIN_ORDER_PROBABILITIES, strict=True)
            },
        },
        "model": {
            "name": "set_transformer_patch_predictor",
            "parameter_count": parameter_count,
            "architecture": {
                "d_model": D_MODEL,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "feedforward_dim": FEEDFORWARD_DIM,
                "type_embedding_dim": TYPE_EMBED_DIM,
                "dropout": DROPOUT,
                "max_context_patches": MAX_CONTEXT_PATCHES,
                "uses_causal_mask": False,
                "uses_sequence_position_embedding": False,
                "query_patch_representation": "learned_mask_token",
                "coordinates": "normalized_row_col_concat",
            },
            "optimizer": {
                "type": "AdamW",
                "learning_rate": LEARNING_RATE,
                "min_learning_rate": MIN_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip": GRADIENT_CLIP,
            },
            "runtime_seconds": round_metric(runtime_seconds, digits=2),
            "history": history,
            "evaluation": {
                "by_context_patches": eval_by_context,
            },
        },
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

    model = SetTransformerPatchPredictor().to(device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(train_patches.shape[0] / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS * steps_per_epoch,
        eta_min=MIN_LEARNING_RATE,
    )

    history: list[dict[str, float | int]] = []
    started_at = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        train_mse = train_one_epoch(model, optimizer, scheduler, train_patches, coordinates, batch_size=BATCH_SIZE)
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
            "learning_rate": round_metric(optimizer.param_groups[0]["lr"], digits=8),
        }
        history.append(epoch_record)
        print(
            f"[{timestamp()}] epoch={epoch}/{EPOCHS} train_mse={train_mse:.6f} "
            f"val_mse={val_metrics['mse']:.6f} val_psnr={val_metrics['psnr']:.3f}",
            flush=True,
        )

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
        model=model,
        parameter_count=parameter_count,
        runtime_seconds=runtime_seconds,
        history=history,
        eval_by_context=eval_by_context,
        train_patch_count=int(train_patches.shape[0]),
        val_patch_count=int(val_patches.shape[0]),
        device=device,
    )
    OUTPUT_FILE.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[{timestamp()}] wrote_results path={OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
