from __future__ import annotations

import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.datasets import MNIST


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "arbitrary_order_mnist"
DATA_DIR = EXPERIMENT_DIR / "data.ignore"

SEED = 42
IMAGE_SIZE = 32
PATCH_SIZE = 4
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
PATCH_DIM = PATCH_SIZE * PATCH_SIZE
PATCH_COUNT = GRID_SIZE * GRID_SIZE

BATCH_SIZE = 32
EPOCHS = 50
WARMUP_EPOCHS = 2
LEARNING_RATE = 1e-3

D_MODEL = 128
N_HEADS = 4
N_ENCODER_LAYERS = 4
N_DECODER_LAYERS = 2
FEEDFORWARD_DIM = 256


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


class ArbitraryOrderPatchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(PATCH_DIM, D_MODEL)
        self.position_projection = nn.Sequential(
            nn.Linear(2, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, D_MODEL),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FEEDFORWARD_DIM,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FEEDFORWARD_DIM,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_ENCODER_LAYERS)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=N_DECODER_LAYERS)
        self.output_norm = nn.LayerNorm(D_MODEL)
        self.output_head = nn.Linear(D_MODEL, PATCH_DIM)

    def forward(
        self,
        observed_patches: Tensor,
        observed_coordinates: Tensor,
        target_coordinates: Tensor,
        *,
        observed_padding_mask: Tensor,
        target_padding_mask: Tensor,
    ) -> Tensor:
        encoder_input = self.patch_projection(observed_patches) + self.position_projection(observed_coordinates)
        memory = self.encoder(encoder_input, src_key_padding_mask=observed_padding_mask)
        queries = self.position_projection(target_coordinates)
        decoded = self.decoder(
            queries,
            memory,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=observed_padding_mask,
        )
        return self.output_head(self.output_norm(decoded))


def build_batch(patches: Tensor, coordinates: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch_size = patches.shape[0]
    device = patches.device
    observation_counts = torch.randint(1, PATCH_COUNT, (batch_size,), device=device)
    permutation_scores = torch.rand(batch_size, PATCH_COUNT, device=device)
    patch_order = permutation_scores.argsort(dim=1)

    max_observed = int(observation_counts.max().item())
    target_counts = PATCH_COUNT - observation_counts
    max_target = int(target_counts.max().item())

    observed_indices = patch_order[:, :max_observed]
    target_indices = patch_order.flip(dims=(1,))[:, :max_target]

    observed_patches = patches.gather(1, observed_indices[:, :, None].expand(-1, -1, PATCH_DIM))
    observed_coordinates = coordinates[observed_indices]
    target_patches = patches.gather(1, target_indices[:, :, None].expand(-1, -1, PATCH_DIM))
    target_coordinates = coordinates[target_indices]

    observed_positions = torch.arange(max_observed, device=device)[None, :]
    target_positions = torch.arange(max_target, device=device)[None, :]
    observed_padding_mask = observed_positions >= observation_counts[:, None]
    target_padding_mask = target_positions >= target_counts[:, None]

    loss_mask = (~target_padding_mask).unsqueeze(-1)
    return (
        observed_patches,
        observed_coordinates,
        observed_padding_mask,
        target_patches,
        target_coordinates,
        target_padding_mask,
        loss_mask,
    )


def masked_mse(predictions: Tensor, targets: Tensor, loss_mask: Tensor) -> Tensor:
    squared_error = (predictions - targets).pow(2)
    masked_error = squared_error * loss_mask
    valid_pixels = loss_mask.sum().clamp_min(1) * PATCH_DIM
    return masked_error.sum() / valid_pixels


def build_lr_lambda(*, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if total_steps <= 1:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        cosine_steps = max(total_steps - warmup_steps, 1)
        progress = float(step - warmup_steps) / float(cosine_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    return lr_lambda


def evaluate(model: ArbitraryOrderPatchModel, patches: Tensor, coordinates: Tensor) -> float:
    model.eval()
    total_squared_error = 0.0
    total_values = 0.0

    with torch.no_grad():
        for start in range(0, patches.shape[0], BATCH_SIZE):
            stop = min(start + BATCH_SIZE, patches.shape[0])
            batch = patches[start:stop]
            (
                observed_patches,
                observed_coordinates,
                observed_padding_mask,
                target_patches,
                target_coordinates,
                target_padding_mask,
                loss_mask,
            ) = build_batch(batch, coordinates)
            predictions = model(
                observed_patches,
                observed_coordinates,
                target_coordinates,
                observed_padding_mask=observed_padding_mask,
                target_padding_mask=target_padding_mask,
            )
            squared_error = ((predictions - target_patches).pow(2) * loss_mask).sum().item()
            total_squared_error += squared_error
            total_values += loss_mask.sum().item() * PATCH_DIM

    model.train()
    return total_squared_error / max(total_values, 1.0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(
        f"[{timestamp()}] device={device} device_name={cuda_name} git_sha={current_git_sha()} python={sys.version.split()[0]} torch={torch.__version__}",
        flush=True,
    )
    print(
        f"[{timestamp()}] config batch_size={BATCH_SIZE} epochs={EPOCHS} lr={LEARNING_RATE} d_model={D_MODEL} heads={N_HEADS} encoder_layers={N_ENCODER_LAYERS} decoder_layers={N_DECODER_LAYERS}",
        flush=True,
    )

    train_patches, val_patches = load_mnist_patches(device=device)
    coordinates = build_patch_coordinates(device=device)
    print(
        f"[{timestamp()}] dataset train_images={train_patches.shape[0]} val_images={val_patches.shape[0]} patch_count={PATCH_COUNT} patch_dim={PATCH_DIM}",
        flush=True,
    )

    model = ArbitraryOrderPatchModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    steps_per_epoch = (train_patches.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda(total_steps=total_steps, warmup_steps=warmup_steps),
    )

    started_at = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_start = time.perf_counter()
        shuffle = torch.randperm(train_patches.shape[0], device=device)
        shuffled_patches = train_patches[shuffle]
        total_train_loss = 0.0
        total_train_batches = 0

        for start in range(0, shuffled_patches.shape[0], BATCH_SIZE):
            stop = min(start + BATCH_SIZE, shuffled_patches.shape[0])
            batch = shuffled_patches[start:stop]
            (
                observed_patches,
                observed_coordinates,
                observed_padding_mask,
                target_patches,
                target_coordinates,
                target_padding_mask,
                loss_mask,
            ) = build_batch(batch, coordinates)

            predictions = model(
                observed_patches,
                observed_coordinates,
                target_coordinates,
                observed_padding_mask=observed_padding_mask,
                target_padding_mask=target_padding_mask,
            )
            loss = masked_mse(predictions, target_patches, loss_mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            total_train_batches += 1

        val_mse = evaluate(model, val_patches, coordinates)
        epoch_seconds = time.perf_counter() - epoch_start
        mean_train_mse = total_train_loss / max(total_train_batches, 1)
        print(
            f"[{timestamp()}] epoch={epoch}/{EPOCHS} train_mse={mean_train_mse:.6f} val_mse={val_mse:.6f} lr={optimizer.param_groups[0]['lr']:.8f} epoch_seconds={epoch_seconds:.2f}",
            flush=True,
        )

    total_seconds = time.perf_counter() - started_at
    final_val_mse = evaluate(model, val_patches, coordinates)
    print(f"[{timestamp()}] final_val_mse={final_val_mse:.6f} total_seconds={total_seconds:.2f}", flush=True)


if __name__ == "__main__":
    main()
