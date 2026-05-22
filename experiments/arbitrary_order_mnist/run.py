from __future__ import annotations

import subprocess
import sys
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
OBSERVED_PATCH_COUNT = PATCH_COUNT // 2
PREDICTED_PATCH_COUNT = PATCH_COUNT - OBSERVED_PATCH_COUNT

BATCH_SIZE = 32
TRAIN_STEPS = 2000
PRINT_EVERY = 100
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


def load_mnist_patches(*, device: torch.device) -> Tensor:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = MNIST(root=DATA_DIR, train=True, download=True)
    images = dataset.data.to(dtype=torch.float32).div_(255.0).unsqueeze(1)
    padded = F.pad(images, (2, 2, 2, 2))
    patches = padded.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
    patches = patches.contiguous().view(-1, GRID_SIZE, GRID_SIZE, PATCH_DIM).view(-1, PATCH_COUNT, PATCH_DIM)
    return patches.to(device)


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
    ) -> Tensor:
        encoder_input = self.patch_projection(observed_patches) + self.position_projection(observed_coordinates)
        memory = self.encoder(encoder_input)
        queries = self.position_projection(target_coordinates)
        decoded = self.decoder(queries, memory)
        return self.output_head(self.output_norm(decoded))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(
        f"[{timestamp()}] device={device} device_name={cuda_name} git_sha={current_git_sha()} python={sys.version.split()[0]} torch={torch.__version__}",
        flush=True,
    )
    print(
        f"[{timestamp()}] config batch_size={BATCH_SIZE} steps={TRAIN_STEPS} lr={LEARNING_RATE} d_model={D_MODEL} heads={N_HEADS} encoder_layers={N_ENCODER_LAYERS} decoder_layers={N_DECODER_LAYERS}",
        flush=True,
    )

    patches = load_mnist_patches(device=device)
    coordinates = build_patch_coordinates(device=device)
    batch = patches[:BATCH_SIZE]

    observed_indices = torch.arange(OBSERVED_PATCH_COUNT, device=device)
    target_indices = torch.arange(OBSERVED_PATCH_COUNT, PATCH_COUNT, device=device)
    observed_patches = batch[:, observed_indices, :]
    target_patches = batch[:, target_indices, :]
    observed_coordinates = coordinates[observed_indices].unsqueeze(0).expand(BATCH_SIZE, -1, -1)
    target_coordinates = coordinates[target_indices].unsqueeze(0).expand(BATCH_SIZE, -1, -1)

    model = ArbitraryOrderPatchModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for step in range(1, TRAIN_STEPS + 1):
        predictions = model(observed_patches, observed_coordinates, target_coordinates)
        loss = F.mse_loss(predictions, target_patches)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % PRINT_EVERY == 0:
            print(f"[{timestamp()}] step={step} mse={loss.item():.8f}", flush=True)

    with torch.no_grad():
        final_predictions = model(observed_patches, observed_coordinates, target_coordinates)
        final_mse = F.mse_loss(final_predictions, target_patches).item()

    print(f"[{timestamp()}] final_mse={final_mse:.8f}", flush=True)


if __name__ == "__main__":
    main()
