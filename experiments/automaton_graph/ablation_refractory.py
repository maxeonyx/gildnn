from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.automaton_graph import GraphCellularAutomaton


torch.backends.cuda.matmul.allow_tf32 = True


@dataclass(frozen=True)
class TinyShakespeareData:
    encoded_text: Tensor
    vocab_size: int


@dataclass(frozen=True)
class TrajectoryPoint:
    step: int
    ce_loss: float
    total_prediction_loss: float


@dataclass(frozen=True)
class AblationResult:
    name: str
    trajectory: list[TrajectoryPoint]
    per_band_prediction_losses: list[float]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tinyshakespeare(repo_root: Path) -> TinyShakespeareData:
    text_path = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    raw_text = text_path.read_text(encoding="utf-8")
    vocab = sorted(set(raw_text))
    stoi = {char: index for index, char in enumerate(vocab)}
    encoded_text = torch.tensor([stoi[char] for char in raw_text], dtype=torch.long)
    return TinyShakespeareData(encoded_text=encoded_text, vocab_size=len(vocab))


def sample_batch(
    encoded_text: Tensor,
    *,
    batch_size: int,
    chunk_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    max_start = encoded_text.numel() - chunk_size - 1
    if max_start < 0:
        raise ValueError(
            f"Corpus must be at least chunk_size + 1 tokens long, got {encoded_text.numel()} and chunk_size={chunk_size}."
        )

    starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
    offsets = torch.arange(chunk_size + 1, dtype=torch.long)
    windows = encoded_text[starts[:, None] + offsets]
    inputs = windows[:, :-1]
    targets = windows[:, 1:]

    pin_memory = device.type == "cuda"
    if pin_memory:
        inputs = inputs.pin_memory()
        targets = targets.pin_memory()

    return (
        inputs.to(device=device, dtype=torch.long, non_blocking=pin_memory),
        targets.to(device=device, dtype=torch.long, non_blocking=pin_memory),
    )


def mean_per_band_prediction_losses(model: GraphCellularAutomaton, prediction_losses: Tensor) -> list[float]:
    return (
        rearrange(
            prediction_losses.detach(),
            "(band module) -> band module",
            band=model.n_bands,
            module=model.n_cols,
        )
        .mean(dim=1)
        .cpu()
        .tolist()
    )


def run_overfit(
    *,
    name: str,
    vocab_size: int,
    inputs: Tensor,
    targets: Tensor,
    steps: int,
    refractory: bool,
) -> AblationResult:
    device = inputs.device
    model = GraphCellularAutomaton(vocab_size=vocab_size, refractory=refractory).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trajectory: list[TrajectoryPoint] = []
    final_per_band_prediction_losses: list[float] = []

    for step in range(steps):
        output = model(inputs)
        ce_loss = F.cross_entropy(output.logits.reshape(-1, output.logits.shape[-1]), targets.reshape(-1))
        total_prediction_loss = output.total_prediction_loss
        loss = ce_loss + total_prediction_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        trajectory.append(
            TrajectoryPoint(
                step=step,
                ce_loss=ce_loss.item(),
                total_prediction_loss=total_prediction_loss.item(),
            )
        )
        final_per_band_prediction_losses = mean_per_band_prediction_losses(model, output.prediction_losses)

    return AblationResult(
        name=name,
        trajectory=trajectory,
        per_band_prediction_losses=final_per_band_prediction_losses,
    )


def format_per_band_losses(losses: list[float]) -> str:
    return ", ".join(f"b{band}={value:.4f}" for band, value in enumerate(losses))


def print_trajectory_table(baseline: AblationResult, refractory: AblationResult) -> None:
    print("step | baseline_ce | baseline_pred | refractory_ce | refractory_pred")
    for baseline_point, refractory_point in zip(baseline.trajectory, refractory.trajectory, strict=True):
        print(
            f"{baseline_point.step:>4} | "
            f"{baseline_point.ce_loss:>11.4f} | "
            f"{baseline_point.total_prediction_loss:>13.4f} | "
            f"{refractory_point.ce_loss:>13.4f} | "
            f"{refractory_point.total_prediction_loss:>15.4f}"
        )


def main() -> None:
    device = torch.device("cuda")
    repo_root = Path(__file__).resolve().parents[2]
    data = load_tinyshakespeare(repo_root)

    seed = 0
    batch_size = 4
    chunk_size = 32
    steps = 20

    set_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs, targets = sample_batch(
        data.encoded_text,
        batch_size=batch_size,
        chunk_size=chunk_size,
        device=device,
        generator=generator,
    )

    set_seed(seed)
    baseline = run_overfit(
        name="baseline",
        vocab_size=data.vocab_size,
        inputs=inputs,
        targets=targets,
        steps=steps,
        refractory=False,
    )

    set_seed(seed)
    refractory = run_overfit(
        name="refractory",
        vocab_size=data.vocab_size,
        inputs=inputs,
        targets=targets,
        steps=steps,
        refractory=True,
    )

    print_trajectory_table(baseline, refractory)
    print()
    print(f"baseline per_band_prediction_loss:   {format_per_band_losses(baseline.per_band_prediction_losses)}")
    print(f"refractory per_band_prediction_loss: {format_per_band_losses(refractory.per_band_prediction_losses)}")


if __name__ == "__main__":
    main()
