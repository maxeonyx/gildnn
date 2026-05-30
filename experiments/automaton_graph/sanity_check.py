from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import torch
from einops import rearrange
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.automaton_graph import GraphCellularAutomaton


def sample_batch(
    encoded_text: torch.Tensor,
    *,
    batch_size: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = encoded_text.numel() - chunk_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    inputs = torch.stack([encoded_text[start : start + chunk_size] for start in starts.tolist()])
    targets = torch.stack([encoded_text[start + 1 : start + chunk_size + 1] for start in starts.tolist()])
    return inputs.to(device=device), targets.to(device=device)


def main() -> None:
    device = torch.device("cuda")
    repo_root = Path(__file__).resolve().parents[2]
    text_path = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    raw_text = text_path.read_text(encoding="utf-8")

    vocab = sorted(set(raw_text))
    stoi = {char: index for index, char in enumerate(vocab)}
    encoded_text = torch.tensor([stoi[char] for char in raw_text], dtype=torch.long)

    model = GraphCellularAutomaton(vocab_size=len(vocab), loss_type="infonce").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 4
    chunk_size = 32
    band_count = model.n_bands
    modules_per_band = model.n_cols
    inputs, targets = sample_batch(
        encoded_text,
        batch_size=batch_size,
        chunk_size=chunk_size,
        device=device,
    )

    for step in range(20):
        step_start = perf_counter()
        output = model(inputs)
        ce_loss = F.cross_entropy(output.logits.reshape(-1, output.logits.shape[-1]), targets.reshape(-1))
        loss = ce_loss + output.total_prediction_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step_duration = perf_counter() - step_start
        per_band_prediction_losses = rearrange(
            output.prediction_losses,
            "(band module) -> band module",
            band=band_count,
            module=modules_per_band,
        ).mean(dim=1)
        per_band_prediction_counts = rearrange(
            output.prediction_counts,
            "(band module) -> band module",
            band=band_count,
            module=modules_per_band,
        ).sum(dim=1)
        losses_text = ", ".join(f"b{band}={value.item():.4f}" for band, value in enumerate(per_band_prediction_losses))
        counts_text = ", ".join(f"b{band}={value.item()}" for band, value in enumerate(per_band_prediction_counts))

        print(
            f"step={step} ce_loss={ce_loss.item():.4f} total_prediction_loss={output.total_prediction_loss.item():.4f} "
            f"time_per_step={step_duration:.4f}s"
        )
        print(f"  per_band_prediction_loss: {losses_text}")
        print(f"  per_band_prediction_count: {counts_text}")


if __name__ == "__main__":
    main()
