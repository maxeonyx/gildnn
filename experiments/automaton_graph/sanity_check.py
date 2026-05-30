from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

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

    model = GraphCellularAutomaton(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 4
    chunk_size = 32

    for step in range(5):
        inputs, targets = sample_batch(
            encoded_text,
            batch_size=batch_size,
            chunk_size=chunk_size,
            device=device,
        )
        output = model(inputs)
        ce_loss = F.cross_entropy(output.logits.reshape(-1, output.logits.shape[-1]), targets.reshape(-1))
        loss = ce_loss + output.total_prediction_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        print(
            f"step={step} ce_loss={ce_loss.item():.4f} mean_prediction_loss={output.prediction_losses.mean().item():.4f}"
        )


if __name__ == "__main__":
    main()
