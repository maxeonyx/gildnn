from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 192-module graph automaton on TinyShakespeare.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}.")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}.")
    if args.chunk_size <= 0:
        raise ValueError(f"--chunk-size must be positive, got {args.chunk_size}.")
    if args.lr <= 0.0:
        raise ValueError(f"--lr must be positive, got {args.lr}.")
    if args.log_every <= 0:
        raise ValueError(f"--log-every must be positive, got {args.log_every}.")
    return args


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


def format_per_band_losses(prediction_losses: Tensor, *, band_count: int, modules_per_band: int) -> str:
    per_band_mean_losses = rearrange(
        prediction_losses,
        "(band module) -> band module",
        band=band_count,
        module=modules_per_band,
    ).mean(dim=1)
    return ", ".join(f"b{band}={value.item():.4f}" for band, value in enumerate(per_band_mean_losses))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda")
    repo_root = Path(__file__).resolve().parents[2]
    data = load_tinyshakespeare(repo_root)

    model = GraphCellularAutomaton(vocab_size=data.vocab_size).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    print(
        f"starting training steps={args.steps} batch_size={args.batch_size} chunk_size={args.chunk_size} device={device}",
        flush=True,
    )

    start_time = time.perf_counter()
    final_ce_loss: float | None = None
    final_total_prediction_loss: float | None = None
    final_per_band_mean_losses: list[float] | None = None

    for step in range(1, args.steps + 1):
        inputs, targets = sample_batch(
            data.encoded_text,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            device=device,
            generator=generator,
        )

        states, global_buffer, predictions, has_predicted, refractory_levels = model.initial_recurrent_state(
            args.batch_size,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _, _, _, _, _, prediction_loss_sums, prediction_counts = model.forward_chunk(
            inputs,
            states,
            global_buffer,
            predictions,
            has_predicted,
            refractory_levels,
            global_step_offset=0,
        )
        prediction_losses = model.prediction_losses_from_sums(prediction_loss_sums, prediction_counts)
        total_prediction_loss = prediction_losses.sum()
        ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss = ce_loss + total_prediction_loss
        loss.backward()
        optimizer.step()

        final_ce_loss = ce_loss.item()
        final_total_prediction_loss = total_prediction_loss.item()
        final_per_band_mean_losses = rearrange(
            prediction_losses.detach(),
            "(band module) -> band module",
            band=model.n_bands,
            module=model.n_cols,
        ).mean(dim=1).cpu().tolist()

        if step % args.log_every == 0 or step == args.steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_s = time.perf_counter() - start_time
            print(
                f"step={step} ce_loss={final_ce_loss:.4f} total_prediction_loss={final_total_prediction_loss:.4f} elapsed_s={elapsed_s:.2f}",
                flush=True,
            )
            print(
                f"  per_band_mean_prediction_loss: {format_per_band_losses(prediction_losses.detach(), band_count=model.n_bands, modules_per_band=model.n_cols)}",
                flush=True,
            )

    total_time_s = time.perf_counter() - start_time
    summary = {
        "step_count": args.steps,
        "total_time_s": round(total_time_s, 6),
        "final_ce_loss": round(final_ce_loss if final_ce_loss is not None else float("nan"), 6),
        "final_total_prediction_loss": round(
            final_total_prediction_loss if final_total_prediction_loss is not None else float("nan"),
            6,
        ),
        "final_per_band_mean_prediction_loss": [
            round(value, 6) for value in (final_per_band_mean_losses if final_per_band_mean_losses is not None else [])
        ],
    }
    print(json.dumps(summary), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
