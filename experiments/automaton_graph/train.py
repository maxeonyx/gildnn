from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.automaton_graph import GraphCellularAutomaton


torch.backends.cuda.matmul.allow_tf32 = True

PER_BAND_CE_WEIGHT = 0.1


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
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--save-dir", type=Path, default=Path("runs/checkpoints.ignore/"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi-scale-input", action="store_true", help="EMA token injection per band")
    parser.add_argument("--temporal-targets", action="store_true", help="Predict future token embedding instead of neighbor sum")
    parser.add_argument("--cross-band-negatives", action="store_true", help="Use other bands as negatives in InfoNCE (anti-redundancy)")
    parser.add_argument("--attention-readout", action="store_true", help="Use attention over all module states instead of mean band-0")
    parser.add_argument("--per-band-ce", action="store_true", help="Add per-band horizon cross-entropy loss")
    parser.add_argument("--streaming", action="store_true", help="Stateful streaming training (no state reset between chunks)")
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
    if args.save_every <= 0:
        raise ValueError(f"--save-every must be positive, got {args.save_every}.")
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


class StreamingData:
    """Split text into batch_size contiguous streams. Yield chunk_size tokens endlessly."""

    def __init__(self, encoded_text: Tensor, batch_size: int, chunk_size: int, device: torch.device) -> None:
        # Trim text to be evenly divisible by batch_size
        n = (encoded_text.numel() // batch_size) * batch_size
        self.streams = encoded_text[:n].reshape(batch_size, -1)  # [batch, stream_len]
        self.chunk_size = chunk_size
        self.device = device
        self.pos = 0  # current position in each stream
        self.stream_len = self.streams.shape[1]

    def next_chunk(self) -> tuple[Tensor, Tensor]:
        end = self.pos + self.chunk_size + 1
        if end > self.stream_len:
            self.pos = 0
            end = self.chunk_size + 1
        window = self.streams[:, self.pos : end]  # [batch, chunk_size+1]
        self.pos += self.chunk_size
        inputs = window[:, :-1].to(device=self.device, dtype=torch.long)
        targets = window[:, 1:].to(device=self.device, dtype=torch.long)
        return inputs, targets


def format_per_band_losses(prediction_losses: Tensor, *, band_count: int, modules_per_band: int) -> str:
    per_band_mean_losses = prediction_losses.reshape(band_count, modules_per_band).mean(dim=1)
    return ", ".join(f"b{band}={value.item():.4f}" for band, value in enumerate(per_band_mean_losses))


def compute_per_band_ce(
    *,
    per_band_logits: Tensor | None,
    targets: Tensor,
    chunk_size: int,
) -> tuple[Tensor, Tensor | None]:
    if per_band_logits is None:
        return targets.new_zeros((), dtype=torch.float32), None

    batch_size, n_bands, seq_len, vocab_size = per_band_logits.shape
    per_band_losses = torch.zeros((n_bands,), device=per_band_logits.device, dtype=torch.float32)
    horizon_cap = max(1, chunk_size // 2)
    for band in range(n_bands):
        horizon = min(2**band, horizon_cap)
        target_shift = horizon - 1
        valid_positions = seq_len - target_shift
        if valid_positions <= 0:
            continue
        band_logits = per_band_logits[:, band, :valid_positions, :].reshape(batch_size * valid_positions, vocab_size)
        band_targets = targets[:, target_shift:].reshape(batch_size * valid_positions)
        per_band_losses[band] = F.cross_entropy(band_logits, band_targets)

    return per_band_losses.sum() * PER_BAND_CE_WEIGHT, per_band_losses


def save_checkpoint(
    *,
    model: GraphCellularAutomaton,
    optimizer: torch.optim.Optimizer,
    step: int,
    ce_loss: float,
    save_dir: Path,
    args: object | None = None,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f"step_{step:06d}.pt"
    checkpoint_data: dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "ce_loss": ce_loss,
    }
    if args is not None:
        checkpoint_data["args"] = vars(args) if hasattr(args, "__dict__") else args
    torch.save(checkpoint_data, checkpoint_path)
    return checkpoint_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda")
    repo_root = Path(__file__).resolve().parents[2]
    data = load_tinyshakespeare(repo_root)

    model = GraphCellularAutomaton(vocab_size=data.vocab_size, multi_scale_input=args.multi_scale_input, temporal_targets=args.temporal_targets, cross_band_negatives=args.cross_band_negatives, attention_readout=args.attention_readout, per_band_ce=args.per_band_ce).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    initial_step = 0

    if args.resume is not None:
        print(f"loading checkpoint {args.resume}", flush=True)
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_step = int(checkpoint["step"])

    print(
        f"starting training steps={args.steps} batch_size={args.batch_size} chunk_size={args.chunk_size} device={device} streaming={args.streaming}",
        flush=True,
    )

    # Set up streaming or random-chunk data
    if args.streaming:
        stream = StreamingData(data.encoded_text, args.batch_size, args.chunk_size, device)
        # Persistent state for streaming (carried across steps, detached for TBPTT)
        carry_states, carry_buffer, carry_preds, carry_has_pred, carry_refrac = model.initial_recurrent_state(
            args.batch_size, device=device,
        )
        carry_step_offset = torch.zeros((), device=device, dtype=torch.long)

    start_time = time.perf_counter()
    final_ce_loss: float | None = None
    final_total_prediction_loss: float | None = None
    final_per_band_ce_loss: float | None = None
    final_per_band_mean_losses: list[float] | None = None
    final_per_band_ce_losses: list[float] | None = None

    for step in range(initial_step + 1, args.steps + 1):
        if args.streaming:
            inputs, targets = stream.next_chunk()
            # Detach carried state (TBPTT boundary)
            states = carry_states.detach()
            global_buffer = carry_buffer.detach()
            predictions = carry_preds.detach()
            has_predicted = carry_has_pred
            refractory_levels = carry_refrac.detach()
            step_offset = carry_step_offset
        else:
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
            step_offset = torch.zeros((), device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits, per_band_logits, new_states, new_buffer, new_preds, new_has_pred, new_refrac, prediction_loss_sums, prediction_counts = model.forward_chunk(
            inputs,
            states,
            global_buffer,
            predictions,
            has_predicted,
            refractory_levels,
            global_step_offset=step_offset,
        )
        prediction_losses = model.prediction_losses_from_sums(prediction_loss_sums, prediction_counts)
        total_prediction_loss = prediction_losses.sum()
        ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        per_band_ce_loss, per_band_ce_losses = compute_per_band_ce(
            per_band_logits=per_band_logits,
            targets=targets,
            chunk_size=args.chunk_size,
        )
        loss = ce_loss + total_prediction_loss + per_band_ce_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if args.streaming:
            carry_states = new_states
            carry_buffer = new_buffer
            carry_preds = new_preds
            carry_has_pred = new_has_pred
            carry_refrac = new_refrac
            carry_step_offset = step_offset + args.chunk_size * model.steps_per_token

        final_ce_loss = ce_loss.item()
        final_total_prediction_loss = total_prediction_loss.item()
        final_per_band_ce_loss = per_band_ce_loss.item()
        final_per_band_mean_losses = (
            prediction_losses.detach().reshape(model.n_bands, model.n_cols).mean(dim=1).cpu().tolist()
        )
        final_per_band_ce_losses = None if per_band_ce_losses is None else per_band_ce_losses.detach().cpu().tolist()

        if step % args.log_every == 0 or step == args.steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_s = time.perf_counter() - start_time
            print(
                f"step={step} ce_loss={final_ce_loss:.4f} total_prediction_loss={final_total_prediction_loss:.4f} per_band_ce_loss={final_per_band_ce_loss:.4f} elapsed_s={elapsed_s:.2f}",
                flush=True,
            )
            print(
                f"  per_band_mean_prediction_loss: {format_per_band_losses(prediction_losses.detach(), band_count=model.n_bands, modules_per_band=model.n_cols)}",
                flush=True,
            )
            if per_band_ce_losses is not None:
                print(
                    "  per_band_ce: " + ", ".join(
                        f"b{band}={value.item():.4f}" for band, value in enumerate(per_band_ce_losses.detach())
                    ),
                    flush=True,
                )

        if step % args.save_every == 0:
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                ce_loss=final_ce_loss,
                save_dir=args.save_dir,
                args=args,
            )
            print(f"saved checkpoint {checkpoint_path}", flush=True)

    total_time_s = time.perf_counter() - start_time
    summary = {
        "step_count": max(args.steps, initial_step),
        "total_time_s": round(total_time_s, 6),
        "final_ce_loss": round(final_ce_loss if final_ce_loss is not None else float("nan"), 6),
        "final_total_prediction_loss": round(
            final_total_prediction_loss if final_total_prediction_loss is not None else float("nan"),
            6,
        ),
        "final_per_band_ce_loss": round(final_per_band_ce_loss if final_per_band_ce_loss is not None else float("nan"), 6),
        "final_per_band_mean_prediction_loss": [
            round(value, 6) for value in (final_per_band_mean_losses if final_per_band_mean_losses is not None else [])
        ],
        "final_per_band_ce": [
            round(value, 6) for value in (final_per_band_ce_losses if final_per_band_ce_losses is not None else [])
        ],
    }
    print(json.dumps(summary), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
