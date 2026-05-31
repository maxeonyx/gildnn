from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Int
from torch import Tensor

if __package__ in {None, ""}:
    from model import SpecGraphCarry, SpecGraphConfig, SpecGraphModel
else:
    from .model import SpecGraphCarry, SpecGraphConfig, SpecGraphModel


torch.backends.cuda.matmul.allow_tf32 = True


@dataclass(frozen=True)
class TinyShakespeareData:
    encoded_text: Int[Tensor, "tokens"]
    vocab_size: int


class StreamingData:
    def __init__(self, encoded_text: Int[Tensor, "tokens"], batch_size: int, chunk_size: int, device: torch.device) -> None:
        usable_tokens = (encoded_text.numel() // batch_size) * batch_size
        if usable_tokens <= chunk_size:
            raise ValueError(
                f"Corpus is too short for batch_size={batch_size} and chunk_size={chunk_size}: {encoded_text.numel()} tokens."
            )
        self.streams = encoded_text[:usable_tokens].reshape(batch_size, -1)
        self.chunk_size = chunk_size
        self.device = device
        self.position = 0
        self.stream_len = self.streams.shape[1]

    def next_chunk(self) -> tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"]]:
        end = self.position + self.chunk_size + 1
        if end > self.stream_len:
            self.position = 0
            end = self.chunk_size + 1
        window = self.streams[:, self.position:end]
        self.position += self.chunk_size
        return (
            window[:, :-1].to(self.device, dtype=torch.long),
            window[:, 1:].to(self.device, dtype=torch.long),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SPEC local-learning graph skeleton.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grid-h", type=int, default=4)
    parser.add_argument("--grid-w", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--predict-horizon", type=int, default=4)
    parser.add_argument("--local-loss-weight", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--allow-head-gradient", action="store_true", help="Allow CE gradient to flow into graph nodes (scaffolding)")
    parser.add_argument("--no-reward", action="store_true", help="Disable reward modulation (constant reward=1.0)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sanity-check-only", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}.")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}.")
    if args.chunk_size <= 0:
        raise ValueError(f"--chunk-size must be positive, got {args.chunk_size}.")
    if args.lr <= 0.0:
        raise ValueError(f"--lr must be positive, got {args.lr}.")
    if args.grid_h <= 0 or args.grid_w <= 0:
        raise ValueError(f"Grid dimensions must be positive, got {args.grid_h}x{args.grid_w}.")
    if args.d_model <= 0:
        raise ValueError(f"--d-model must be positive, got {args.d_model}.")
    if args.rollout_steps <= 0:
        raise ValueError(f"--rollout-steps must be positive, got {args.rollout_steps}.")
    if args.noise_std < 0.0:
        raise ValueError(f"--noise-std must be non-negative, got {args.noise_std}.")
    if args.local_loss_weight < 0.0:
        raise ValueError(f"--local-loss-weight must be non-negative, got {args.local_loss_weight}.")
    if args.log_every <= 0:
        raise ValueError(f"--log-every must be positive, got {args.log_every}.")
    if args.sanity_check_only:
        args.steps = min(args.steps, 3)
        args.log_every = 1
    return args


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tinyshakespeare(repo_root: Path) -> TinyShakespeareData:
    text_path = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    raw_text = text_path.read_text(encoding="utf-8")
    vocab = sorted(set(raw_text))
    stoi = {character: index for index, character in enumerate(vocab)}
    encoded_text = torch.tensor([stoi[character] for character in raw_text], dtype=torch.long)
    return TinyShakespeareData(encoded_text=encoded_text, vocab_size=len(vocab))


def format_horizon_metrics(horizon_steps: tuple[int, ...], per_horizon_ce: Tensor) -> str:
    return ", ".join(
        f"h{horizon}={value.item():.4f}" for horizon, value in zip(horizon_steps, per_horizon_ce, strict=True)
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[2]
    data = load_tinyshakespeare(repo_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpecGraphModel(
        SpecGraphConfig(
            vocab_size=data.vocab_size,
            grid_h=args.grid_h,
            grid_w=args.grid_w,
            d_model=args.d_model,
            rollout_steps=args.rollout_steps,
            noise_std=args.noise_std,
            ema_decay=args.ema_decay,
            predict_horizon=args.predict_horizon,
            disable_reward=args.no_reward,
            detach_head_input=not args.allow_head_gradient,
        )
    ).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stream = StreamingData(data.encoded_text, args.batch_size, args.chunk_size, device)
    carry_state = model.initial_carry(args.batch_size, device)

    print(
        f"starting training steps={args.steps} batch_size={args.batch_size} chunk_size={args.chunk_size} device={device.type}",
        flush=True,
    )
    start_time = time.perf_counter()

    for step in range(1, args.steps + 1):
        inputs, targets = stream.next_chunk()
        carry_state = carry_state.detach_all()

        optimizer.zero_grad(set_to_none=True)
        output, new_state = model.forward_chunk(inputs, targets, carry_state)
        loss = output.head_ce_loss + args.local_loss_weight * output.local_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        carry_state = new_state

        if step % args.log_every == 0 or step == args.steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start_time
            print(
                f"step={step} head_ce={output.head_ce_loss.item():.4f} local_loss={output.local_loss.item():.4f} reward={output.reward_scalar.item():.4f}[{output.reward_min.item():.3f},{output.reward_max.item():.3f}] time_elapsed={elapsed:.2f}s",
                flush=True,
            )
            print(
                f"  per_horizon_ce: {format_horizon_metrics(model.config.horizon_steps, output.per_horizon_ce)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
