from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn
from torch.nn import functional as F

from core.fixed_window_char import load_dataset, set_seed


torch.backends.cuda.matmul.allow_tf32 = True


class CausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        chunk_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(chunk_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=tokens.device),
            diagonal=1,
        )
        hidden = self.layers(hidden, mask=causal_mask)
        hidden = self.final_norm(hidden)
        return self.lm_head(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a causal transformer baseline on TinyShakespeare.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}.")
    if args.log_every <= 0:
        raise ValueError(f"--log-every must be positive, got {args.log_every}.")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}.")
    if args.chunk_size <= 1:
        raise ValueError(f"--chunk-size must be greater than 1, got {args.chunk_size}.")

    return args


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def prepare_dataset(*, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    (train_inputs, train_next_tokens), (val_inputs, val_next_tokens), vocab_size = load_dataset(context_size=chunk_size)
    train_targets = torch.cat((train_inputs[:, 1:], train_next_tokens.unsqueeze(1)), dim=1)
    val_targets = torch.cat((val_inputs[:, 1:], val_next_tokens.unsqueeze(1)), dim=1)
    return train_inputs, train_targets, val_inputs, val_targets, vocab_size


def compute_val_ce(model: nn.Module, val_inputs: torch.Tensor, val_targets: torch.Tensor, vocab_size: int, device: torch.device) -> float:
    """Compute validation CE on a random 256-sample batch."""
    indices = torch.randint(0, val_inputs.shape[0], (256,), generator=torch.Generator().manual_seed(17241))
    batch_inputs = val_inputs[indices].to(device)
    batch_targets = val_targets[indices].to(device)
    with torch.inference_mode():
        logits = model(batch_inputs)
        return F.cross_entropy(logits.reshape(-1, vocab_size), batch_targets.reshape(-1)).item()


def main() -> None:
    args = parse_args()
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_inputs, train_targets, val_inputs, val_targets, vocab_size = prepare_dataset(chunk_size=args.chunk_size)
    model = CausalTransformer(vocab_size=vocab_size, chunk_size=args.chunk_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    parameter_count = count_parameters(model)
    sample_count = train_inputs.shape[0]

    print(f"params={parameter_count}", flush=True)

    start_time = time.perf_counter()
    final_ce_loss = float("nan")

    for step in range(1, args.steps + 1):
        batch_indices = torch.randint(0, sample_count, (args.batch_size,))
        batch_inputs = train_inputs[batch_indices].to(device)
        batch_targets = train_targets[batch_indices].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), batch_targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        final_ce_loss = loss.item()
        if step % args.log_every == 0 or step == args.steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_s = time.perf_counter() - start_time
            val_ce = compute_val_ce(model, val_inputs, val_targets, vocab_size, device)
            print(f"step={step} train_ce={final_ce_loss:.4f} val_ce={val_ce:.4f} elapsed_s={elapsed_s:.2f}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_time_s = time.perf_counter() - start_time
    final_val_ce = compute_val_ce(model, val_inputs, val_targets, vocab_size, device)
    print(
        json.dumps(
            {
                "step_count": args.steps,
                "final_train_ce": round(final_ce_loss, 6),
                "final_val_ce": round(final_val_ce, 6),
                "total_time_s": round(total_time_s, 6),
                "model": "transformer",
                "params": parameter_count,
            }
        ),
        flush=True,
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
