"""Gradient audit: measure per-band gradient norms from CE vs local loss.

This answers the key question: is CE gradient a gentle nudge or a steamroller
relative to local InfoNCE loss? Also measures gradient cosine between the two
signals per band.

Usage:
    .venv/bin/python experiments/automaton_graph/grad_audit.py --checkpoint runs/checkpoints_attn.ignore/step_000500.pt
    .venv/bin/python experiments/automaton_graph/grad_audit.py --random  # control with random model
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import Tensor
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.automaton_graph import GraphCellularAutomaton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradient audit: CE vs local loss per band.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--checkpoint", type=Path)
    source_group.add_argument("--random", action="store_true")
    parser.add_argument("--tokens", type=int, default=512, help="Tokens to process")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def load_tinyshakespeare() -> tuple[Tensor, int]:
    repo_root = Path(__file__).resolve().parents[2]
    text_path = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    raw_text = text_path.read_text(encoding="utf-8")
    vocab = sorted(set(raw_text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    encoded = torch.tensor([stoi[ch] for ch in raw_text], dtype=torch.long)
    return encoded, len(vocab)


def get_per_band_grad_norms(model: GraphCellularAutomaton) -> dict[str, list[float]]:
    """Extract per-band gradient L2 norms for w1, w2 (the main computation params)."""
    n_bands = model.n_bands
    n_cols = model.n_cols
    result = {}
    for name in ["w1", "w2", "b1", "b2"]:
        param = getattr(model, name)
        if param.grad is None:
            result[name] = [0.0] * n_bands
            continue
        # param shape: [n_modules, ...] where n_modules = n_bands * n_cols
        grad = param.grad.reshape(n_bands, n_cols, *param.grad.shape[1:])
        # Per-band L2 norm (flatten all dims except band, then norm)
        per_band = grad.float().reshape(n_bands, -1).norm(dim=1).tolist()
        result[name] = per_band
    return result


def get_flat_grad_vector(model: GraphCellularAutomaton, params: list[str]) -> Tensor:
    """Flatten gradients from specified params into one vector."""
    parts = []
    for name in params:
        param = getattr(model, name)
        if param.grad is not None:
            parts.append(param.grad.flatten())
        else:
            parts.append(torch.zeros(param.numel(), device=param.device))
    return torch.cat(parts)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoded_text, vocab_size = load_tinyshakespeare()

    # Load model
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        ckpt_args = checkpoint.get("args", {})
        model = GraphCellularAutomaton(
            vocab_size=vocab_size,
            multi_scale_input=ckpt_args.get("multi_scale_input", False),
            temporal_targets=ckpt_args.get("temporal_targets", False),
            cross_band_negatives=ckpt_args.get("cross_band_negatives", False),
            attention_readout=ckpt_args.get("attention_readout", False),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        step = checkpoint.get("step", "?")
        ce_at_ckpt = checkpoint.get("ce_loss", "?")
        print(f"Loaded checkpoint step={step}, ce_loss={ce_at_ckpt}")
    else:
        model = GraphCellularAutomaton(
            vocab_size=vocab_size,
            multi_scale_input=True,
            attention_readout=True,
        ).to(device)
        print("Using random model")

    model.train()
    chunk_size = min(args.tokens, 128)

    # Prepare data
    max_start = encoded_text.numel() - chunk_size - 1
    starts = torch.randint(0, max_start + 1, (args.batch_size,))
    offsets = torch.arange(chunk_size + 1, dtype=torch.long)
    windows = encoded_text[starts[:, None] + offsets]
    inputs = windows[:, :-1].to(device)
    targets = windows[:, 1:].to(device)

    # Forward pass
    states, global_buffer, predictions, has_predicted, refractory_levels = model.initial_recurrent_state(
        args.batch_size, device=device
    )
    step_offset = torch.zeros((), device=device, dtype=torch.long)

    logits, *_, prediction_loss_sums, prediction_counts = model.forward_chunk(
        inputs, states, global_buffer, predictions, has_predicted, refractory_levels,
        global_step_offset=step_offset,
    )
    prediction_losses = model.prediction_losses_from_sums(prediction_loss_sums, prediction_counts)
    total_prediction_loss = prediction_losses.sum()
    ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    print(f"\nCE loss: {ce_loss.item():.4f}")
    print(f"Total prediction loss: {total_prediction_loss.item():.4f}")

    # --- Gradient from CE only ---
    model.zero_grad(set_to_none=True)
    ce_loss.backward(retain_graph=True)
    ce_grads = get_per_band_grad_norms(model)
    ce_grad_vector = get_flat_grad_vector(model, ["w1", "w2", "b1", "b2"]).clone()

    # --- Gradient from local loss only ---
    model.zero_grad(set_to_none=True)
    total_prediction_loss.backward()
    local_grads = get_per_band_grad_norms(model)
    local_grad_vector = get_flat_grad_vector(model, ["w1", "w2", "b1", "b2"]).clone()

    # --- Compute cosine similarity ---
    cos_sim = F.cosine_similarity(ce_grad_vector.unsqueeze(0), local_grad_vector.unsqueeze(0)).item()

    # --- Report ---
    print("\n" + "=" * 70)
    print("GRADIENT AUDIT: CE gradient vs Local InfoNCE gradient per band")
    print("=" * 70)

    print(f"\n{'':8s} {'CE grad L2':>12s} {'Local grad L2':>14s} {'Ratio (local/CE)':>18s}")
    print("-" * 56)

    for band in range(model.n_bands):
        ce_total = sum(ce_grads[p][band] for p in ["w1", "w2", "b1", "b2"])
        local_total = sum(local_grads[p][band] for p in ["w1", "w2", "b1", "b2"])
        ratio = local_total / ce_total if ce_total > 0 else float("inf")
        print(f"band {band:2d}  {ce_total:12.6f} {local_total:14.6f} {ratio:18.1f}x")

    print(f"\nGlobal gradient cosine (CE vs local): {cos_sim:.4f}")
    print(f"  (0 = orthogonal, 1 = aligned, -1 = opposed)")

    # Per-param breakdown
    print(f"\nPer-param breakdown (total L2 norm across all bands):")
    for p in ["w1", "w2", "b1", "b2"]:
        ce_sum = sum(ce_grads[p])
        local_sum = sum(local_grads[p])
        print(f"  {p}: CE={ce_sum:.6f}, Local={local_sum:.6f}, ratio={local_sum/ce_sum:.1f}x" if ce_sum > 0 else f"  {p}: CE=0, Local={local_sum:.6f}")


if __name__ == "__main__":
    main()
