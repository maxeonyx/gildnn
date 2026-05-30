"""Readout comparison probe: does an attention readout over all bands beat band-0-only?

Tests whether multi-timescale representations (from per-band CE training) are
actually useful when combined. Trains a fresh detached attention readout on
frozen model states and compares CE against the default band-0 mean readout.

Usage:
    .venv/bin/python experiments/automaton_graph/readout_probe.py --checkpoint runs/checkpoints_pbc.ignore/step_000500.pt
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

from core.automaton_graph import GraphCellularAutomaton, l2_normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Readout comparison: attention vs band-0-only.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4096, help="Training tokens for the probe")
    parser.add_argument("--eval-tokens", type=int, default=2048, help="Eval tokens")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--probe-steps", type=int, default=200, help="Steps to train the attention probe")
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def load_tinyshakespeare() -> tuple[Tensor, int]:
    repo_root = Path(__file__).resolve().parents[2]
    text_path = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    raw_text = text_path.read_text(encoding="utf-8")
    vocab = sorted(set(raw_text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    encoded = torch.tensor([stoi[ch] for ch in raw_text], dtype=torch.long)
    return encoded, len(vocab)


class AttentionReadoutProbe(torch.nn.Module):
    """Learned attention over frozen module states -> logits."""

    def __init__(self, d_stream: int, n_modules: int, vocab_size: int, temperature: float = 0.07):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(1, 1, d_stream) * 0.02)
        self.key_proj = torch.nn.Linear(d_stream, d_stream, bias=False)
        self.value_proj = torch.nn.Linear(d_stream, d_stream, bias=False)
        self.d_stream = d_stream
        self.temperature = temperature
        # Use the same fixed embedding readout
        self.register_buffer("embedding_weight", torch.zeros(vocab_size, d_stream))

    def set_embedding(self, embedding_weight: Tensor) -> None:
        self.embedding_weight.copy_(embedding_weight)

    def forward(self, module_states: Tensor) -> Tensor:
        """module_states: [batch, n_modules, d_stream] -> logits: [batch, vocab]"""
        keys = self.key_proj(module_states)  # [batch, n_modules, d_stream]
        values = self.value_proj(module_states)  # [batch, n_modules, d_stream]
        query = self.query.expand(module_states.shape[0], -1, -1)  # [batch, 1, d_stream]
        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / (self.d_stream ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, 1, n_modules]
        readout_state = torch.bmm(attn_weights, values).squeeze(1)  # [batch, d_stream]
        # Logits via normalized dot product with fixed embeddings
        normalized_hidden = l2_normalize(readout_state)
        normalized_embedding = l2_normalize(self.embedding_weight)
        logits = F.linear(normalized_hidden, normalized_embedding) / self.temperature
        return logits, attn_weights.squeeze(1)  # [batch, n_modules]


def collect_states_and_targets(
    model: GraphCellularAutomaton,
    encoded_text: Tensor,
    n_tokens: int,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Run model on text, collect module states at each token boundary and next-token targets."""
    chunk_size = min(n_tokens, 128)
    n_chunks = max(1, n_tokens // chunk_size)

    all_states = []
    all_targets = []

    with torch.no_grad():
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size * batch_size
            # Get batch of sequences
            windows = []
            for b in range(batch_size):
                s = start + b * chunk_size
                if s + chunk_size + 1 > encoded_text.numel():
                    s = s % (encoded_text.numel() - chunk_size - 1)
                windows.append(encoded_text[s : s + chunk_size + 1])
            windows_t = torch.stack(windows).to(device)
            inputs = windows_t[:, :-1]
            targets = windows_t[:, 1:]

            states, global_buffer, predictions, has_predicted, refractory_levels = model.initial_recurrent_state(
                batch_size, device=device
            )
            step_offset = torch.zeros((), device=device, dtype=torch.long)

            # Run forward, collect states at each token boundary
            seq_len = inputs.shape[1]
            token_embeddings = model.token_embedding(inputs)
            total_steps = seq_len * model.steps_per_token
            current_states = states

            # Simplified: just run full forward_chunk and collect final states
            logits_out, _per_band_logits, final_states, *_ = model.forward_chunk(
                inputs, states, global_buffer, predictions, has_predicted,
                refractory_levels, global_step_offset=step_offset,
            )

            # Collect mean per-module state (averaged over the sequence via final state)
            # Actually we want per-token-boundary states. Use a simpler approach:
            # Just use the final state as the representation for the last token in chunk
            # Shape: [n_modules, batch, d_stream] -> [batch, n_modules, d_stream]
            chunk_states = final_states.permute(1, 0, 2)  # [batch, n_modules, d_stream]
            chunk_targets = targets[:, -1]  # [batch] — last token in chunk

            all_states.append(chunk_states.cpu())
            all_targets.append(chunk_targets.cpu())

    return torch.cat(all_states, dim=0), torch.cat(all_targets, dim=0)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoded_text, vocab_size = load_tinyshakespeare()

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    model = GraphCellularAutomaton(
        vocab_size=vocab_size,
        multi_scale_input=ckpt_args.get("multi_scale_input", False),
        temporal_targets=ckpt_args.get("temporal_targets", False),
        cross_band_negatives=ckpt_args.get("cross_band_negatives", False),
        attention_readout=ckpt_args.get("attention_readout", False),
        per_band_ce=ckpt_args.get("per_band_ce", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    step = checkpoint.get("step", "?")
    print(f"Loaded checkpoint step={step}")

    # Collect states
    print("Collecting module states...")
    train_states, train_targets = collect_states_and_targets(
        model, encoded_text, args.tokens, args.batch_size, device
    )
    eval_states, eval_targets = collect_states_and_targets(
        model, encoded_text[len(encoded_text)//2:], args.eval_tokens, args.batch_size, device
    )
    print(f"  Train: {train_states.shape[0]} samples, Eval: {eval_states.shape[0]} samples")

    # Baseline: band-0-only CE
    with torch.no_grad():
        band0_mask = model.band0_mask.cpu()
        band0_states = eval_states[:, band0_mask, :]  # [samples, n_cols, d_stream]
        band0_mean = band0_states.mean(dim=1).to(device)  # [samples, d_stream]
        band0_logits = model.logits_from_hidden(band0_mean)
        band0_ce = F.cross_entropy(band0_logits, eval_targets.to(device)).item()
    print(f"\nBaseline (band-0 mean readout): CE = {band0_ce:.4f}")

    # Train attention readout probe
    probe = AttentionReadoutProbe(
        d_stream=model.d_stream,
        n_modules=model.n_modules,
        vocab_size=vocab_size,
    ).to(device)
    probe.set_embedding(model.token_embedding.weight.data)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)

    print(f"\nTraining attention readout probe ({args.probe_steps} steps)...")
    n_train = train_states.shape[0]
    for step_i in range(args.probe_steps):
        # Random batch
        idx = torch.randint(0, n_train, (min(32, n_train),))
        batch_states = train_states[idx].to(device)
        batch_targets = train_targets[idx].to(device)

        optimizer.zero_grad()
        logits, _ = probe(batch_states)
        loss = F.cross_entropy(logits, batch_targets)
        loss.backward()
        optimizer.step()

        if (step_i + 1) % 50 == 0:
            print(f"  probe step {step_i+1}: train CE = {loss.item():.4f}")

    # Evaluate probe
    probe.eval()
    with torch.no_grad():
        eval_logits, eval_attn = probe(eval_states.to(device))
        probe_ce = F.cross_entropy(eval_logits, eval_targets.to(device)).item()

    print(f"\n{'='*60}")
    print(f"READOUT COMPARISON (checkpoint step={checkpoint.get('step', '?')})")
    print(f"{'='*60}")
    print(f"  Band-0 mean readout CE:     {band0_ce:.4f}")
    print(f"  Attention readout probe CE:  {probe_ce:.4f}")
    improvement = band0_ce - probe_ce
    print(f"  Improvement:                 {improvement:+.4f} nats")
    print(f"  Relative:                    {improvement/band0_ce*100:+.1f}%")

    # Show attention distribution over bands
    n_bands = model.n_bands
    n_cols = model.n_cols
    # Reshape attention: [samples, n_modules] -> mean over samples, reshape to [n_bands, n_cols]
    mean_attn = eval_attn.mean(dim=0).cpu().reshape(n_bands, n_cols).mean(dim=1)
    print(f"\n  Attention weight per band (mean):")
    for b in range(n_bands):
        bar = "█" * int(mean_attn[b].item() * 80)
        print(f"    band {b}: {mean_attn[b].item():.4f} {bar}")


if __name__ == "__main__":
    main()
