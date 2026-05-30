"""Per-band CE probe: measures how much language-modeling information each band's hidden state carries.

For each band, decode its mean hidden state through the fixed embedding readout and compute CE loss
against the next token. This reveals whether bands encode useful info even if the lag probe can't detect it.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.automaton_graph import GraphCellularAutomaton, l2_normalize


def load_tinyshakespeare(repo_root: Path):
    text_path = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    raw_text = text_path.read_text(encoding="utf-8")
    vocab = sorted(set(raw_text))
    stoi = {char: index for index, char in enumerate(vocab)}
    encoded_text = torch.tensor([stoi[char] for char in raw_text], dtype=torch.long)
    return encoded_text, len(vocab)


def infer_model_kwargs(checkpoint: dict) -> dict:
    state_dict = checkpoint["model_state_dict"]
    w1 = state_dict["w1"]
    token_embedding = state_dict["token_embedding.weight"]
    n_modules, d_stream, d_hidden = w1.shape
    kwargs = {
        "vocab_size": int(token_embedding.shape[0]),
        "d_stream": int(d_stream),
        "d_hidden": int(d_hidden),
        "n_bands": 8,
        "n_cols": int(n_modules // 8),
    }
    train_args = checkpoint.get("args")
    if isinstance(train_args, dict):
        if train_args.get("multi_scale_input"):
            kwargs["multi_scale_input"] = True
        if train_args.get("temporal_targets"):
            kwargs["temporal_targets"] = True
        if train_args.get("cross_band_negatives"):
            kwargs["cross_band_negatives"] = True
    return kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokens", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_kwargs = infer_model_kwargs(checkpoint)
    model = GraphCellularAutomaton(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    repo_root = Path(__file__).resolve().parents[2]
    encoded_text, _ = load_tinyshakespeare(repo_root)
    text_tensor = encoded_text[:args.tokens + 1].unsqueeze(0).to(device)
    tokens = text_tensor[:, :-1]  # [1, seq]
    targets = text_tensor[:, 1:]  # [1, seq]

    n_bands = model.n_bands
    n_cols = model.n_cols

    with torch.no_grad():
        states, global_buffer, predictions, has_predicted, refractory_levels = model.initial_recurrent_state(1, device=device)
        logits, final_states, _, _, _, _, _, _ = model.forward_chunk(
            tokens, states, global_buffer, predictions, has_predicted, refractory_levels,
            global_step_offset=0,
        )

        # Model CE (band 0 mean, as normal)
        model_ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).item()

        # Per-band CE: decode each band's mean state at end of sequence
        # Actually we need per-token states. The model only returns final states.
        # Instead, let's run the model and collect logits from each band at each token.
        # Simpler: just decode the final_states through the readout for a rough estimate.
        # But that only gives one token. Let me instead modify to collect per-token per-band logits.

        # Actually, let's just compute logits from each band's final state and compare to the LAST target token.
        # This is a rough proxy but fast.
        print(f"Model CE (normal readout): {model_ce:.4f}")
        print(f"Random baseline CE: {torch.log(torch.tensor(float(model.vocab_size))).item():.4f}")
        print()

        # For a proper per-band CE, we need per-token band states.
        # Let's run a custom forward that collects all band states at each token boundary.
        seq_len = tokens.shape[1]
        token_embeddings = model.token_embedding(tokens)  # [1, seq, d_stream]

        current_states = states
        current_global_buffer = global_buffer
        current_predictions = predictions
        current_has_predicted = has_predicted
        current_refractory_levels = refractory_levels

        if model.multi_scale_input:
            ema_alphas = (1.0 / model.module_rates.float()).unsqueeze(-1)
            token_ema = torch.zeros((model.n_modules, 1, model.d_stream), device=device)

        step_offset = torch.zeros((), device=device, dtype=torch.long)
        total_steps = seq_len * model.steps_per_token
        local_step_offsets = torch.arange(total_steps, device=device, dtype=torch.long)
        fires_at = (
            torch.remainder(
                step_offset + local_step_offsets[:, None] + model.module_phases[None, :],
                model.module_rates[None, :],
            ) == 0
        )

        # Collect per-band logits at each token boundary
        band_logits_per_token = [[] for _ in range(n_bands)]  # [band][token_idx] = logits

        for timestep in range(total_steps):
            token_index = timestep // model.steps_per_token
            fires = fires_at[timestep]
            fire_mask = fires[:, None, None]

            if model.refractory:
                current_refractory_levels = current_refractory_levels * model.refractory_decay

            neighbor_sum = model._neighbor_sum(current_global_buffer, current_refractory_levels)
            combined = current_states + neighbor_sum
            combined = combined.clone()
            if model.multi_scale_input:
                if timestep % model.steps_per_token == 0:
                    tok_emb = token_embeddings[:, token_index, :]
                    tok_emb_expanded = tok_emb.unsqueeze(0).expand(model.n_modules, -1, -1)
                    token_ema = (1.0 - ema_alphas.unsqueeze(-1)) * token_ema + ema_alphas.unsqueeze(-1) * tok_emb_expanded
                combined = combined + token_ema
            else:
                combined[model.band0_mask] = combined[model.band0_mask] + token_embeddings[:, token_index, :]
            combined = l2_normalize(combined)

            hidden = F.gelu(model._stacked_linear(combined, model.w1, model.b1))
            output = model._stacked_linear(hidden, model.w2, model.b2)

            current_states = torch.where(fire_mask, output, current_states)
            current_global_buffer = torch.where(fire_mask, output.detach(), current_global_buffer)

            # At last microstep of each token, collect per-band logits
            if timestep % model.steps_per_token == model.steps_per_token - 1:
                for band in range(n_bands):
                    band_start = band * n_cols
                    band_end = (band + 1) * n_cols
                    band_states = current_states[band_start:band_end]  # [n_cols, 1, d_stream]
                    band_mean = band_states.mean(dim=0)  # [1, d_stream]
                    band_logit = model.logits_from_hidden(band_mean)  # [1, vocab]
                    band_logits_per_token[band].append(band_logit)

        # Compute per-band CE
        print("Per-band CE loss (lower = more language-modeling info in that band):")
        print(f"{'band':<6} {'CE loss':<10} {'vs random':<12}")
        random_ce = torch.log(torch.tensor(float(model.vocab_size))).item()
        for band in range(n_bands):
            all_logits = torch.cat(band_logits_per_token[band], dim=0)  # [seq, vocab]
            band_ce = F.cross_entropy(all_logits, targets.reshape(-1)).item()
            improvement = random_ce - band_ce
            print(f"band{band:<2} {band_ce:<10.4f} {improvement:+.4f} nats vs random")


if __name__ == "__main__":
    main()
