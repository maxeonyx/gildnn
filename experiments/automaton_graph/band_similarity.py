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


TOKENS_TO_ANALYZE = 2048
CHUNK_SIZE = 128
BATCH_SIZE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Band representation cosine-similarity diagnostic.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def load_tinyshakespeare() -> tuple[Tensor, int]:
    repo_root = Path(__file__).resolve().parents[2]
    text_path = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    raw_text = text_path.read_text(encoding="utf-8")
    vocab = sorted(set(raw_text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    encoded = torch.tensor([stoi[ch] for ch in raw_text], dtype=torch.long)
    return encoded, len(vocab)


def infer_model_kwargs(checkpoint: dict[str, object]) -> dict[str, int | bool]:
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing model_state_dict.")

    token_embedding = state_dict.get("token_embedding.weight")
    w1 = state_dict.get("w1")
    if not isinstance(token_embedding, Tensor) or not isinstance(w1, Tensor):
        raise ValueError("Checkpoint model_state_dict is missing token_embedding.weight or w1.")

    n_modules, d_stream, d_hidden = w1.shape
    kwargs: dict[str, int | bool] = {
        "vocab_size": int(token_embedding.shape[0]),
        "d_stream": int(d_stream),
        "d_hidden": int(d_hidden),
        "n_bands": 8,
        "n_cols": int(n_modules // 8),
    }

    train_args = checkpoint.get("args")
    if isinstance(train_args, dict):
        for key in (
            "multi_scale_input",
            "temporal_targets",
            "cross_band_negatives",
            "attention_readout",
            "detach_readout",
            "per_band_ce",
        ):
            if train_args.get(key):
                kwargs[key] = True

    return kwargs


def make_windows(encoded_text: Tensor) -> Tensor:
    required_tokens = TOKENS_TO_ANALYZE + 1
    if encoded_text.numel() < required_tokens:
        raise ValueError(
            f"TinyShakespeare corpus has only {encoded_text.numel()} tokens, need at least {required_tokens}."
        )

    starts = torch.arange(0, TOKENS_TO_ANALYZE, CHUNK_SIZE, dtype=torch.long)
    windows = [encoded_text[start : start + CHUNK_SIZE + 1] for start in starts.tolist()]
    return torch.stack(windows)


def collect_band_means(model: GraphCellularAutomaton, windows: Tensor) -> Tensor:
    all_band_means: list[Tensor] = []

    with torch.inference_mode():
        for batch_start in range(0, windows.shape[0], BATCH_SIZE):
            batch_windows = windows[batch_start : batch_start + BATCH_SIZE]
            inputs = batch_windows[:, :-1]
            states, global_buffer, predictions, has_predicted, refractory_levels = model.initial_recurrent_state(
                inputs.shape[0],
                device=torch.device("cpu"),
            )
            _, _, final_states, *_ = model.forward_chunk(
                inputs,
                states,
                global_buffer,
                predictions,
                has_predicted,
                refractory_levels,
                global_step_offset=0,
            )

            batch_band_means = []
            for band_idx in range(model.n_bands):
                band_start = band_idx * model.n_cols
                band_end = (band_idx + 1) * model.n_cols
                band_mean = final_states[band_start:band_end].mean(dim=0)
                batch_band_means.append(band_mean)
            all_band_means.append(torch.stack(batch_band_means, dim=1))

    return torch.cat(all_band_means, dim=0)


def format_similarity_matrix(similarity: Tensor) -> str:
    header = "       " + " ".join(f"b{band:>6}" for band in range(similarity.shape[0]))
    rows = [header]
    for band_idx, row in enumerate(similarity):
        values = " ".join(f"{value.item():7.4f}" for value in row)
        rows.append(f"b{band_idx:<2}    {values}")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    encoded_text, _ = load_tinyshakespeare()
    windows = make_windows(encoded_text)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GraphCellularAutomaton(**infer_model_kwargs(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    band_means = collect_band_means(model, windows)
    mean_band_representations = band_means.mean(dim=0)
    normalized_band_representations = F.normalize(mean_band_representations.float(), dim=-1, eps=1e-6)
    similarity = normalized_band_representations @ normalized_band_representations.T
    mean_vector_norms = mean_band_representations.float().norm(dim=-1)
    average_sample_norms = band_means.float().norm(dim=-1).mean(dim=0)

    print(
        f"checkpoint={args.checkpoint} samples={band_means.shape[0]} chunks={windows.shape[0]} chunk_size={CHUNK_SIZE} tokens={TOKENS_TO_ANALYZE}"
    )
    print()
    print("Pairwise cosine similarity between mean band representations")
    print(format_similarity_matrix(similarity))
    print()
    print("Per-band norms")
    for band_idx in range(model.n_bands):
        print(
            f"band {band_idx}: mean-vector={mean_vector_norms[band_idx].item():.4f} avg-sample={average_sample_norms[band_idx].item():.4f}"
        )


if __name__ == "__main__":
    main()
