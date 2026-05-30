from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import Tensor
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.automaton_graph import GraphCellularAutomaton, l2_normalize


LAGS = (1, 2, 4, 8, 16, 32, 64)


@dataclass(frozen=True)
class TinyShakespeareData:
    encoded_text: Tensor
    vocab_size: int


@dataclass(frozen=True)
class BandEvents:
    features: Tensor
    token_indices: Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lag-spectrum linear probe for the graph cellular automaton.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--checkpoint", type=Path, help="Checkpoint to probe.")
    source_group.add_argument("--random", action="store_true", help="Use a fresh random model as a control.")
    parser.add_argument("--tokens", type=int, default=2048, help="Number of TinyShakespeare tokens to analyze.")
    parser.add_argument(
        "--probe-vocab-size",
        type=int,
        default=128,
        help="Number of classes in the linear probe.",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="Override TinyShakespeare text path.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.1,
        help="Runtime noise_std for random models or checkpoints without stored config.",
    )
    parser.add_argument(
        "--refractory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Runtime refractory setting for random models or checkpoints without stored config.",
    )
    parser.add_argument(
        "--refractory-threshold",
        type=float,
        default=1.0,
        help="Runtime refractory threshold.",
    )
    parser.add_argument(
        "--refractory-decay",
        type=float,
        default=0.8,
        help="Runtime refractory decay.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1.0,
        help="L2 regularization strength for the linear probe.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.tokens <= max(LAGS):
        raise ValueError(f"--tokens must be greater than {max(LAGS)}, got {args.tokens}.")
    if args.probe_vocab_size <= 0:
        raise ValueError(f"--probe-vocab-size must be positive, got {args.probe_vocab_size}.")
    if args.noise_std < 0.0:
        raise ValueError(f"--noise-std must be non-negative, got {args.noise_std}.")
    if args.ridge < 0.0:
        raise ValueError(f"--ridge must be non-negative, got {args.ridge}.")
    return args


def load_tinyshakespeare(repo_root: Path, *, text_file: Path | None) -> TinyShakespeareData:
    resolved_text_file = text_file or (repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt")
    raw_text = resolved_text_file.read_text(encoding="utf-8")
    vocab = sorted(set(raw_text))
    stoi = {char: index for index, char in enumerate(vocab)}
    encoded_text = torch.tensor([stoi[char] for char in raw_text], dtype=torch.long)
    return TinyShakespeareData(encoded_text=encoded_text, vocab_size=len(vocab))


def infer_model_kwargs(checkpoint: dict[str, object]) -> dict[str, int | float | bool]:
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing model_state_dict.")
    token_embedding = state_dict.get("token_embedding.weight")
    w1 = state_dict.get("w1")
    if not isinstance(token_embedding, Tensor) or not isinstance(w1, Tensor):
        raise ValueError("Checkpoint model_state_dict is missing token_embedding.weight or w1.")
    n_modules, d_stream, d_hidden = w1.shape
    return {
        "vocab_size": int(token_embedding.shape[0]),
        "d_stream": int(d_stream),
        "d_hidden": int(d_hidden),
        "n_bands": 8,
        "n_cols": int(n_modules // 8),
    }


def build_model(args: argparse.Namespace, *, data_vocab_size: int) -> tuple[GraphCellularAutomaton, str]:
    device = torch.device("cpu")
    if args.random:
        model = GraphCellularAutomaton(
            vocab_size=max(data_vocab_size, args.probe_vocab_size),
            noise_std=args.noise_std,
            refractory=args.refractory,
            refractory_threshold=args.refractory_threshold,
            refractory_decay=args.refractory_decay,
        ).to(device)
        return model, "random model"

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_kwargs = infer_model_kwargs(checkpoint)
    model = GraphCellularAutomaton(
        **model_kwargs,
        noise_std=args.noise_std,
        refractory=args.refractory,
        refractory_threshold=args.refractory_threshold,
        refractory_decay=args.refractory_decay,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    step = checkpoint.get("step")
    ce_loss = checkpoint.get("ce_loss")
    summary_bits = [f"checkpoint {args.checkpoint}"]
    if step is not None:
        summary_bits.append(f"step={step}")
    if ce_loss is not None:
        summary_bits.append(f"ce_loss={float(ce_loss):.4f}")
    return model, ", ".join(summary_bits)


def collect_band_events(model: GraphCellularAutomaton, tokens: Tensor) -> list[BandEvents]:
    if tokens.ndim != 1:
        raise ValueError(f"Expected 1D token tensor, got shape {tuple(tokens.shape)}.")

    device = torch.device("cpu")
    batch_tokens = tokens.to(device=device, dtype=torch.long).unsqueeze(0)
    band_masks = [(model.module_rows == band) for band in range(model.n_bands)]
    feature_lists: list[list[Tensor]] = [[] for _ in range(model.n_bands)]
    token_index_lists: list[list[int]] = [[] for _ in range(model.n_bands)]

    with torch.inference_mode():
        model.eval()
        states, global_buffer, predictions, has_predicted, refractory_levels = model.initial_recurrent_state(
            1,
            device=device,
        )
        token_embeddings = model.token_embedding(batch_tokens)
        total_steps = batch_tokens.shape[1] * model.steps_per_token
        local_step_offsets = torch.arange(total_steps, device=device, dtype=torch.long)
        fires_at = (
            torch.remainder(
                local_step_offsets[:, None] + model.module_phases[None, :],
                model.module_rates[None, :],
            )
            == 0
        )

        for timestep in range(total_steps):
            token_index = timestep // model.steps_per_token
            fires = fires_at[timestep]
            fire_mask = fires[:, None, None]

            if model.refractory:
                refractory_levels = refractory_levels * model.refractory_decay

            neighbor_sum = model._neighbor_sum(global_buffer, refractory_levels)
            combined = states + neighbor_sum
            combined = combined.clone()
            combined[model.band0_mask] = combined[model.band0_mask] + token_embeddings[:, token_index, :]
            combined = l2_normalize(combined)

            hidden = F.gelu(model._stacked_linear(combined, model.w1, model.b1))
            output = model._stacked_linear(hidden, model.w2, model.b2)
            new_predictions = model._stacked_linear(output, model.pred_w, model.pred_b)

            for band, band_mask in enumerate(band_masks):
                band_fires = fires & band_mask
                if torch.any(band_fires):
                    feature_lists[band].append(output[band_fires, 0].mean(dim=0).cpu())
                    token_index_lists[band].append(token_index)

            states = torch.where(fire_mask, output, states)
            predictions = torch.where(fire_mask, new_predictions, predictions)
            global_buffer = torch.where(fire_mask, output.detach(), global_buffer)
            has_predicted = has_predicted | fires

            if model.refractory:
                output_norms = output.float().norm(dim=-1).amax(dim=1)
                fired_strongly = fires & (output_norms > model.refractory_threshold)
                refractory_levels = torch.where(
                    fired_strongly,
                    torch.ones_like(refractory_levels),
                    refractory_levels,
                )

    band_events: list[BandEvents] = []
    for band in range(model.n_bands):
        if len(feature_lists[band]) == 0:
            raise ValueError(f"Band {band} produced no fire events.")
        band_events.append(
            BandEvents(
                features=torch.stack(feature_lists[band]),
                token_indices=torch.tensor(token_index_lists[band], dtype=torch.long),
            )
        )
    return band_events


def fit_ridge_probe(
    features: Tensor,
    labels: Tensor,
    *,
    num_classes: int,
    ridge: float,
) -> float:
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {tuple(features.shape)}.")
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape {tuple(labels.shape)}.")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Feature and label counts do not match.")
    if features.shape[0] < 4:
        raise ValueError(f"Need at least 4 samples for a probe, got {features.shape[0]}.")

    sample_count = features.shape[0]
    train_count = max(2, int(sample_count * 0.8))
    train_count = min(train_count, sample_count - 1)
    eval_count = sample_count - train_count
    if eval_count <= 0:
        raise ValueError(f"Need at least one evaluation sample, got {sample_count} total samples.")

    train_x = features[:train_count].float()
    eval_x = features[train_count:].float()
    train_y = labels[:train_count]
    eval_y = labels[train_count:]

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    eval_x = (eval_x - mean) / std

    train_x = torch.cat((train_x, torch.ones((train_count, 1), dtype=train_x.dtype)), dim=1)
    eval_x = torch.cat((eval_x, torch.ones((eval_count, 1), dtype=eval_x.dtype)), dim=1)
    train_targets = F.one_hot(train_y, num_classes=num_classes).to(dtype=train_x.dtype)

    gram = train_x.T @ train_x
    regularizer = torch.eye(gram.shape[0], dtype=train_x.dtype) * ridge
    regularizer[-1, -1] = 0.0
    weights = torch.linalg.solve(gram + regularizer, train_x.T @ train_targets)
    logits = eval_x @ weights
    predictions = logits.argmax(dim=1)
    return float((predictions == eval_y).float().mean().item())


def compute_probe_matrix(
    band_events: list[BandEvents],
    tokens: Tensor,
    *,
    lags: tuple[int, ...],
    probe_vocab_size: int,
    ridge: float,
) -> list[list[float]]:
    results: list[list[float]] = []
    for events in band_events:
        band_results: list[float] = []
        for lag in lags:
            valid = events.token_indices >= lag
            lag_features = events.features[valid]
            lag_labels = tokens[events.token_indices[valid] - lag]
            band_results.append(
                fit_ridge_probe(
                    lag_features,
                    lag_labels,
                    num_classes=probe_vocab_size,
                    ridge=ridge,
                )
            )
        results.append(band_results)
    return results


def print_results_table(
    results: list[list[float]],
    *,
    model_label: str,
    token_count: int,
    lags: tuple[int, ...],
) -> None:
    print(f"Lag-spectrum probe results ({model_label}, {token_count} tokens)")
    print()
    header = "       " + " ".join(f"lag={lag:>3}" for lag in lags)
    print(header)
    for band, band_results in enumerate(results):
        row = " ".join(f"{100.0 * value:5.1f}%" for value in band_results)
        print(f"band{band}: {row}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[2]
    data = load_tinyshakespeare(repo_root, text_file=args.text_file)
    if data.encoded_text.numel() < args.tokens:
        raise ValueError(
            f"TinyShakespeare corpus has only {data.encoded_text.numel()} tokens, cannot take {args.tokens}."
        )

    tokens = data.encoded_text[: args.tokens]
    model, model_label = build_model(args, data_vocab_size=data.vocab_size)
    band_events = collect_band_events(model, tokens)
    results = compute_probe_matrix(
        band_events,
        tokens,
        lags=LAGS,
        probe_vocab_size=args.probe_vocab_size,
        ridge=args.ridge,
    )
    print_results_table(results, model_label=model_label, token_count=args.tokens, lags=LAGS)


if __name__ == "__main__":
    main()
