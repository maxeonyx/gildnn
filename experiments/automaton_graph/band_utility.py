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

from core.automaton_graph import GraphCellularAutomaton


ALPHAS = tuple(step / 10 for step in range(11))


@dataclass
class LossAccumulator:
    total_loss: float = 0.0
    total_count: int = 0

    def add(self, logits: Tensor, targets: Tensor) -> None:
        vocab_size = logits.shape[-1]
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="sum")
        self.total_loss += float(loss.item())
        self.total_count += int(targets.numel())

    @property
    def mean_ce(self) -> float:
        if self.total_count <= 0:
            raise ValueError("Cannot report mean CE with zero targets.")
        return self.total_loss / self.total_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parameter-free band utility diagnostic.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
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
    state_dict = checkpoint.get("model_state_dict")
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
        "per_band_ce": True,
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

    kwargs["per_band_ce"] = True
    return kwargs


def iter_windows(encoded_text: Tensor, *, n_chunks: int, batch_size: int, chunk_size: int):
    required = chunk_size + 1
    if encoded_text.numel() < required:
        raise ValueError(
            f"Encoded corpus has only {encoded_text.numel()} tokens, need at least {required}."
        )

    wrap_mod = encoded_text.numel() - required + 1
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size * batch_size
        windows = []
        for batch_idx in range(batch_size):
            offset = start + batch_idx * chunk_size
            if offset + required > encoded_text.numel():
                offset = offset % wrap_mod
            windows.append(encoded_text[offset : offset + required])
        yield torch.stack(windows)


def update_from_per_band_logits(
    *,
    accumulators: dict[str, LossAccumulator],
    alpha_accumulators: dict[float, LossAccumulator],
    per_band_logits: Tensor,
    targets: Tensor,
) -> None:
    band0_logits = per_band_logits[:, 0]
    other_bands_logits = per_band_logits[:, 1:].mean(dim=1)

    accumulators["band 0 only"].add(band0_logits, targets)
    accumulators["all bands equal weight"].add(per_band_logits.mean(dim=1), targets)
    accumulators["bands 0-3 only"].add(per_band_logits[:, :4].mean(dim=1), targets)
    accumulators["bands 4-7 only"].add(per_band_logits[:, 4:].mean(dim=1), targets)

    for alpha, accumulator in alpha_accumulators.items():
        combined_logits = (alpha * band0_logits) + ((1.0 - alpha) * other_bands_logits)
        accumulator.add(combined_logits, targets)


def update_from_final_states(
    *,
    model: GraphCellularAutomaton,
    accumulators: dict[str, LossAccumulator],
    alpha_accumulators: dict[float, LossAccumulator],
    final_states: Tensor,
    targets: Tensor,
) -> None:
    band_means = final_states.reshape(model.n_bands, model.n_cols, final_states.shape[1], final_states.shape[2]).mean(dim=1)

    band0_hidden = band_means[0]
    other_bands_hidden = band_means[1:].mean(dim=0)

    accumulators["band 0 only"].add(model.logits_from_hidden(band0_hidden), targets)
    accumulators["all bands equal weight"].add(model.logits_from_hidden(band_means.mean(dim=0)), targets)
    accumulators["bands 0-3 only"].add(model.logits_from_hidden(band_means[:4].mean(dim=0)), targets)
    accumulators["bands 4-7 only"].add(model.logits_from_hidden(band_means[4:].mean(dim=0)), targets)

    for alpha, accumulator in alpha_accumulators.items():
        combined_hidden = (alpha * band0_hidden) + ((1.0 - alpha) * other_bands_hidden)
        accumulator.add(model.logits_from_hidden(combined_hidden), targets)


def main() -> None:
    args = parse_args()
    if args.tokens <= 0:
        raise ValueError(f"--tokens must be positive, got {args.tokens}.")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}.")
    if args.chunk_size <= 0:
        raise ValueError(f"--chunk-size must be positive, got {args.chunk_size}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoded_text, _ = load_tinyshakespeare()

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GraphCellularAutomaton(**infer_model_kwargs(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    n_chunks = max(1, args.tokens // args.chunk_size)
    accumulators = {
        "band 0 only": LossAccumulator(),
        "all bands equal weight": LossAccumulator(),
        "bands 0-3 only": LossAccumulator(),
        "bands 4-7 only": LossAccumulator(),
    }
    alpha_accumulators = {alpha: LossAccumulator() for alpha in ALPHAS}
    used_per_band_logits = False

    with torch.inference_mode():
        for windows in iter_windows(
            encoded_text,
            n_chunks=n_chunks,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        ):
            windows = windows.to(device)
            inputs = windows[:, :-1]
            targets = windows[:, 1:]

            states, global_buffer, predictions, has_predicted, refractory_levels = model.initial_recurrent_state(
                inputs.shape[0],
                device=device,
            )
            _logits, per_band_logits, final_states, *_ = model.forward_chunk(
                inputs,
                states,
                global_buffer,
                predictions,
                has_predicted,
                refractory_levels,
                global_step_offset=torch.zeros((), device=device, dtype=torch.long),
            )

            if per_band_logits is not None:
                used_per_band_logits = True
                update_from_per_band_logits(
                    accumulators=accumulators,
                    alpha_accumulators=alpha_accumulators,
                    per_band_logits=per_band_logits,
                    targets=targets,
                )
            else:
                update_from_final_states(
                    model=model,
                    accumulators=accumulators,
                    alpha_accumulators=alpha_accumulators,
                    final_states=final_states,
                    targets=targets[:, -1],
                )

    best_alpha, best_alpha_accumulator = min(alpha_accumulators.items(), key=lambda item: item[1].mean_ce)
    mode = "per-token band-logit ensemble" if used_per_band_logits else "final-state fallback"
    sample_count = next(iter(accumulators.values())).total_count

    print(f"checkpoint={args.checkpoint}")
    print(f"step={checkpoint.get('step', '?')} device={device.type} mode={mode}")
    print(
        f"tokens={args.tokens} batch_size={args.batch_size} chunk_size={args.chunk_size} "
        f"chunks={n_chunks} evaluated_targets={sample_count}"
    )
    print()
    print(f"{'strategy':<28} {'ce':>10}")
    print(f"{'-' * 28} {'-' * 10}")
    for label, accumulator in accumulators.items():
        print(f"{label:<28} {accumulator.mean_ce:>10.4f}")
    print(f"{'optimal α blend':<28} {best_alpha_accumulator.mean_ce:>10.4f}  α={best_alpha:.1f}")


if __name__ == "__main__":
    main()
