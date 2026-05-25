from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional as F

from core.fixed_window_char import set_seed
from core.training import (
    batched_pairs,
    current_git_sha,
    current_git_status_short,
    evaluate_model,
    write_json,
)
from experiments.wide_recurrent_vs_transformer.run import (
    RunConfig,
    TiedDepthTransformer,
    build_corpus,
    dataset_tensors,
    effective_vocab_size,
    replace_config,
)

DEFAULT_DELTA = 0.01
DEFAULT_MATCH_TOLERANCE = 1e-5


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=repo_root
        / "experiments"
        / "wide_recurrent_vs_transformer"
        / "artifacts"
        / "per_token_depth_analysis"
        / "model_state.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "experiments"
        / "wide_recurrent_vs_transformer"
        / "artifacts"
        / "clean_depth_eval",
    )
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def load_run_config(checkpoint_path: Path) -> RunConfig:
    config_path = checkpoint_path.with_name("config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing checkpoint config alongside {checkpoint_path}: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    valid_field_names = {field.name for field in fields(RunConfig)}
    overrides: dict[str, object] = {}
    for key, value in payload.items():
        if key == "learning_rate":
            overrides["learning_rates"] = (value,)
            continue
        if key == "learning_rates":
            overrides["learning_rates"] = tuple(value)
            continue
        if key in valid_field_names:
            overrides[key] = value

    return replace_config(RunConfig(), **overrides)


def encode_corpus(text: str, *, char_to_idx: dict[str, int]) -> Int[Tensor, "tokens"]:
    unknown_index = char_to_idx["<unk>"]
    return torch.tensor([char_to_idx.get(char, unknown_index) for char in text], dtype=torch.long)


def all_next_token_examples(
    encoded_corpus: Int[Tensor, "tokens"],
    *,
    context_size: int,
) -> tuple[Int[Tensor, "examples context"], Int[Tensor, "examples"]]:
    total_examples = encoded_corpus.numel() - context_size
    if total_examples <= 0:
        raise ValueError(
            "Corpus is too short for next-token evaluation. "
            f"Got length {encoded_corpus.numel()} and context_size {context_size}."
        )
    inputs = encoded_corpus.unfold(0, context_size, 1)[:-1].clone()
    targets = encoded_corpus[context_size:].clone()
    return inputs, targets


def evenly_spaced_subset(
    inputs: Int[Tensor, "examples context"],
    targets: Int[Tensor, "examples"],
    *,
    subset_size: int,
) -> tuple[Int[Tensor, "subset context"], Int[Tensor, "subset"], Int[Tensor, "subset"]]:
    total_examples = inputs.shape[0]
    if subset_size > total_examples:
        raise ValueError(
            f"Requested subset_size {subset_size}, but only {total_examples} examples are available."
        )
    subset_indices = torch.linspace(0, total_examples - 1, steps=subset_size, dtype=torch.float64)
    subset_indices = subset_indices.round().to(dtype=torch.long)
    return inputs[subset_indices], targets[subset_indices], subset_indices


def per_example_losses_by_depth(
    *,
    model: TiedDepthTransformer,
    inputs: Int[Tensor, "examples context"],
    targets: Int[Tensor, "examples"],
    batch_size: int,
) -> Float[Tensor, "examples depth"]:
    was_training = model.training
    model.eval()
    depth_batches: list[Float[Tensor, "batch depth"]] = []
    with torch.inference_mode():
        for batch_inputs, batch_targets in batched_pairs(inputs, targets, batch_size=batch_size):
            logits_by_depth = model.depth_logits(batch_inputs)
            batch_losses = [
                F.cross_entropy(logits, batch_targets, reduction="none") for logits in logits_by_depth
            ]
            depth_batches.append(torch.stack(batch_losses, dim=1).cpu())
    if was_training:
        model.train()
    return torch.cat(depth_batches, dim=0)


def histogram(depths: Int[Tensor, "examples"], *, max_depth: int) -> dict[str, object]:
    counts = torch.bincount(depths, minlength=max_depth + 1)[1:]
    total = int(depths.numel())
    return {
        "counts": {f"depth_{depth}": int(count.item()) for depth, count in enumerate(counts, start=1)},
        "fractions": {
            f"depth_{depth}": round(count.item() / total, 6)
            for depth, count in enumerate(counts, start=1)
        },
    }


def summarize_split(
    *,
    losses_by_depth: Float[Tensor, "examples depth"],
    delta: float,
) -> dict[str, object]:
    max_depth = losses_by_depth.shape[1]
    mean_losses = losses_by_depth.mean(dim=0)
    oracle_best_losses, oracle_depths_zero_based = losses_by_depth.min(dim=1)
    oracle_depths = oracle_depths_zero_based + 1
    final_losses = losses_by_depth[:, -1]
    within_delta = losses_by_depth <= (final_losses.unsqueeze(1) + delta)
    no_regret_depths = within_delta.to(dtype=torch.int64).argmax(dim=1) + 1

    return {
        "num_examples": int(losses_by_depth.shape[0]),
        "mean_loss_by_depth": {
            f"depth_{depth}": round(mean_loss.item(), 6)
            for depth, mean_loss in enumerate(mean_losses, start=1)
        },
        "oracle_best_loss": round(oracle_best_losses.mean().item(), 6),
        "oracle_depth_histogram": histogram(oracle_depths, max_depth=max_depth),
        "no_regret_delta": delta,
        "no_regret_shallowest_depth": {
            "mean": round(no_regret_depths.float().mean().item(), 6),
            "histogram": histogram(no_regret_depths, max_depth=max_depth),
        },
        "oracle_speedup": round(max_depth / no_regret_depths.float().mean().item(), 6),
        "tokens_harmed_by_extra_depth_fraction": round(
            (oracle_best_losses < final_losses).float().mean().item(),
            6,
        ),
    }


def loss_match_summary(*, measured_loss: float, standard_loss: float) -> dict[str, object]:
    absolute_difference = abs(measured_loss - standard_loss)
    return {
        "measured_loss": round(measured_loss, 6),
        "standard_loss": round(standard_loss, 6),
        "absolute_difference": round(absolute_difference, 6),
        "matches_within_tolerance": absolute_difference <= DEFAULT_MATCH_TOLERANCE,
    }


def main() -> int:
    args = parse_args()
    repo_root: Path = args.repo_root
    checkpoint_path: Path = args.checkpoint_path
    output_dir: Path = args.output_dir
    device = torch.device(args.device)

    config = load_run_config(checkpoint_path)
    set_seed(config.seed)

    prepared_dir = repo_root / "experiments" / "wide_recurrent_vs_transformer" / "prepared.ignore"
    corpus, raw_text, train_text = build_corpus(
        repo_root=repo_root,
        config=config,
        prepared_dir=prepared_dir,
    )
    vocab_size = effective_vocab_size(corpus)

    model = TiedDepthTransformer(vocab_size=vocab_size, config=config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    train_inputs_all, train_targets_all = dataset_tensors(corpus.train_dataset)
    val_text = (prepared_dir / "tinyshakespeare_val.txt").read_text(encoding="utf-8")
    val_encoded = encode_corpus(val_text, char_to_idx=corpus.char_to_idx)
    val_inputs, val_targets = all_next_token_examples(val_encoded, context_size=config.context_size)

    train_inputs, train_targets, train_subset_indices = evenly_spaced_subset(
        train_inputs_all,
        train_targets_all,
        subset_size=val_inputs.shape[0],
    )

    val_inputs = val_inputs.to(device)
    val_targets = val_targets.to(device)
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)

    val_losses = per_example_losses_by_depth(
        model=model,
        inputs=val_inputs,
        targets=val_targets,
        batch_size=config.eval_batch_size,
    )
    train_losses = per_example_losses_by_depth(
        model=model,
        inputs=train_inputs,
        targets=train_targets,
        batch_size=config.eval_batch_size,
    )

    val_standard_eval = evaluate_model(
        model,
        val_inputs,
        val_targets,
        batch_size=config.eval_batch_size,
    )
    train_standard_eval = evaluate_model(
        model,
        train_inputs,
        train_targets,
        batch_size=config.eval_batch_size,
    )

    val_depth_8_loss = val_losses[:, -1].mean().item()
    train_depth_8_loss = train_losses[:, -1].mean().item()

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "val_losses_by_depth": val_losses,
            "train_subset_losses_by_depth": train_losses,
            "train_subset_indices": train_subset_indices,
        },
        output_dir / "per_example_losses.pt",
    )

    write_json(
        output_dir / "config.json",
        {
            "delta": args.delta,
            "device": str(device),
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(output_dir),
            "train_subset_strategy": "evenly spaced deterministic subset across full train corpus",
            "model_config": asdict(config),
        },
    )
    git_status_short = current_git_status_short()
    write_json(
        output_dir / "environment.json",
        {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": git_status_short == [],
            "git_status_short": git_status_short,
            "torch_version": torch.__version__,
            "device": str(device),
        },
    )
    write_json(
        output_dir / "corpus_summary.json",
        {
            "source_characters": len(raw_text),
            "train_characters": len(train_text),
            "validation_characters": len(val_text),
            "context_size": config.context_size,
            "full_validation_examples": int(val_inputs.shape[0]),
            "train_examples_total": int(train_inputs_all.shape[0]),
            "train_subset_examples": int(train_inputs.shape[0]),
        },
    )
    write_json(
        output_dir / "summary.json",
        {
            "validation": {
                **summarize_split(losses_by_depth=val_losses, delta=args.delta),
                "standard_depth_8_eval": {
                    "loss": round(val_standard_eval["loss"], 6),
                    "accuracy": round(val_standard_eval["accuracy"], 6),
                },
                "depth_8_loss_match": loss_match_summary(
                    measured_loss=val_depth_8_loss,
                    standard_loss=val_standard_eval["loss"],
                ),
            },
            "train_subset": {
                **summarize_split(losses_by_depth=train_losses, delta=args.delta),
                "standard_depth_8_eval": {
                    "loss": round(train_standard_eval["loss"], 6),
                    "accuracy": round(train_standard_eval["accuracy"], 6),
                },
                "depth_8_loss_match": loss_match_summary(
                    measured_loss=train_depth_8_loss,
                    standard_loss=train_standard_eval["loss"],
                ),
            },
        },
    )

    print((output_dir / "summary.json").read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
