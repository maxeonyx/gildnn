from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure repo root is importable regardless of how this script is launched
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from jaxtyping import Int
from torch import Tensor
from torch.nn import functional as F

from core.fixed_window_char import set_seed
from core.training import batched_pairs, current_git_sha, current_git_status_short, write_json
from experiments.wide_recurrent_vs_transformer.run import (
    RunConfig,
    TiedDepthTransformer,
    build_corpus,
    corpus_summary,
    effective_vocab_size,
    evaluate_model,
    replace_config,
    resolve_device,
    train_full_run,
)

DEFAULT_EPSILONS = (0.1, 0.05)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "experiments"
        / "wide_recurrent_vs_transformer"
        / "artifacts"
        / "per_token_depth_analysis",
    )
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPSILONS))
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def analysis_config() -> RunConfig:
    return replace_config(RunConfig(), depth=8, d_model=72, seed=42, learning_rates=(0.003,))


def all_position_validation_windows(
    encoded_validation: Int[Tensor, "tokens"],
    *,
    context_size: int,
) -> tuple[Int[Tensor, "examples context"], Int[Tensor, "examples context"]]:
    total_windows = encoded_validation.numel() - context_size
    if total_windows <= 0:
        raise ValueError(
            "Validation corpus is too short for per-token depth analysis. "
            f"Got length {encoded_validation.numel()} and context_size {context_size}."
        )
    inputs = encoded_validation.unfold(0, context_size, 1)[:total_windows].clone()
    targets = encoded_validation[1:].unfold(0, context_size, 1).clone()
    return inputs, targets


def per_token_losses_by_depth(
    *,
    model: TiedDepthTransformer,
    inputs: Int[Tensor, "examples context"],
    targets: Int[Tensor, "examples context"],
    batch_size: int,
) -> list[Tensor]:
    was_training = model.training
    model.eval()
    losses_by_depth: list[list[Tensor]] = [[] for _ in range(model.config.depth)]
    with torch.inference_mode():
        for batch_inputs, batch_targets in batched_pairs(inputs, targets, batch_size=batch_size):
            logits_by_depth = model.depth_logits_all_positions(batch_inputs)
            flat_targets = batch_targets.reshape(-1)
            for depth_index, depth_logits in enumerate(logits_by_depth):
                flat_logits = depth_logits.reshape(-1, depth_logits.shape[-1])
                flat_losses = F.cross_entropy(flat_logits, flat_targets, reduction="none")
                losses_by_depth[depth_index].append(flat_losses.reshape(batch_targets.shape))
    if was_training:
        model.train()
    return [torch.cat(depth_losses, dim=0) for depth_losses in losses_by_depth]


def tensor_stats(values: Tensor) -> dict[str, float]:
    return {
        "mean": round(values.mean().item(), 6),
        "std": round(values.std(unbiased=False).item(), 6),
        "q25": round(torch.quantile(values, 0.25).item(), 6),
        "median": round(torch.quantile(values, 0.5).item(), 6),
        "q75": round(torch.quantile(values, 0.75).item(), 6),
        "min": round(values.min().item(), 6),
        "max": round(values.max().item(), 6),
    }


def fraction_done_by_depth(
    *,
    losses_by_depth: list[Tensor],
    epsilon: float,
) -> dict[str, float]:
    final_losses = losses_by_depth[-1]
    fractions: dict[str, float] = {}
    total_tokens = final_losses.numel()
    for depth_index, depth_losses in enumerate(losses_by_depth, start=1):
        done_fraction = ((depth_losses - final_losses) <= epsilon).sum().item() / total_tokens
        fractions[f"depth_{depth_index}"] = round(done_fraction, 6)
    return fractions


def earliest_depth_distribution(
    *,
    losses_by_depth: list[Tensor],
    epsilon: float,
) -> dict[str, object]:
    final_losses = losses_by_depth[-1]
    stacked = torch.stack(losses_by_depth, dim=0)
    within_epsilon = (stacked - final_losses.unsqueeze(0)) <= epsilon
    earliest_depth = within_epsilon.float().argmax(dim=0) + 1
    total_tokens = earliest_depth.numel()
    counts = torch.bincount(earliest_depth.reshape(-1), minlength=len(losses_by_depth) + 1)[1:]
    fractions = {
        f"depth_{depth_index}": round(count.item() / total_tokens, 6)
        for depth_index, count in enumerate(counts, start=1)
    }
    return {
        "epsilon": epsilon,
        "fractions": fractions,
        "mean": round(earliest_depth.float().mean().item(), 6),
        "std": round(earliest_depth.float().std(unbiased=False).item(), 6),
    }


def marginal_improvement_summary(losses_by_depth: list[Tensor]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for depth_index in range(len(losses_by_depth) - 1):
        improvements = (losses_by_depth[depth_index] - losses_by_depth[depth_index + 1]).reshape(-1)
        positive_fraction = (improvements > 0).float().mean().item()
        negative_fraction = (improvements < 0).float().mean().item()
        summaries.append(
            {
                "transition": f"depth_{depth_index + 1}_to_{depth_index + 2}",
                "positive_fraction": round(positive_fraction, 6),
                "negative_fraction": round(negative_fraction, 6),
                **tensor_stats(improvements),
            }
        )
    return summaries


def heterogeneity_summary(losses_by_depth: list[Tensor]) -> dict[str, object]:
    final_losses = losses_by_depth[-1].reshape(-1)
    total_improvement = (losses_by_depth[0] - losses_by_depth[-1]).reshape(-1)
    return {
        "total_depth_1_to_8_improvement": tensor_stats(total_improvement),
        "final_depth_8_loss": tensor_stats(final_losses),
        "correlation_total_improvement_vs_final_loss": round(
            torch.corrcoef(torch.stack((total_improvement, final_losses)))[0, 1].item(),
            6,
        ),
        "tokens_with_negative_total_improvement_fraction": round(
            (total_improvement < 0).float().mean().item(),
            6,
        ),
    }


def validation_char_tensor(val_path: Path, *, char_to_idx: dict[str, int]) -> Int[Tensor, "tokens"]:
    validation_text = val_path.read_text(encoding="utf-8")
    unknown_index = char_to_idx["<unk>"]
    return torch.tensor([char_to_idx.get(char, unknown_index) for char in validation_text], dtype=torch.long)


def main() -> int:
    args = parse_args()
    repo_root: Path = args.repo_root
    config = analysis_config()
    device = resolve_device(args.device)
    output_dir: Path = args.output_dir
    prepared_dir = repo_root / "experiments" / "wide_recurrent_vs_transformer" / "prepared.ignore"

    set_seed(config.seed)
    corpus, raw_text, train_text = build_corpus(
        repo_root=repo_root,
        config=config,
        prepared_dir=prepared_dir,
    )
    vocab_size = effective_vocab_size(corpus)
    train_dataset = corpus.train_dataset
    if not hasattr(train_dataset, "encoded_corpus"):
        raise TypeError("Expected RandomWindowCharDataset with encoded_corpus for training input construction.")
    train_encoded = train_dataset.encoded_corpus
    train_inputs = train_encoded.unfold(0, config.context_size, 1)[:-1].clone().to(device)
    train_targets = train_encoded[config.context_size:].clone().to(device)
    val_inputs = corpus.val_inputs.to(device)
    val_targets = corpus.val_targets.to(device)
    prompt = train_text[: config.context_size]

    run_result = train_full_run(
        corpus=corpus,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        prompt=prompt,
        learning_rate=config.learning_rates[0],
        device=device,
        config=config,
        vocab_size=vocab_size,
        checkpoint_path=output_dir / "model_state.pt",
    )

    model = TiedDepthTransformer(vocab_size=vocab_size, config=config).to(device)
    model.load_state_dict(torch.load(output_dir / "model_state.pt", map_location=device))

    train_path = prepared_dir / "tinyshakespeare_train.txt"
    val_path = prepared_dir / "tinyshakespeare_val.txt"
    val_encoded = validation_char_tensor(val_path, char_to_idx=corpus.char_to_idx)
    all_val_inputs, all_val_targets = all_position_validation_windows(val_encoded, context_size=config.context_size)
    all_val_inputs = all_val_inputs.to(device)
    all_val_targets = all_val_targets.to(device)

    losses_by_depth = per_token_losses_by_depth(
        model=model,
        inputs=all_val_inputs,
        targets=all_val_targets,
        batch_size=config.eval_batch_size,
    )
    marginal_improvements_by_depth = [
        losses_by_depth[depth_index] - losses_by_depth[depth_index + 1]
        for depth_index in range(len(losses_by_depth) - 1)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "losses_by_depth": [depth_losses.cpu() for depth_losses in losses_by_depth],
            "marginal_improvements_by_depth": [
                improvements.cpu() for improvements in marginal_improvements_by_depth
            ],
        },
        output_dir / "per_token_tensors.pt",
    )

    epsilon_summaries = []
    for epsilon in args.epsilons:
        epsilon_summaries.append(
            {
                "epsilon": epsilon,
                "fraction_done_by_depth": fraction_done_by_depth(
                    losses_by_depth=losses_by_depth,
                    epsilon=epsilon,
                ),
                "earliest_depth_distribution": earliest_depth_distribution(
                    losses_by_depth=losses_by_depth,
                    epsilon=epsilon,
                ),
            }
        )

    marginal_summaries = marginal_improvement_summary(losses_by_depth)
    heterogeneity = heterogeneity_summary(losses_by_depth)
    depth_loss_summary = {
        f"depth_{depth_index}": round(depth_losses.mean().item(), 6)
        for depth_index, depth_losses in enumerate(losses_by_depth, start=1)
    }
    fixed_window_eval = evaluate_model(
        model,
        val_inputs,
        val_targets,
        batch_size=config.eval_batch_size,
    )

    write_json(output_dir / "config.json", asdict(config))
    write_json(
        output_dir / "environment.json",
        {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": current_git_status_short() == [],
            "git_status_short": current_git_status_short(),
            "device": str(device),
            "torch_version": torch.__version__,
        },
    )
    write_json(
        output_dir / "corpus_summary.json",
        {
            **corpus_summary(
                repo_root=repo_root,
                raw_text=raw_text,
                train_text=train_text,
                corpus=corpus,
                config=config,
            ),
            "all_validation_positions": int(all_val_inputs.shape[0] * all_val_inputs.shape[1]),
            "all_validation_windows": int(all_val_inputs.shape[0]),
        },
    )
    write_json(
        output_dir / "training_summary.json",
        {
            "best_epoch": run_result["best_epoch"],
            "best_val_loss": run_result["best_val_loss"],
            "best_depth_val_losses": run_result["best_depth_val_losses"],
            "runtime_seconds": run_result["runtime_seconds"],
            "parameter_count": run_result["parameter_count"],
        },
    )
    write_json(
        output_dir / "analysis_summary.json",
        {
            "depth_mean_losses_all_validation_positions": depth_loss_summary,
            "epsilon_summaries": epsilon_summaries,
            "marginal_improvement": marginal_summaries,
            "heterogeneity": heterogeneity,
            "fixed_window_validation": {
                "loss": round(fixed_window_eval["loss"], 6),
                "accuracy": round(fixed_window_eval["accuracy"], 6),
            },
        },
    )
    (output_dir / "stdout_summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "best_val_loss": run_result["best_val_loss"],
                "depth_mean_losses_all_validation_positions": depth_loss_summary,
                "epsilon_summaries": epsilon_summaries,
                "marginal_improvement": marginal_summaries,
                "heterogeneity": heterogeneity,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print((output_dir / "stdout_summary.json").read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
