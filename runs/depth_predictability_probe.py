from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import set_seed
from core.training import batched_pairs, current_git_sha, current_git_status_short, write_json
from experiments.wide_recurrent_vs_transformer.run import TiedDepthTransformer, build_corpus, effective_vocab_size
from runs.clean_depth_eval import all_next_token_examples, encode_corpus, load_run_config

DEFAULT_DELTA = 0.01
DEFAULT_BATCH_SIZE = 512
DEFAULT_HIDDEN_DIM = 32
DEFAULT_MAX_EPOCHS = 400
DEFAULT_PATIENCE = 30
DEFAULT_MIN_DELTA = 1e-4
DEFAULT_MLP_TRIGGER_MARGIN = 0.05


@dataclass(frozen=True)
class ProbeConfig:
    train_fraction: float = 0.8
    train_eval_fraction: float = 0.1
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    learning_rate: float = 0.05
    weight_decay: float = 0.001
    mlp_hidden_dim: int = DEFAULT_HIDDEN_DIM
    mlp_trigger_margin: float = DEFAULT_MLP_TRIGGER_MARGIN
    min_delta: float = DEFAULT_MIN_DELTA
    delta: float = DEFAULT_DELTA
    seed: int = 42


@dataclass(frozen=True)
class TargetDefinition:
    name: str
    num_classes: int
    label_names: tuple[str, ...]


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: Float[Tensor, "batch hidden"]) -> Float[Tensor, "batch classes"]:
        return self.classifier(inputs)


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: Float[Tensor, "batch hidden"]) -> Float[Tensor, "batch classes"]:
        return self.network(inputs)


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
        "--losses-path",
        type=Path,
        default=repo_root
        / "experiments"
        / "wide_recurrent_vs_transformer"
        / "artifacts"
        / "clean_depth_eval"
        / "per_example_losses.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "experiments"
        / "wide_recurrent_vs_transformer"
        / "artifacts"
        / "depth_predictability_probe",
    )
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--mlp-hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--mlp-trigger-margin", type=float, default=DEFAULT_MLP_TRIGGER_MARGIN)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def resolve_probe_config(args: argparse.Namespace) -> ProbeConfig:
    return ProbeConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_trigger_margin=args.mlp_trigger_margin,
        delta=args.delta,
        seed=args.seed,
    )


def final_hidden_states_by_depth(
    model: TiedDepthTransformer,
    inputs: Int[Tensor, "examples context"],
    *,
    depths: tuple[int, ...],
    batch_size: int,
) -> dict[int, Float[Tensor, "examples hidden"]]:
    requested_depths = set(depths)
    invalid_depths = [depth for depth in requested_depths if not 1 <= depth <= model.config.depth]
    if len(invalid_depths) > 0:
        raise ValueError(
            f"Requested hidden states for depths {sorted(invalid_depths)}, but model depth is {model.config.depth}."
        )

    was_training = model.training
    model.eval()
    hidden_batches: dict[int, list[Tensor]] = {depth: [] for depth in requested_depths}
    with torch.inference_mode():
        for batch_inputs, _ in batched_pairs(inputs, inputs, batch_size=batch_size):
            stream = model.embedded_tokens(batch_inputs)
            for depth_index in range(model.config.depth):
                stream = stream + model.attention(model.attention_norms[depth_index](stream))
                stream = stream + model.feedforward(model.feedforward_norms[depth_index](stream))
                depth = depth_index + 1
                if depth in requested_depths:
                    final_state = model.final_norm(stream)
                    hidden_batches[depth].append(final_state[:, -1, :].cpu())
    if was_training:
        model.train()
    return {depth: torch.cat(batches, dim=0) for depth, batches in hidden_batches.items()}


def oracle_depth_targets(losses_by_depth: Float[Tensor, "examples depth"]) -> Int[Tensor, "examples"]:
    return losses_by_depth.argmin(dim=1).to(dtype=torch.long)


def harmed_by_depth_8_targets(losses_by_depth: Float[Tensor, "examples depth"]) -> Int[Tensor, "examples"]:
    oracle_depth = oracle_depth_targets(losses_by_depth)
    return (oracle_depth < losses_by_depth.shape[1] - 1).to(dtype=torch.long)


def no_regret_depth_targets(
    losses_by_depth: Float[Tensor, "examples depth"],
    *,
    delta: float,
) -> Int[Tensor, "examples"]:
    final_losses = losses_by_depth[:, -1]
    within_delta = losses_by_depth <= (final_losses.unsqueeze(1) + delta)
    return within_delta.to(dtype=torch.int64).argmax(dim=1).to(dtype=torch.long)


def done_by_depth_3_targets(
    losses_by_depth: Float[Tensor, "examples depth"],
    *,
    delta: float,
) -> Int[Tensor, "examples"]:
    no_regret_depth = no_regret_depth_targets(losses_by_depth, delta=delta)
    return (no_regret_depth <= 2).to(dtype=torch.long)


def target_definitions() -> tuple[TargetDefinition, ...]:
    return (
        TargetDefinition(name="harmed_by_depth_8", num_classes=2, label_names=("not_harmed", "harmed")),
        TargetDefinition(name="done_by_depth_3_delta_0p01", num_classes=2, label_names=("needs_more_than_3", "done_by_3")),
        TargetDefinition(
            name="oracle_depth_argmin",
            num_classes=8,
            label_names=tuple(f"depth_{depth}" for depth in range(1, 9)),
        ),
    )


def build_targets(
    losses_by_depth: Float[Tensor, "examples depth"],
    *,
    delta: float,
) -> dict[str, Int[Tensor, "examples"]]:
    return {
        "harmed_by_depth_8": harmed_by_depth_8_targets(losses_by_depth),
        "done_by_depth_3_delta_0p01": done_by_depth_3_targets(losses_by_depth, delta=delta),
        "oracle_depth_argmin": oracle_depth_targets(losses_by_depth),
    }


def split_indices(
    total_examples: int,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[Int[Tensor, "train"], Int[Tensor, "test"]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(total_examples, generator=generator)
    train_size = int(total_examples * train_fraction)
    if train_size <= 0 or train_size >= total_examples:
        raise ValueError(
            f"Train split must be non-empty and smaller than the full dataset, got {train_size} of {total_examples}."
        )
    return permutation[:train_size], permutation[train_size:]


def split_train_and_eval(
    train_indices: Int[Tensor, "train"],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[Int[Tensor, "fit"], Int[Tensor, "eval"]]:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be between 0 and 1, got {eval_fraction}.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    permutation = train_indices[torch.randperm(train_indices.shape[0], generator=generator)]
    eval_size = int(permutation.shape[0] * eval_fraction)
    if eval_size <= 0 or eval_size >= permutation.shape[0]:
        raise ValueError(
            f"Inner eval split must be non-empty and smaller than the train split, got {eval_size} of {permutation.shape[0]}."
        )
    return permutation[eval_size:], permutation[:eval_size]


def standardize_features(
    train_features: Float[Tensor, "examples hidden"],
    eval_features: Float[Tensor, "examples hidden"],
    test_features: Float[Tensor, "examples hidden"],
) -> tuple[Float[Tensor, "examples hidden"], Float[Tensor, "examples hidden"], Float[Tensor, "examples hidden"], dict[str, list[float]]]:
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False)
    safe_std = torch.where(std > 0, std, torch.ones_like(std))
    return (
        (train_features - mean) / safe_std,
        (eval_features - mean) / safe_std,
        (test_features - mean) / safe_std,
        {
            "mean": mean.tolist(),
            "std": safe_std.tolist(),
        },
    )


def class_distribution(targets: Int[Tensor, "examples"], label_names: tuple[str, ...]) -> dict[str, object]:
    counts = torch.bincount(targets, minlength=len(label_names))
    total = int(targets.shape[0])
    return {
        "counts": {label_names[index]: int(count.item()) for index, count in enumerate(counts)},
        "fractions": {
            label_names[index]: round(count.item() / total, 6) for index, count in enumerate(counts)
        },
    }


def accuracy_from_logits(
    logits: Float[Tensor, "examples classes"],
    targets: Int[Tensor, "examples"],
) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def evaluate_probe(
    model: nn.Module,
    features: Float[Tensor, "examples hidden"],
    targets: Int[Tensor, "examples"],
) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        logits = model(features)
        return {
            "loss": F.cross_entropy(logits, targets).item(),
            "accuracy": accuracy_from_logits(logits, targets),
        }


def train_probe(
    probe: nn.Module,
    *,
    train_features: Float[Tensor, "train hidden"],
    train_targets: Int[Tensor, "train"],
    eval_features: Float[Tensor, "eval hidden"],
    eval_targets: Int[Tensor, "eval"],
    config: ProbeConfig,
) -> tuple[nn.Module, dict[str, object]]:
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state = {key: value.detach().clone() for key, value in probe.state_dict().items()}
    best_eval_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.max_epochs + 1):
        probe.train()
        logits = probe(train_features)
        train_loss = F.cross_entropy(logits, train_targets)
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()

        train_accuracy = accuracy_from_logits(logits.detach(), train_targets)
        eval_metrics = evaluate_probe(probe, eval_features, eval_targets)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss.item(), 6),
                "train_accuracy": round(train_accuracy, 6),
                "eval_loss": round(eval_metrics["loss"], 6),
                "eval_accuracy": round(eval_metrics["accuracy"], 6),
            }
        )

        if eval_metrics["loss"] < best_eval_loss - config.min_delta:
            best_eval_loss = eval_metrics["loss"]
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in probe.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    probe.load_state_dict(best_state)
    return probe, {
        "best_epoch": best_epoch,
        "best_eval_loss": round(best_eval_loss, 6),
        "epochs_ran": len(history),
        "history": history,
    }


def majority_baseline(
    train_targets: Int[Tensor, "train"],
    test_targets: Int[Tensor, "test"],
    label_names: tuple[str, ...],
) -> dict[str, object]:
    train_counts = torch.bincount(train_targets, minlength=len(label_names))
    majority_class = int(train_counts.argmax().item())
    accuracy = (test_targets == majority_class).float().mean().item()
    return {
        "majority_class_index": majority_class,
        "majority_class_label": label_names[majority_class],
        "accuracy": round(accuracy, 6),
    }


def fit_probe_family(
    *,
    features: Float[Tensor, "examples hidden"],
    targets: Int[Tensor, "examples"],
    target_definition: TargetDefinition,
    config: ProbeConfig,
) -> dict[str, object]:
    train_indices, test_indices = split_indices(
        features.shape[0],
        train_fraction=config.train_fraction,
        seed=config.seed,
    )
    fit_indices, eval_indices = split_train_and_eval(
        train_indices,
        eval_fraction=config.train_eval_fraction,
        seed=config.seed,
    )

    fit_features = features[fit_indices]
    eval_features = features[eval_indices]
    test_features = features[test_indices]
    fit_targets = targets[fit_indices]
    eval_targets = targets[eval_indices]
    test_targets = targets[test_indices]

    standardized_fit, standardized_eval, standardized_test, normalization = standardize_features(
        fit_features,
        eval_features,
        test_features,
    )
    baseline = majority_baseline(fit_targets, test_targets, target_definition.label_names)

    linear_probe, linear_training = train_probe(
        LinearProbe(features.shape[1], target_definition.num_classes),
        train_features=standardized_fit,
        train_targets=fit_targets,
        eval_features=standardized_eval,
        eval_targets=eval_targets,
        config=config,
    )
    linear_test = evaluate_probe(linear_probe, standardized_test, test_targets)

    result: dict[str, object] = {
        "split": {
            "fit_size": int(fit_indices.shape[0]),
            "inner_eval_size": int(eval_indices.shape[0]),
            "test_size": int(test_indices.shape[0]),
        },
        "label_distribution": {
            "fit": class_distribution(fit_targets, target_definition.label_names),
            "test": class_distribution(test_targets, target_definition.label_names),
        },
        "baseline": baseline,
        "normalization": normalization,
        "linear_probe": {
            "training": linear_training,
            "test": {
                "loss": round(linear_test["loss"], 6),
                "accuracy": round(linear_test["accuracy"], 6),
                "accuracy_gain_over_baseline": round(
                    linear_test["accuracy"] - float(baseline["accuracy"]),
                    6,
                ),
            },
        },
    }

    if linear_test["accuracy"] < float(baseline["accuracy"]) + config.mlp_trigger_margin:
        mlp_probe, mlp_training = train_probe(
            MLPProbe(features.shape[1], config.mlp_hidden_dim, target_definition.num_classes),
            train_features=standardized_fit,
            train_targets=fit_targets,
            eval_features=standardized_eval,
            eval_targets=eval_targets,
            config=config,
        )
        mlp_test = evaluate_probe(mlp_probe, standardized_test, test_targets)
        result["mlp_probe"] = {
            "training": mlp_training,
            "test": {
                "loss": round(mlp_test["loss"], 6),
                "accuracy": round(mlp_test["accuracy"], 6),
                "accuracy_gain_over_baseline": round(
                    mlp_test["accuracy"] - float(baseline["accuracy"]),
                    6,
                ),
            },
        }

    return result


def build_summary_table(results: dict[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for feature_name, feature_results in results.items():
        summary[feature_name] = {}
        for target_name, target_results in feature_results.items():
            target_payload = {
                "baseline_accuracy": target_results["baseline"]["accuracy"],
                "linear_accuracy": target_results["linear_probe"]["test"]["accuracy"],
                "linear_gain": target_results["linear_probe"]["test"]["accuracy_gain_over_baseline"],
            }
            if "mlp_probe" in target_results:
                target_payload["mlp_accuracy"] = target_results["mlp_probe"]["test"]["accuracy"]
                target_payload["mlp_gain"] = target_results["mlp_probe"]["test"][
                    "accuracy_gain_over_baseline"
                ]
            summary[feature_name][target_name] = target_payload
    return summary


def main() -> int:
    args = parse_args()
    repo_root: Path = args.repo_root
    probe_config = resolve_probe_config(args)
    device = torch.device(args.device)
    checkpoint_path: Path = args.checkpoint_path
    losses_path: Path = args.losses_path
    output_dir: Path = args.output_dir

    set_seed(probe_config.seed)
    run_config = load_run_config(checkpoint_path)
    prepared_dir = repo_root / "experiments" / "wide_recurrent_vs_transformer" / "prepared.ignore"
    corpus, _, _ = build_corpus(
        repo_root=repo_root,
        config=run_config,
        prepared_dir=prepared_dir,
    )
    vocab_size = effective_vocab_size(corpus)

    losses_payload = torch.load(losses_path, map_location="cpu")
    val_losses_by_depth = losses_payload["val_losses_by_depth"].to(dtype=torch.float32)

    model = TiedDepthTransformer(vocab_size=vocab_size, config=run_config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    val_text = (prepared_dir / "tinyshakespeare_val.txt").read_text(encoding="utf-8")
    val_encoded = encode_corpus(val_text, char_to_idx=corpus.char_to_idx)
    val_inputs, _ = all_next_token_examples(val_encoded, context_size=run_config.context_size)
    if val_inputs.shape[0] != val_losses_by_depth.shape[0]:
        raise ValueError(
            "Validation inputs and saved losses disagree on example count: "
            f"{val_inputs.shape[0]} vs {val_losses_by_depth.shape[0]}."
        )

    hidden_states = final_hidden_states_by_depth(
        model,
        val_inputs.to(device),
        depths=(1, 2),
        batch_size=probe_config.batch_size,
    )
    targets = build_targets(val_losses_by_depth, delta=probe_config.delta)
    target_specs = {target.name: target for target in target_definitions()}

    feature_results: dict[str, object] = {}
    for depth, features in hidden_states.items():
        feature_name = f"depth_{depth}_hidden"
        feature_results[feature_name] = {}
        for target_name, target_values in targets.items():
            feature_results[feature_name][target_name] = fit_probe_family(
                features=features.to(dtype=torch.float32),
                targets=target_values,
                target_definition=target_specs[target_name],
                config=probe_config,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "depth_1_hidden": hidden_states[1],
            "depth_2_hidden": hidden_states[2],
            "targets": targets,
        },
        output_dir / "probe_features.pt",
    )
    write_json(
        output_dir / "config.json",
        {
            "probe_config": asdict(probe_config),
            "run_config": asdict(run_config),
            "checkpoint_path": str(checkpoint_path),
            "losses_path": str(losses_path),
            "device": str(device),
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
        output_dir / "summary.json",
        {
            "overall_target_distribution": {
                name: class_distribution(target_tensor, target_specs[name].label_names)
                for name, target_tensor in targets.items()
            },
            "probe_results": feature_results,
            "summary_table": build_summary_table(feature_results),
            "notes": {
                "binary_harmed_target_definition": "1 when argmin_d loss_t(d) occurs before depth 8; ties go to the earliest depth via argmin.",
                "binary_done_by_depth_3_definition": "1 when the shallowest depth within delta of depth-8 loss is <= 3.",
                "multiclass_definition": "0-based class index for argmin_d loss_t(d), corresponding to depth_1..depth_8.",
                "mlp_trigger_rule": "Run MLP when linear-probe test accuracy is less than baseline + mlp_trigger_margin.",
            },
        },
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary_table": build_summary_table(feature_results),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
