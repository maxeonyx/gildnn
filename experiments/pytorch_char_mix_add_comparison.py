from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from core.fixed_window_char import (
    FixedWindowCharDataset as CharDataset,
    generate_text,
    render_predictions,
    resolve_device,
    set_seed,
    train_fixed_batch,
    train_tiny_dataset,
)
from core.tiny_char_transformer import (
    PlainResidualCombine,
    ScalarMixAddResidualCombine,
    TinyTransformerCharModel,
    count_parameters,
)


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 5
    d_model: int = 24
    feedforward_dim: int = 112
    num_heads: int = 4
    num_layers: int = 1
    overfit_batch_size: int = 16
    overfit_steps: int = 2000
    overfit_learning_rate: float = 0.02
    tiny_batch_size: int = 64
    tiny_steps: int = 1500
    tiny_learning_rate: float = 0.01
    sample_length: int = 80
    seed: int = 7
    gate_saturation_low: float = 0.05
    gate_saturation_high: float = 0.95


def current_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def current_git_status_short() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.rstrip() for line in result.stdout.splitlines() if line.strip()]
    return lines


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def feedforward_reference_parameter_count(vocab_size: int) -> int:
    context_size = 5
    embedding_dim = 24
    hidden_dim = 64
    return (
        vocab_size * embedding_dim
        + (context_size * embedding_dim) * hidden_dim
        + hidden_dim
        + hidden_dim * vocab_size
        + vocab_size
    )


def build_model(
    *,
    dataset: CharDataset,
    config: RunConfig,
    residual_mode: str,
) -> TinyTransformerCharModel:
    if residual_mode == "plain_residual":
        residual_factory = PlainResidualCombine
    elif residual_mode == "scalar_mix_add":
        residual_factory = lambda: ScalarMixAddResidualCombine(
            saturation_low=config.gate_saturation_low,
            saturation_high=config.gate_saturation_high,
        )
    else:
        raise ValueError(f"Unsupported residual mode: {residual_mode}")

    return TinyTransformerCharModel(
        vocab_size=dataset.vocab_size,
        context_size=config.context_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        feedforward_dim=config.feedforward_dim,
        residual_factory=residual_factory,
    )


def write_branch_provenance(
    *,
    branch_dir: Path,
    config: RunConfig,
    residual_mode: str,
    device: torch.device,
    dataset: CharDataset,
    raw_text: str,
    git_sha: str,
    git_status_short: list[str],
) -> None:
    write_json(
        branch_dir / "config.json",
        {
            **asdict(config),
            "host_surface": "tiny_transformer_baseline",
            "bounded_task_surface": "tiny_fixed_window_character_next_token",
            "primary_comparator": "plain_residual",
            "variant_under_test": "scalar_m_mix_add",
            "residual_mode": residual_mode,
        },
    )
    write_json(
        branch_dir / "environment.json",
        {
            "git_sha": git_sha,
            "git_working_tree_clean": len(git_status_short) == 0,
            "git_status_short": git_status_short,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_is_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None,
            "text_characters": len(raw_text),
            "vocab_size": dataset.vocab_size,
            "window_count": int(dataset.inputs.shape[0]),
        },
    )


def write_model_summary(
    *,
    branch_dir: Path,
    model: nn.Module,
    config: RunConfig,
    dataset: CharDataset,
    residual_mode: str,
) -> None:
    parameter_count = count_parameters(model)
    plain_residual_parameter_count = feedforward_reference_parameter_count(
        dataset.vocab_size
    )
    summary = {
        "model_family": "tiny_transformer",
        "host_surface": "tiny_transformer_baseline",
        "residual_mode": residual_mode,
        "parameter_count": parameter_count,
        "feedforward_reference_parameter_count": plain_residual_parameter_count,
        "parameter_delta_vs_feedforward": parameter_count
        - plain_residual_parameter_count,
        "parameter_ratio_vs_feedforward": round(
            parameter_count / plain_residual_parameter_count, 6
        ),
        "d_model": config.d_model,
        "feedforward_dim": config.feedforward_dim,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "context_size": config.context_size,
        "vocab_size": dataset.vocab_size,
    }
    if residual_mode == "scalar_mix_add":
        mix_parameters = [
            module.m.item()
            for module in model.modules()
            if isinstance(module, ScalarMixAddResidualCombine)
        ]
        summary["mix_parameter_count"] = len(mix_parameters)
        summary["mix_parameter_initial_values"] = mix_parameters
    write_json(branch_dir / "model_summary.json", summary)


def run_branch(
    *,
    dataset: CharDataset,
    raw_text: str,
    config: RunConfig,
    output_dir: Path,
    device: torch.device,
    residual_mode: str,
    git_sha: str,
    git_status_short: list[str],
) -> None:
    branch_dir = output_dir / residual_mode
    branch_dir.mkdir(parents=True, exist_ok=True)
    write_branch_provenance(
        branch_dir=branch_dir,
        config=config,
        residual_mode=residual_mode,
        device=device,
        dataset=dataset,
        raw_text=raw_text,
        git_sha=git_sha,
        git_status_short=git_status_short,
    )

    overfit_model = build_model(
        dataset=dataset,
        config=config,
        residual_mode=residual_mode,
    ).to(device)
    write_model_summary(
        branch_dir=branch_dir,
        model=overfit_model,
        config=config,
        dataset=dataset,
        residual_mode=residual_mode,
    )

    overfit_inputs = dataset.inputs[: config.overfit_batch_size].to(device)
    overfit_targets = dataset.targets[: config.overfit_batch_size].to(device)
    overfit_trace, overfit_loss, overfit_accuracy = train_fixed_batch(
        overfit_model,
        overfit_inputs,
        overfit_targets,
        steps=config.overfit_steps,
        learning_rate=config.overfit_learning_rate,
    )
    overfit_logits = overfit_model(overfit_inputs)
    overfit_predictions = overfit_logits.argmax(dim=1)
    write_json(
        branch_dir / "overfit_metrics.json",
        {
            "final_loss": overfit_loss,
            "final_accuracy": overfit_accuracy,
            "reached_memorization_bar": overfit_accuracy == 1.0 and overfit_loss < 1e-3,
            "trace": overfit_trace,
        },
    )
    (branch_dir / "overfit_predictions.txt").write_text(
        render_predictions(
            dataset,
            overfit_inputs.cpu(),
            overfit_targets.cpu(),
            overfit_predictions.cpu(),
        ),
        encoding="utf-8",
    )
    overfit_observables = overfit_model.residual_observations()

    tiny_model = build_model(
        dataset=dataset,
        config=config,
        residual_mode=residual_mode,
    ).to(device)
    tiny_inputs = dataset.inputs.to(device)
    tiny_targets = dataset.targets.to(device)
    tiny_trace, tiny_loss, tiny_accuracy = train_tiny_dataset(
        tiny_model,
        tiny_inputs,
        tiny_targets,
        batch_size=config.tiny_batch_size,
        steps=config.tiny_steps,
        learning_rate=config.tiny_learning_rate,
    )
    write_json(
        branch_dir / "tiny_run_metrics.json",
        {
            "final_loss": tiny_loss,
            "final_accuracy": tiny_accuracy,
            "trace": tiny_trace,
        },
    )

    prompts = ["hello", "small", " text"]
    samples = {
        prompt.replace("\n", "\\n"): generate_text(
            tiny_model,
            dataset,
            prompt,
            length=config.sample_length,
            device=device,
        )
        for prompt in prompts
    }
    write_json(branch_dir / "tiny_samples.json", samples)
    tiny_model(tiny_inputs)
    tiny_observables = tiny_model.residual_observations()

    observable_name = (
        "plain_branch_observables.json"
        if residual_mode == "plain_residual"
        else "mix_branch_observables.json"
    )
    write_json(
        branch_dir / observable_name,
        {
            "host_surface": "tiny_transformer_baseline",
            "residual_mode": residual_mode,
            "overfit_surface": overfit_observables,
            "tiny_surface": tiny_observables,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    config = RunConfig()
    set_seed(config.seed)
    device = resolve_device(args.device)
    raw_text = args.text_file.read_text(encoding="utf-8")
    dataset = CharDataset(raw_text, context_size=config.context_size)
    git_sha = current_git_sha()
    git_status_short = current_git_status_short()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "comparison_scope.json",
        {
            "host_surface": "tiny_transformer_baseline",
            "variant_under_test": "scalar_m_mix_add",
            "primary_comparator": "plain_residual",
            "bounded_task_surface": "tiny_fixed_window_character_next_token",
            "branches": ["plain_residual", "scalar_mix_add"],
        },
    )

    for residual_mode in ("plain_residual", "scalar_mix_add"):
        set_seed(config.seed)
        run_branch(
            dataset=dataset,
            raw_text=raw_text,
            config=config,
            output_dir=args.output_dir,
            device=device,
            residual_mode=residual_mode,
            git_sha=git_sha,
            git_status_short=git_status_short,
        )


if __name__ == "__main__":
    main()
