from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

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
    depth_settings: tuple[int, ...] = (1, 2, 4)
    overfit_batch_size: int = 16
    overfit_steps: int = 2000
    overfit_learning_rate: float = 0.02
    tiny_batch_size: int = 64
    tiny_steps: int = 1500
    tiny_learning_rate: float = 0.01
    sample_length: int = 80
    seed: int = 7


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
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_model(
    *, dataset: CharDataset, config: RunConfig, depth: int
) -> TinyTransformerCharModel:
    return TinyTransformerCharModel(
        vocab_size=dataset.vocab_size,
        context_size=config.context_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        feedforward_dim=config.feedforward_dim,
        residual_factory=PlainResidualCombine,
        within_token_reuse_depth=depth,
    )


def write_branch_provenance(
    *,
    branch_dir: Path,
    config: RunConfig,
    depth: int,
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
            "primary_comparator": "depth_1_one_pass",
            "within_token_reuse_depth": depth,
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
    model: TinyTransformerCharModel,
    config: RunConfig,
    dataset: CharDataset,
    depth: int,
) -> None:
    write_json(
        branch_dir / "model_summary.json",
        {
            "model_family": "tiny_transformer",
            "host_surface": "tiny_transformer_baseline",
            "within_token_reuse_depth": depth,
            "parameter_count": count_parameters(model),
            "d_model": config.d_model,
            "feedforward_dim": config.feedforward_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "context_size": config.context_size,
            "vocab_size": dataset.vocab_size,
        },
    )


def prediction_rows(
    *,
    dataset: CharDataset,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
) -> list[dict[str, str | int | bool]]:
    rows: list[dict[str, str | int | bool]] = []
    for index, (input_tokens, target_token, predicted_token) in enumerate(
        zip(inputs.tolist(), targets.tolist(), predictions.tolist(), strict=True)
    ):
        rows.append(
            {
                "index": index,
                "context": dataset.decode(input_tokens).replace("\n", "\\n"),
                "target": dataset.decode([target_token]).replace("\n", "\\n"),
                "prediction": dataset.decode([predicted_token]).replace("\n", "\\n"),
                "correct": target_token == predicted_token,
            }
        )
    return rows


def write_depth_surface(
    *,
    branch_dir: Path,
    dataset: CharDataset,
    tiny_inputs: torch.Tensor,
    tiny_targets: torch.Tensor,
    tiny_model: TinyTransformerCharModel,
    tiny_loss: float,
    tiny_accuracy: float,
) -> dict[str, object]:
    tiny_logits = tiny_model(tiny_inputs)
    tiny_predictions = tiny_logits.argmax(dim=1)
    rows = prediction_rows(
        dataset=dataset,
        inputs=tiny_inputs.cpu(),
        targets=tiny_targets.cpu(),
        predictions=tiny_predictions.cpu(),
    )
    mismatches = [row for row in rows if not row["correct"]]
    artifact = {
        "final_loss": tiny_loss,
        "final_accuracy": tiny_accuracy,
        "prediction_row_count": len(rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    write_json(branch_dir / "depth_surface.json", artifact)
    return artifact


def run_branch(
    *,
    dataset: CharDataset,
    raw_text: str,
    config: RunConfig,
    output_dir: Path,
    device: torch.device,
    depth: int,
    git_sha: str,
    git_status_short: list[str],
) -> dict[str, object]:
    branch_dir = output_dir / f"depth_{depth}"
    branch_dir.mkdir(parents=True, exist_ok=True)
    write_branch_provenance(
        branch_dir=branch_dir,
        config=config,
        depth=depth,
        device=device,
        dataset=dataset,
        raw_text=raw_text,
        git_sha=git_sha,
        git_status_short=git_status_short,
    )

    overfit_model = build_model(dataset=dataset, config=config, depth=depth).to(device)
    write_model_summary(
        branch_dir=branch_dir,
        model=overfit_model,
        config=config,
        dataset=dataset,
        depth=depth,
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
    overfit_predictions = overfit_model(overfit_inputs).argmax(dim=1)
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

    tiny_model = build_model(dataset=dataset, config=config, depth=depth).to(device)
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
    depth_surface = write_depth_surface(
        branch_dir=branch_dir,
        dataset=dataset,
        tiny_inputs=tiny_inputs,
        tiny_targets=tiny_targets,
        tiny_model=tiny_model,
        tiny_loss=tiny_loss,
        tiny_accuracy=tiny_accuracy,
    )

    return {
        "depth": depth,
        "final_loss": tiny_loss,
        "final_accuracy": tiny_accuracy,
        "mismatch_count": depth_surface["mismatch_count"],
        "sample_hello": samples["hello"],
    }


def write_comparison_summary(
    *, output_dir: Path, summaries: list[dict[str, object]]
) -> None:
    comparator = next(summary for summary in summaries if summary["depth"] == 1)
    rows = []
    for summary in summaries:
        rows.append(
            {
                **summary,
                "loss_delta_vs_depth_1": summary["final_loss"]
                - comparator["final_loss"],
                "accuracy_delta_vs_depth_1": summary["final_accuracy"]
                - comparator["final_accuracy"],
                "mismatch_delta_vs_depth_1": summary["mismatch_count"]
                - comparator["mismatch_count"],
                "sample_hello_matches_depth_1": summary["sample_hello"]
                == comparator["sample_hello"],
            }
        )
    write_json(output_dir / "comparison_summary.json", {"depth_rows": rows})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    config = RunConfig()
    if 1 not in config.depth_settings:
        raise ValueError("RunConfig.depth_settings must include depth 1 comparator.")

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
            "bounded_question": "Does repeated within-token reuse of the same tiny-transformer weights produce a readable depth-related outcome surface on the tiny fixed-window character next-token task relative to the ordinary one-pass comparator?",
            "host_surface": "tiny_transformer_baseline",
            "bounded_task_surface": "tiny_fixed_window_character_next_token",
            "primary_comparator": "depth_1_one_pass",
            "depth_settings": list(config.depth_settings),
            "artifact_class": "per-depth bounded outcome artifacts with comparator-readable loss, accuracy, saved samples, and final prediction mismatches",
        },
    )

    summaries = []
    for depth in config.depth_settings:
        set_seed(config.seed)
        summaries.append(
            run_branch(
                dataset=dataset,
                raw_text=raw_text,
                config=config,
                output_dir=args.output_dir,
                device=device,
                depth=depth,
                git_sha=git_sha,
                git_status_short=git_status_short,
            )
        )
    write_comparison_summary(output_dir=args.output_dir, summaries=summaries)


if __name__ == "__main__":
    main()
