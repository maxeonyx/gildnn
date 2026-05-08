from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import (
    FixedWindowCharDataset as CharDataset,
    generate_text,
    render_predictions,
    resolve_device,
    set_seed,
    train_fixed_batch,
    train_tiny_dataset,
)


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 5
    embedding_dim: int = 24
    hidden_dim: int = 64
    num_layers: int = 1
    nonlinearity: str = "tanh"
    overfit_batch_size: int = 16
    overfit_steps: int = 2000
    overfit_learning_rate: float = 0.05
    tiny_batch_size: int = 64
    tiny_steps: int = 1500
    tiny_learning_rate: float = 0.03
    sample_length: int = 80
    seed: int = 7


class TinyRnnCharModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        nonlinearity: str,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        embedded = self.embedding(tokens)
        outputs, _final_hidden = self.rnn(embedded)
        return self.output(outputs[:, -1, :])


def current_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def current_git_status_short() -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


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


def transformer_reference_parameter_count(vocab_size: int) -> int:
    d_model = 24
    feedforward_dim = 112
    num_heads = 4
    num_layers = 1

    embedding_parameters = vocab_size * d_model + 5 * d_model
    attention_parameters = (
        3 * d_model * d_model + 3 * d_model + d_model * d_model + d_model
    )
    feedforward_parameters = (
        d_model * feedforward_dim
        + feedforward_dim
        + feedforward_dim * d_model
        + d_model
    )
    layer_norm_parameters = 2 * d_model * 2
    block_parameters = num_layers * (
        attention_parameters + feedforward_parameters + layer_norm_parameters
    )
    final_norm_parameters = 2 * d_model
    output_parameters = d_model * vocab_size + vocab_size
    _ = num_heads
    return (
        embedding_parameters
        + block_parameters
        + final_norm_parameters
        + output_parameters
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
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    git_status_short = current_git_status_short()

    write_json(output_dir / "config.json", asdict(config))
    write_json(
        output_dir / "environment.json",
        {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": git_status_short == "",
            "git_status_short": git_status_short.splitlines(),
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

    overfit_model = TinyRnnCharModel(
        vocab_size=dataset.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        nonlinearity=config.nonlinearity,
    ).to(device)
    write_json(
        output_dir / "model_summary.json",
        {
            "model_family": "ordinary_rnn",
            "rnn_type": "nn.RNN",
            "rnn_parameter_count": count_parameters(overfit_model),
            "feedforward_reference_parameter_count": feedforward_reference_parameter_count(
                dataset.vocab_size
            ),
            "transformer_reference_parameter_count": transformer_reference_parameter_count(
                dataset.vocab_size
            ),
            "parameter_delta_vs_feedforward": count_parameters(overfit_model)
            - feedforward_reference_parameter_count(dataset.vocab_size),
            "parameter_delta_vs_transformer": count_parameters(overfit_model)
            - transformer_reference_parameter_count(dataset.vocab_size),
            "parameter_ratio_vs_feedforward": round(
                count_parameters(overfit_model)
                / feedforward_reference_parameter_count(dataset.vocab_size),
                6,
            ),
            "parameter_ratio_vs_transformer": round(
                count_parameters(overfit_model)
                / transformer_reference_parameter_count(dataset.vocab_size),
                6,
            ),
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "nonlinearity": config.nonlinearity,
            "context_size": config.context_size,
            "vocab_size": dataset.vocab_size,
            "hidden_state_handling": "reset_to_zero_per_forward_call_no_carry_between_windows_or_batches",
        },
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
        output_dir / "overfit_metrics.json",
        {
            "final_loss": overfit_loss,
            "final_accuracy": overfit_accuracy,
            "reached_memorization_bar": overfit_accuracy == 1.0 and overfit_loss < 1e-3,
            "trace": overfit_trace,
            "hidden_state_handling": "reset_to_zero_per_forward_call_no_carry_between_windows_or_batches",
        },
    )
    (output_dir / "overfit_predictions.txt").write_text(
        render_predictions(
            dataset,
            overfit_inputs.cpu(),
            overfit_targets.cpu(),
            overfit_predictions.cpu(),
        ),
        encoding="utf-8",
    )

    tiny_model = TinyRnnCharModel(
        vocab_size=dataset.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        nonlinearity=config.nonlinearity,
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
        output_dir / "tiny_run_metrics.json",
        {
            "final_loss": tiny_loss,
            "final_accuracy": tiny_accuracy,
            "trace": tiny_trace,
            "hidden_state_handling": "reset_to_zero_per_forward_call_no_carry_between_windows_or_batches",
        },
    )

    prompts = ["hello", "small", " text"]
    samples = {
        prompt.replace("\n", "\\n"): generate_text(
            tiny_model, dataset, prompt, length=config.sample_length, device=device
        )
        for prompt in prompts
    }
    write_json(output_dir / "tiny_samples.json", samples)


if __name__ == "__main__":
    main()
