from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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


class CharDataset:
    def __init__(self, text: str, context_size: int) -> None:
        if len(text) <= context_size:
            raise ValueError("Text sample must be longer than the context size.")

        self.text = text
        self.context_size = context_size
        self.vocab = sorted(set(text))
        self.stoi = {char: index for index, char in enumerate(self.vocab)}
        self.itos = {index: char for index, char in enumerate(self.vocab)}

        encoded = torch.tensor([self.stoi[char] for char in text], dtype=torch.long)
        inputs = []
        targets = []
        for start in range(len(encoded) - context_size):
            stop = start + context_size
            inputs.append(encoded[start:stop])
            targets.append(encoded[stop])

        self.inputs = torch.stack(inputs)
        self.targets = torch.stack(targets)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[token] for token in tokens)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, d_model),
        )

    def forward(self, x: Tensor, *, causal_mask: Tensor) -> Tensor:
        normalized = self.attention_norm(x)
        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + attention_output
        x = x + self.feedforward(self.feedforward_norm(x))
        return x


class TinyTransformerCharModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        context_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
    ) -> None:
        super().__init__()
        self.context_size = context_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.shape[1]
        if sequence_length != self.context_size:
            raise ValueError(
                f"Expected context length {self.context_size}, got {sequence_length}."
            )

        positions = torch.arange(sequence_length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(
            0
        )
        causal_mask = torch.triu(
            torch.ones(
                sequence_length, sequence_length, device=tokens.device, dtype=torch.bool
            ),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, causal_mask=causal_mask)

        x = self.final_norm(x)
        return self.output(x[:, -1, :])


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device request: {requested}")


def train_fixed_batch(
    model: TinyTransformerCharModel,
    batch_inputs: Tensor,
    batch_targets: Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> tuple[list[dict[str, float | int]], float, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    trace: list[dict[str, float | int]] = []

    for step in range(steps + 1):
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == batch_targets).float().mean().item()

        if step % 50 == 0 or step == steps:
            trace.append(
                {
                    "step": step,
                    "loss": round(loss.item(), 6),
                    "accuracy": round(accuracy, 6),
                }
            )

        if accuracy == 1.0 and loss.item() < 1e-3:
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_logits = model(batch_inputs)
    final_loss = F.cross_entropy(final_logits, batch_targets).item()
    final_accuracy = (final_logits.argmax(dim=1) == batch_targets).float().mean().item()
    final_step = trace[-1]["step"] if trace else None
    if final_step != step:
        trace.append(
            {
                "step": step,
                "loss": round(final_loss, 6),
                "accuracy": round(final_accuracy, 6),
            }
        )
    return trace, final_loss, final_accuracy


def train_tiny_dataset(
    model: TinyTransformerCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
    steps: int,
    learning_rate: float,
) -> tuple[list[dict[str, float | int]], float, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    sample_count = inputs.shape[0]
    trace: list[dict[str, float | int]] = []

    for step in range(steps + 1):
        if step % 50 == 0 or step == steps:
            full_logits = model(inputs)
            full_loss = F.cross_entropy(full_logits, targets)
            full_accuracy = (full_logits.argmax(dim=1) == targets).float().mean().item()
            trace.append(
                {
                    "step": step,
                    "loss": round(full_loss.item(), 6),
                    "accuracy": round(full_accuracy, 6),
                }
            )

        if step == steps:
            break

        batch_indices = torch.randint(
            0, sample_count, (batch_size,), device=inputs.device
        )
        batch_inputs = inputs[batch_indices]
        batch_targets = targets[batch_indices]

        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_logits = model(inputs)
    final_loss = F.cross_entropy(final_logits, targets).item()
    final_accuracy = (final_logits.argmax(dim=1) == targets).float().mean().item()
    return trace, final_loss, final_accuracy


def render_predictions(
    dataset: CharDataset, inputs: Tensor, targets: Tensor, predictions: Tensor
) -> str:
    rows = ["context | target | prediction"]
    for input_tokens, target_token, predicted_token in zip(
        inputs.tolist(), targets.tolist(), predictions.tolist(), strict=True
    ):
        context = dataset.decode(input_tokens).replace("\n", "\\n")
        target = dataset.decode([target_token]).replace("\n", "\\n")
        prediction = dataset.decode([predicted_token]).replace("\n", "\\n")
        rows.append(f"{context} | {target} | {prediction}")
    return "\n".join(rows) + "\n"


def generate_text(
    model: TinyTransformerCharModel,
    dataset: CharDataset,
    prompt: str,
    *,
    length: int,
    device: torch.device,
) -> str:
    if len(prompt) != dataset.context_size:
        raise ValueError(
            f"Prompt must be exactly {dataset.context_size} characters long."
        )

    window = dataset.encode(prompt)
    generated = prompt
    for _ in range(length):
        tokens = torch.tensor([window], dtype=torch.long, device=device)
        next_token = model(tokens).argmax(dim=1).item()
        generated += dataset.decode([next_token])
        window = window[1:] + [next_token]
    return generated


def current_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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

    write_json(output_dir / "config.json", asdict(config))
    write_json(
        output_dir / "environment.json",
        {
            "git_sha": current_git_sha(),
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

    overfit_model = TinyTransformerCharModel(
        vocab_size=dataset.vocab_size,
        context_size=config.context_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        feedforward_dim=config.feedforward_dim,
    ).to(device)
    transformer_parameter_count = count_parameters(overfit_model)
    feedforward_parameter_count = feedforward_reference_parameter_count(
        dataset.vocab_size
    )
    write_json(
        output_dir / "model_summary.json",
        {
            "model_family": "ordinary_transformer",
            "transformer_parameter_count": transformer_parameter_count,
            "feedforward_reference_parameter_count": feedforward_parameter_count,
            "parameter_delta_vs_feedforward": transformer_parameter_count
            - feedforward_parameter_count,
            "parameter_ratio_vs_feedforward": round(
                transformer_parameter_count / feedforward_parameter_count, 6
            ),
            "d_model": config.d_model,
            "feedforward_dim": config.feedforward_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "context_size": config.context_size,
            "vocab_size": dataset.vocab_size,
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

    tiny_model = TinyTransformerCharModel(
        vocab_size=dataset.vocab_size,
        context_size=config.context_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        feedforward_dim=config.feedforward_dim,
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
