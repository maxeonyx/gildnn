from __future__ import annotations

from pathlib import Path
import random

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FixedWindowCharDataset:
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


def _encode_windows(
    text: str,
    *,
    context_size: int,
    stoi: dict[str, int],
) -> tuple[Tensor, Tensor]:
    if len(text) <= context_size:
        raise ValueError("Text split must be longer than the context size.")
    encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    inputs = []
    targets = []
    for start in range(len(encoded) - context_size):
        stop = start + context_size
        inputs.append(encoded[start:stop])
        targets.append(encoded[stop])
    return torch.stack(inputs), torch.stack(targets)


def _choose_validation_text(
    text: str,
    *,
    train_text: str,
    val_characters: int,
) -> str:
    start = len(train_text)
    stop = start + val_characters
    candidate = text[start:stop]
    if len(candidate) < val_characters:
        raise ValueError(
            f"Need validation slice of {val_characters} characters, got {len(candidate)}."
        )
    if set(candidate).issubset(set(train_text)):
        return candidate

    max_start = len(text) - val_characters
    for candidate_start in range(start + 1, max_start + 1):
        candidate_stop = candidate_start + val_characters
        candidate = text[candidate_start:candidate_stop]
        if set(candidate).issubset(set(train_text)):
            return candidate

    missing = sorted(set(text[start : max_start + val_characters]) - set(train_text))
    raise ValueError(
        "Could not find a validation slice whose characters are all present in the training slice. "
        f"Missing training vocabulary coverage for: {missing}"
    )


def load_dataset(
    *,
    context_size: int,
    text_file: Path | None = None,
    train_characters: int = 100_000,
    val_characters: int = 20_000,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], int]:
    repo_root = Path(__file__).resolve().parents[1]
    resolved_text_file = text_file or (
        repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    )
    raw_text = resolved_text_file.read_text(encoding="utf-8")
    required_characters = train_characters + val_characters
    if len(raw_text) < required_characters:
        raise ValueError(
            f"Need at least {required_characters} characters, got {len(raw_text)}."
        )

    train_text = raw_text[:train_characters]
    val_text = _choose_validation_text(
        raw_text,
        train_text=train_text,
        val_characters=val_characters,
    )
    train_dataset = FixedWindowCharDataset(train_text, context_size=context_size)
    missing_val_chars = sorted(set(val_text) - set(train_text))
    if missing_val_chars:
        raise ValueError(
            f"Validation text contains characters absent from training text: {missing_val_chars}"
        )

    val_inputs, val_targets = _encode_windows(
        val_text,
        context_size=context_size,
        stoi=train_dataset.stoi,
    )
    return (
        (train_dataset.inputs, train_dataset.targets),
        (val_inputs, val_targets),
        train_dataset.vocab_size,
    )


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
    model: nn.Module,
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
    model: nn.Module,
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
    dataset: FixedWindowCharDataset,
    inputs: Tensor,
    targets: Tensor,
    predictions: Tensor,
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
    model: nn.Module,
    dataset: FixedWindowCharDataset,
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
