from __future__ import annotations

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
