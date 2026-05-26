from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
from jaxtyping import Int
from torch import Tensor

from core.recurrent_depth import RecurrentDepthLM


@dataclass(frozen=True)
class GenerationResult:
    text: str
    generated_text: str
    generated_token_ids: tuple[int, ...]
    halting_depths: tuple[int, ...]


def _trim_prompt(prompt_tokens: list[int], *, context_size: int) -> list[int]:
    if len(prompt_tokens) <= context_size:
        return prompt_tokens
    return prompt_tokens[-context_size:]


def _top_k_filter(logits: Tensor, *, top_k: int | None) -> Tensor:
    if top_k is None:
        return logits
    if top_k <= 0:
        raise ValueError(f"top_k must be positive when provided, got {top_k}")
    limited_top_k = min(top_k, logits.shape[-1])
    top_values = torch.topk(logits, k=limited_top_k, dim=-1).values
    cutoff = top_values[:, -1:]
    return logits.masked_fill(logits < cutoff, float("-inf"))


def _top_p_filter(logits: Tensor, *, top_p: float | None) -> Tensor:
    if top_p is None or top_p >= 1.0:
        return logits
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_remove_mask = cumulative_probs > top_p
    sorted_remove_mask[:, 1:] = sorted_remove_mask[:, :-1].clone()
    sorted_remove_mask[:, 0] = False
    remove_mask = torch.zeros_like(logits, dtype=torch.bool)
    remove_mask.scatter_(1, sorted_indices, sorted_remove_mask)
    return logits.masked_fill(remove_mask, float("-inf"))


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
) -> Int[Tensor, "batch"]:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    filtered_logits = logits.float() / temperature
    filtered_logits = _top_k_filter(filtered_logits, top_k=top_k)
    filtered_logits = _top_p_filter(filtered_logits, top_p=top_p)
    probabilities = torch.softmax(filtered_logits, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("sampling probabilities became non-finite")
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


@torch.inference_mode()
def generate_with_halting(
    model: RecurrentDepthLM,
    *,
    prompt: str,
    stoi: dict[str, int],
    itos: dict[int, str],
    length: int,
    threshold: float,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
    calibration: dict[str, object] | Tensor | None = None,
) -> GenerationResult:
    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    if len(prompt) == 0:
        raise ValueError("prompt must not be empty")
    missing_chars = sorted({char for char in prompt if char not in stoi})
    if len(missing_chars) > 0:
        raise ValueError(f"prompt contains characters outside the training vocabulary: {missing_chars}")

    device = next(model.parameters()).device
    encoded_prompt = [stoi[char] for char in prompt]
    window = _trim_prompt(encoded_prompt, context_size=model.config.context_size)
    generated_token_ids: list[int] = []
    halting_depths: list[int] = []

    was_training = model.training
    model.eval()
    for _ in range(length):
        tokens = torch.tensor([window], dtype=torch.long, device=device)
        halting_output = model.forward_sequence_with_halting(
            tokens,
            epsilon=threshold,
            calibration=calibration,
            right_align=True,
        )
        next_token = int(
            sample_next_token(halting_output.logits, temperature=temperature, top_k=top_k, top_p=top_p)[0].item()
        )
        generated_token_ids.append(next_token)
        halting_depths.append(int(halting_output.halt_depths[0].item()))
        window.append(next_token)
        window = _trim_prompt(window, context_size=model.config.context_size)

    if was_training:
        model.train()

    generated_text = "".join(itos[token_id] for token_id in generated_token_ids)
    return GenerationResult(
        text=prompt + generated_text,
        generated_text=generated_text,
        generated_token_ids=tuple(generated_token_ids),
        halting_depths=tuple(halting_depths),
    )


def halting_depth_histogram(depths: tuple[int, ...]) -> dict[int, int]:
    return dict(sorted(Counter(depths).items()))


__all__ = [
    "GenerationResult",
    "generate_with_halting",
    "halting_depth_histogram",
    "sample_next_token",
]
