"""Residual stream across time with partial detach (truncated BPTT).

Same as the stop-grad variant, but only detaches the stream every N timesteps.
Within a window of N consecutive steps, the stream carries gradient.
History entries are always detached (prevents unbounded graph from attention).

This isolates: how much temporal gradient flow along the main stream chain
is needed for quality?
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _rms(tensor: Tensor) -> float:
    return torch.sqrt(torch.mean(tensor.detach().float().square())).item()


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError(f"Mix probability must be between 0 and 1, got {probability}.")
    return math.log(probability / (1.0 - probability))


@dataclass(frozen=True)
class PartialDetachConfig:
    context_size: int
    d_model: int
    feedforward_dim: int
    detach_every_n: int  # detach stream every N timesteps; 1 = full stop-grad, 0 = never
    temporal_window: int = 4
    num_heads: int = 4
    token_mix_init: float = 0.5
    block_mix_init: float = 0.9
    time_mix_init: float = 0.9


class MixAdd(nn.Module):
    def __init__(self, *, init: float) -> None:
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.tensor(_logit(init), dtype=torch.float32))

    def coefficient(self) -> Tensor:
        return torch.sigmoid(self.alpha_logit)

    def coefficient_value(self) -> float:
        return self.coefficient().detach().item()

    def forward(self, stream: Tensor, delta: Tensor) -> Tensor:
        mix = self.coefficient().to(device=stream.device, dtype=stream.dtype)
        return (mix * stream) + ((1.0 - mix) * delta)


class TemporalWindowAttention(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(
        self,
        query_source: Tensor,
        past_states: Tensor,
    ) -> Tensor:
        if past_states.shape[1] == 0:
            return torch.zeros_like(query_source)

        batch_size, _, d_model = past_states.shape
        query = self.query(query_source).reshape(batch_size, self.num_heads, self.head_dim)
        keys = self.key(past_states).reshape(batch_size, past_states.shape[1], self.num_heads, self.head_dim)
        values = self.value(past_states).reshape(batch_size, past_states.shape[1], self.num_heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        logits = torch.einsum("bhd,bhwd->bhw", query, keys) * (self.head_dim ** -0.5)
        weights = torch.softmax(logits, dim=-1)
        attended = torch.einsum("bhw,bhwd->bhd", weights, values).reshape(batch_size, d_model)
        return self.output(attended)


class ResidualFeedForwardBlock(nn.Module):
    def __init__(self, *, d_model: int, feedforward_dim: int) -> None:
        super().__init__()
        self.proj_in = nn.Linear(d_model, feedforward_dim)
        self.activation = nn.GELU()
        self.proj_out = nn.Linear(feedforward_dim, d_model)

    def forward(self, stream: Tensor) -> Tensor:
        return self.proj_out(self.activation(self.proj_in(stream)))


class PartialDetachCharModel(nn.Module):
    def __init__(self, *, vocab_size: int, config: PartialDetachConfig) -> None:
        super().__init__()
        self.config = config
        self.context_size = config.context_size
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_size, config.d_model)
        self.mix_token = MixAdd(init=config.token_mix_init)
        self.mix_block = MixAdd(init=config.block_mix_init)
        self.mix_time = MixAdd(init=config.time_mix_init)
        self.temporal_attention = TemporalWindowAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
        )
        self.block = ResidualFeedForwardBlock(
            d_model=config.d_model,
            feedforward_dim=config.feedforward_dim,
        )
        self.output = nn.Linear(config.d_model, vocab_size)

    def embedded_tokens(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.shape[1]
        if sequence_length != self.context_size:
            raise ValueError(f"Expected context length {self.context_size}, got {sequence_length}.")
        positions = torch.arange(sequence_length, device=tokens.device)
        return self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)

    def mix_coefficients(self) -> dict[str, float]:
        return {
            "token": self.mix_token.coefficient_value(),
            "block": self.mix_block.coefficient_value(),
            "time": self.mix_time.coefficient_value(),
        }

    def forward(self, tokens: Tensor) -> Tensor:
        embeddings = self.embedded_tokens(tokens)
        batch_size = tokens.shape[0]
        stream = torch.zeros(batch_size, self.config.d_model, device=tokens.device, dtype=embeddings.dtype)
        history: list[Tensor] = []
        detach_n = self.config.detach_every_n

        for time_index in range(self.context_size):
            # Partial detach: cut gradient every N steps along the stream chain.
            # detach_n=1 is full stop-grad; detach_n=0 means never detach (full gradient).
            if detach_n > 0 and time_index > 0 and time_index % detach_n == 0:
                stream = stream.detach()

            block_input = self.mix_token(stream, embeddings[:, time_index, :])

            # Attention over past states (always detached to bound graph size).
            past_states = history[-self.config.temporal_window:]
            if past_states:
                stacked_past = torch.stack(past_states, dim=1)
            else:
                stacked_past = torch.empty(
                    batch_size, 0, self.config.d_model,
                    device=tokens.device, dtype=embeddings.dtype,
                )
            temporal_context = self.temporal_attention(block_input, stacked_past)

            block_delta = self.block(block_input)
            post_block = self.mix_block(block_input, block_delta)
            stream = self.mix_time(post_block, temporal_context)

            # History always detached — prevents unbounded graph from attention reads.
            history.append(stream.detach())

        return self.output(stream)
