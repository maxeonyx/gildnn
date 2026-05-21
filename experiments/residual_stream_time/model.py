from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _rms(tensor: Tensor) -> float:
    return torch.sqrt(torch.mean(tensor.detach().float().square())).item()


@dataclass(frozen=True)
class ResidualStepTrace:
    time_index: int
    history_length: int
    stream_rms: float
    block_delta_rms: float
    temporal_context_rms: float
    attention_weights: list[float]


@dataclass(frozen=True)
class ResidualRunTrace:
    step_traces: list[ResidualStepTrace]
    final_stream_rms: float


@dataclass(frozen=True)
class ResidualStreamTimeConfig:
    context_size: int
    d_model: int
    feedforward_dim: int
    temporal_window: int = 4
    num_heads: int = 4


class TemporalWindowAttention(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(
        self,
        query_source: Tensor,
        past_states: Tensor,
        *,
        capture_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if past_states.shape[1] == 0:
            return torch.zeros_like(query_source), None

        batch_size, _, d_model = past_states.shape
        normalized_query = self.query_norm(query_source)
        normalized_memory = self.memory_norm(past_states)
        query = self.query(normalized_query).reshape(batch_size, self.num_heads, self.head_dim)
        keys = self.key(normalized_memory).reshape(batch_size, past_states.shape[1], self.num_heads, self.head_dim)
        values = self.value(normalized_memory).reshape(batch_size, past_states.shape[1], self.num_heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        logits = torch.einsum("bhd,bhwd->bhw", query, keys) * (self.head_dim ** -0.5)
        weights = torch.softmax(logits, dim=-1)
        attended = torch.einsum("bhw,bhwd->bhd", weights, values).reshape(batch_size, d_model)
        context = self.output(attended)
        if not capture_weights:
            return context, None
        return context, weights


class ResidualFeedForwardBlock(nn.Module):
    def __init__(self, *, d_model: int, feedforward_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj_in = nn.Linear(d_model, feedforward_dim)
        self.activation = nn.GELU()
        self.proj_out = nn.Linear(feedforward_dim, d_model)

    def forward(self, stream: Tensor) -> Tensor:
        normalized = self.norm(stream)
        return self.proj_out(self.activation(self.proj_in(normalized)))


class ResidualStreamTimeCharModel(nn.Module):
    def __init__(self, *, vocab_size: int, config: ResidualStreamTimeConfig) -> None:
        super().__init__()
        self.config = config
        self.context_size = config.context_size
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_size, config.d_model)
        self.temporal_attention = TemporalWindowAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
        )
        self.block = ResidualFeedForwardBlock(
            d_model=config.d_model,
            feedforward_dim=config.feedforward_dim,
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size)

    def embedded_tokens(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.shape[1]
        if sequence_length != self.context_size:
            raise ValueError(f"Expected context length {self.context_size}, got {sequence_length}.")
        positions = torch.arange(sequence_length, device=tokens.device)
        return self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)

    def _run(self, tokens: Tensor, *, capture_trace: bool) -> tuple[Tensor, ResidualRunTrace | None]:
        embeddings = self.embedded_tokens(tokens)
        batch_size = tokens.shape[0]
        stream = torch.zeros(batch_size, self.config.d_model, device=tokens.device, dtype=embeddings.dtype)
        history: list[Tensor] = []
        step_traces: list[ResidualStepTrace] = []

        for time_index in range(self.context_size):
            block_input = stream + embeddings[:, time_index, :]
            past_states = history[-self.config.temporal_window :]
            if past_states:
                stacked_past = torch.stack(past_states, dim=1)
            else:
                stacked_past = torch.empty(
                    batch_size,
                    0,
                    self.config.d_model,
                    device=tokens.device,
                    dtype=embeddings.dtype,
                )
            temporal_context, attention_weights = self.temporal_attention(
                block_input,
                stacked_past,
                capture_weights=capture_trace,
            )
            block_delta = self.block(block_input)
            stream = block_input + block_delta + temporal_context
            history.append(stream)
            if capture_trace:
                if attention_weights is None:
                    first_attention = []
                else:
                    first_attention = attention_weights[0].mean(dim=0).detach().cpu().tolist()
                step_traces.append(
                    ResidualStepTrace(
                        time_index=time_index,
                        history_length=stacked_past.shape[1],
                        stream_rms=_rms(stream),
                        block_delta_rms=_rms(block_delta),
                        temporal_context_rms=_rms(temporal_context),
                        attention_weights=[round(weight, 6) for weight in first_attention],
                    )
                )

        logits = self.output(self.final_norm(stream))
        if not capture_trace:
            return logits, None
        return logits, ResidualRunTrace(
            step_traces=step_traces,
            final_stream_rms=_rms(stream),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        logits, _ = self._run(tokens, capture_trace=False)
        return logits

    def forward_with_trace(self, tokens: Tensor) -> tuple[Tensor, ResidualRunTrace]:
        logits, trace = self._run(tokens, capture_trace=True)
        if trace is None:
            raise RuntimeError("Trace missing.")
        return logits, trace
