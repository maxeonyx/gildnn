from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from experiments.residual_stream_time.model import TemporalWindowAttention, _rms


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@dataclass(frozen=True)
class ResidualStepTrace:
    time_index: int
    history_length: int
    stream_rms: float
    block1_delta_rms: float
    diagonal_input_rms: float
    block2_delta_rms: float
    temporal_context_rms: float
    attention_weights: list[float]


@dataclass(frozen=True)
class ResidualRunTrace:
    step_traces: list[ResidualStepTrace]
    final_stream_rms: float


@dataclass(frozen=True)
class DiagonalResidualStreamTimeConfig:
    context_size: int
    d_model: int
    feedforward_dim: int
    temporal_window: int = 4
    num_heads: int = 4


class ResidualFeedForward(nn.Module):
    def __init__(self, *, d_model: int, feedforward_dim: int) -> None:
        super().__init__()
        self.proj_in = nn.Linear(d_model, feedforward_dim)
        self.activation = nn.GELU()
        self.proj_out = nn.Linear(feedforward_dim, d_model)

    def forward(self, stream: Tensor) -> Tensor:
        return self.proj_out(self.activation(self.proj_in(stream)))


class DiagonalResidualStreamTimeCharModel(nn.Module):
    def __init__(self, *, vocab_size: int, config: DiagonalResidualStreamTimeConfig) -> None:
        super().__init__()
        self.config = config
        self.context_size = config.context_size
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_size, config.d_model)
        self.stream_norm = nn.LayerNorm(config.d_model)
        self.temporal_attention = TemporalWindowAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
        )
        self.block1_norm = nn.LayerNorm(config.d_model)
        self.block1 = ResidualFeedForward(
            d_model=config.d_model,
            feedforward_dim=config.feedforward_dim,
        )
        self.block2_norm = nn.LayerNorm(config.d_model)
        self.block2 = ResidualFeedForward(
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
        d_model = self.config.d_model
        stream = torch.zeros(batch_size, d_model, device=tokens.device, dtype=embeddings.dtype)
        previous_block1_delta = torch.zeros_like(stream)
        history: list[Tensor] = []
        step_traces: list[ResidualStepTrace] = []

        for time_index in range(self.context_size):
            normalized_stream = self.stream_norm(stream)
            base_input = normalized_stream + embeddings[:, time_index, :]
            past_states = history[-self.config.temporal_window :]
            if past_states:
                stacked_past = torch.stack(past_states, dim=1)
            else:
                stacked_past = torch.empty(
                    batch_size,
                    0,
                    d_model,
                    device=tokens.device,
                    dtype=embeddings.dtype,
                )
            temporal_context, attention_weights = self.temporal_attention(
                base_input,
                stacked_past,
                capture_weights=capture_trace,
            )
            block1_input = base_input + temporal_context
            block1_delta = self.block1(self.block1_norm(block1_input))
            diagonal_input = previous_block1_delta
            block2_input = block1_input + block1_delta + diagonal_input
            block2_delta = self.block2(self.block2_norm(block2_input))
            stream = block2_input + block2_delta
            history.append(stream)
            previous_block1_delta = block1_delta

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
                        block1_delta_rms=_rms(block1_delta),
                        diagonal_input_rms=_rms(diagonal_input),
                        block2_delta_rms=_rms(block2_delta),
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
