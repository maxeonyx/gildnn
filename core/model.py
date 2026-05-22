from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from einops import rearrange, repeat
from jaxtyping import Float, Int
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
class ResidualStepTrace:
    time_index: int
    history_length: int
    stream_rms: float
    block_delta_rms: float
    temporal_context_rms: float
    attention_weights: list[float]
    mix_token: float
    mix_block: float
    mix_time: float


@dataclass(frozen=True)
class ResidualRunTrace:
    step_traces: list[ResidualStepTrace]
    final_stream_rms: float
    mix_coefficients: dict[str, float]


@dataclass(frozen=True)
class ResidualStreamTimeMixAddConfig:
    context_size: int
    d_model: int
    feedforward_dim: int
    temporal_window: int = 4
    num_heads: int = 4
    token_mix_init: float = 0.5
    block_mix_init: float = 0.9
    time_mix_init: float = 0.9


@dataclass(frozen=True)
class MultiRateBlockTrace:
    block_index: int
    rate: int
    executed: bool
    used_delta_rms: float
    diagonal_input_rms: float
    diagonal_applied: bool
    diagonal_matches_previous_delta: bool


@dataclass(frozen=True)
class MultiRateTimeStepTrace:
    time_index: int
    blocks: list[MultiRateBlockTrace]


@dataclass(frozen=True)
class MultiRateForwardTrace:
    step_traces: list[MultiRateTimeStepTrace]


class MixAdd(nn.Module):
    def __init__(self, *, init: float) -> None:
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.tensor(_logit(init), dtype=torch.float32))

    def coefficient(self) -> Tensor:
        return torch.sigmoid(self.alpha_logit)

    @torch.no_grad()
    def coefficient_value(self) -> float:
        return self.coefficient().item()

    def forward(
        self,
        stream: Float[Tensor, "batch d_model"],
        delta: Float[Tensor, "batch d_model"],
    ) -> Float[Tensor, "batch d_model"]:
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
        query_source: Float[Tensor, "batch d_model"],
        past_states: Float[Tensor, "batch window d_model"],
        *,
        capture_weights: bool = False,
    ) -> tuple[
        Float[Tensor, "batch d_model"],
        Float[Tensor, "batch num_heads window"] | None,
    ]:
        if past_states.shape[1] == 0:
            return torch.zeros_like(query_source), None

        batch_size, _, d_model = past_states.shape
        query = rearrange(
            self.query(query_source),
            "batch (num_heads head_dim) -> batch num_heads head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        keys = rearrange(
            self.key(past_states),
            "batch window (num_heads head_dim) -> batch num_heads window head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        values = rearrange(
            self.value(past_states),
            "batch window (num_heads head_dim) -> batch num_heads window head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        logits = torch.einsum("bhd,bhwd->bhw", query, keys) * (self.head_dim ** -0.5)
        weights = torch.softmax(logits, dim=-1)
        attended = rearrange(
            torch.einsum("bhw,bhwd->bhd", weights, values),
            "batch num_heads head_dim -> batch (num_heads head_dim)",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        context = self.output(attended)
        if not capture_weights:
            return context, None
        return context, weights


class ResidualFeedForwardBlock(nn.Module):
    def __init__(self, *, d_model: int, feedforward_dim: int) -> None:
        super().__init__()
        self.proj_in = nn.Linear(d_model, feedforward_dim)
        self.activation = nn.GELU()
        self.proj_out = nn.Linear(feedforward_dim, d_model)

    def forward(
        self,
        stream: Float[Tensor, "batch d_model"],
    ) -> Float[Tensor, "batch d_model"]:
        return self.proj_out(self.activation(self.proj_in(stream)))


class ResidualStreamTimeMixAddCharModel(nn.Module):
    def __init__(self, *, vocab_size: int, config: ResidualStreamTimeMixAddConfig) -> None:
        super().__init__()
        self.config = config
        self.context_size = config.context_size
        self.d_model = config.d_model
        self.feedforward_dim = config.feedforward_dim
        self.temporal_window = config.temporal_window
        self.num_heads = config.num_heads
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

    def embedded_tokens(self, tokens: Int[Tensor, "batch context"]) -> Float[Tensor, "batch context d_model"]:
        sequence_length = tokens.shape[1]
        if sequence_length != self.context_size:
            raise ValueError(f"Expected context length {self.context_size}, got {sequence_length}.")
        positions = torch.arange(sequence_length, device=tokens.device)
        return self.token_embedding(tokens) + repeat(
            self.position_embedding(positions),
            "context d_model -> batch context d_model",
            batch=tokens.shape[0],
        )

    @torch.no_grad()
    def mix_coefficients(self) -> dict[str, float]:
        return {
            "token": self.mix_token.coefficient_value(),
            "block": self.mix_block.coefficient_value(),
            "time": self.mix_time.coefficient_value(),
        }

    def _run(self, tokens: Tensor, *, capture_trace: bool) -> tuple[Tensor, ResidualRunTrace | None]:
        embeddings = self.embedded_tokens(tokens)
        batch_size = tokens.shape[0]
        stream = torch.zeros(batch_size, self.config.d_model, device=tokens.device, dtype=embeddings.dtype)
        history: list[Tensor] = []
        step_traces: list[ResidualStepTrace] = []

        for time_index in range(self.context_size):
            block_input = self.mix_token(stream, embeddings[:, time_index, :])
            past_states = history[-self.config.temporal_window :]
            if len(past_states) == 0:
                stacked_past = torch.empty(
                    batch_size,
                    0,
                    self.config.d_model,
                    device=tokens.device,
                    dtype=embeddings.dtype,
                )
            else:
                stacked_past = torch.stack(past_states, dim=1)
            temporal_context, attention_weights = self.temporal_attention(
                block_input,
                stacked_past,
                capture_weights=capture_trace,
            )
            block_delta = self.block(block_input)
            post_block = self.mix_block(block_input, block_delta)
            stream = self.mix_time(post_block, temporal_context)
            history.append(stream)
            if capture_trace:
                if attention_weights is None:
                    first_attention = []
                else:
                    first_attention = attention_weights[0].mean(dim=0).detach().cpu().tolist()
                mix_coefficients = self.mix_coefficients()
                step_traces.append(
                    ResidualStepTrace(
                        time_index=time_index,
                        history_length=stacked_past.shape[1],
                        stream_rms=_rms(stream),
                        block_delta_rms=_rms(block_delta),
                        temporal_context_rms=_rms(temporal_context),
                        attention_weights=[round(weight, 6) for weight in first_attention],
                        mix_token=mix_coefficients["token"],
                        mix_block=mix_coefficients["block"],
                        mix_time=mix_coefficients["time"],
                    )
                )

        logits = self.output(stream)
        if not capture_trace:
            return logits, None
        return logits, ResidualRunTrace(
            step_traces=step_traces,
            final_stream_rms=_rms(stream),
            mix_coefficients=self.mix_coefficients(),
        )

    def forward(self, tokens: Int[Tensor, "batch context"]) -> Float[Tensor, "batch vocab"]:
        logits, _ = self._run(tokens, capture_trace=False)
        return logits

    @torch.no_grad()
    def forward_with_trace(
        self,
        tokens: Int[Tensor, "batch context"],
    ) -> tuple[Float[Tensor, "batch vocab"], ResidualRunTrace]:
        logits, trace = self._run(tokens, capture_trace=True)
        if trace is None:
            raise RuntimeError("Trace missing.")
        return logits, trace


class MultiRateResidualModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        context_size: int,
        d_model: int,
        feedforward_dim: int,
        temporal_window: int,
        num_heads: int,
        rates: tuple[int, ...],
        diagonal_mode: str = "off",
        diagonal_enabled: bool | None = None,
        token_mix_init: float = 0.5,
        block_mix_init: float = 0.9,
        time_mix_init: float = 0.9,
    ) -> None:
        super().__init__()
        if len(rates) == 0:
            raise ValueError("MultiRateResidualModel requires at least one block rate.")
        if any(rate <= 0 for rate in rates):
            raise ValueError(f"Block rates must be positive, got {rates}.")
        if diagonal_enabled is not None:
            diagonal_mode = "raw" if diagonal_enabled else "off"
        valid_diagonal_modes = {"off", "raw", "scaled"}
        if diagonal_mode not in valid_diagonal_modes:
            raise ValueError(
                "MultiRateResidualModel diagonal_mode must be one of "
                f"{sorted(valid_diagonal_modes)}, got {diagonal_mode!r}."
            )
        self.context_size = context_size
        self.d_model = d_model
        self.feedforward_dim = feedforward_dim
        self.temporal_window = temporal_window
        self.num_heads = num_heads
        self.rates = tuple(rates)
        self.diagonal_mode = diagonal_mode
        self.diagonal_enabled = diagonal_mode != "off"
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.mix_token = MixAdd(init=token_mix_init)
        self.mix_time = MixAdd(init=time_mix_init)
        self.temporal_attention = TemporalWindowAttention(
            d_model=d_model,
            num_heads=num_heads,
        )
        self.blocks = nn.ModuleList(
            [
                ResidualFeedForwardBlock(
                    d_model=d_model,
                    feedforward_dim=feedforward_dim,
                )
                for _ in self.rates
            ]
        )
        self.block_mixes = nn.ModuleList([MixAdd(init=block_mix_init) for _ in self.rates])
        self.diagonal_scales = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, dtype=torch.float32)) for _ in range(len(self.rates) - 1)]
        )
        self.output = nn.Linear(d_model, vocab_size)

    def embedded_tokens(self, tokens: Int[Tensor, "batch context"]) -> Float[Tensor, "batch context d_model"]:
        sequence_length = tokens.shape[1]
        if sequence_length != self.context_size:
            raise ValueError(f"Expected context length {self.context_size}, got {sequence_length}.")
        positions = torch.arange(sequence_length, device=tokens.device)
        return self.token_embedding(tokens) + repeat(
            self.position_embedding(positions),
            "context d_model -> batch context d_model",
            batch=tokens.shape[0],
        )

    @torch.no_grad()
    def mix_coefficients(self) -> dict[str, object]:
        return {
            "token": self.mix_token.coefficient_value(),
            "time": self.mix_time.coefficient_value(),
            "blocks": [block_mix.coefficient_value() for block_mix in self.block_mixes],
            "diagonal_mode": self.diagonal_mode,
            "diagonal_enabled": self.diagonal_enabled,
        }

    def _run(self, tokens: Tensor, *, capture_trace: bool) -> tuple[Tensor, MultiRateForwardTrace | None]:
        embeddings = self.embedded_tokens(tokens)
        batch_size = tokens.shape[0]
        stream = torch.zeros(batch_size, self.d_model, device=tokens.device, dtype=embeddings.dtype)
        history: list[Tensor] = []
        cached_outputs = [torch.zeros_like(stream) for _ in self.blocks]
        previous_timestep_deltas = [torch.zeros_like(stream) for _ in self.blocks]
        step_traces: list[MultiRateTimeStepTrace] = []

        for time_index in range(self.context_size):
            stream = self.mix_token(stream, embeddings[:, time_index, :])
            past_states = history[-self.temporal_window :]
            if len(past_states) == 0:
                stacked_past = torch.empty(
                    batch_size,
                    0,
                    self.d_model,
                    device=tokens.device,
                    dtype=embeddings.dtype,
                )
            else:
                stacked_past = torch.stack(past_states, dim=1)
            temporal_context, _ = self.temporal_attention(stream, stacked_past, capture_weights=False)

            current_timestep_deltas: list[Tensor] = []
            block_traces: list[MultiRateBlockTrace] = []
            for block_index, (block, block_mix, rate) in enumerate(
                zip(self.blocks, self.block_mixes, self.rates, strict=True)
            ):
                has_diagonal_input = self.diagonal_mode != "off" and block_index > 0
                if has_diagonal_input:
                    diagonal_input = previous_timestep_deltas[block_index - 1]
                    if self.diagonal_mode == "scaled":
                        alpha = self.diagonal_scales[block_index - 1].to(
                            device=stream.device,
                            dtype=stream.dtype,
                        )
                        block_input = stream + (alpha * diagonal_input)
                    else:
                        block_input = stream + diagonal_input
                else:
                    diagonal_input = torch.zeros_like(stream)
                    block_input = stream
                should_execute = time_index % rate == 0
                if should_execute:
                    cached_outputs[block_index] = block(block_input)
                block_output = cached_outputs[block_index]
                stream = block_mix(stream, block_output)
                current_timestep_deltas.append(block_output)
                if capture_trace:
                    block_traces.append(
                        MultiRateBlockTrace(
                            block_index=block_index,
                            rate=rate,
                            executed=should_execute,
                            used_delta_rms=_rms(block_output),
                            diagonal_input_rms=_rms(diagonal_input),
                            diagonal_applied=has_diagonal_input and should_execute,
                            diagonal_matches_previous_delta=True,
                        )
                    )

            previous_timestep_deltas = current_timestep_deltas
            stream = self.mix_time(stream, temporal_context)
            history.append(stream)
            if capture_trace:
                step_traces.append(MultiRateTimeStepTrace(time_index=time_index, blocks=block_traces))

        logits = self.output(stream)
        if not capture_trace:
            return logits, None
        return logits, MultiRateForwardTrace(step_traces=step_traces)

    def forward(self, tokens: Int[Tensor, "batch context"]) -> Float[Tensor, "batch vocab"]:
        logits, _ = self._run(tokens, capture_trace=False)
        return logits

    @torch.no_grad()
    def forward_with_trace(
        self,
        tokens: Int[Tensor, "batch context"],
    ) -> tuple[Float[Tensor, "batch vocab"], MultiRateForwardTrace]:
        logits, trace = self._run(tokens, capture_trace=True)
        if trace is None:
            raise RuntimeError("Trace missing.")
        return logits, trace


class ParallelDiagonalModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        context_size: int,
        d_model: int,
        feedforward_dim: int,
        num_blocks: int,
        internal_steps: int = 1,
        readout_mode: str = "last",
        token_mix_init: float = 0.5,
        block_mix_init: float = 0.9,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError(f"ParallelDiagonalModel requires at least one block, got {num_blocks}.")
        if internal_steps <= 0:
            raise ValueError(
                f"ParallelDiagonalModel requires at least one internal step, got {internal_steps}."
            )
        valid_readout_modes = {"last", "all"}
        if readout_mode not in valid_readout_modes:
            raise ValueError(
                "ParallelDiagonalModel readout_mode must be one of "
                f"{sorted(valid_readout_modes)}, got {readout_mode!r}."
            )
        self.context_size = context_size
        self.d_model = d_model
        self.feedforward_dim = feedforward_dim
        self.num_blocks = num_blocks
        self.internal_steps = internal_steps
        self.readout_mode = readout_mode
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.token_mixes = nn.ModuleList([MixAdd(init=token_mix_init) for _ in range(num_blocks)])
        self.blocks = nn.ModuleList(
            [
                ResidualFeedForwardBlock(
                    d_model=d_model,
                    feedforward_dim=feedforward_dim,
                )
                for _ in range(num_blocks)
            ]
        )
        self.block_mixes = nn.ModuleList([MixAdd(init=block_mix_init) for _ in range(num_blocks)])
        self.readout_logits = (
            nn.Parameter(torch.zeros(num_blocks, dtype=torch.float32))
            if readout_mode == "all"
            else None
        )
        self.output = nn.Linear(d_model, vocab_size)

    def embedded_tokens(self, tokens: Int[Tensor, "batch context"]) -> Float[Tensor, "batch context d_model"]:
        sequence_length = tokens.shape[1]
        if sequence_length != self.context_size:
            raise ValueError(f"Expected context length {self.context_size}, got {sequence_length}.")
        positions = torch.arange(sequence_length, device=tokens.device)
        return self.token_embedding(tokens) + repeat(
            self.position_embedding(positions),
            "context d_model -> batch context d_model",
            batch=tokens.shape[0],
        )

    @torch.no_grad()
    def mix_coefficients(self) -> dict[str, object]:
        readout_weights = None
        if self.readout_logits is not None:
            readout_weights = torch.softmax(self.readout_logits, dim=0).detach().cpu().tolist()
        return {
            "token": [mix.coefficient_value() for mix in self.token_mixes],
            "blocks": [mix.coefficient_value() for mix in self.block_mixes],
            "internal_steps": self.internal_steps,
            "readout_mode": self.readout_mode,
            "readout_weights": readout_weights,
        }

    def _readout_state(
        self,
        states: list[Float[Tensor, "batch d_model"]],
    ) -> Float[Tensor, "batch d_model"]:
        if self.readout_mode == "last":
            return states[-1]
        if self.readout_logits is None:
            raise RuntimeError("readout_logits missing for readout_mode='all'.")
        stacked_states = torch.stack(states, dim=1)
        readout_weights = torch.softmax(
            self.readout_logits.to(device=stacked_states.device, dtype=stacked_states.dtype),
            dim=0,
        )
        return (stacked_states * rearrange(readout_weights, "blocks -> 1 blocks 1")).sum(dim=1)

    def forward(self, tokens: Int[Tensor, "batch context"]) -> Float[Tensor, "batch vocab"]:
        embeddings = self.embedded_tokens(tokens)
        batch_size = tokens.shape[0]
        previous_states = [
            torch.zeros(batch_size, self.d_model, device=tokens.device, dtype=embeddings.dtype)
            for _ in range(self.num_blocks)
        ]

        for time_index in range(self.context_size):
            token_state = embeddings[:, time_index, :]
            seeded_states = [
                token_mix(previous_state, token_state)
                for token_mix, previous_state in zip(self.token_mixes, previous_states, strict=True)
            ]
            current_states = previous_states
            for internal_step in range(self.internal_steps):
                next_states: list[Tensor] = []
                for block_index, (block, block_mix) in enumerate(
                    zip(self.blocks, self.block_mixes, strict=True)
                ):
                    state_input = seeded_states[block_index] if internal_step == 0 else current_states[block_index]
                    if block_index == 0:
                        block_input = state_input
                    else:
                        if internal_step == 0:
                            neighbor_state = previous_states[block_index - 1]
                        else:
                            neighbor_state = current_states[block_index - 1]
                        block_input = 0.5 * (state_input + neighbor_state)
                    block_delta = block(block_input)
                    next_states.append(block_mix(block_input, block_delta))
                current_states = next_states
            previous_states = current_states

        return self.output(self._readout_state(previous_states))
