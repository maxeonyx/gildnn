from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from core.tied_readout import CausalSelfAttention, FeedForward, normalize_hidden, tied_logits


@dataclass(frozen=True)
class RecurrentDepthConfig:
    context_size: int
    d_model: int
    n_heads: int
    ff_dim: int
    iterations: int
    temperature: float
    dropout: float = 0.0
    normalize: bool = True


@dataclass(frozen=True)
class HaltingOutput:
    logits: Float[Tensor, "batch vocab"]
    halt_predictions: Float[Tensor, "batch depth"]
    calibrated_halt_predictions: Float[Tensor, "batch depth"]
    halt_depths: Int[Tensor, "batch"]


class SharedRecurrentCore(nn.Module):
    """Shared transformer block iterated with per-depth norms."""

    def __init__(self, *, d_model: int, n_heads: int, ff_dim: int, dropout: float, iterations: int) -> None:
        super().__init__()
        if iterations <= 0:
            raise ValueError(f"iterations must be positive, got {iterations}")
        self.iterations = iterations
        self.attn_norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(iterations))
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(iterations))
        self.ffn = FeedForward(d_model=d_model, ff_dim=ff_dim)
        self.ffn_dropout = nn.Dropout(dropout)

    def step(
        self,
        hidden: Float[Tensor, "batch context d_model"],
        *,
        iteration_index: int,
    ) -> Float[Tensor, "batch context d_model"]:
        hidden = hidden + self.attn_dropout(self.attn(self.attn_norms[iteration_index](hidden)))
        hidden = hidden + self.ffn_dropout(self.ffn(self.ffn_norms[iteration_index](hidden)))
        return hidden

    def forward(
        self,
        hidden: Float[Tensor, "batch context d_model"],
        *,
        collect_iteration_states: bool,
    ) -> tuple[Float[Tensor, "batch context d_model"], list[Float[Tensor, "batch context d_model"]]]:
        iteration_states: list[Float[Tensor, "batch context d_model"]] = []
        for iteration_index in range(self.iterations):
            hidden = self.step(hidden, iteration_index=iteration_index)
            if collect_iteration_states:
                iteration_states.append(hidden)
        return hidden, iteration_states


class HaltHead(nn.Module):
    """Regression head predicting remaining loss gain for the current depth."""

    def __init__(self, *, d_model: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.input_proj = nn.Linear(d_model + 1, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        last_hidden: Float[Tensor, "batch d_model"],
        *,
        depth_index: int,
        total_depth: int,
    ) -> Float[Tensor, "batch"]:
        normalized_hidden = self.norm(last_hidden)
        depth_fraction = torch.full(
            (last_hidden.shape[0], 1),
            fill_value=depth_index / total_depth,
            device=last_hidden.device,
            dtype=normalized_hidden.dtype,
        )
        hidden = torch.cat([normalized_hidden, depth_fraction], dim=-1)
        hidden = torch.nn.functional.gelu(self.input_proj(hidden))
        return self.output_proj(hidden).squeeze(-1)


class RecurrentDepthLM(nn.Module):
    """Recurrent-depth language model with optional calibrated halting."""

    def __init__(
        self,
        *,
        vocab_size: int,
        config: RecurrentDepthConfig,
        calibration_params: dict[str, Any] | Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.temperature = config.temperature
        self.normalize = config.normalize
        self.calibration_params = calibration_params
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.core = SharedRecurrentCore(
            d_model=config.d_model,
            n_heads=config.n_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            iterations=config.iterations,
        )
        self.halt_head = HaltHead(d_model=config.d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.config.d_model**-0.5)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=self.config.d_model**-0.5)

    @property
    def halting_depth_count(self) -> int:
        return max(self.config.iterations - 1, 0)

    def embed(self, tokens: Int[Tensor, "batch context"]) -> Float[Tensor, "batch context d_model"]:
        if tokens.shape[1] != self.config.context_size:
            raise ValueError(f"Expected context {self.config.context_size}, got {tokens.shape[1]}")
        positions = torch.arange(self.config.context_size, device=tokens.device)
        token_hidden = self.token_embedding(tokens)
        if self.normalize:
            token_hidden = normalize_hidden(token_hidden)
        hidden = token_hidden + self.position_embedding(positions)
        return self.embedding_dropout(hidden)

    def iteration_states(
        self,
        tokens: Int[Tensor, "batch context"],
        *,
        collect_iteration_states: bool,
    ) -> tuple[Float[Tensor, "batch context d_model"], list[Float[Tensor, "batch context d_model"]]]:
        hidden = self.embed(tokens)
        return self.core(hidden, collect_iteration_states=collect_iteration_states)

    def lm_logits_from_hidden(self, last_hidden: Float[Tensor, "batch d_model"]) -> Float[Tensor, "batch vocab"]:
        return tied_logits(
            last_hidden,
            self.token_embedding,
            temperature=self.temperature,
            normalize=self.normalize,
        )

    def predicted_gain_from_hidden(
        self,
        last_hidden: Float[Tensor, "batch d_model"],
        *,
        depth_index: int,
    ) -> Float[Tensor, "batch"]:
        return self.halt_head(last_hidden, depth_index=depth_index, total_depth=self.config.iterations)

    def _stack_halt_predictions(
        self,
        predictions: list[Float[Tensor, "batch"]],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Float[Tensor, "batch depth"]:
        if len(predictions) == 0:
            return torch.empty((batch_size, 0), device=device, dtype=dtype)
        return torch.stack(predictions, dim=1)

    def halt_predictions_from_iteration_states(
        self,
        iteration_states: list[Float[Tensor, "batch context d_model"]],
    ) -> Float[Tensor, "batch depth"]:
        if len(iteration_states) != self.config.iterations:
            raise ValueError(
                f"Expected {self.config.iterations} iteration states, got {len(iteration_states)}"
            )
        predictions = [
            self.predicted_gain_from_hidden(iteration_hidden[:, -1, :], depth_index=depth_index)
            for depth_index, iteration_hidden in enumerate(iteration_states[:-1], start=1)
        ]
        batch_size = iteration_states[0].shape[0]
        return self._stack_halt_predictions(
            predictions,
            batch_size=batch_size,
            device=iteration_states[0].device,
            dtype=iteration_states[0].dtype,
        )

    def resolve_calibration(
        self,
        calibration: dict[str, Any] | Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Float[Tensor, "depth"], Float[Tensor, "depth"]]:
        depth_count = self.halting_depth_count
        scale = torch.ones(depth_count, device=device, dtype=dtype)
        bias = torch.zeros(depth_count, device=device, dtype=dtype)
        if calibration is None:
            return scale, bias
        if isinstance(calibration, dict):
            if "scale" in calibration:
                scale = torch.as_tensor(calibration["scale"], device=device, dtype=dtype)
            if "bias" in calibration:
                bias = torch.as_tensor(calibration["bias"], device=device, dtype=dtype)
        else:
            calibration_tensor = torch.as_tensor(calibration, device=device, dtype=dtype)
            if calibration_tensor.shape == (depth_count, 2):
                scale = calibration_tensor[:, 0]
                bias = calibration_tensor[:, 1]
            elif calibration_tensor.shape == (2, depth_count):
                scale = calibration_tensor[0]
                bias = calibration_tensor[1]
            else:
                raise ValueError(
                    "Calibration tensor must have shape "
                    f"({depth_count}, 2) or (2, {depth_count}), got {tuple(calibration_tensor.shape)}"
                )
        if scale.shape != (depth_count,):
            raise ValueError(f"Calibration scale must have shape ({depth_count},), got {tuple(scale.shape)}")
        if bias.shape != (depth_count,):
            raise ValueError(f"Calibration bias must have shape ({depth_count},), got {tuple(bias.shape)}")
        return scale, bias

    def apply_calibration(
        self,
        predictions: Float[Tensor, "batch depth"],
        calibration: dict[str, Any] | Tensor | None = None,
    ) -> Float[Tensor, "batch depth"]:
        scale, bias = self.resolve_calibration(
            self.calibration_params if calibration is None else calibration,
            device=predictions.device,
            dtype=predictions.dtype,
        )
        return predictions * scale.unsqueeze(0) + bias.unsqueeze(0)

    def forward(
        self,
        tokens: Int[Tensor, "batch context"],
    ) -> tuple[Float[Tensor, "batch vocab"], Float[Tensor, "batch depth"]]:
        hidden, iteration_states = self.iteration_states(tokens, collect_iteration_states=True)
        logits = self.lm_logits_from_hidden(hidden[:, -1, :])
        halt_predictions = self.halt_predictions_from_iteration_states(iteration_states)
        return logits, halt_predictions

    def forward_with_halting(
        self,
        tokens: Int[Tensor, "batch context"],
        *,
        epsilon: float,
        calibration: dict[str, Any] | Tensor | None = None,
    ) -> HaltingOutput:
        hidden = self.embed(tokens)
        batch_size = tokens.shape[0]
        device = hidden.device
        dtype = hidden.dtype
        raw_predictions = torch.full((batch_size, self.halting_depth_count), torch.nan, device=device, dtype=dtype)
        calibrated_predictions = torch.full(
            (batch_size, self.halting_depth_count),
            torch.nan,
            device=device,
            dtype=dtype,
        )
        final_last_hidden = torch.empty((batch_size, self.config.d_model), device=device, dtype=dtype)
        halt_depths = torch.full((batch_size,), self.config.iterations, device=device, dtype=torch.int64)
        halted_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        scale, bias = self.resolve_calibration(
            self.calibration_params if calibration is None else calibration,
            device=device,
            dtype=dtype,
        )

        for depth_index in range(1, self.config.iterations + 1):
            active_indices = (~halted_mask).nonzero(as_tuple=False).squeeze(-1)
            if active_indices.numel() == 0:
                break
            updated_hidden = self.core.step(hidden.index_select(0, active_indices), iteration_index=depth_index - 1)
            hidden = hidden.clone()
            hidden[active_indices] = updated_hidden
            last_hidden = updated_hidden[:, -1, :]

            if depth_index == self.config.iterations:
                final_last_hidden[active_indices] = last_hidden
                halt_depths[active_indices] = depth_index
                continue

            raw_gain = self.predicted_gain_from_hidden(last_hidden, depth_index=depth_index)
            calibrated_gain = (raw_gain * scale[depth_index - 1]) + bias[depth_index - 1]
            raw_predictions[active_indices, depth_index - 1] = raw_gain
            calibrated_predictions[active_indices, depth_index - 1] = calibrated_gain

            newly_halted_local = calibrated_gain < epsilon
            newly_halted_indices = active_indices[newly_halted_local]
            if newly_halted_indices.numel() > 0:
                final_last_hidden[newly_halted_indices] = last_hidden[newly_halted_local]
                halt_depths[newly_halted_indices] = depth_index
                halted_mask[newly_halted_indices] = True

        logits = self.lm_logits_from_hidden(final_last_hidden)
        return HaltingOutput(
            logits=logits,
            halt_predictions=raw_predictions,
            calibrated_halt_predictions=calibrated_predictions,
            halt_depths=halt_depths,
        )


__all__ = [
    "HaltHead",
    "HaltingOutput",
    "RecurrentDepthConfig",
    "RecurrentDepthLM",
    "SharedRecurrentCore",
]
