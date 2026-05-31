from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F


def l2_normalize(hidden: Tensor) -> Tensor:
    return F.normalize(hidden.float(), dim=-1, eps=1e-6).to(hidden.dtype)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@dataclass(frozen=True)
class ParameterSlice:
    parameter: nn.Parameter
    level: int

    @property
    def tensor(self) -> Tensor:
        return self.parameter[self.level]


@dataclass(frozen=True)
class AutomatonOutput:
    logits: Float[Tensor, "batch seq vocab"]
    prediction_losses: Float[Tensor, "levels"]
    prediction_counts: Int[Tensor, "levels"]
    total_prediction_loss: Float[Tensor, ""]


class CellularAutomaton(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_stream: int = 128,
        n_levels: int = 8,
        steps_per_token: int = 8,
        rates: list[int] | tuple[int, ...] | None = None,
        noise_std: float = 0.1,
        d_hidden: int | None = None,
        readout_temperature: float = 0.07,
        loss_type: str = "mse",
        info_nce_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
        if d_stream <= 0:
            raise ValueError(f"d_stream must be positive, got {d_stream}.")
        if n_levels <= 0:
            raise ValueError(f"n_levels must be positive, got {n_levels}.")
        if steps_per_token <= 0:
            raise ValueError(f"steps_per_token must be positive, got {steps_per_token}.")
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}.")
        if readout_temperature <= 0.0:
            raise ValueError(f"readout_temperature must be positive, got {readout_temperature}.")
        if loss_type not in {"mse", "infonce"}:
            raise ValueError(f"loss_type must be 'mse' or 'infonce', got {loss_type!r}.")
        if info_nce_temperature <= 0.0:
            raise ValueError(f"info_nce_temperature must be positive, got {info_nce_temperature}.")

        resolved_rates = tuple(rates) if rates is not None else tuple(2**level for level in range(n_levels))
        if len(resolved_rates) != n_levels:
            raise ValueError(f"Expected {n_levels} rates, got {len(resolved_rates)}.")
        if any(rate <= 0 for rate in resolved_rates):
            raise ValueError(f"All rates must be positive, got {resolved_rates}.")

        self.vocab_size = vocab_size
        self.d_stream = d_stream
        self.n_levels = n_levels
        self.steps_per_token = steps_per_token
        self.rates = resolved_rates
        self.register_buffer("rate_tensor", torch.tensor(resolved_rates, dtype=torch.long), persistent=False)
        # Mask for skipping level 0 in local prediction loss (CUDA-graph-compatible: no CPU→CUDA copy at runtime)
        level0_mask = torch.ones(n_levels, dtype=torch.bool)
        level0_mask[0] = False
        self.register_buffer("_skip_level0", level0_mask, persistent=False)
        self.noise_std = noise_std
        self.d_hidden = d_hidden if d_hidden is not None else 4 * d_stream
        self.readout_temperature = readout_temperature
        self.loss_type = loss_type
        self.info_nce_temperature = info_nce_temperature
        self.contrastive_buffer_size = 64

        self.token_embedding = nn.Embedding(vocab_size, d_stream)
        self.w1 = nn.Parameter(torch.empty(n_levels, d_stream, self.d_hidden))
        self.b1 = nn.Parameter(torch.empty(n_levels, 1, self.d_hidden))
        self.w2 = nn.Parameter(torch.empty(n_levels, self.d_hidden, d_stream))
        self.b2 = nn.Parameter(torch.empty(n_levels, 1, d_stream))
        self.pred_w = nn.Parameter(torch.empty(n_levels, d_stream, d_stream))
        self.pred_b = nn.Parameter(torch.empty(n_levels, 1, d_stream))
        if self.loss_type == "infonce":
            buffer = l2_normalize(torch.randn(n_levels, self.contrastive_buffer_size, d_stream))
            self.register_buffer("contrastive_buffer", buffer, persistent=False)
            self.register_buffer(
                "contrastive_buffer_ptr",
                torch.zeros((n_levels,), dtype=torch.long),
                persistent=False,
            )
        self.reset_parameters()

    def _reset_stacked_linear(
        self,
        weight: Float[Tensor, "levels in_dim out_dim"],
        bias: Float[Tensor, "levels 1 out_dim"],
    ) -> None:
        for level in range(self.n_levels):
            nn.init.kaiming_uniform_(weight[level], a=math.sqrt(5))
            fan_in = weight[level].shape[0]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias[level], -bound, bound)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.token_embedding.weight.normal_()
            self.token_embedding.weight.copy_(l2_normalize(self.token_embedding.weight))
            self._reset_stacked_linear(self.w1, self.b1)
            self._reset_stacked_linear(self.w2, self.b2)
            self._reset_stacked_linear(self.pred_w, self.pred_b)
        self.token_embedding.weight.requires_grad_(False)

    def level_parameters(self, level: int) -> list[ParameterSlice]:
        if level < 0 or level >= self.n_levels:
            raise ValueError(f"level must be in [0, {self.n_levels}), got {level}.")
        return [
            ParameterSlice(self.w1, level),
            ParameterSlice(self.b1, level),
            ParameterSlice(self.w2, level),
            ParameterSlice(self.b2, level),
            ParameterSlice(self.pred_w, level),
            ParameterSlice(self.pred_b, level),
        ]

    def _add_noise(self, hidden: Float[Tensor, "levels batch d_stream"]) -> Float[Tensor, "levels batch d_stream"]:
        if self.noise_std == 0.0:
            return hidden
        return hidden + (torch.randn_like(hidden) * self.noise_std)

    def _stacked_linear(
        self,
        inputs: Float[Tensor, "levels batch in_dim"],
        weight: Float[Tensor, "levels in_dim out_dim"],
        bias: Float[Tensor, "levels 1 out_dim"],
    ) -> Float[Tensor, "levels batch out_dim"]:
        return torch.bmm(inputs, weight) + bias

    def logits_from_hidden(self, hidden: Float[Tensor, "batch d_stream"]) -> Float[Tensor, "batch vocab"]:
        normalized_hidden = l2_normalize(hidden)
        normalized_embedding = l2_normalize(self.token_embedding.weight)
        return F.linear(normalized_hidden, normalized_embedding) / self.readout_temperature

    def initial_recurrent_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
    ) -> tuple[
        Float[Tensor, "levels batch d_stream"],
        Float[Tensor, "levels batch d_stream"],
        Float[Tensor, "levels batch d_stream"],
        torch.Tensor,
    ]:
        dtype = self.token_embedding.weight.dtype
        resolved_device = device if device is not None else self.token_embedding.weight.device
        state_shape = (self.n_levels, batch_size, self.d_stream)
        states = torch.zeros(state_shape, device=resolved_device, dtype=dtype)
        lateral_buffers = torch.zeros_like(states)
        predictions = torch.zeros_like(states)
        has_predicted = torch.zeros((self.n_levels,), device=resolved_device, dtype=torch.bool)
        return states, lateral_buffers, predictions, has_predicted

    def detach_recurrent_state(
        self,
        states: Float[Tensor, "levels batch d_stream"],
        lateral_buffers: Float[Tensor, "levels batch d_stream"],
        predictions: Float[Tensor, "levels batch d_stream"],
        has_predicted: torch.Tensor,
    ) -> tuple[
        Float[Tensor, "levels batch d_stream"],
        Float[Tensor, "levels batch d_stream"],
        Float[Tensor, "levels batch d_stream"],
        torch.Tensor,
    ]:
        return states.detach(), lateral_buffers.detach(), predictions.detach(), has_predicted

    def prediction_losses_from_sums(
        self,
        prediction_loss_sums: Float[Tensor, "levels"],
        prediction_counts: Int[Tensor, "levels"],
    ) -> Float[Tensor, "levels"]:
        safe_counts = prediction_counts.clamp_min(1).to(dtype=prediction_loss_sums.dtype)
        return torch.where(
            prediction_counts > 0,
            prediction_loss_sums / safe_counts,
            torch.zeros_like(prediction_loss_sums),
        )

    def _prediction_errors(
        self,
        predictions: Float[Tensor, "levels batch d_stream"],
        targets: Float[Tensor, "levels batch d_stream"],
    ) -> Float[Tensor, "levels"]:
        if self.loss_type == "mse":
            return F.mse_loss(
                predictions.float(),
                targets.float(),
                reduction="none",
            ).mean(dim=(1, 2))

        normalized_predictions = F.normalize(predictions.float(), dim=-1, eps=1e-6)
        normalized_targets = F.normalize(targets.float(), dim=-1, eps=1e-6)
        normalized_buffer = F.normalize(self.contrastive_buffer.float(), dim=-1, eps=1e-6)
        positive_logits = (normalized_predictions * normalized_targets).sum(dim=-1, keepdim=True)
        negative_logits = torch.einsum("lbd,lkd->lbk", normalized_predictions, normalized_buffer)
        logits = torch.cat((positive_logits, negative_logits), dim=-1) / self.info_nce_temperature
        return (torch.logsumexp(logits, dim=-1) - logits[..., 0]).mean(dim=1)

    def _update_contrastive_buffer(
        self,
        targets: Float[Tensor, "levels batch d_stream"],
        active_predictions: torch.Tensor,
    ) -> None:
        if self.loss_type != "infonce":
            return

        batch_size = targets.shape[1]
        write_offsets = torch.arange(batch_size, device=targets.device, dtype=torch.long)
        positions = torch.remainder(self.contrastive_buffer_ptr[:, None] + write_offsets[None, :], self.contrastive_buffer_size)
        scatter_index = positions.unsqueeze(-1).expand(-1, -1, self.d_stream)

        with torch.no_grad():
            existing_values = self.contrastive_buffer.gather(1, scatter_index)
            source = torch.where(active_predictions[:, None, None], targets.to(self.contrastive_buffer.dtype), existing_values)
            self.contrastive_buffer.scatter_(1, scatter_index, source)
            self.contrastive_buffer_ptr.copy_(
                torch.remainder(
                    self.contrastive_buffer_ptr + active_predictions.to(dtype=torch.long) * batch_size,
                    self.contrastive_buffer_size,
                )
            )

    def forward_chunk(
        self,
        tokens: Int[Tensor, "batch seq"],
        states: Float[Tensor, "levels batch d_stream"],
        lateral_buffers: Float[Tensor, "levels batch d_stream"],
        predictions: Float[Tensor, "levels batch d_stream"],
        has_predicted: torch.Tensor,
        global_step_offset: Int[Tensor, ""],
    ) -> tuple[
        Float[Tensor, "batch seq vocab"],
        Float[Tensor, "levels batch d_stream"],
        Float[Tensor, "levels batch d_stream"],
        Float[Tensor, "levels batch d_stream"],
        torch.Tensor,
        Float[Tensor, "levels"],
        Int[Tensor, "levels"],
    ]:
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens with shape [batch, seq], got {tuple(tokens.shape)}.")
        if tokens.dtype != torch.long:
            raise ValueError(f"Expected tokens dtype torch.long, got {tokens.dtype}.")

        batch_size, seq_len = tokens.shape
        token_embeddings: Float[Tensor, "batch seq d_stream"] = self.token_embedding(tokens)
        logits = torch.empty((seq_len, batch_size, self.vocab_size), device=tokens.device, dtype=token_embeddings.dtype)
        prediction_loss_sums = torch.zeros((self.n_levels,), device=tokens.device, dtype=token_embeddings.dtype)
        prediction_counts = torch.zeros((self.n_levels,), device=tokens.device, dtype=torch.long)

        current_states = states
        current_predictions = predictions
        current_lateral_buffers = lateral_buffers
        current_has_predicted = has_predicted

        total_steps = seq_len * self.steps_per_token
        local_step_offsets = torch.arange(total_steps, device=tokens.device, dtype=torch.long)
        fires_at = torch.remainder(global_step_offset + local_step_offsets[:, None], self.rate_tensor[None, :]) == 0

        for timestep in range(total_steps):
            token_index = timestep // self.steps_per_token
            step_index = timestep % self.steps_per_token
            fires = fires_at[timestep]
            fire_mask = fires[:, None, None]
            active_predictions = fires & current_has_predicted

            combined = current_states + current_lateral_buffers
            if step_index == 0:
                combined = combined.clone()
                combined[0] = combined[0] + token_embeddings[:, token_index, :]
            combined = l2_normalize(combined)

            detached_combined = combined.detach()
            prediction_errors = self._prediction_errors(current_predictions, detached_combined)
            # Level 0 uses CE for grounding - no local prediction loss (InfoNCE conflicts with CE)
            # Use precomputed mask instead of scalar assignment for CUDA graph compatibility
            prediction_errors = prediction_errors * self._skip_level0.to(dtype=prediction_errors.dtype)
            prediction_loss_sums = prediction_loss_sums + (
                prediction_errors * active_predictions.to(dtype=prediction_errors.dtype)
            )
            prediction_counts = prediction_counts + active_predictions.to(dtype=torch.long)
            active_predictions = active_predictions.clone()
            active_predictions = active_predictions & self._skip_level0
            self._update_contrastive_buffer(detached_combined, active_predictions)

            hidden = self._stacked_linear(combined, self.w1, self.b1)
            hidden = F.gelu(hidden)
            output = self._stacked_linear(hidden, self.w2, self.b2)
            new_predictions = self._stacked_linear(output, self.pred_w, self.pred_b)

            current_states = torch.where(fire_mask, l2_normalize(current_states + output), current_states)
            current_predictions = torch.where(fire_mask, new_predictions, current_predictions)
            current_has_predicted = current_has_predicted | fires

            next_lateral_buffers = torch.where(fire_mask, torch.zeros_like(current_lateral_buffers), current_lateral_buffers)
            upward = self._add_noise(output.detach())
            arrivals = torch.zeros_like(next_lateral_buffers)
            arrivals[1:] = upward[:-1] * fire_mask[:-1].to(dtype=upward.dtype)
            current_lateral_buffers = next_lateral_buffers + arrivals

            if step_index == self.steps_per_token - 1:
                logits[token_index] = self.logits_from_hidden(current_states[0])

        return (
            rearrange(logits, "seq batch vocab -> batch seq vocab"),
            current_states,
            current_lateral_buffers,
            current_predictions,
            current_has_predicted,
            prediction_loss_sums,
            prediction_counts,
        )

    def forward(self, tokens: Int[Tensor, "batch seq"]) -> AutomatonOutput:
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens with shape [batch, seq], got {tuple(tokens.shape)}.")
        if tokens.dtype != torch.long:
            raise ValueError(f"Expected tokens dtype torch.long, got {tokens.dtype}.")

        batch_size = tokens.shape[0]
        states, lateral_buffers, predictions, has_predicted = self.initial_recurrent_state(batch_size, device=tokens.device)
        (
            logits,
            _states,
            _lateral_buffers,
            _predictions,
            _has_predicted,
            prediction_loss_sums,
            prediction_counts,
        ) = self.forward_chunk(
            tokens,
            states,
            lateral_buffers,
            predictions,
            has_predicted,
            torch.zeros((), device=tokens.device, dtype=torch.long),
        )
        prediction_loss_tensor = self.prediction_losses_from_sums(prediction_loss_sums, prediction_counts)
        return AutomatonOutput(
            logits=logits,
            prediction_losses=prediction_loss_tensor,
            prediction_counts=prediction_counts,
            total_prediction_loss=prediction_loss_tensor.sum(),
        )


__all__ = ["AutomatonOutput", "CellularAutomaton", "ParameterSlice", "count_parameters", "l2_normalize"]
