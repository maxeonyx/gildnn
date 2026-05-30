from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from einops import rearrange, repeat
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
    module: int

    @property
    def tensor(self) -> Tensor:
        return self.parameter[self.module]


@dataclass(frozen=True)
class GraphAutomatonOutput:
    logits: Float[Tensor, "batch seq vocab"]
    prediction_losses: Float[Tensor, "modules"]
    prediction_counts: Int[Tensor, "modules"]
    total_prediction_loss: Float[Tensor, ""]


class GraphCellularAutomaton(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_stream: int = 96,
        d_hidden: int | None = None,
        n_bands: int = 8,
        n_cols: int = 24,
        steps_per_token: int = 8,
        noise_std: float = 0.1,
        readout_temperature: float = 0.07,
        loss_type: str = "infonce",
        info_nce_temperature: float = 0.07,
        contrastive_buffer_size: int = 64,
        refractory: bool = False,
        refractory_threshold: float = 1.0,
        refractory_decay: float = 0.8,
        multi_scale_input: bool = False,
        temporal_targets: bool = False,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
        if d_stream <= 0:
            raise ValueError(f"d_stream must be positive, got {d_stream}.")
        if n_bands <= 0:
            raise ValueError(f"n_bands must be positive, got {n_bands}.")
        if n_cols <= 0:
            raise ValueError(f"n_cols must be positive, got {n_cols}.")
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
        if contrastive_buffer_size <= 0:
            raise ValueError(
                f"contrastive_buffer_size must be positive, got {contrastive_buffer_size}."
            )
        if refractory_threshold < 0.0:
            raise ValueError(f"refractory_threshold must be non-negative, got {refractory_threshold}.")
        if not 0.0 <= refractory_decay <= 1.0:
            raise ValueError(f"refractory_decay must be in [0, 1], got {refractory_decay}.")

        rates = (1, 2, 4, 8, 16, 32, 64, 128)
        if n_bands != len(rates):
            raise ValueError(f"Expected n_bands={len(rates)} to match the fixed rate schedule, got {n_bands}.")

        self.vocab_size = vocab_size
        self.d_stream = d_stream
        self.d_hidden = d_hidden if d_hidden is not None else 4 * d_stream
        self.n_bands = n_bands
        self.n_cols = n_cols
        self.n_modules = n_bands * n_cols
        self.steps_per_token = steps_per_token
        self.noise_std = noise_std
        self.readout_temperature = readout_temperature
        self.loss_type = loss_type
        self.info_nce_temperature = info_nce_temperature
        self.contrastive_buffer_size = contrastive_buffer_size
        self.refractory = refractory
        self.refractory_threshold = refractory_threshold
        self.refractory_decay = refractory_decay
        self.multi_scale_input = multi_scale_input
        self.temporal_targets = temporal_targets

        self.token_embedding = nn.Embedding(vocab_size, d_stream)
        self.w1 = nn.Parameter(torch.empty(self.n_modules, d_stream, self.d_hidden))
        self.b1 = nn.Parameter(torch.empty(self.n_modules, 1, self.d_hidden))
        self.w2 = nn.Parameter(torch.empty(self.n_modules, self.d_hidden, d_stream))
        self.b2 = nn.Parameter(torch.empty(self.n_modules, 1, d_stream))
        self.pred_w = nn.Parameter(torch.empty(self.n_modules, d_stream, d_stream))
        self.pred_b = nn.Parameter(torch.empty(self.n_modules, 1, d_stream))
        if self.loss_type == "infonce":
            contrastive_buffer = l2_normalize(
                torch.randn(self.n_modules, self.contrastive_buffer_size, d_stream)
            )
            self.register_buffer("contrastive_buffer", contrastive_buffer, persistent=False)
            self.register_buffer(
                "contrastive_buffer_ptr",
                torch.zeros((self.n_modules,), dtype=torch.long),
                persistent=False,
            )

        module_rows = repeat(torch.arange(n_bands, dtype=torch.long), "band -> (band col)", col=n_cols)
        module_cols = repeat(torch.arange(n_cols, dtype=torch.long), "col -> (band col)", band=n_bands)
        module_rates = torch.tensor([rates[row] for row in module_rows.tolist()], dtype=torch.long)
        module_phases = torch.remainder(module_cols, module_rates)
        self.register_buffer("module_rows", module_rows, persistent=False)
        self.register_buffer("module_cols", module_cols, persistent=False)
        self.register_buffer("module_rates", module_rates, persistent=False)
        self.register_buffer("module_phases", module_phases, persistent=False)
        self.register_buffer("band0_mask", module_rows == 0, persistent=False)

        neighbor_indices, neighbor_mask = self._build_neighbor_index()
        self.register_buffer("neighbor_indices", neighbor_indices, persistent=False)
        self.register_buffer("neighbor_mask", neighbor_mask, persistent=False)
        self.register_buffer("clamped_neighbor_indices", neighbor_indices.clamp_min(0), persistent=False)

        self.reset_parameters()

    def _build_neighbor_index(self) -> tuple[Int[Tensor, "modules 4"], Float[Tensor, "modules 4 1 1"]]:
        neighbors = torch.full((self.n_modules, 4), -1, dtype=torch.long)
        for band in range(self.n_bands):
            for col in range(self.n_cols):
                module = band * self.n_cols + col
                left = band * self.n_cols + ((col - 1) % self.n_cols)
                right = band * self.n_cols + ((col + 1) % self.n_cols)
                up = (band - 1) * self.n_cols + col if band > 0 else -1
                down = (band + 1) * self.n_cols + col if band < self.n_bands - 1 else -1
                neighbors[module] = torch.tensor((left, right, up, down), dtype=torch.long)
        neighbor_mask = (neighbors >= 0).to(dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        return neighbors, neighbor_mask

    def _reset_stacked_linear(
        self,
        weight: Float[Tensor, "modules in_dim out_dim"],
        bias: Float[Tensor, "modules 1 out_dim"],
    ) -> None:
        for module in range(self.n_modules):
            nn.init.kaiming_uniform_(weight[module], a=math.sqrt(5))
            fan_in = weight[module].shape[0]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias[module], -bound, bound)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.token_embedding.weight.normal_()
            self.token_embedding.weight.copy_(l2_normalize(self.token_embedding.weight))
            self._reset_stacked_linear(self.w1, self.b1)
            self._reset_stacked_linear(self.w2, self.b2)
            self._reset_stacked_linear(self.pred_w, self.pred_b)
        self.token_embedding.weight.requires_grad_(False)

    def module_parameters(self, module: int) -> list[ParameterSlice]:
        if module < 0 or module >= self.n_modules:
            raise ValueError(f"module must be in [0, {self.n_modules}), got {module}.")
        return [
            ParameterSlice(self.w1, module),
            ParameterSlice(self.b1, module),
            ParameterSlice(self.w2, module),
            ParameterSlice(self.b2, module),
            ParameterSlice(self.pred_w, module),
            ParameterSlice(self.pred_b, module),
        ]

    def initial_recurrent_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
    ) -> tuple[
        Float[Tensor, "modules batch d_stream"],
        Float[Tensor, "modules batch d_stream"],
        Float[Tensor, "modules batch d_stream"],
        torch.Tensor,
        Float[Tensor, "modules"],
    ]:
        resolved_device = device if device is not None else self.token_embedding.weight.device
        dtype = self.token_embedding.weight.dtype
        state_shape = (self.n_modules, batch_size, self.d_stream)
        states = torch.zeros(state_shape, device=resolved_device, dtype=dtype)
        global_buffer = torch.zeros_like(states)
        predictions = torch.zeros_like(states)
        has_predicted = torch.zeros((self.n_modules,), device=resolved_device, dtype=torch.bool)
        refractory_levels = torch.zeros((self.n_modules,), device=resolved_device, dtype=dtype)
        return states, global_buffer, predictions, has_predicted, refractory_levels

    def _stacked_linear(
        self,
        inputs: Float[Tensor, "modules batch in_dim"],
        weight: Float[Tensor, "modules in_dim out_dim"],
        bias: Float[Tensor, "modules 1 out_dim"],
    ) -> Float[Tensor, "modules batch out_dim"]:
        return torch.bmm(inputs, weight) + bias

    def _neighbor_sum(
        self,
        global_buffer: Float[Tensor, "modules batch d_stream"],
        refractory_levels: Float[Tensor, "modules"],
    ) -> Float[Tensor, "modules batch d_stream"]:
        gathered = global_buffer.detach()[self.clamped_neighbor_indices]
        if self.noise_std > 0.0:
            gathered = gathered + (torch.randn_like(gathered) * self.noise_std)
        if self.refractory:
            neighbor_refractory = refractory_levels[self.clamped_neighbor_indices].to(dtype=gathered.dtype)
            gathered = gathered * (1.0 - neighbor_refractory[..., None, None])
        masked = gathered * self.neighbor_mask.to(dtype=gathered.dtype)
        return masked.sum(dim=1)

    def logits_from_hidden(self, hidden: Tensor) -> Tensor:
        normalized_hidden = l2_normalize(hidden)
        normalized_embedding = l2_normalize(self.token_embedding.weight)
        return F.linear(normalized_hidden, normalized_embedding) / self.readout_temperature

    def prediction_losses_from_sums(
        self,
        prediction_loss_sums: Float[Tensor, "modules"],
        prediction_counts: Int[Tensor, "modules"],
    ) -> Float[Tensor, "modules"]:
        safe_counts = prediction_counts.clamp_min(1).to(dtype=prediction_loss_sums.dtype)
        return torch.where(
            prediction_counts > 0,
            prediction_loss_sums / safe_counts,
            torch.zeros_like(prediction_loss_sums),
        )

    def _prediction_errors(
        self,
        predictions: Float[Tensor, "modules batch d_stream"],
        targets: Float[Tensor, "modules batch d_stream"],
    ) -> Float[Tensor, "modules"]:
        if self.loss_type == "mse":
            return F.mse_loss(predictions.float(), targets.float(), reduction="none").mean(dim=(1, 2))

        norm_pred = F.normalize(predictions.float(), dim=-1, eps=1e-6)
        norm_target = F.normalize(targets.float(), dim=-1, eps=1e-6)
        norm_buffer = F.normalize(self.contrastive_buffer.float(), dim=-1, eps=1e-6)

        positive_logits = (norm_pred * norm_target).sum(dim=-1, keepdim=True)
        negative_logits = torch.einsum("mbd,mkd->mbk", norm_pred, norm_buffer)
        logits = torch.cat((positive_logits, negative_logits), dim=-1) / self.info_nce_temperature
        return (torch.logsumexp(logits, dim=-1) - logits[..., 0]).mean(dim=1)

    def _update_contrastive_buffer(
        self,
        targets: Float[Tensor, "modules batch d_stream"],
        active_modules: torch.Tensor,
    ) -> None:
        if self.loss_type != "infonce":
            return

        batch_size = targets.shape[1]
        write_offsets = torch.arange(batch_size, device=targets.device, dtype=torch.long)
        positions = torch.remainder(
            self.contrastive_buffer_ptr[:, None] + write_offsets[None, :],
            self.contrastive_buffer_size,
        )
        scatter_index = positions.unsqueeze(-1).expand(-1, -1, self.d_stream)

        with torch.no_grad():
            existing = self.contrastive_buffer.gather(1, scatter_index)
            source = torch.where(active_modules[:, None, None], targets.to(self.contrastive_buffer.dtype), existing)
            self.contrastive_buffer.scatter_(1, scatter_index, source)
            self.contrastive_buffer_ptr.copy_(
                torch.remainder(
                    self.contrastive_buffer_ptr + active_modules.to(dtype=torch.long) * batch_size,
                    self.contrastive_buffer_size,
                )
            )

    def forward_chunk(
        self,
        tokens: Int[Tensor, "batch seq"],
        states: Float[Tensor, "modules batch d_stream"],
        global_buffer: Float[Tensor, "modules batch d_stream"],
        predictions: Float[Tensor, "modules batch d_stream"],
        has_predicted: torch.Tensor,
        refractory_levels: Float[Tensor, "modules"],
        global_step_offset: int | Int[Tensor, ""],
    ) -> tuple[
        Float[Tensor, "batch seq vocab"],
        Float[Tensor, "modules batch d_stream"],
        Float[Tensor, "modules batch d_stream"],
        Float[Tensor, "modules batch d_stream"],
        torch.Tensor,
        Float[Tensor, "modules"],
        Float[Tensor, "modules"],
        Int[Tensor, "modules"],
    ]:
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens with shape [batch, seq], got {tuple(tokens.shape)}.")
        if tokens.dtype != torch.long:
            raise ValueError(f"Expected tokens dtype torch.long, got {tokens.dtype}.")

        batch_size, seq_len = tokens.shape
        token_embeddings: Float[Tensor, "batch seq d_stream"] = self.token_embedding(tokens)
        logits = torch.empty((seq_len, batch_size, self.vocab_size), device=tokens.device, dtype=token_embeddings.dtype)
        prediction_loss_sums = torch.zeros((self.n_modules,), device=tokens.device, dtype=token_embeddings.dtype)
        prediction_counts = torch.zeros((self.n_modules,), device=tokens.device, dtype=torch.long)

        current_states = states
        current_global_buffer = global_buffer
        current_predictions = predictions
        current_has_predicted = has_predicted
        current_refractory_levels = refractory_levels

        # Multi-scale input: EMA of token embeddings per band
        if self.multi_scale_input:
            # ema_alphas[band] = 1.0 / rate[band], so band 0 (rate 1) gets α=1 (raw token)
            ema_alphas = (1.0 / self.module_rates.float()).unsqueeze(-1)  # [n_modules, 1]
            token_ema = torch.zeros(
                (self.n_modules, batch_size, self.d_stream),
                device=tokens.device,
                dtype=token_embeddings.dtype,
            )

        if isinstance(global_step_offset, int):
            step_offset = torch.tensor(global_step_offset, device=tokens.device, dtype=torch.long)
        else:
            step_offset = global_step_offset.to(device=tokens.device, dtype=torch.long)

        total_steps = seq_len * self.steps_per_token
        local_step_offsets = torch.arange(total_steps, device=tokens.device, dtype=torch.long)
        fires_at = (
            torch.remainder(
                step_offset + local_step_offsets[:, None] + self.module_phases[None, :],
                self.module_rates[None, :],
            )
            == 0
        )

        for timestep in range(total_steps):
            token_index = timestep // self.steps_per_token
            fires = fires_at[timestep]
            fire_mask = fires[:, None, None]

            if self.refractory:
                current_refractory_levels = current_refractory_levels * self.refractory_decay

            neighbor_sum = self._neighbor_sum(current_global_buffer, current_refractory_levels)
            prediction_target = l2_normalize(neighbor_sum)
            combined = current_states + neighbor_sum
            combined = combined.clone()
            if self.multi_scale_input:
                # Update EMA at token boundaries
                if timestep % self.steps_per_token == 0:
                    tok_emb = token_embeddings[:, token_index, :]  # [batch, d_stream]
                    tok_emb_expanded = tok_emb.unsqueeze(0).expand(self.n_modules, -1, -1)
                    token_ema = (1.0 - ema_alphas.unsqueeze(-1)) * token_ema + ema_alphas.unsqueeze(-1) * tok_emb_expanded
                # All modules get their band's temporal view
                combined = combined + token_ema
            else:
                combined[self.band0_mask] = combined[self.band0_mask] + token_embeddings[:, token_index, :]
            combined = l2_normalize(combined)

            active_predictions = fires & current_has_predicted
            if self.temporal_targets:
                # Each module's prediction target = current token embedding (L2-normalized)
                # Since different bands fire at different rates, their predictions were made
                # rate/steps_per_token tokens ago — naturally forcing timescale separation.
                temporal_target = l2_normalize(token_embeddings[:, token_index, :])  # [batch, d_stream]
                temporal_target_expanded = temporal_target.unsqueeze(0).expand(self.n_modules, -1, -1)
                prediction_errors = self._prediction_errors(current_predictions, temporal_target_expanded.detach())
            else:
                prediction_errors = self._prediction_errors(current_predictions, prediction_target.detach())
            prediction_errors[self.band0_mask] = 0.0
            prediction_loss_sums = prediction_loss_sums + (
                prediction_errors * active_predictions.to(dtype=prediction_errors.dtype)
            )
            prediction_counts = prediction_counts + active_predictions.to(dtype=torch.long)
            non_band0_active = active_predictions.clone()
            non_band0_active[self.band0_mask] = False
            self._update_contrastive_buffer(prediction_target.detach(), non_band0_active)

            hidden = F.gelu(self._stacked_linear(combined, self.w1, self.b1))
            output = self._stacked_linear(hidden, self.w2, self.b2)
            new_predictions = self._stacked_linear(output, self.pred_w, self.pred_b)

            current_states = torch.where(fire_mask, output, current_states)
            current_predictions = torch.where(fire_mask, new_predictions, current_predictions)
            current_global_buffer = torch.where(fire_mask, output.detach(), current_global_buffer)
            current_has_predicted = current_has_predicted | fires
            if self.refractory:
                output_norms = output.float().norm(dim=-1).amax(dim=1)
                fired_strongly = fires & (output_norms > self.refractory_threshold)
                current_refractory_levels = torch.where(
                    fired_strongly,
                    torch.ones_like(current_refractory_levels),
                    current_refractory_levels,
                )

            if timestep % self.steps_per_token == self.steps_per_token - 1:
                band0_logits = self.logits_from_hidden(current_states[self.band0_mask])
                logits[token_index] = band0_logits.mean(dim=0)

        return (
            rearrange(logits, "seq batch vocab -> batch seq vocab"),
            current_states,
            current_global_buffer,
            current_predictions,
            current_has_predicted,
            current_refractory_levels,
            prediction_loss_sums,
            prediction_counts,
        )

    def forward(self, tokens: Int[Tensor, "batch seq"]) -> GraphAutomatonOutput:
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens with shape [batch, seq], got {tuple(tokens.shape)}.")
        if tokens.dtype != torch.long:
            raise ValueError(f"Expected tokens dtype torch.long, got {tokens.dtype}.")

        states, global_buffer, predictions, has_predicted, refractory_levels = self.initial_recurrent_state(
            tokens.shape[0],
            device=tokens.device,
        )
        logits, _, _, _, _, _, prediction_loss_sums, prediction_counts = self.forward_chunk(
            tokens,
            states,
            global_buffer,
            predictions,
            has_predicted,
            refractory_levels,
            torch.zeros((), device=tokens.device, dtype=torch.long),
        )
        prediction_losses = self.prediction_losses_from_sums(prediction_loss_sums, prediction_counts)
        return GraphAutomatonOutput(
            logits=logits,
            prediction_losses=prediction_losses,
            prediction_counts=prediction_counts,
            total_prediction_loss=prediction_losses.sum(),
        )


__all__ = [
    "GraphAutomatonOutput",
    "GraphCellularAutomaton",
    "ParameterSlice",
    "count_parameters",
    "l2_normalize",
]
