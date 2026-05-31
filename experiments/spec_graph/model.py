from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from einops import repeat
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SpecGraphConfig:
    vocab_size: int
    grid_h: int = 4
    grid_w: int = 6
    d_model: int = 128
    rollout_steps: int = 8
    horizon_steps: tuple[int, ...] = (2, 4, 6, 8)
    noise_std: float = 0.1
    ema_decay: float = 0.999
    reward_gain: float = 64.0
    predict_horizon: int = 4
    detach_head_input: bool = True
    readout_temperature: float = 1.0
    input_rows: tuple[int, ...] = (0,)

    @property
    def num_nodes(self) -> int:
        return self.grid_h * self.grid_w


@dataclass
class SpecGraphCarry:
    node_states: Float[Tensor, "batch node d_model"]
    lateral_buffer: Float[Tensor, "batch node d_model"]
    prediction_ring: Float[Tensor, "horizon batch node d_model"]
    ring_position: int
    ring_full: bool
    delayed_reward: Float[Tensor, ""]
    ema_ce: Float[Tensor, ""]
    ema_initialized: Bool[Tensor, ""]
    neighbor_loss_buffer: Float[Tensor, "node"]

    def detach_all(self) -> SpecGraphCarry:
        return SpecGraphCarry(
            node_states=self.node_states.detach(),
            lateral_buffer=self.lateral_buffer.detach(),
            prediction_ring=self.prediction_ring.detach(),
            ring_position=self.ring_position,
            ring_full=self.ring_full,
            delayed_reward=self.delayed_reward.detach(),
            ema_ce=self.ema_ce.detach(),
            ema_initialized=self.ema_initialized,
            neighbor_loss_buffer=self.neighbor_loss_buffer.detach(),
        )


@dataclass
class SpecGraphOutput:
    head_ce_loss: Float[Tensor, ""]
    local_loss: Float[Tensor, ""]
    reward_scalar: Float[Tensor, ""]
    reward_min: Float[Tensor, ""]
    reward_max: Float[Tensor, ""]
    per_horizon_ce: Float[Tensor, "horizon"]


class SharedNodeCell(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = d_model * 4
        self.input_norm = nn.LayerNorm(d_model * 3 + 2)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3 + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(
        self,
        cell_input: Float[Tensor, "batch node features"],
        current_state: Float[Tensor, "batch node d_model"],
    ) -> tuple[Float[Tensor, "batch node d_model"], Float[Tensor, "batch node d_model"]]:
        normed_input = self.input_norm(cell_input)
        update = self.mlp(normed_input)
        new_state = self.output_norm(current_state + update)
        predicted_next_lateral = self.predictor(new_state)
        return new_state, predicted_next_lateral


class CentralHead(nn.Module):
    def __init__(self, d_model: int, token_embedding: nn.Embedding, temperature: float) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.temperature = temperature

    def forward(self, node_states: Float[Tensor, "batch node d_model"]) -> Float[Tensor, "batch vocab"]:
        keys = self.key(node_states)
        values = self.value(node_states)
        attention_scores = torch.einsum("d,bnd->bn", self.query, keys) / math.sqrt(keys.shape[-1])
        attention = attention_scores.softmax(dim=1)
        context = torch.einsum("bn,bnd->bd", attention, values)
        hidden = self.mlp(context)
        normalized_hidden = F.normalize(hidden, dim=-1)
        normalized_embedding = F.normalize(self.token_embedding.weight, dim=-1)
        return normalized_hidden @ normalized_embedding.T / self.temperature


class SpecGraphModel(nn.Module):
    def __init__(self, config: SpecGraphConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.cell = SharedNodeCell(config.d_model)
        self.head = CentralHead(config.d_model, self.token_embedding, config.readout_temperature)

        if config.grid_h <= 0 or config.grid_w <= 0:
            raise ValueError(f"Grid dimensions must be positive, got {config.grid_h}x{config.grid_w}.")
        if config.rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be positive, got {config.rollout_steps}.")
        if len(config.horizon_steps) == 0:
            raise ValueError("horizon_steps must be non-empty.")
        if any(step <= 0 or step > config.rollout_steps for step in config.horizon_steps):
            raise ValueError(
                f"All horizon steps must be in [1, {config.rollout_steps}], got {config.horizon_steps}."
            )
        if config.noise_std < 0.0:
            raise ValueError(f"noise_std must be non-negative, got {config.noise_std}.")
        if not 0.0 < config.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {config.ema_decay}.")

        self.register_buffer("neighbor_index", self._build_neighbor_index(config.grid_h, config.grid_w), persistent=False)
        self.register_buffer("input_mask", self._build_input_mask(config.grid_h, config.grid_w, config.input_rows), persistent=False)
        self.horizon_to_index = {step: index for index, step in enumerate(config.horizon_steps)}

    @staticmethod
    def _build_neighbor_index(grid_h: int, grid_w: int) -> Int[Tensor, "node four"]:
        neighbors: list[list[int]] = []
        for row in range(grid_h):
            for col in range(grid_w):
                left = row * grid_w + ((col - 1) % grid_w)
                right = row * grid_w + ((col + 1) % grid_w)
                up = ((row - 1) % grid_h) * grid_w + col
                down = ((row + 1) % grid_h) * grid_w + col
                neighbors.append([left, right, up, down])
        return torch.tensor(neighbors, dtype=torch.long)

    @staticmethod
    def _build_input_mask(grid_h: int, grid_w: int, input_rows: tuple[int, ...]) -> Bool[Tensor, "node"]:
        mask = torch.zeros(grid_h * grid_w, dtype=torch.bool)
        for row in input_rows:
            if row < 0 or row >= grid_h:
                raise ValueError(f"Input row {row} is outside grid height {grid_h}.")
            start = row * grid_w
            mask[start : start + grid_w] = True
        return mask

    def initial_carry(self, batch_size: int, device: torch.device) -> SpecGraphCarry:
        zeros_state = torch.zeros(batch_size, self.config.num_nodes, self.config.d_model, device=device)
        prediction_ring = torch.zeros(
            self.config.predict_horizon, batch_size, self.config.num_nodes, self.config.d_model, device=device
        )
        return SpecGraphCarry(
            node_states=zeros_state,
            lateral_buffer=zeros_state.clone(),
            prediction_ring=prediction_ring,
            ring_position=0,
            ring_full=False,
            delayed_reward=torch.tensor(0.5, device=device),
            ema_ce=torch.tensor(0.0, device=device),
            ema_initialized=torch.tensor(False, device=device),
            neighbor_loss_buffer=torch.zeros(self.config.num_nodes, device=device),
        )

    def _clean_lateral_from_buffer(
        self,
        lateral_buffer: Float[Tensor, "batch node d_model"],
    ) -> Float[Tensor, "batch node d_model"]:
        neighbor_values = lateral_buffer[:, self.neighbor_index, :]
        return neighbor_values.mean(dim=2)

    def _neighbor_loss_signal(
        self,
        neighbor_loss_buffer: Float[Tensor, "node"],
    ) -> Float[Tensor, "node"]:
        return neighbor_loss_buffer[self.neighbor_index].mean(dim=1)

    def _token_inputs(
        self,
        token_embedding: Float[Tensor, "batch d_model"],
    ) -> Float[Tensor, "batch node d_model"]:
        raw_token_input = token_embedding.new_zeros(token_embedding.shape[0], self.config.num_nodes, self.config.d_model)
        if bool(self.input_mask.any().item()):
            raw_token_input[:, self.input_mask, :] = repeat(token_embedding, "b d -> b n d", n=int(self.input_mask.sum().item()))
        return raw_token_input

    def _reward_from_ce(
        self,
        current_ce: Float[Tensor, ""],
        ema_ce: Float[Tensor, ""],
        ema_initialized: Bool[Tensor, ""],
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Bool[Tensor, ""]]:
        baseline_ce = torch.where(ema_initialized, ema_ce, current_ce.detach())
        delta = baseline_ce - current_ce.detach()
        reward = torch.sigmoid(self.config.reward_gain * delta)
        updated_ema = torch.where(
            ema_initialized,
            self.config.ema_decay * ema_ce + (1.0 - self.config.ema_decay) * current_ce.detach(),
            current_ce.detach(),
        )
        return reward, updated_ema, torch.tensor(True, device=current_ce.device)

    def forward_chunk(
        self,
        inputs: Int[Tensor, "batch seq"],
        targets: Int[Tensor, "batch seq"],
        carry: SpecGraphCarry,
    ) -> tuple[SpecGraphOutput, SpecGraphCarry]:
        batch_size, seq_len = inputs.shape
        token_embeddings = self.token_embedding(inputs)
        head_losses: list[Float[Tensor, ""]] = []
        per_horizon_terms: list[list[Float[Tensor, ""]]] = [[] for _ in self.config.horizon_steps]
        local_losses: list[Float[Tensor, ""]] = []
        reward_values: list[Float[Tensor, ""]] = []

        current = carry
        ring_pos = carry.ring_position
        ring_full = carry.ring_full
        for token_index in range(seq_len):
            token_embedding = token_embeddings[:, token_index, :]
            target = targets[:, token_index]
            raw_token_input = self._token_inputs(token_embedding)

            for rollout_step in range(1, self.config.rollout_steps + 1):
                reward_values.append(current.delayed_reward)
                clean_lateral = self._clean_lateral_from_buffer(current.lateral_buffer)

                # Score the prediction from K steps ago against current clean lateral
                if ring_full:
                    old_prediction = current.prediction_ring[ring_pos]
                    per_node_local_loss = F.mse_loss(old_prediction, clean_lateral, reduction="none").mean(dim=-1)
                    weighted_local_loss = per_node_local_loss * current.delayed_reward.detach()
                    local_losses.append(weighted_local_loss.mean())
                    next_neighbor_loss_buffer = per_node_local_loss.mean(dim=0).detach()
                else:
                    next_neighbor_loss_buffer = torch.zeros_like(current.neighbor_loss_buffer)

                noisy_lateral = clean_lateral
                if self.config.noise_std > 0.0:
                    noisy_lateral = noisy_lateral + torch.randn_like(noisy_lateral) * self.config.noise_std

                neighbor_loss_signal = self._neighbor_loss_signal(current.neighbor_loss_buffer)
                reward_channel = current.delayed_reward.view(1, 1, 1).expand(batch_size, self.config.num_nodes, 1)
                neighbor_loss_channel = neighbor_loss_signal.view(1, self.config.num_nodes, 1).expand(batch_size, -1, -1)
                cell_input = torch.cat(
                    [
                        current.node_states,
                        noisy_lateral,
                        raw_token_input,
                        reward_channel,
                        neighbor_loss_channel,
                    ],
                    dim=-1,
                )
                new_state, predicted_next_lateral = self.cell(cell_input, current.node_states)

                # Store current prediction in ring buffer at current position
                new_ring = current.prediction_ring.clone()
                new_ring[ring_pos] = predicted_next_lateral.detach()
                next_ring_pos = (ring_pos + 1) % self.config.predict_horizon
                next_ring_full = ring_full or (next_ring_pos == 0)

                next_reward = current.delayed_reward.detach()
                next_ema = current.ema_ce.detach()
                next_ema_initialized = current.ema_initialized
                if rollout_step in self.horizon_to_index:
                    head_input = new_state.detach() if self.config.detach_head_input else new_state
                    logits = self.head(head_input)
                    ce = F.cross_entropy(logits, target)
                    head_losses.append(ce)
                    per_horizon_terms[self.horizon_to_index[rollout_step]].append(ce)
                    next_reward, next_ema, next_ema_initialized = self._reward_from_ce(
                        ce,
                        current.ema_ce,
                        current.ema_initialized,
                    )

                current = SpecGraphCarry(
                    node_states=new_state,
                    lateral_buffer=new_state.detach(),
                    prediction_ring=new_ring,
                    ring_position=next_ring_pos,
                    ring_full=next_ring_full,
                    delayed_reward=next_reward.detach(),
                    ema_ce=next_ema.detach(),
                    ema_initialized=next_ema_initialized,
                    neighbor_loss_buffer=next_neighbor_loss_buffer,
                )
                ring_pos = next_ring_pos
                ring_full = next_ring_full

        device = inputs.device
        head_ce_loss = torch.stack(head_losses).mean() if head_losses else torch.zeros((), device=device)
        local_loss = torch.stack(local_losses).mean() if local_losses else torch.zeros((), device=device)
        reward_stack = torch.stack(reward_values) if reward_values else torch.zeros(1, device=device)
        reward_scalar = reward_stack.mean()
        reward_min = reward_stack.min()
        reward_max = reward_stack.max()
        per_horizon_ce = torch.stack(
            [
                torch.stack(terms).mean() if terms else torch.zeros((), device=device)
                for terms in per_horizon_terms
            ]
        )
        return SpecGraphOutput(
            head_ce_loss=head_ce_loss,
            local_loss=local_loss,
            reward_scalar=reward_scalar,
            reward_min=reward_min,
            reward_max=reward_max,
            per_horizon_ce=per_horizon_ce,
        ), current
