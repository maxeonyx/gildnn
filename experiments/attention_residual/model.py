from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


Family = Literal["external_control", "depth_only", "internal_control"]


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def rms(tensor: Tensor) -> float:
    return torch.sqrt(torch.mean(tensor.detach().float().square())).item()


@dataclass(frozen=True)
class VariantSpec:
    family: Family
    num_layers: int
    num_heads: int
    feedforward_dim: int

    @property
    def label(self) -> str:
        return self.family


@dataclass(frozen=True)
class BlockTrace:
    block_index: int
    memory_length: int
    depth_update_rms: float
    sequence_update_rms: float
    feedforward_update_rms: float
    depth_attention_weights: Tensor | None
    internal_mix_weights: Tensor | None


@dataclass(frozen=True)
class ModelRun:
    embedded: Tensor
    boundary_states: list[Tensor]
    block_traces: list[BlockTrace]
    final_hidden: Tensor
    full_logits: Tensor

    @property
    def last_logits(self) -> Tensor:
        return self.full_logits[:, -1, :]


class NoDepthReadout(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(
        self,
        current_state: Tensor,
        *,
        boundary_memory: list[Tensor],
        capture_details: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        del boundary_memory
        del capture_details
        return torch.zeros_like(current_state), None, None


class DepthAttentionReadout(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(
        self,
        current_state: Tensor,
        *,
        boundary_memory: list[Tensor],
        capture_details: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if not boundary_memory:
            return torch.zeros_like(current_state), None, None

        batch_size, sequence_length, d_model = current_state.shape
        normalized_query = self.norm(current_state).reshape(batch_size * sequence_length, 1, d_model)
        normalized_memory = torch.stack([self.norm(state) for state in boundary_memory], dim=2)
        memory = normalized_memory.reshape(batch_size * sequence_length, len(boundary_memory), d_model)
        update, weights = self.attention(
            normalized_query,
            memory,
            memory,
            need_weights=capture_details,
            average_attn_weights=False,
        )
        reshaped_update = update.reshape(batch_size, sequence_length, d_model)
        if weights is None:
            return reshaped_update, None, None
        return (
            reshaped_update,
            weights.reshape(batch_size, sequence_length, self.attention.num_heads, len(boundary_memory)),
            None,
        )


class TokenIndependentDepthMixReadout(nn.Module):
    def __init__(self, d_model: int, memory_slots: int) -> None:
        super().__init__()
        if memory_slots < 1:
            raise ValueError(f"memory_slots must be positive, got {memory_slots}.")
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.learned_logits = nn.Parameter(torch.zeros(memory_slots))

    def forward(
        self,
        current_state: Tensor,
        *,
        boundary_memory: list[Tensor],
        capture_details: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        del current_state
        if len(boundary_memory) != self.learned_logits.shape[0]:
            raise ValueError(
                "Boundary memory length does not match the locked learned mixing width: "
                f"expected {self.learned_logits.shape[0]}, got {len(boundary_memory)}."
            )
        normalized_memory = torch.stack([self.norm(state) for state in boundary_memory], dim=2)
        projected_memory = self.projection(normalized_memory)
        mix_weights = torch.softmax(self.learned_logits, dim=0)
        mixed = (projected_memory * mix_weights.view(1, 1, -1, 1)).sum(dim=2)
        if capture_details:
            return mixed, None, mix_weights.detach().cpu()
        return mixed, None, None


class AttentionResidualBlock(nn.Module):
    def __init__(
        self,
        *,
        block_index: int,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        family: Family,
    ) -> None:
        super().__init__()
        self.block_index = block_index
        if family == "external_control" or block_index == 0:
            self.depth_readout: nn.Module = NoDepthReadout(d_model)
        elif family == "depth_only":
            self.depth_readout = DepthAttentionReadout(d_model=d_model, num_heads=num_heads)
        elif family == "internal_control":
            self.depth_readout = TokenIndependentDepthMixReadout(
                d_model=d_model,
                memory_slots=block_index,
            )
        else:
            raise ValueError(f"Unsupported family: {family}")

        self.sequence_norm = nn.LayerNorm(d_model)
        self.sequence_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, d_model),
        )

    def forward(
        self,
        current_state: Tensor,
        *,
        boundary_memory: list[Tensor],
        causal_mask: Tensor,
        capture_details: bool = False,
    ) -> tuple[Tensor, BlockTrace]:
        depth_update, depth_attention_weights, internal_mix_weights = self.depth_readout(
            current_state,
            boundary_memory=boundary_memory,
            capture_details=capture_details,
        )
        after_depth = current_state + depth_update
        sequence_input = self.sequence_norm(after_depth)
        sequence_update, _ = self.sequence_attention(
            sequence_input,
            sequence_input,
            sequence_input,
            attn_mask=causal_mask,
            need_weights=False,
        )
        after_sequence = after_depth + sequence_update
        feedforward_update = self.feedforward(self.feedforward_norm(after_sequence))
        output_state = after_sequence + feedforward_update
        trace = BlockTrace(
            block_index=self.block_index,
            memory_length=len(boundary_memory),
            depth_update_rms=rms(depth_update),
            sequence_update_rms=rms(sequence_update),
            feedforward_update_rms=rms(feedforward_update),
            depth_attention_weights=depth_attention_weights,
            internal_mix_weights=internal_mix_weights,
        )
        return output_state, trace


class AttentionResidualCharModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        context_size: int,
        d_model: int,
        variant: VariantSpec,
    ) -> None:
        super().__init__()
        self.context_size = context_size
        self.variant = variant
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.blocks = nn.ModuleList(
            [
                AttentionResidualBlock(
                    block_index=block_index,
                    d_model=d_model,
                    num_heads=variant.num_heads,
                    feedforward_dim=variant.feedforward_dim,
                    family=variant.family,
                )
                for block_index in range(variant.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def causal_mask(self, *, device: torch.device, sequence_length: int) -> Tensor:
        return torch.triu(
            torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def embedded_tokens(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.shape[1]
        if sequence_length != self.context_size:
            raise ValueError(
                f"Expected context length {self.context_size}, got {sequence_length}."
            )
        positions = torch.arange(sequence_length, device=tokens.device)
        return self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)

    def run(self, tokens: Tensor, *, capture_details: bool = False) -> ModelRun:
        embedded = self.embedded_tokens(tokens)
        causal_mask = self.causal_mask(device=tokens.device, sequence_length=tokens.shape[1])
        current_state = embedded
        boundary_states = [embedded]
        block_traces: list[BlockTrace] = []
        for block in self.blocks:
            current_state, trace = block(
                current_state,
                boundary_memory=boundary_states[:-1],
                causal_mask=causal_mask,
                capture_details=capture_details,
            )
            boundary_states.append(current_state)
            block_traces.append(trace)
        final_hidden = self.final_norm(current_state)
        full_logits = self.output(final_hidden)
        return ModelRun(
            embedded=embedded,
            boundary_states=boundary_states,
            block_traces=block_traces,
            final_hidden=final_hidden,
            full_logits=full_logits,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return self.run(tokens).last_logits
