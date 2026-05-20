from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


VariantName = Literal[
    "synchronous_control",
    "learned_gate",
    "random_skip",
    "forced_open",
]


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def rms(tensor: Tensor) -> float:
    return torch.sqrt(torch.mean(tensor.detach().float().square())).item()


def hard_sigmoid(logits: Tensor) -> Tensor:
    return ((logits + 1.0) / 2.0).clamp_(0.0, 1.0)


@dataclass(frozen=True)
class VariantSpec:
    name: VariantName
    target_open_rate: float | None = None
    budget_weight: float = 1.0
    num_layers: int = 3
    num_heads: int = 4
    feedforward_dims: tuple[int, int, int] = (256, 255, 255)

    @property
    def label(self) -> str:
        if self.target_open_rate is None:
            return self.name
        return f"{self.name}_r{int(round(self.target_open_rate * 100)):02d}"


@dataclass(frozen=True)
class GateTrace:
    mode: str
    logits: Tensor | None
    probabilities: Tensor | None
    values: Tensor


@dataclass(frozen=True)
class BlockTrace:
    block_index: int
    gate: GateTrace
    attention_update_rms: float
    feedforward_update_rms: float
    block_delta_rms: float
    executed_delta_rms: float


@dataclass(frozen=True)
class ModelRun:
    embedded: Tensor
    hidden_states: list[Tensor]
    block_traces: list[BlockTrace]
    final_hidden: Tensor
    full_logits: Tensor
    budget_loss: Tensor

    @property
    def last_logits(self) -> Tensor:
        return self.full_logits[:, -1, :]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, *, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, hidden: Tensor) -> Tensor:
        rms_value = hidden.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return hidden * rms_value * self.scale


class SelectiveGate(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.projection = nn.Linear(d_model, 1)

    def forward(
        self,
        hidden: Tensor,
        *,
        variant: VariantSpec,
        gate_override: Tensor | None = None,
    ) -> GateTrace:
        if gate_override is not None:
            override = gate_override.to(device=hidden.device, dtype=hidden.dtype)
            return GateTrace(
                mode="override",
                logits=None,
                probabilities=override,
                values=override,
            )

        shape = (*hidden.shape[:2], 1)
        if variant.name in {"synchronous_control", "forced_open"}:
            ones = torch.ones(shape, device=hidden.device, dtype=hidden.dtype)
            return GateTrace(mode="always_open", logits=None, probabilities=ones, values=ones)

        if variant.target_open_rate is None:
            raise ValueError(f"Variant {variant.name} requires target_open_rate.")

        if variant.name == "random_skip":
            probabilities = torch.full(
                shape,
                fill_value=variant.target_open_rate,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            values = torch.bernoulli(probabilities)
            return GateTrace(
                mode="random_skip",
                logits=None,
                probabilities=probabilities,
                values=values,
            )

        if variant.name != "learned_gate":
            raise ValueError(f"Unsupported gate variant: {variant.name}")

        logits = self.projection(self.norm(hidden))
        probabilities = hard_sigmoid(logits)
        if self.training:
            hard_values = torch.bernoulli(probabilities)
        else:
            hard_values = (probabilities >= 0.5).to(hidden.dtype)
        straight_through_values = hard_values + probabilities - probabilities.detach()
        return GateTrace(
            mode="learned_gate",
            logits=logits,
            probabilities=probabilities,
            values=straight_through_values,
        )


class SelectiveTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        block_index: int,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
    ) -> None:
        super().__init__()
        self.block_index = block_index
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
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
        self.gate = SelectiveGate(d_model) if block_index > 0 else None

    def forward(
        self,
        hidden: Tensor,
        *,
        causal_mask: Tensor,
        variant: VariantSpec,
        gate_override: Tensor | None = None,
    ) -> tuple[Tensor, BlockTrace]:
        if self.gate is None:
            gate_values = torch.ones(
                (*hidden.shape[:2], 1),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            gate_trace = GateTrace(
                mode="block_1_always_open",
                logits=None,
                probabilities=gate_values,
                values=gate_values,
            )
        else:
            gate_trace = self.gate(hidden, variant=variant, gate_override=gate_override)

        attention_input = self.attention_norm(hidden)
        attention_update, _ = self.attention(
            attention_input,
            attention_input,
            attention_input,
            attn_mask=causal_mask,
            need_weights=False,
        )
        after_attention = hidden + attention_update
        feedforward_update = self.feedforward(self.feedforward_norm(after_attention))
        full_output = after_attention + feedforward_update
        block_delta = full_output - hidden
        output = hidden + gate_trace.values * block_delta
        trace = BlockTrace(
            block_index=self.block_index,
            gate=gate_trace,
            attention_update_rms=rms(attention_update),
            feedforward_update_rms=rms(feedforward_update),
            block_delta_rms=rms(block_delta),
            executed_delta_rms=rms(gate_trace.values * block_delta),
        )
        return output, trace


def budget_regularizer(
    gate_values: list[Tensor],
    *,
    target_open_rate: float | None,
    budget_weight: float,
) -> Tensor:
    if not gate_values:
        raise ValueError("budget_regularizer requires at least one gate tensor.")
    reference = gate_values[0]
    if target_open_rate is None:
        return reference.new_zeros(())
    mean_open = torch.stack([gate.float().mean() for gate in gate_values]).mean()
    return (mean_open - target_open_rate).square() * budget_weight


class AsyncSelectiveCharModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        context_size: int,
        d_model: int,
        variant: VariantSpec,
    ) -> None:
        super().__init__()
        if len(variant.feedforward_dims) != variant.num_layers:
            raise ValueError(
                "feedforward_dims length must match num_layers: "
                f"expected {variant.num_layers}, got {len(variant.feedforward_dims)}."
            )
        self.context_size = context_size
        self.variant = variant
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.blocks = nn.ModuleList(
            [
                SelectiveTransformerBlock(
                    block_index=block_index,
                    d_model=d_model,
                    num_heads=variant.num_heads,
                    feedforward_dim=variant.feedforward_dims[block_index],
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

    def run(
        self,
        tokens: Tensor,
        *,
        gate_overrides: dict[int, Tensor] | None = None,
    ) -> ModelRun:
        embedded = self.embedded_tokens(tokens)
        hidden = embedded
        hidden_states = [embedded]
        block_traces: list[BlockTrace] = []
        causal_mask = self.causal_mask(device=tokens.device, sequence_length=tokens.shape[1])
        for block in self.blocks:
            hidden, trace = block(
                hidden,
                causal_mask=causal_mask,
                variant=self.variant,
                gate_override=None if gate_overrides is None else gate_overrides.get(block.block_index),
            )
            hidden_states.append(hidden)
            block_traces.append(trace)
        final_hidden = self.final_norm(hidden)
        full_logits = self.output(final_hidden)
        gate_values = [trace.gate.values for trace in block_traces[1:]]
        budget_loss = budget_regularizer(
            gate_values,
            target_open_rate=self.variant.target_open_rate if self.variant.name == "learned_gate" else None,
            budget_weight=self.variant.budget_weight,
        )
        return ModelRun(
            embedded=embedded,
            hidden_states=hidden_states,
            block_traces=block_traces,
            final_hidden=final_hidden,
            full_logits=full_logits,
            budget_loss=budget_loss,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return self.run(tokens).last_logits
