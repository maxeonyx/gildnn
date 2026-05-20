from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


Family = Literal["end_to_end", "local"]


@dataclass(frozen=True)
class VariantSpec:
    family: Family
    num_blocks: int
    ff_hidden: int
    num_heads: int

    @property
    def label(self) -> str:
        return f"{self.family}_{self.num_blocks}b"


@dataclass(frozen=True)
class BlockState:
    input_residual: Tensor
    delta: Tensor
    output_residual: Tensor
    predicted_delta: Tensor | None
    local_target_delta: Tensor | None


@dataclass(frozen=True)
class ModelRun:
    embedded: Tensor
    block_states: list[BlockState]
    final_hidden: Tensor
    logits: Tensor


@dataclass(frozen=True)
class LossBundle:
    total_loss: Tensor
    lm_loss: Tensor
    local_loss_total: Tensor
    local_losses: list[Tensor]
    logits: Tensor
    accuracy: float


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def rms(tensor: Tensor) -> float:
    return torch.sqrt(torch.mean(tensor.detach().float().square())).item()


class ResidualFfnBlock(nn.Module):
    def __init__(self, *, d_model: int, ff_hidden: int, num_heads: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, ff_hidden)
        self.down = nn.Linear(ff_hidden, d_model)

    def forward(self, residual: Tensor, *, causal_mask: Tensor) -> tuple[Tensor, Tensor]:
        attention_input = self.attention_norm(residual)
        attention_delta, _ = self.attention(
            attention_input,
            attention_input,
            attention_input,
            attn_mask=causal_mask,
            need_weights=False,
        )
        residual_after_attention = residual + attention_delta
        ffn_delta = self.down(F.gelu(self.up(self.ffn_norm(residual_after_attention))))
        output_residual = residual_after_attention + ffn_delta
        total_delta = output_residual - residual
        return output_residual, total_delta


class ResidualLocalLearningModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        context_size: int,
        d_model: int,
        variant: VariantSpec,
    ) -> None:
        super().__init__()
        if variant.num_blocks < 1:
            raise ValueError(f"num_blocks must be at least 1, got {variant.num_blocks}.")
        self.variant = variant
        self.context_size = context_size
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.blocks = nn.ModuleList(
            [
                ResidualFfnBlock(
                    d_model=d_model,
                    ff_hidden=variant.ff_hidden,
                    num_heads=variant.num_heads,
                )
                for _ in range(variant.num_blocks)
            ]
        )
        self.local_heads = (
            nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(variant.num_blocks)])
            if variant.family == "local"
            else None
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def causal_mask(self, *, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(
                self.context_size,
                self.context_size,
                device=device,
                dtype=torch.bool,
            ),
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

    def run(self, tokens: Tensor) -> ModelRun:
        residual = self.embedded_tokens(tokens)
        causal_mask = self.causal_mask(device=tokens.device)
        block_states: list[BlockState] = []
        for block_index, block in enumerate(self.blocks):
            block_input = residual
            block_output, delta = block(block_input, causal_mask=causal_mask)
            predicted_delta = None
            if self.local_heads is not None:
                predicted_delta = self.local_heads[block_index](block_output)
            block_states.append(
                BlockState(
                    input_residual=block_input,
                    delta=delta,
                    output_residual=block_output,
                    predicted_delta=predicted_delta,
                    local_target_delta=None,
                )
            )
            if self.variant.family == "local" and block_index < len(self.blocks) - 1:
                residual = block_output.detach()
            else:
                residual = block_output
        final_hidden = self.final_norm(residual)
        logits = self.lm_head(final_hidden[:, -1, :])
        if self.local_heads is not None:
            target_deltas: list[Tensor] = []
            for block_index, block_state in enumerate(block_states):
                if block_index < len(block_states) - 1:
                    target_deltas.append(block_states[block_index + 1].delta.detach())
                else:
                    target_deltas.append(block_state.delta.detach())
            block_states = [
                BlockState(
                    input_residual=block_state.input_residual,
                    delta=block_state.delta,
                    output_residual=block_state.output_residual,
                    predicted_delta=block_state.predicted_delta,
                    local_target_delta=target_delta,
                )
                for block_state, target_delta in zip(block_states, target_deltas, strict=True)
            ]

        return ModelRun(
            embedded=self.embedded_tokens(tokens),
            block_states=block_states,
            final_hidden=final_hidden,
            logits=logits,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return self.run(tokens).logits


def compute_loss_bundle(
    model: ResidualLocalLearningModel,
    tokens: Tensor,
    targets: Tensor,
    *,
    local_loss_weight: float,
) -> LossBundle:
    run = model.run(tokens)
    lm_loss = F.cross_entropy(run.logits, targets)
    local_losses: list[Tensor] = []
    for block_state in run.block_states:
        if block_state.predicted_delta is None:
            continue
        local_losses.append(
            F.mse_loss(block_state.predicted_delta, block_state.local_target_delta)
        )
    local_loss_total = (
        torch.stack(local_losses).sum()
        if local_losses
        else lm_loss.new_zeros(())
    )
    total_loss = lm_loss + local_loss_weight * local_loss_total
    accuracy = (run.logits.argmax(dim=1) == targets).float().mean().item()
    return LossBundle(
        total_loss=total_loss,
        lm_loss=lm_loss,
        local_loss_total=local_loss_total,
        local_losses=local_losses,
        logits=run.logits,
        accuracy=accuracy,
    )
