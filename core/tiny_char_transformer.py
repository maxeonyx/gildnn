from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _rms(values: Tensor) -> float:
    return torch.sqrt(torch.mean(values.detach().float().square())).item()


def _branch_correlation(input_branch: Tensor, update_branch: Tensor) -> float:
    input_values = input_branch.detach().float().reshape(-1)
    update_values = update_branch.detach().float().reshape(-1)
    input_rms = torch.sqrt(torch.mean(input_values.square()))
    update_rms = torch.sqrt(torch.mean(update_values.square()))
    denominator = input_rms * update_rms
    if denominator.item() == 0.0:
        raise ValueError(
            "Residual branch correlation is undefined for zero-RMS inputs."
        )
    return torch.mean(input_values * update_values).div(denominator).item()


def _base_residual_observation(
    input_branch: Tensor,
    update_branch: Tensor,
    output_branch: Tensor,
) -> dict[str, float]:
    return {
        "input_branch_rms": _rms(input_branch),
        "update_branch_rms": _rms(update_branch),
        "combined_output_rms": _rms(output_branch),
        "branch_correlation": _branch_correlation(input_branch, update_branch),
    }


class PlainResidualCombine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._last_observation: dict[str, float] | None = None

    def forward(self, input_branch: Tensor, update_branch: Tensor) -> Tensor:
        output_branch = input_branch + update_branch
        self._last_observation = _base_residual_observation(
            input_branch=input_branch,
            update_branch=update_branch,
            output_branch=output_branch,
        )
        return output_branch

    def last_observation(self) -> dict[str, float]:
        if self._last_observation is None:
            raise RuntimeError(
                "Plain residual observation requested before any forward pass."
            )
        return dict(self._last_observation)


class ScalarMixAddResidualCombine(nn.Module):
    def __init__(
        self,
        *,
        saturation_low: float = 0.05,
        saturation_high: float = 0.95,
    ) -> None:
        super().__init__()
        self.m = nn.Parameter(torch.tensor(0.0))
        self.saturation_low = saturation_low
        self.saturation_high = saturation_high
        self._last_observation: dict[str, float] | None = None

    def forward(self, input_branch: Tensor, update_branch: Tensor) -> Tensor:
        gate = torch.sigmoid(self.m)
        input_weight = torch.sqrt(gate)
        update_weight = torch.sqrt(1.0 - gate)
        output_branch = input_branch * input_weight + update_branch * update_weight
        gate_value = gate.item()
        self._last_observation = {
            **_base_residual_observation(
                input_branch=input_branch,
                update_branch=update_branch,
                output_branch=output_branch,
            ),
            "effective_gate": gate_value,
            "input_weight": input_weight.item(),
            "update_weight": update_weight.item(),
            "gate_saturation_frequency": float(
                gate_value <= self.saturation_low or gate_value >= self.saturation_high
            ),
            "gate_saturation_low": self.saturation_low,
            "gate_saturation_high": self.saturation_high,
        }
        return output_branch

    def last_observation(self) -> dict[str, float]:
        if self._last_observation is None:
            raise RuntimeError("Mix-Add observation requested before any forward pass.")
        return dict(self._last_observation)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        *,
        residual_factory: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.attention_residual = residual_factory()
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, d_model),
        )
        self.feedforward_residual = residual_factory()

    def forward(self, x: Tensor, *, causal_mask: Tensor) -> Tensor:
        normalized = self.attention_norm(x)
        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = self.attention_residual(x, attention_output)
        x = self.feedforward_residual(x, self.feedforward(self.feedforward_norm(x)))
        return x

    def residual_observations(self) -> dict[str, dict[str, float]]:
        return {
            "attention_residual": self.attention_residual.last_observation(),
            "feedforward_residual": self.feedforward_residual.last_observation(),
        }


class TinyTransformerCharModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        context_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        residual_factory: Callable[[], nn.Module],
        within_token_reuse_depth: int = 1,
    ) -> None:
        super().__init__()
        if within_token_reuse_depth < 1:
            raise ValueError(
                "within_token_reuse_depth must be at least 1 for TinyTransformerCharModel."
            )
        self.context_size = context_size
        self.within_token_reuse_depth = within_token_reuse_depth
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    residual_factory=residual_factory,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, *, sequence_length: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

    def _embedded_tokens(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.shape[1]
        if sequence_length != self.context_size:
            raise ValueError(
                f"Expected context length {self.context_size}, got {sequence_length}."
            )
        positions = torch.arange(sequence_length, device=tokens.device)
        return self.token_embedding(tokens) + self.position_embedding(
            positions
        ).unsqueeze(0)

    def hidden_states(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.shape[1]
        x = self._embedded_tokens(tokens)
        causal_mask = self._causal_mask(
            sequence_length=sequence_length,
            device=tokens.device,
        )

        for _ in range(self.within_token_reuse_depth):
            for block in self.blocks:
                x = block(x, causal_mask=causal_mask)

        return self.final_norm(x)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.hidden_states(tokens)
        return self.output(x[:, -1, :])

    def residual_observations(self) -> dict[str, dict[str, dict[str, float]]]:
        return {
            f"block_{block_index}": block.residual_observations()
            for block_index, block in enumerate(self.blocks)
        }
