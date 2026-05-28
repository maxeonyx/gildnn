from __future__ import annotations

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


class Block(nn.Module):
    def __init__(self, d_stream: int, d_hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_stream, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_stream),
        )

    def forward(self, x: Float[Tensor, "batch d_stream"]) -> Float[Tensor, "batch d_stream"]:
        return self.net(x)


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
        self.noise_std = noise_std
        self.d_hidden = d_hidden if d_hidden is not None else 4 * d_stream
        self.readout_temperature = readout_temperature

        self.token_embedding = nn.Embedding(vocab_size, d_stream)
        self.blocks = nn.ModuleList(Block(d_stream, self.d_hidden) for _ in range(n_levels))
        self.prediction_heads = nn.ModuleList(nn.Linear(d_stream, d_stream) for _ in range(n_levels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.token_embedding.weight.normal_()
            self.token_embedding.weight.copy_(l2_normalize(self.token_embedding.weight))
        self.token_embedding.weight.requires_grad_(False)

    def level_parameters(self, level: int) -> list[nn.Parameter]:
        if level < 0 or level >= self.n_levels:
            raise ValueError(f"level must be in [0, {self.n_levels}), got {level}.")
        return list(self.blocks[level].parameters()) + list(self.prediction_heads[level].parameters())

    def _add_noise(self, hidden: Float[Tensor, "batch d_stream"]) -> Float[Tensor, "batch d_stream"]:
        if self.noise_std == 0.0:
            return hidden
        return hidden + (torch.randn_like(hidden) * self.noise_std)

    def logits_from_hidden(self, hidden: Float[Tensor, "batch d_stream"]) -> Float[Tensor, "batch vocab"]:
        normalized_hidden = l2_normalize(hidden)
        normalized_embedding = l2_normalize(self.token_embedding.weight)
        return F.linear(normalized_hidden, normalized_embedding) / self.readout_temperature

    def forward(self, tokens: Int[Tensor, "batch seq"]) -> AutomatonOutput:
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens with shape [batch, seq], got {tuple(tokens.shape)}.")
        if tokens.dtype != torch.long:
            raise ValueError(f"Expected tokens dtype torch.long, got {tokens.dtype}.")

        batch_size, seq_len = tokens.shape
        token_embeddings: Float[Tensor, "batch seq d_stream"] = self.token_embedding(tokens)
        zero_state = torch.zeros(batch_size, self.d_stream, device=tokens.device, dtype=token_embeddings.dtype)
        states = [zero_state.clone() for _ in range(self.n_levels)]
        lateral_buffers = [zero_state.clone() for _ in range(self.n_levels)]
        pending_predictions: list[Float[Tensor, "batch d_stream"] | None] = [None for _ in range(self.n_levels)]
        prediction_loss_sums = [torch.zeros((), device=tokens.device) for _ in range(self.n_levels)]
        prediction_counts = [torch.zeros((), device=tokens.device, dtype=torch.long) for _ in range(self.n_levels)]
        logits_per_token: list[Float[Tensor, "batch vocab"]] = []

        for token_index in range(seq_len):
            token_drive = token_embeddings[:, token_index, :]
            for step_index in range(self.steps_per_token):
                next_arrivals = [zero_state.clone() for _ in range(self.n_levels)]
                next_buffers = list(lateral_buffers)

                for level in range(self.n_levels):
                    if step_index % self.rates[level] != 0:
                        continue

                    combined_input = states[level] + lateral_buffers[level]
                    if level == 0 and step_index == 0:
                        combined_input = combined_input + token_drive
                    combined_input = l2_normalize(combined_input)

                    pending_prediction = pending_predictions[level]
                    if pending_prediction is not None:
                        prediction_loss_sums[level] = prediction_loss_sums[level] + F.mse_loss(
                            pending_prediction.float(),
                            combined_input.detach().float(),
                        )
                        prediction_counts[level] = prediction_counts[level] + 1

                    output = self.blocks[level](combined_input)
                    states[level] = output
                    pending_predictions[level] = self.prediction_heads[level](output)
                    next_buffers[level] = zero_state.clone()

                    if level + 1 < self.n_levels:
                        next_arrivals[level + 1] = next_arrivals[level + 1] + self._add_noise(output)

                lateral_buffers = [next_buffers[level] + next_arrivals[level] for level in range(self.n_levels)]

            logits_per_token.append(self.logits_from_hidden(states[0]))

        prediction_losses = []
        for level in range(self.n_levels):
            count = prediction_counts[level]
            if count.item() == 0:
                prediction_losses.append(torch.zeros((), device=tokens.device))
                continue
            prediction_losses.append(prediction_loss_sums[level] / count)

        logits = rearrange(torch.stack(logits_per_token, dim=0), "seq batch vocab -> batch seq vocab")
        prediction_loss_tensor = torch.stack(prediction_losses)
        prediction_count_tensor = torch.stack(prediction_counts)
        return AutomatonOutput(
            logits=logits,
            prediction_losses=prediction_loss_tensor,
            prediction_counts=prediction_count_tensor,
            total_prediction_loss=prediction_loss_tensor.sum(),
        )


__all__ = ["AutomatonOutput", "Block", "CellularAutomaton", "count_parameters", "l2_normalize"]
