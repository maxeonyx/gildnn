from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F


class SelfPredictionLoss(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_aux: int = 32,
        num_blocks: int,
        rates: tuple[int, ...] | list[int],
    ) -> None:
        super().__init__()
        resolved_rates = tuple(rates)
        if len(resolved_rates) != num_blocks:
            raise ValueError(f"Expected {num_blocks} rates, got {resolved_rates}.")
        if num_blocks < 2:
            raise ValueError(f"SelfPredictionLoss requires at least 2 blocks, got {num_blocks}.")
        if any(rate <= 0 for rate in resolved_rates):
            raise ValueError(f"Rates must be positive, got {resolved_rates}.")

        self.d_model = d_model
        self.d_aux = d_aux
        self.num_blocks = num_blocks
        self.rates = resolved_rates
        self.block_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_blocks)])
        self.align = nn.Linear(d_model, d_aux, bias=False)

    def _pair_mask(
        self,
        *,
        batch_size: int,
        context_size: int,
        slow_rate: int,
        device: torch.device,
    ) -> Tensor:
        fire_steps = torch.arange(context_size, device=device) % slow_rate == 0
        return fire_steps.unsqueeze(0).expand(batch_size, context_size)

    def forward(
        self,
        block_outputs: list[Float[Tensor, "batch context d_model"]],
    ) -> Float[Tensor, ""]:
        if len(block_outputs) != self.num_blocks:
            raise ValueError(f"Expected {self.num_blocks} block outputs, got {len(block_outputs)}.")

        batch_size, context_size, _ = block_outputs[0].shape
        pair_losses: list[Tensor] = []
        for block_index in range(self.num_blocks - 1):
            fast_output = block_outputs[block_index]
            slow_output = block_outputs[block_index + 1]
            if fast_output.shape != (batch_size, context_size, self.d_model):
                raise ValueError(
                    f"Expected block {block_index} output shape {(batch_size, context_size, self.d_model)}, "
                    f"got {tuple(fast_output.shape)}."
                )
            if slow_output.shape != (batch_size, context_size, self.d_model):
                raise ValueError(
                    f"Expected block {block_index + 1} output shape {(batch_size, context_size, self.d_model)}, "
                    f"got {tuple(slow_output.shape)}."
                )

            fast_normalized = self.block_norms[block_index](fast_output)
            slow_normalized = self.block_norms[block_index + 1](slow_output)
            fast_projected = F.normalize(self.align(fast_normalized), dim=-1)
            slow_projected = F.normalize(self.align(slow_normalized), dim=-1)
            cosine_distance = 1.0 - (fast_projected * slow_projected.detach()).sum(dim=-1)
            fire_mask = self._pair_mask(
                batch_size=batch_size,
                context_size=context_size,
                slow_rate=self.rates[block_index + 1],
                device=slow_output.device,
            )
            masked_loss = cosine_distance.masked_select(fire_mask)
            if masked_loss.numel() == 0:
                continue
            pair_losses.append(masked_loss.mean())

        if len(pair_losses) == 0:
            return block_outputs[0].new_zeros(())
        return torch.stack(pair_losses).mean()
