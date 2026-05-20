from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PrototypeConfig:
    batch_size: int = 1
    num_slots: int = 2
    d_model: int = 4
    num_modules: int = 3
    hidden_dim: int = 6
    num_ticks: int = 4
    seed: int = 7


class DenseModuleBank(nn.Module):
    def __init__(self, config: PrototypeConfig) -> None:
        super().__init__()
        self.config = config
        scale_in = 0.35 / math.sqrt(config.d_model)
        scale_out = 0.20 / math.sqrt(config.hidden_dim)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)

        self.in_proj = nn.Parameter(
            torch.randn(
                config.num_modules,
                config.d_model,
                config.hidden_dim,
                generator=generator,
            )
            * scale_in
        )
        self.in_bias = nn.Parameter(
            torch.randn(config.num_modules, config.hidden_dim, generator=generator) * 0.05
        )
        self.out_proj = nn.Parameter(
            torch.randn(
                config.num_modules,
                config.hidden_dim,
                config.d_model,
                generator=generator,
            )
            * scale_out
        )
        self.out_bias = nn.Parameter(
            torch.randn(config.num_modules, config.d_model, generator=generator) * 0.03
        )
        self.module_gain = nn.Parameter(torch.linspace(0.6, 1.0, steps=config.num_modules))

    def forward(self, visible_memory: Tensor) -> Tensor:
        hidden = torch.einsum("mbsd,mdh->mbsh", visible_memory, self.in_proj)
        hidden = torch.tanh(hidden + self.in_bias[:, None, None, :])
        delta = torch.einsum("mbsh,mhd->mbsd", hidden, self.out_proj)
        delta = delta + self.out_bias[:, None, None, :]
        return delta * self.module_gain[:, None, None, None]


def make_initial_memory(config: PrototypeConfig, device: torch.device) -> Tensor:
    base = torch.tensor(
        [
            [
                [0.25, -0.50, 0.75, -1.00],
                [1.25, -1.50, 1.75, -2.00],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )
    if config.batch_size != 1 or config.num_slots != 2 or config.d_model != 4:
        raise ValueError("This Stage 2 prototype locks batch_size=1, num_slots=2, d_model=4.")
    return base


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
