from __future__ import annotations

from dataclasses import dataclass

from legacy_models.async_gru_char_model import (
    AsyncGRUCharModel,
    ForwardTrace,
    TickTrace,
    VariantSpec,
    count_parameters,
    make_async_stale_variant,
    make_async_zero_variant,
    make_sync_variant,
)


@dataclass(frozen=True)
class TrainableConfig:
    context_size: int = 32
    train_characters: int = 100_000
    val_characters: int = 20_000
    d_model: int = 152
    num_modules: int = 6
    num_ticks: int = 4
    batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 0.003
    gradient_clip_norm: float = 1.0
    memorization_batch_size: int = 32
    memorization_steps: int = 300
    memorization_learning_rate: float = 0.02
    seed: int = 42
