from __future__ import annotations

from dataclasses import dataclass, replace

from core.async_gru_char_model import (
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
    train_characters: int = 900_000
    val_characters: int = 100_000
    d_model: int = 512
    num_modules: int = 6
    num_ticks: int = 4
    batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 0.0015
    gradient_clip_norm: float = 1.0
    seed: int = 42


def with_overrides(
    config: TrainableConfig,
    *,
    train_characters: int | None = None,
    val_characters: int | None = None,
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
) -> TrainableConfig:
    return replace(
        config,
        train_characters=config.train_characters if train_characters is None else train_characters,
        val_characters=config.val_characters if val_characters is None else val_characters,
        batch_size=config.batch_size if batch_size is None else batch_size,
        eval_batch_size=config.eval_batch_size if eval_batch_size is None else eval_batch_size,
        learning_rate=config.learning_rate if learning_rate is None else learning_rate,
        seed=config.seed if seed is None else seed,
    )
