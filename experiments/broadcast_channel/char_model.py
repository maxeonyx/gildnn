from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from legacy_models.async_gru_char_model import ForwardTrace, ResidualGRUModule, TickTrace, count_parameters


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


@dataclass(frozen=True)
class BroadcastVariantSpec:
    label: str
    read_lags: tuple[int, ...]
    use_broadcast: bool


def make_sync_variant(config: TrainableConfig) -> BroadcastVariantSpec:
    return BroadcastVariantSpec(
        label="sync_no_broadcast",
        read_lags=tuple(0 for _ in range(config.num_modules)),
        use_broadcast=False,
    )


def make_async_zero_variant(config: TrainableConfig) -> BroadcastVariantSpec:
    return BroadcastVariantSpec(
        label="async_zero_no_broadcast",
        read_lags=tuple(0 for _ in range(config.num_modules)),
        use_broadcast=False,
    )


def make_async_stale_variant(config: TrainableConfig) -> BroadcastVariantSpec:
    return BroadcastVariantSpec(
        label="async_no_broadcast",
        read_lags=tuple(range(config.num_modules)),
        use_broadcast=False,
    )


def make_async_broadcast_variant(config: TrainableConfig) -> BroadcastVariantSpec:
    return BroadcastVariantSpec(
        label="async_with_broadcast",
        read_lags=tuple(range(config.num_modules)),
        use_broadcast=True,
    )


class BroadcastAsyncGRUCharModel(nn.Module):
    def __init__(self, *, vocab_size: int, config: TrainableConfig, variant: BroadcastVariantSpec) -> None:
        super().__init__()
        if len(variant.read_lags) != config.num_modules:
            raise ValueError("Variant lag count must match num_modules.")
        self.config = config
        self.variant = variant
        self.context_size = config.context_size
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_size, config.d_model)
        delta_scale = 1.0 / config.num_modules
        self.modules_bank = nn.ModuleList(
            [
                ResidualGRUModule(d_model=config.d_model, delta_scale=delta_scale)
                for _ in range(config.num_modules)
            ]
        )
        self.broadcast_proj = nn.Linear(config.d_model, config.d_model) if variant.use_broadcast else None
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size)

    def _embedded_tokens(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.shape[1]
        if sequence_length != self.context_size:
            raise ValueError(f"Expected context length {self.context_size}, got {sequence_length}.")
        positions = torch.arange(sequence_length, device=tokens.device)
        return self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)

    def _forward_impl(self, tokens: Tensor, *, capture_trace: bool) -> tuple[Tensor, ForwardTrace | None]:
        hidden = self._embedded_tokens(tokens)
        history: list[Tensor] = [hidden]
        tick_traces: list[TickTrace] = []

        for tick in range(self.config.num_ticks):
            latest_version_id = len(history) - 1
            read_version_ids = [max(0, latest_version_id - lag) for lag in self.variant.read_lags]
            visible_memories = [history[version_id] for version_id in read_version_ids]
            module_deltas = [
                module(visible_memory)
                for module, visible_memory in zip(self.modules_bank, visible_memories, strict=True)
            ]
            stacked_deltas = torch.stack(module_deltas, dim=0)
            next_hidden = history[-1] + stacked_deltas.sum(dim=0)
            if self.broadcast_proj is not None:
                next_hidden = next_hidden + self.broadcast_proj(stacked_deltas.mean(dim=0))

            if capture_trace:
                max_diffs = tuple(
                    float((visible_memory - history[-1]).abs().max().item())
                    for visible_memory in visible_memories
                )
                tick_traces.append(
                    TickTrace(
                        tick=tick,
                        latest_version_id=latest_version_id,
                        read_version_ids=tuple(read_version_ids),
                        max_read_vs_latest_abs_diff=max_diffs,
                    )
                )

            history.append(next_hidden)

        logits = self.output(self.final_norm(history[-1])[:, -1, :])
        if not capture_trace:
            return logits, None
        return logits, ForwardTrace(history=history, tick_traces=tick_traces)

    def forward(self, tokens: Tensor) -> Tensor:
        logits, _ = self._forward_impl(tokens, capture_trace=False)
        return logits

    def forward_with_trace(self, tokens: Tensor) -> tuple[Tensor, ForwardTrace]:
        logits, trace = self._forward_impl(tokens, capture_trace=True)
        if trace is None:
            raise RuntimeError("Trace missing.")
        return logits, trace


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
