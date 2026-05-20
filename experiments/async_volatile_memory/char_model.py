from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TrainableConfig:
    context_size: int = 32
    train_characters: int = 100_000
    val_characters: int = 20_000
    d_model: int = 72
    num_heads: int = 4
    num_modules: int = 3
    num_ticks: int = 4
    feedforward_dim: int = 192
    batch_size: int = 256
    benchmark_warmup_steps: int = 15
    benchmark_steps: int = 40
    memorization_batch_size: int = 32
    memorization_steps: int = 300
    learning_rate: float = 0.003
    memorization_learning_rate: float = 0.02
    gradient_clip_norm: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class VariantSpec:
    label: str
    read_lags: tuple[int, ...]


SYNC_CONTROL = VariantSpec(label="synchronous_control", read_lags=(0, 0, 0))
ASYNC_STALE = VariantSpec(label="async_stale_reads", read_lags=(0, 1, 2))
ASYNC_ZERO = VariantSpec(label="async_zero_staleness", read_lags=(0, 0, 0))


@dataclass(frozen=True)
class ForwardProfile:
    read_visibility_ms: float
    module_compute_ms: float
    commit_ms: float
    forward_wall_ms: float

    @property
    def accounted_ms(self) -> float:
        return self.read_visibility_ms + self.module_compute_ms + self.commit_ms

    @property
    def untracked_ms(self) -> float:
        return max(0.0, self.forward_wall_ms - self.accounted_ms)


class ResidualMemoryModule(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int, feedforward_dim: int, delta_scale: float) -> None:
        super().__init__()
        self.delta_scale = delta_scale
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

    def forward(self, visible_memory: Tensor, *, causal_mask: Tensor) -> Tensor:
        normalized = self.attention_norm(visible_memory)
        attention_update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden = visible_memory + attention_update
        feedforward_update = self.feedforward(self.feedforward_norm(hidden))
        return (attention_update + feedforward_update) * self.delta_scale


class AsyncVolatileMemoryCharModel(nn.Module):
    def __init__(self, *, vocab_size: int, config: TrainableConfig, variant: VariantSpec) -> None:
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
                ResidualMemoryModule(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    feedforward_dim=config.feedforward_dim,
                    delta_scale=delta_scale,
                )
                for _ in range(config.num_modules)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size)

    def _causal_mask(self, *, sequence_length: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def _embedded_tokens(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.shape[1]
        if sequence_length != self.context_size:
            raise ValueError(f"Expected context length {self.context_size}, got {sequence_length}.")
        positions = torch.arange(sequence_length, device=tokens.device)
        return self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)

    def _empty_profile(self) -> dict[str, float]:
        return {
            "read_visibility_ms": 0.0,
            "module_compute_ms": 0.0,
            "commit_ms": 0.0,
        }

    def _record_cuda_segment(self, bucket: dict[str, float], key: str, fn) -> Tensor:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = fn()
        end_event.record()
        torch.cuda.synchronize()
        bucket[key] += start_event.elapsed_time(end_event)
        return result

    def _record_cpu_segment(self, bucket: dict[str, float], key: str, fn) -> Tensor:
        start = time.perf_counter()
        result = fn()
        bucket[key] += (time.perf_counter() - start) * 1000.0
        return result

    def _forward_impl(self, tokens: Tensor, *, profile: bool) -> tuple[Tensor, ForwardProfile | None]:
        forward_start = time.perf_counter()
        hidden = self._embedded_tokens(tokens)
        causal_mask = self._causal_mask(sequence_length=tokens.shape[1], device=tokens.device)
        history: list[Tensor] = [hidden]
        profile_bucket = self._empty_profile() if profile else None

        for _ in range(self.config.num_ticks):
            latest_version_id = len(history) - 1
            read_version_ids = [max(0, latest_version_id - lag) for lag in self.variant.read_lags]
            if profile and tokens.device.type == "cuda":
                visible_memories = self._record_cuda_segment(
                    profile_bucket,
                    "read_visibility_ms",
                    lambda: [history[version_id] for version_id in read_version_ids],
                )
                module_deltas = self._record_cuda_segment(
                    profile_bucket,
                    "module_compute_ms",
                    lambda: [
                        module(visible_memory, causal_mask=causal_mask)
                        for module, visible_memory in zip(self.modules_bank, visible_memories, strict=True)
                    ],
                )
                next_hidden = self._record_cuda_segment(
                    profile_bucket,
                    "commit_ms",
                    lambda: history[-1] + torch.stack(module_deltas, dim=0).sum(dim=0),
                )
            elif profile:
                visible_memories = self._record_cpu_segment(
                    profile_bucket,
                    "read_visibility_ms",
                    lambda: [history[version_id] for version_id in read_version_ids],
                )
                module_deltas = self._record_cpu_segment(
                    profile_bucket,
                    "module_compute_ms",
                    lambda: [
                        module(visible_memory, causal_mask=causal_mask)
                        for module, visible_memory in zip(self.modules_bank, visible_memories, strict=True)
                    ],
                )
                next_hidden = self._record_cpu_segment(
                    profile_bucket,
                    "commit_ms",
                    lambda: history[-1] + torch.stack(module_deltas, dim=0).sum(dim=0),
                )
            else:
                visible_memories = [history[version_id] for version_id in read_version_ids]
                module_deltas = [
                    module(visible_memory, causal_mask=causal_mask)
                    for module, visible_memory in zip(self.modules_bank, visible_memories, strict=True)
                ]
                next_hidden = history[-1] + torch.stack(module_deltas, dim=0).sum(dim=0)
            history.append(next_hidden)

        logits = self.output(self.final_norm(history[-1])[:, -1, :])
        if not profile:
            return logits, None

        if tokens.device.type == "cuda":
            torch.cuda.synchronize()
        forward_wall_ms = (time.perf_counter() - forward_start) * 1000.0
        return logits, ForwardProfile(forward_wall_ms=forward_wall_ms, **profile_bucket)

    def forward(self, tokens: Tensor) -> Tensor:
        logits, _ = self._forward_impl(tokens, profile=False)
        return logits

    def forward_with_profile(self, tokens: Tensor) -> tuple[Tensor, ForwardProfile]:
        logits, profile = self._forward_impl(tokens, profile=True)
        if profile is None:
            raise RuntimeError("Profile data missing.")
        return logits, profile


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
