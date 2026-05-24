from __future__ import annotations

import argparse
from collections.abc import Iterator
import gc
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.dataset import CorpusData, load_corpus
from core.fixed_window_char import set_seed
from core.model import count_parameters
from core.training import capturable_adamw, current_git_sha, current_git_status_short, write_json

CONTEXT_SIZE = 128
TRAINING_STEPS = 20_000
EVAL_INTERVAL = 1_000
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 3e-4
DEFAULT_SEEDS = (42, 43)
D_MODEL = 256
FEEDFORWARD_DIM = 512
EVAL_SAMPLES = 4096
PREDICTION_LOSS_WEIGHT = 0.1
LOCAL_CE_WEIGHT = 1.0
SANITY_STEPS = 100
SANITY_EVAL_INTERVAL = 25
WARMUP_STEPS = 3


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    num_blocks: int
    rates: tuple[int, ...]
    readout_mode: str
    closed_loop: bool
    strict_local: bool = False
    local_ce: bool = False
    phases: tuple[int, ...] = ()
    per_helper_prediction_loss: bool = False

    def __post_init__(self) -> None:
        if len(self.rates) != self.num_blocks:
            raise ValueError(
                f"VariantSpec {self.key!r} requires len(rates) == num_blocks, got {len(self.rates)} and {self.num_blocks}."
            )
        phases = self.phases if len(self.phases) > 0 else tuple(0 for _ in range(self.num_blocks))
        if len(phases) != self.num_blocks:
            raise ValueError(
                f"VariantSpec {self.key!r} requires len(phases) == num_blocks, got {len(phases)} and {self.num_blocks}."
            )
        for index, (phase, rate) in enumerate(zip(phases, self.rates, strict=True)):
            if not 0 <= phase < rate:
                raise ValueError(
                    f"VariantSpec {self.key!r} requires 0 <= phases[{index}] < rates[{index}], got phase={phase}, rate={rate}."
                )
        object.__setattr__(self, "phases", phases)


@dataclass(frozen=True)
class LossBreakdown:
    ce_loss: Tensor
    pred_loss: Tensor
    local_ce_loss: Tensor
    total_loss: Tensor


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "wikitext_103" / "artifacts" / "closed_loop_prediction"
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--eval-samples", type=int, default=EVAL_SAMPLES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["A_single", "B_spectator", "C_closed_loop", "D_strict_local", "E_grounded"],
    )
    parser.add_argument("--compile", dest="compile_model", action="store_true")
    parser.add_argument("--no-compile", dest="compile_model", action="store_false")
    parser.set_defaults(compile_model=True)
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--sanity-steps", type=int, default=SANITY_STEPS)
    parser.add_argument("--sanity-eval-interval", type=int, default=SANITY_EVAL_INTERVAL)
    parser.add_argument(
        "--train-path",
        type=Path,
        default=repo_root / "data" / "wikitext-103-raw" / "wiki.train.raw",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=repo_root / "data" / "wikitext-103-raw" / "wiki.valid.raw",
    )
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.jsonl")
    return parser.parse_args()


def variant_specs() -> dict[str, VariantSpec]:
    return {
        "A_single": VariantSpec(
            key="A_single",
            label="closed_loop_prediction_A_single",
            num_blocks=1,
            rates=(1,),
            phases=(0,),
            readout_mode="block0",
            closed_loop=False,
        ),
        "B_spectator": VariantSpec(
            key="B_spectator",
            label="closed_loop_prediction_B_spectator",
            num_blocks=2,
            rates=(1, 2),
            phases=(0, 0),
            readout_mode="weighted",
            closed_loop=False,
        ),
        "C_closed_loop": VariantSpec(
            key="C_closed_loop",
            label="closed_loop_prediction_C_closed_loop",
            num_blocks=2,
            rates=(1, 2),
            phases=(0, 0),
            readout_mode="block0",
            closed_loop=True,
        ),
        "D_strict_local": VariantSpec(
            key="D_strict_local",
            label="closed_loop_prediction_D_strict_local",
            num_blocks=2,
            rates=(1, 2),
            phases=(0, 0),
            readout_mode="block0",
            closed_loop=True,
            strict_local=True,
        ),
        "E_grounded": VariantSpec(
            key="E_grounded",
            label="closed_loop_prediction_E_grounded",
            num_blocks=2,
            rates=(1, 2),
            phases=(0, 0),
            readout_mode="block0",
            closed_loop=True,
            strict_local=True,
            local_ce=True,
        ),
        "F_star_3block": VariantSpec(
            key="F_star_3block",
            label="closed_loop_prediction_F_star_3block",
            num_blocks=3,
            rates=(1, 2, 4),
            phases=(0, 0, 0),
            readout_mode="block0",
            closed_loop=True,
        ),
        "G_rate4_only": VariantSpec(
            key="G_rate4_only",
            label="closed_loop_prediction_G_rate4_only",
            num_blocks=2,
            rates=(1, 4),
            phases=(0, 0),
            readout_mode="block0",
            closed_loop=True,
        ),
        "I_phase_offset": VariantSpec(
            key="I_phase_offset",
            label="closed_loop_prediction_I_phase_offset",
            num_blocks=3,
            rates=(1, 2, 2),
            phases=(0, 0, 1),
            readout_mode="block0",
            closed_loop=True,
            per_helper_prediction_loss=True,
        ),
        "I_control": VariantSpec(
            key="I_control",
            label="closed_loop_prediction_I_control",
            num_blocks=3,
            rates=(1, 2, 2),
            phases=(0, 0, 0),
            readout_mode="block0",
            closed_loop=True,
            per_helper_prediction_loss=True,
        ),
    }


def append_log(log_path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


class MixAdd(nn.Module):
    def __init__(self, init_keep: float = 0.9) -> None:
        super().__init__()
        if not 0.0 < init_keep < 1.0:
            raise ValueError(f"MixAdd requires 0 < init_keep < 1, got {init_keep}.")
        self.mix_logit = nn.Parameter(torch.tensor(math.log(init_keep / (1.0 - init_keep)), dtype=torch.float32))

    def forward(self, keep: Tensor, add: Tensor) -> Tensor:
        mix = torch.sigmoid(self.mix_logit)
        return keep * mix.sqrt() + add * (1.0 - mix).sqrt()

    @torch.no_grad()
    def coefficient_value(self) -> float:
        return float(torch.sigmoid(self.mix_logit).item())


class FeedForwardBlock(nn.Module):
    def __init__(self, *, d_model: int, feedforward_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear_in = nn.Linear(d_model, feedforward_dim)
        self.activation = nn.GELU()
        self.linear_out = nn.Linear(feedforward_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_out(self.activation(self.linear_in(self.norm(x))))


class ClosedLoopPredictionModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        context_size: int,
        d_model: int,
        feedforward_dim: int,
        spec: VariantSpec,
    ) -> None:
        super().__init__()
        if spec.num_blocks not in {1, 2, 3}:
            raise ValueError(f"ClosedLoopPredictionModel supports 1, 2, or 3 blocks, got {spec.num_blocks}.")
        if spec.num_blocks != len(spec.rates):
            raise ValueError(
                "ClosedLoopPredictionModel requires one rate per block. "
                f"Got num_blocks={spec.num_blocks} and rates={spec.rates}."
            )
        if spec.closed_loop and spec.num_blocks not in {2, 3}:
            raise ValueError("Closed-loop prediction requires exactly two or three blocks.")
        if spec.readout_mode not in {"block0", "weighted"}:
            raise ValueError(f"Unsupported readout_mode {spec.readout_mode!r}.")

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model
        self.feedforward_dim = feedforward_dim
        self.spec = spec
        self.num_blocks = spec.num_blocks

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.token_mix = MixAdd(init_keep=0.9)
        self.block0_ffn = FeedForwardBlock(d_model=d_model, feedforward_dim=feedforward_dim)
        self.block0_mix = MixAdd(init_keep=0.9)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.block1_ffn: FeedForwardBlock | None = None
        self.block1_mix: MixAdd | None = None
        self.block2_ffn: FeedForwardBlock | None = None
        self.block2_mix: MixAdd | None = None
        self.prior_gain_1: nn.Parameter | None = None
        self.prior_gain_2: nn.Parameter | None = None
        self.prior_norm_1: nn.LayerNorm | None = None
        self.prior_norm_2: nn.LayerNorm | None = None
        self.prediction_head_1: nn.Linear | None = None
        self.prediction_head_2: nn.Linear | None = None
        self.readout_logits: nn.Parameter | None = None
        self.block1_norm: nn.LayerNorm | None = None
        self.block1_lm_head: nn.Linear | None = None

        if spec.num_blocks >= 2:
            self.block1_ffn = FeedForwardBlock(d_model=d_model, feedforward_dim=feedforward_dim)
            self.block1_mix = MixAdd(init_keep=0.9)
        if spec.num_blocks == 3:
            self.block2_ffn = FeedForwardBlock(d_model=d_model, feedforward_dim=feedforward_dim)
            self.block2_mix = MixAdd(init_keep=0.9)
        if spec.closed_loop:
            self.prior_gain_1 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.prior_norm_1 = nn.LayerNorm(d_model)
            self.prediction_head_1 = nn.Linear(d_model, spec.rates[1] * d_model)
            if spec.num_blocks == 3:
                self.prior_gain_2 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
                self.prior_norm_2 = nn.LayerNorm(d_model)
                self.prediction_head_2 = nn.Linear(d_model, spec.rates[2] * d_model)
        if spec.local_ce:
            self.block1_norm = nn.LayerNorm(d_model)
            self.block1_lm_head = nn.Linear(d_model, vocab_size)
        if spec.readout_mode == "weighted":
            self.readout_logits = nn.Parameter(torch.zeros(spec.num_blocks, dtype=torch.float32))

    def embedded_tokens(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"embedded_tokens expects [batch, context] tokens, got shape {tuple(tokens.shape)}.")
        if tokens.shape[1] != self.context_size:
            raise ValueError(f"Expected context length {self.context_size}, got {tokens.shape[1]}.")
        positions = torch.arange(self.context_size, device=tokens.device)
        return self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)

    @staticmethod
    def _pop_prior_buffer(prior_buffer: Tensor, prior_valid_buffer: Tensor) -> tuple[Tensor, Tensor]:
        prior_t = prior_buffer[:, 0, :].clone()
        prior_valid_t = prior_valid_buffer[0].clone()
        prior_buffer[:, :-1, :].copy_(prior_buffer[:, 1:, :])
        prior_buffer[:, -1, :].zero_()
        prior_valid_buffer[:-1].copy_(prior_valid_buffer[1:])
        prior_valid_buffer[-1:].fill_(False)
        return prior_t, prior_valid_t

    def readout_state(self, states: tuple[Tensor, ...]) -> Tensor:
        if self.spec.readout_mode == "block0":
            return states[0]
        if self.readout_logits is None:
            raise RuntimeError("Weighted readout requested but readout_logits is missing.")
        stacked_states = torch.stack(states, dim=1)
        readout_weights = torch.softmax(self.readout_logits.to(device=states[0].device, dtype=states[0].dtype), dim=0)
        return (stacked_states * readout_weights.view(1, self.num_blocks, 1)).sum(dim=1)

    def forward(
        self, tokens: Tensor, *, use_priors: bool = True
    ) -> tuple[Tensor, Tensor | None, Tensor, tuple[Tensor, ...], tuple[Tensor, ...], Tensor]:
        embeddings = self.embedded_tokens(tokens)
        batch_size = tokens.shape[0]
        device = embeddings.device

        s0 = embeddings.new_zeros((batch_size, self.d_model))
        s1 = embeddings.new_zeros((batch_size, self.d_model))
        s2 = embeddings.new_zeros((batch_size, self.d_model))
        zero_state = embeddings.new_zeros((batch_size, self.d_model))
        block1_rate = self.spec.rates[1] if self.spec.num_blocks >= 2 else 1
        prior_buffer_1 = embeddings.new_zeros((batch_size, block1_rate, self.d_model))
        prior_valid_buffer_1 = torch.zeros((block1_rate,), device=device, dtype=torch.bool)
        prior_buffer_2 = None
        prior_valid_buffer_2 = None
        if self.spec.num_blocks == 3:
            prior_buffer_2 = embeddings.new_zeros((batch_size, self.spec.rates[2], self.d_model))
            prior_valid_buffer_2 = torch.zeros((self.spec.rates[2],), device=device, dtype=torch.bool)
        state0_history = embeddings.new_zeros((batch_size, self.context_size, self.d_model))
        prior_history_1 = embeddings.new_zeros((batch_size, self.context_size, self.d_model))
        prior_valid_history_1 = torch.zeros((self.context_size,), device=device, dtype=torch.bool)
        prior_history_2 = None
        prior_valid_history_2 = None
        if self.spec.num_blocks == 3:
            prior_history_2 = embeddings.new_zeros((batch_size, self.context_size, self.d_model))
            prior_valid_history_2 = torch.zeros((self.context_size,), device=device, dtype=torch.bool)

        allow_priors = self.spec.closed_loop and use_priors
        for time_index in range(self.context_size):
            has_prior = allow_priors and time_index > 0
            if has_prior:
                prior_t_1, prior_valid_t_1 = self._pop_prior_buffer(prior_buffer_1, prior_valid_buffer_1)
                if prior_buffer_2 is None or prior_valid_buffer_2 is None:
                    prior_t_2 = zero_state
                    prior_valid_t_2 = torch.zeros((), device=device, dtype=torch.bool)
                else:
                    prior_t_2, prior_valid_t_2 = self._pop_prior_buffer(prior_buffer_2, prior_valid_buffer_2)
            else:
                prior_t_1 = zero_state
                prior_valid_t_1 = torch.zeros((), device=device, dtype=torch.bool)
                prior_t_2 = zero_state
                prior_valid_t_2 = torch.zeros((), device=device, dtype=torch.bool)

            token_state = embeddings[:, time_index, :]
            seed0 = self.token_mix(s0, token_state)
            x0 = seed0
            if has_prior and self.prior_gain_1 is not None and self.prior_norm_1 is not None:
                effective_prior_1 = prior_t_1.detach() if self.spec.strict_local else prior_t_1
                valid_mask_1 = prior_valid_t_1.float()
                x0 = x0 + valid_mask_1 * self.prior_gain_1 * self.prior_norm_1(effective_prior_1)
            if has_prior and self.prior_gain_2 is not None and self.prior_norm_2 is not None:
                effective_prior_2 = prior_t_2.detach() if self.spec.strict_local else prior_t_2
                valid_mask_2 = prior_valid_t_2.float()
                x0 = x0 + valid_mask_2 * self.prior_gain_2 * self.prior_norm_2(effective_prior_2)
            delta0 = self.block0_ffn(x0)
            s0 = self.block0_mix(x0, delta0)

            if (
                self.block1_ffn is not None
                and self.block1_mix is not None
                and (time_index - self.spec.phases[1]) % self.spec.rates[1] == 0
            ):
                x1 = 0.5 * (s1 + s0.detach())
                delta1 = self.block1_ffn(x1)
                s1 = self.block1_mix(x1, delta1)
                if self.prediction_head_1 is not None:
                    pred_pair = self.prediction_head_1(s1).view(batch_size, self.spec.rates[1], self.d_model)
                    prior_buffer_1.copy_(pred_pair)
                    prior_valid_buffer_1.fill_(True)

            if (
                self.block2_ffn is not None
                and self.block2_mix is not None
                and (time_index - self.spec.phases[2]) % self.spec.rates[2] == 0
            ):
                x2 = 0.5 * (s2 + s0.detach())
                delta2 = self.block2_ffn(x2)
                s2 = self.block2_mix(x2, delta2)
                if (
                    self.prediction_head_2 is not None
                    and prior_buffer_2 is not None
                    and prior_valid_buffer_2 is not None
                ):
                    pred_quad = self.prediction_head_2(s2).view(batch_size, self.spec.rates[2], self.d_model)
                    prior_buffer_2.copy_(pred_quad)
                    prior_valid_buffer_2.fill_(True)

            state0_history[:, time_index, :].copy_(s0)
            prior_history_1[:, time_index, :].copy_(prior_t_1)
            prior_valid_history_1[time_index] = prior_valid_t_1
            if prior_history_2 is not None and prior_valid_history_2 is not None:
                prior_history_2[:, time_index, :].copy_(prior_t_2)
                prior_valid_history_2[time_index] = prior_valid_t_2

        states = (s0,) if self.num_blocks == 1 else (s0, s1) if self.num_blocks == 2 else (s0, s1, s2)
        readout_state = self.readout_state(states)
        logits = self.lm_head(self.final_norm(readout_state))
        block1_logits = None
        if self.block1_norm is not None and self.block1_lm_head is not None:
            block1_logits = self.block1_lm_head(self.block1_norm(s1))
        final_states = torch.stack(states, dim=1)
        prior_histories = (prior_history_1,) if prior_history_2 is None else (prior_history_1, prior_history_2)
        prior_valid_histories = (
            (prior_valid_history_1,) if prior_valid_history_2 is None else (prior_valid_history_1, prior_valid_history_2)
        )
        return logits, block1_logits, state0_history, prior_histories, prior_valid_histories, final_states

    @torch.no_grad()
    def mix_coefficients(self) -> dict[str, object]:
        readout_weights = None
        if self.readout_logits is not None:
            readout_weights = torch.softmax(self.readout_logits, dim=0).cpu().tolist()
        prediction_mix_1 = None
        prediction_mix_2 = None
        if self.prior_gain_1 is not None:
            prediction_mix_1 = float(self.prior_gain_1.item())
        if self.prior_gain_2 is not None:
            prediction_mix_2 = float(self.prior_gain_2.item())
        return {
            "token_mix": self.token_mix.coefficient_value(),
            "block0_mix": self.block0_mix.coefficient_value(),
            "block1_mix": self.block1_mix.coefficient_value() if self.block1_mix is not None else None,
            "block2_mix": self.block2_mix.coefficient_value() if self.block2_mix is not None else None,
            "prediction_mix": prediction_mix_1,
            "prediction_mix_1": prediction_mix_1,
            "prediction_mix_2": prediction_mix_2,
            "readout_weights": readout_weights,
            "rates": list(self.spec.rates),
            "phases": list(self.spec.phases),
            "readout_mode": self.spec.readout_mode,
            "closed_loop": self.spec.closed_loop,
            "per_helper_prediction_loss": self.spec.per_helper_prediction_loss,
        }


def prediction_loss_terms(
    *,
    prior_history: Tensor,
    state0_history: Tensor,
    prior_valid_history: Tensor,
) -> tuple[Tensor, Tensor]:
    d_model = state0_history.shape[-1]
    pred_normalized = F.layer_norm(prior_history, (d_model,))
    target_normalized = F.layer_norm(state0_history.detach(), (d_model,))
    cosine_distance = 1.0 - F.cosine_similarity(pred_normalized, target_normalized, dim=-1, eps=1e-6)
    valid_weights = prior_valid_history.to(device=state0_history.device, dtype=state0_history.dtype).view(1, -1)
    loss_sum = (cosine_distance * valid_weights).sum()
    valid_count = valid_weights.sum() * state0_history.shape[0]
    return loss_sum, valid_count


def combined_prior_histories(
    *,
    prior_histories: tuple[Tensor, ...],
    prior_valid_histories: tuple[Tensor, ...],
) -> tuple[Tensor, Tensor]:
    if len(prior_histories) != len(prior_valid_histories):
        raise ValueError(
            "combined_prior_histories requires matching prior history and validity tuples, "
            f"got {len(prior_histories)} and {len(prior_valid_histories)}."
        )
    if len(prior_histories) == 0:
        raise ValueError("combined_prior_histories requires at least one prior history.")
    combined_prior = torch.stack(prior_histories, dim=0).sum(dim=0)
    combined_valid = torch.stack(prior_valid_histories, dim=0).any(dim=0)
    return combined_prior, combined_valid


def compute_losses(
    *,
    logits: Tensor,
    block1_logits: Tensor | None,
    targets: Tensor,
    state0_history: Tensor,
    prior_histories: tuple[Tensor, ...],
    prior_valid_histories: tuple[Tensor, ...],
    prediction_loss_weight: float,
    local_ce_weight: float,
    per_helper_prediction_loss: bool,
) -> LossBreakdown:
    ce_loss = F.cross_entropy(logits, targets)
    if len(prior_histories) != len(prior_valid_histories):
        raise ValueError(
            "compute_losses requires matching prior history and validity tuples, "
            f"got {len(prior_histories)} and {len(prior_valid_histories)}."
        )
    pred_loss = ce_loss.new_zeros(())
    if per_helper_prediction_loss:
        for prior_history, prior_valid_history in zip(prior_histories, prior_valid_histories, strict=True):
            pred_loss_sum, pred_valid_count = prediction_loss_terms(
                prior_history=prior_history,
                state0_history=state0_history,
                prior_valid_history=prior_valid_history,
            )
            pred_loss = pred_loss + (pred_loss_sum / pred_valid_count.clamp_min(1.0))
    else:
        combined_prior, combined_valid = combined_prior_histories(
            prior_histories=prior_histories,
            prior_valid_histories=prior_valid_histories,
        )
        pred_loss_sum, pred_valid_count = prediction_loss_terms(
            prior_history=combined_prior,
            state0_history=state0_history,
            prior_valid_history=combined_valid,
        )
        pred_loss = pred_loss_sum / pred_valid_count.clamp_min(1.0)
    local_ce_loss = ce_loss.new_zeros(())
    if block1_logits is not None:
        local_ce_loss = F.cross_entropy(block1_logits, targets)
    total_loss = ce_loss + (prediction_loss_weight * pred_loss) + (local_ce_weight * local_ce_loss)
    return LossBreakdown(ce_loss=ce_loss, pred_loss=pred_loss, local_ce_loss=local_ce_loss, total_loss=total_loss)


def build_model(
    *,
    device: torch.device,
    vocab_size: int,
    spec: VariantSpec,
) -> ClosedLoopPredictionModel:
    return ClosedLoopPredictionModel(
        vocab_size=vocab_size,
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        spec=spec,
    ).to(device)


def maybe_compile_model(
    model: ClosedLoopPredictionModel,
    *,
    enabled: bool,
) -> nn.Module:
    if not enabled:
        return model
    return torch.compile(model, backend="aot_eager")


class ClosedLoopPredictionGraphTrainer:
    def __init__(
        self,
        *,
        model: ClosedLoopPredictionModel,
        optimizer: torch.optim.AdamW,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> None:
        if device.type != "cuda":
            raise ValueError(f"ClosedLoopPredictionGraphTrainer requires CUDA, got {device}.")
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.capture_stream = torch.cuda.Stream(device=device)
        self.graph = torch.cuda.CUDAGraph()
        self.is_captured = False

        self.static_input = torch.empty((batch_size, seq_len), device=device, dtype=torch.long)
        self.static_target = torch.empty((batch_size,), device=device, dtype=torch.long)
        self.static_ce_loss = torch.zeros((), device=device)
        self.static_pred_loss = torch.zeros((), device=device)
        self.static_local_ce_loss = torch.zeros((), device=device)
        self.static_total_loss = torch.zeros((), device=device)

    def _copy_batch(self, *, batch_input: Tensor, batch_target: Tensor) -> None:
        expected_input_shape = (self.batch_size, self.seq_len)
        expected_target_shape = (self.batch_size,)
        if tuple(batch_input.shape) != expected_input_shape:
            raise ValueError(f"Expected input shape {expected_input_shape}, got {tuple(batch_input.shape)}.")
        if tuple(batch_target.shape) != expected_target_shape:
            raise ValueError(f"Expected target shape {expected_target_shape}, got {tuple(batch_target.shape)}.")
        self.static_input.copy_(batch_input, non_blocking=True)
        self.static_target.copy_(batch_target, non_blocking=True)

    def _training_step(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        logits, block1_logits, state0_history, prior_histories, prior_valid_histories, _ = self.model(
            self.static_input,
            use_priors=True,
        )
        losses = compute_losses(
            logits=logits,
            block1_logits=block1_logits,
            targets=self.static_target,
            state0_history=state0_history,
            prior_histories=prior_histories,
            prior_valid_histories=prior_valid_histories,
            prediction_loss_weight=PREDICTION_LOSS_WEIGHT,
            local_ce_weight=LOCAL_CE_WEIGHT,
            per_helper_prediction_loss=self.model.spec.per_helper_prediction_loss,
        )
        losses.total_loss.backward()
        self.optimizer.step()
        self.static_ce_loss.copy_(losses.ce_loss.detach())
        self.static_pred_loss.copy_(losses.pred_loss.detach())
        self.static_local_ce_loss.copy_(losses.local_ce_loss.detach())
        self.static_total_loss.copy_(losses.total_loss.detach())

    def capture(self, warmup_batches: list[tuple[Tensor, Tensor]]) -> None:
        if self.is_captured:
            raise RuntimeError("capture() may only be called once.")
        if len(warmup_batches) < WARMUP_STEPS:
            raise ValueError(f"Need at least {WARMUP_STEPS} warmup batches, got {len(warmup_batches)}.")

        self.model.train()
        current_stream = torch.cuda.current_stream(device=self.device)
        self.capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.capture_stream):
            for batch_input, batch_target in warmup_batches[:WARMUP_STEPS]:
                self._copy_batch(batch_input=batch_input, batch_target=batch_target)
                self._training_step()
        current_stream.wait_stream(self.capture_stream)
        torch.cuda.synchronize(self.device)

        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self._training_step()

        current_stream.wait_stream(self.capture_stream)
        self.is_captured = True

    def step(self, *, batch_input: Tensor, batch_target: Tensor) -> LossBreakdown:
        if not self.is_captured:
            raise RuntimeError("step() requires capture() first.")
        self._copy_batch(batch_input=batch_input, batch_target=batch_target)
        self.graph.replay()
        return self.snapshot()

    def snapshot(self) -> LossBreakdown:
        return LossBreakdown(
            ce_loss=self.static_ce_loss.detach().clone(),
            pred_loss=self.static_pred_loss.detach().clone(),
            local_ce_loss=self.static_local_ce_loss.detach().clone(),
            total_loss=self.static_total_loss.detach().clone(),
        )

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)


def random_batches(
    encoded_corpus: torch.Tensor,
    *,
    context_size: int,
    batch_size: int,
    device: torch.device,
    rng: torch.Generator,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if encoded_corpus.ndim != 1:
        raise ValueError(f"random_batches expects a 1D corpus tensor, got shape {tuple(encoded_corpus.shape)}.")
    if encoded_corpus.dtype != torch.long:
        raise ValueError(f"random_batches expects torch.long tokens, got {encoded_corpus.dtype}.")

    max_start = encoded_corpus.numel() - context_size
    if max_start <= 0:
        raise ValueError(
            "random_batches needs more encoded tokens than context_size. "
            f"Got corpus length {encoded_corpus.numel()} and context_size {context_size}."
        )

    offsets = torch.arange(context_size, dtype=torch.long)
    pin_memory = device.type == "cuda"
    while True:
        starts = torch.randint(0, max_start, (batch_size,), generator=rng)
        inputs = encoded_corpus[starts[:, None] + offsets]
        targets = encoded_corpus[starts + context_size]
        if pin_memory:
            inputs = inputs.pin_memory()
            targets = targets.pin_memory()
        yield (
            inputs.to(device=device, dtype=torch.long, non_blocking=pin_memory),
            targets.to(device=device, dtype=torch.long, non_blocking=pin_memory),
        )


@torch.inference_mode()
def evaluate_variant(
    *,
    model: ClosedLoopPredictionModel,
    model_call: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    batch_size: int,
    disable_priors: bool,
) -> dict[str, float | None | list[float]]:
    was_training = model.training
    model.eval()
    if model_call is not model:
        model_call.eval()

    mix_coefficients = model.mix_coefficients()

    total_examples = 0
    total_ce_loss = 0.0
    total_correct = 0
    total_pred_loss_sum = 0.0
    total_pred_count = 0.0
    total_pred_loss_sum_by_helper: list[float] | None = None
    total_pred_count_by_helper: list[float] | None = None
    total_block1_ce_loss = 0.0
    total_block1_correct = 0

    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        batch_inputs = inputs[start:stop]
        batch_targets = targets[start:stop]
        logits, block1_logits, state0_history, prior_histories, prior_valid_histories, _ = model_call(
            batch_inputs,
            use_priors=not disable_priors,
        )
        batch_examples = batch_targets.shape[0]
        total_ce_loss += F.cross_entropy(logits, batch_targets, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
        if block1_logits is not None:
            total_block1_ce_loss += F.cross_entropy(block1_logits, batch_targets, reduction="sum").item()
            total_block1_correct += (block1_logits.argmax(dim=1) == batch_targets).sum().item()
        if model.spec.per_helper_prediction_loss:
            if total_pred_loss_sum_by_helper is None or total_pred_count_by_helper is None:
                helper_count = len(prior_histories)
                total_pred_loss_sum_by_helper = [0.0 for _ in range(helper_count)]
                total_pred_count_by_helper = [0.0 for _ in range(helper_count)]
            for helper_index, (prior_history, prior_valid_history) in enumerate(
                zip(prior_histories, prior_valid_histories, strict=True)
            ):
                helper_loss_sum, helper_count = prediction_loss_terms(
                    prior_history=prior_history,
                    state0_history=state0_history,
                    prior_valid_history=prior_valid_history,
                )
                total_pred_loss_sum_by_helper[helper_index] += helper_loss_sum.item()
                total_pred_count_by_helper[helper_index] += float(helper_count.item())
        else:
            combined_prior, combined_valid = combined_prior_histories(
                prior_histories=prior_histories,
                prior_valid_histories=prior_valid_histories,
            )
            pred_loss_sum_tensor, pred_count_tensor = prediction_loss_terms(
                prior_history=combined_prior,
                state0_history=state0_history,
                prior_valid_history=combined_valid,
            )
            pred_loss_sum = pred_loss_sum_tensor.item()
            pred_count = float(pred_count_tensor.item())
            total_pred_loss_sum += pred_loss_sum
            total_pred_count += pred_count
        total_examples += batch_examples

    if was_training:
        model.train()
        if model_call is not model:
            model_call.train()

    if model.spec.per_helper_prediction_loss:
        if total_pred_loss_sum_by_helper is None or total_pred_count_by_helper is None:
            pred_loss = 0.0
        else:
            pred_loss = 0.0
            for helper_loss_sum, helper_count in zip(total_pred_loss_sum_by_helper, total_pred_count_by_helper, strict=True):
                pred_loss += 0.0 if helper_count == 0.0 else helper_loss_sum / helper_count
    else:
        pred_loss = 0.0 if total_pred_count == 0.0 else total_pred_loss_sum / total_pred_count
    return {
        "loss": total_ce_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "pred_loss": pred_loss,
        "block1_ce_loss": None if model.block1_lm_head is None else total_block1_ce_loss / total_examples,
        "block1_accuracy": None if model.block1_lm_head is None else total_block1_correct / total_examples,
        "prediction_mix_coeff": mix_coefficients["prediction_mix_1"],
        "prediction_mix_coeff_2": mix_coefficients["prediction_mix_2"],
        "readout_weights": mix_coefficients["readout_weights"],
    }


def round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def rounded_list(values: list[float] | None) -> list[float] | None:
    if values is None:
        return None
    return [round(float(value), 6) for value in values]


def checkpoint_metrics(
    *,
    model: ClosedLoopPredictionModel,
    model_call: nn.Module,
    val_inputs: Tensor,
    val_targets: Tensor,
    batch_size: int,
    step: int,
) -> dict[str, object]:
    metrics = evaluate_variant(
        model=model,
        model_call=model_call,
        inputs=val_inputs,
        targets=val_targets,
        batch_size=batch_size,
        disable_priors=False,
    )
    ablated_val_loss = None
    if model.spec.closed_loop:
        ablated_metrics = evaluate_variant(
            model=model,
            model_call=model_call,
            inputs=val_inputs,
            targets=val_targets,
            batch_size=batch_size,
            disable_priors=True,
        )
        ablated_val_loss = round_float(ablated_metrics["loss"])
    return {
        "step": step,
        "val_loss": round_float(metrics["loss"]),
        "val_accuracy": round_float(metrics["accuracy"]),
        "pred_loss": round_float(metrics["pred_loss"]),
        "block1_ce_loss": round_float(metrics["block1_ce_loss"]),
        "block1_accuracy": round_float(metrics["block1_accuracy"]),
        "prediction_mix_coeff": round_float(metrics["prediction_mix_coeff"]),
        "prediction_mix_coeff_2": round_float(metrics["prediction_mix_coeff_2"]),
        "ablated_val_loss": ablated_val_loss,
        "readout_weights": rounded_list(metrics["readout_weights"]),
    }


def verify_forward_and_gradients(
    *,
    model: ClosedLoopPredictionModel,
    model_call: nn.Module,
    device: torch.device,
    vocab_size: int,
) -> dict[str, object]:
    saved_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    def collect_gradient_stats() -> tuple[float, int, int, list[str], list[str], LossBreakdown]:
        model.train()
        if model_call is not model:
            model_call.train()
        dummy_inputs = torch.randint(0, vocab_size, (4, CONTEXT_SIZE), device=device)
        dummy_targets = torch.randint(0, vocab_size, (4,), device=device)
        logits, block1_logits, state0_history, prior_histories, prior_valid_histories, _ = model_call(
            dummy_inputs,
            use_priors=True,
        )
        losses = compute_losses(
            logits=logits,
            block1_logits=block1_logits,
            targets=dummy_targets,
            state0_history=state0_history,
            prior_histories=prior_histories,
            prior_valid_histories=prior_valid_histories,
            prediction_loss_weight=PREDICTION_LOSS_WEIGHT,
            local_ce_weight=LOCAL_CE_WEIGHT,
            per_helper_prediction_loss=model.spec.per_helper_prediction_loss,
        )
        if not torch.isfinite(losses.total_loss):
            raise RuntimeError("Closed-loop prediction verification failed: dummy total loss is not finite.")
        model.zero_grad(set_to_none=True)
        losses.total_loss.backward()
        gradient_norm_sum = 0.0
        nonzero_gradient_parameters = 0
        missing_gradient_parameters: list[str] = []
        zero_gradient_parameters: list[str] = []
        total_trainable_parameters = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            total_trainable_parameters += 1
            if parameter.grad is None:
                missing_gradient_parameters.append(name)
                continue
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError("Closed-loop prediction verification failed: dummy gradients contain NaN or Inf.")
            grad_norm = parameter.grad.detach().norm().item()
            gradient_norm_sum += grad_norm
            if grad_norm > 0.0:
                nonzero_gradient_parameters += 1
            else:
                zero_gradient_parameters.append(name)
        return (
            gradient_norm_sum,
            nonzero_gradient_parameters,
            total_trainable_parameters,
            missing_gradient_parameters,
            zero_gradient_parameters,
            losses,
        )

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    model.train()
    if model_call is not model:
        model_call.train()
    (
        initial_gradient_norm_sum,
        initial_nonzero_gradient_parameters,
        total_trainable_parameters,
        initial_missing_gradient_parameters,
        initial_zero_gradient_parameters,
        initial_losses,
    ) = collect_gradient_stats()
    optimizer.step()
    model.zero_grad(set_to_none=True)

    (
        post_step_gradient_norm_sum,
        post_step_nonzero_gradient_parameters,
        _,
        missing_gradient_parameters,
        zero_gradient_parameters,
        post_step_losses,
    ) = collect_gradient_stats()
    if len(missing_gradient_parameters) > 0 or len(zero_gradient_parameters) > 0:
        raise RuntimeError(
            "Closed-loop prediction verification failed: some parameters did not receive gradients after a warmup step. "
            f"missing={missing_gradient_parameters}, zero={zero_gradient_parameters}"
        )

    model.load_state_dict(saved_state)
    model.zero_grad(set_to_none=True)
    if model_call is not model:
        model_call.zero_grad(set_to_none=True)

    return {
        "dummy_total_loss": round(initial_losses.total_loss.item(), 6),
        "dummy_ce_loss": round(initial_losses.ce_loss.item(), 6),
        "dummy_pred_loss": round(initial_losses.pred_loss.item(), 6),
        "dummy_local_ce_loss": round(initial_losses.local_ce_loss.item(), 6),
        "post_step_total_loss": round(post_step_losses.total_loss.item(), 6),
        "total_trainable_parameters": total_trainable_parameters,
        "nonzero_gradient_parameters": initial_nonzero_gradient_parameters,
        "post_step_nonzero_gradient_parameters": post_step_nonzero_gradient_parameters,
        "gradient_norm_sum": round(initial_gradient_norm_sum, 6),
        "post_step_gradient_norm_sum": round(post_step_gradient_norm_sum, 6),
        "initial_missing_gradient_parameters": initial_missing_gradient_parameters,
        "initial_zero_gradient_parameters": initial_zero_gradient_parameters,
    }


def train_single_variant(
    *,
    seed: int,
    variant_key: str,
    spec: VariantSpec,
    args: argparse.Namespace,
    corpus: CorpusData,
    device: torch.device,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    training_steps: int,
    eval_interval: int,
) -> dict[str, object]:
    set_seed(seed)
    encoded_corpus = getattr(corpus.train_dataset, "encoded_corpus", None)
    if not isinstance(encoded_corpus, torch.Tensor):
        raise TypeError(
            "train_single_variant expects corpus.train_dataset to expose encoded_corpus as a torch.Tensor."
        )

    model = build_model(device=device, vocab_size=corpus.vocab_size, spec=spec)
    model_call: nn.Module = model
    trainer: ClosedLoopPredictionGraphTrainer | None = None
    warmup_batches: list[tuple[Tensor, Tensor]] = []
    if args.compile_model:
        model_call = maybe_compile_model(model, enabled=True)
    verification = verify_forward_and_gradients(
        model=model,
        model_call=model_call,
        device=device,
        vocab_size=corpus.vocab_size,
    )
    if args.compile_model:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = capturable_adamw(model, lr=args.learning_rate)

    batch_rng = torch.Generator()
    batch_rng.manual_seed(seed)
    batch_iterator = random_batches(
        encoded_corpus,
        context_size=CONTEXT_SIZE,
        batch_size=args.batch_size,
        device=device,
        rng=batch_rng,
    )
    if not args.compile_model:
        warmup_batches = [next(batch_iterator) for _ in range(WARMUP_STEPS)]
        trainer = ClosedLoopPredictionGraphTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=args.batch_size,
            seq_len=CONTEXT_SIZE,
            device=device,
        )

    checkpoints = [
        checkpoint_metrics(
            model=model,
            model_call=model_call,
            val_inputs=val_inputs,
            val_targets=val_targets,
            batch_size=args.eval_batch_size,
            step=0,
        )
    ]
    parameter_count = count_parameters(model)
    append_log(
        args.log_path,
        {
            "stage": "variant_started",
            "seed": seed,
            "variant": variant_key,
            "label": spec.label,
            "parameter_count": parameter_count,
            "compiled": args.compile_model,
            "verification": verification,
            "initial_checkpoint": checkpoints[-1],
        },
    )

    model.train()
    if model_call is not model:
        model_call.train()
    started_at = perf_counter()
    loss_trace: list[dict[str, float | int]] = []
    last_losses: LossBreakdown | None = None

    if trainer is not None:
        trainer.capture(warmup_batches)
        last_losses = trainer.snapshot()

    for step in range(1, training_steps + 1):
        batch_input, batch_target = next(batch_iterator)
        if trainer is not None:
            losses = trainer.step(batch_input=batch_input, batch_target=batch_target)
        else:
            optimizer.zero_grad(set_to_none=True)
            logits, block1_logits, state0_history, prior_histories, prior_valid_histories, _ = model_call(
                batch_input,
                use_priors=True,
            )
            losses = compute_losses(
                logits=logits,
                block1_logits=block1_logits,
                targets=batch_target,
                state0_history=state0_history,
                prior_histories=prior_histories,
                prior_valid_histories=prior_valid_histories,
                prediction_loss_weight=PREDICTION_LOSS_WEIGHT,
                local_ce_weight=LOCAL_CE_WEIGHT,
                per_helper_prediction_loss=spec.per_helper_prediction_loss,
            )
            if not torch.isfinite(losses.total_loss):
                raise RuntimeError(
                    f"Training diverged for variant {variant_key} at step {step}: total loss is NaN or Inf."
                )
            losses.total_loss.backward()
            optimizer.step()
        last_losses = losses

        if step == 1 or step % eval_interval == 0 or step == training_steps:
            loss_trace.append(
                {
                    "step": step,
                    "ce_loss": round(losses.ce_loss.item(), 6),
                    "pred_loss": round(losses.pred_loss.item(), 6),
                    "local_ce_loss": round(losses.local_ce_loss.item(), 6),
                    "total_loss": round(losses.total_loss.item(), 6),
                }
            )

        if step % eval_interval != 0 and step != training_steps:
            continue

        checkpoint = checkpoint_metrics(
            model=model,
            model_call=model_call,
            val_inputs=val_inputs,
            val_targets=val_targets,
            batch_size=args.eval_batch_size,
            step=step,
        )
        checkpoints.append(checkpoint)
        append_log(
            args.log_path,
            {
                "stage": "checkpoint",
                "seed": seed,
                "variant": variant_key,
                **checkpoint,
            },
        )

    if trainer is not None:
        trainer.synchronize()
    wall_seconds = perf_counter() - started_at
    if last_losses is None:
        raise RuntimeError(f"Variant {variant_key} completed without any training steps.")

    result = {
        "seed": seed,
        "variant": variant_key,
        "label": spec.label,
        "class_name": "ClosedLoopPredictionModel",
        "d_model": D_MODEL,
        "feedforward_dim": FEEDFORWARD_DIM,
        "num_blocks": spec.num_blocks,
        "rates": list(spec.rates),
        "phases": list(spec.phases),
        "readout_mode": spec.readout_mode,
        "closed_loop": spec.closed_loop,
        "per_helper_prediction_loss": spec.per_helper_prediction_loss,
        "parameter_count": parameter_count,
        "compiled": args.compile_model,
        "verification": verification,
        "loss_trace": loss_trace,
        "checkpoints": checkpoints,
        "best_checkpoint": min(checkpoints, key=lambda checkpoint: checkpoint["val_loss"]),
        "final_checkpoint": checkpoints[-1],
        "final_training_ce_loss": round(last_losses.ce_loss.item(), 6),
        "final_training_pred_loss": round(last_losses.pred_loss.item(), 6),
        "final_training_local_ce_loss": round(last_losses.local_ce_loss.item(), 6),
        "final_training_total_loss": round(last_losses.total_loss.item(), 6),
        "wall_seconds": round(wall_seconds, 6),
        "mix_coefficients": model.mix_coefficients(),
    }
    append_log(
        args.log_path,
        {
            "stage": "variant_done",
            "seed": seed,
            "variant": variant_key,
            "final_checkpoint": result["final_checkpoint"],
            "final_training_total_loss": result["final_training_total_loss"],
            "wall_seconds": result["wall_seconds"],
        },
    )

    del optimizer
    del trainer
    del model_call
    del model
    del batch_iterator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def mean_rounded(values: list[float]) -> float:
    return round(mean(values), 6)


def std_rounded(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(pstdev(values), 6)


def summarize_results(
    *,
    per_seed_results: list[dict[str, object]],
    specs: dict[str, VariantSpec],
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {key: [] for key in specs}
    for result in per_seed_results:
        grouped[result["variant"]].append(result)

    summary: dict[str, object] = {}
    for key, runs in grouped.items():
        if len(runs) == 0:
            continue
        final_losses = [run["final_checkpoint"]["val_loss"] for run in runs]
        final_accuracies = [run["final_checkpoint"]["val_accuracy"] for run in runs]
        pred_losses = [run["final_checkpoint"]["pred_loss"] for run in runs]
        block1_accuracies = [
            run["final_checkpoint"]["block1_accuracy"]
            for run in runs
            if run["final_checkpoint"]["block1_accuracy"] is not None
        ]
        wall_seconds = [run["wall_seconds"] for run in runs]
        summary[key] = {
            "label": specs[key].label,
            "num_runs": len(runs),
            "mean_final_val_loss": mean_rounded(final_losses),
            "std_final_val_loss": std_rounded(final_losses),
            "mean_final_val_accuracy": mean_rounded(final_accuracies),
            "mean_final_pred_loss": mean_rounded(pred_losses),
            "mean_final_block1_accuracy": None if len(block1_accuracies) == 0 else mean_rounded(block1_accuracies),
            "mean_wall_seconds": mean_rounded(wall_seconds),
            "runs": runs,
        }
    return summary


def comparison(summary_by_variant: dict[str, object]) -> dict[str, float]:
    comparison_values: dict[str, float] = {}
    if "A_single" in summary_by_variant and "B_spectator" in summary_by_variant:
        comparison_values["mean_final_val_loss_delta_B_minus_A"] = round(
            summary_by_variant["B_spectator"]["mean_final_val_loss"]
            - summary_by_variant["A_single"]["mean_final_val_loss"],
            6,
        )
    if "A_single" in summary_by_variant and "C_closed_loop" in summary_by_variant:
        comparison_values["mean_final_val_loss_delta_C_minus_A"] = round(
            summary_by_variant["C_closed_loop"]["mean_final_val_loss"]
            - summary_by_variant["A_single"]["mean_final_val_loss"],
            6,
        )
    if "B_spectator" in summary_by_variant and "C_closed_loop" in summary_by_variant:
        comparison_values["mean_final_val_loss_delta_C_minus_B"] = round(
            summary_by_variant["C_closed_loop"]["mean_final_val_loss"]
            - summary_by_variant["B_spectator"]["mean_final_val_loss"],
            6,
        )
    if "A_single" in summary_by_variant and "E_grounded" in summary_by_variant:
        comparison_values["mean_final_val_loss_delta_E_minus_A"] = round(
            summary_by_variant["E_grounded"]["mean_final_val_loss"]
            - summary_by_variant["A_single"]["mean_final_val_loss"],
            6,
        )
    if "A_single" in summary_by_variant and "F_star_3block" in summary_by_variant:
        comparison_values["mean_final_val_loss_delta_F_minus_A"] = round(
            summary_by_variant["F_star_3block"]["mean_final_val_loss"]
            - summary_by_variant["A_single"]["mean_final_val_loss"],
            6,
        )
    if "C_closed_loop" in summary_by_variant and "F_star_3block" in summary_by_variant:
        comparison_values["mean_final_val_loss_delta_F_minus_C"] = round(
            summary_by_variant["F_star_3block"]["mean_final_val_loss"]
            - summary_by_variant["C_closed_loop"]["mean_final_val_loss"],
            6,
        )
    if "A_single" in summary_by_variant and "G_rate4_only" in summary_by_variant:
        comparison_values["mean_final_val_loss_delta_G_minus_A"] = round(
            summary_by_variant["G_rate4_only"]["mean_final_val_loss"]
            - summary_by_variant["A_single"]["mean_final_val_loss"],
            6,
        )
    if "C_closed_loop" in summary_by_variant and "G_rate4_only" in summary_by_variant:
        comparison_values["mean_final_val_loss_delta_G_minus_C"] = round(
            summary_by_variant["G_rate4_only"]["mean_final_val_loss"]
            - summary_by_variant["C_closed_loop"]["mean_final_val_loss"],
            6,
        )
    if "I_phase_offset" in summary_by_variant and "I_control" in summary_by_variant:
        comparison_values["mean_final_val_loss_delta_I_phase_offset_minus_I_control"] = round(
            summary_by_variant["I_phase_offset"]["mean_final_val_loss"]
            - summary_by_variant["I_control"]["mean_final_val_loss"],
            6,
        )
    return comparison_values


def main() -> int:
    args = parse_args()
    if len(args.seeds) == 0:
        raise ValueError("At least one seed is required.")
    if args.training_steps <= 0:
        raise ValueError(f"training_steps must be positive, got {args.training_steps}.")
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}.")
    if args.eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {args.eval_batch_size}.")
    if args.eval_samples <= 0:
        raise ValueError(f"eval_samples must be positive, got {args.eval_samples}.")
    if args.sanity_steps <= 0:
        raise ValueError(f"sanity_steps must be positive, got {args.sanity_steps}.")
    if args.sanity_eval_interval <= 0:
        raise ValueError(f"sanity_eval_interval must be positive, got {args.sanity_eval_interval}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/closed_loop_prediction.py.")

    specs = variant_specs()
    unknown_variants = [variant for variant in args.variants if variant not in specs]
    if len(unknown_variants) > 0:
        raise ValueError(f"Unknown variants requested: {unknown_variants}. Available variants: {list(specs)}.")
    selected_specs = {key: specs[key] for key in args.variants}

    training_steps = args.sanity_steps if args.sanity_check else args.training_steps
    eval_interval = args.sanity_eval_interval if args.sanity_check else args.eval_interval

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    if args.log_path.exists():
        previous_log = args.log_path.read_text(encoding="utf-8")
        if previous_log:
            append_log(
                args.log_path,
                {
                    "stage": "run_restarted",
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "previous_lines": len(previous_log.splitlines()),
                },
            )

    device = torch.device("cuda")
    corpus = load_corpus(
        train_path=args.train_path,
        val_path=args.val_path,
        context_size=CONTEXT_SIZE,
        eval_samples=args.eval_samples,
    )
    val_inputs = corpus.val_inputs.to(device=device, dtype=torch.long)
    val_targets = corpus.val_targets.to(device=device, dtype=torch.long)

    append_log(
        args.log_path,
        {
            "stage": "experiment_started",
            "sanity_check": args.sanity_check,
            "seeds": args.seeds,
            "variants": list(selected_specs),
            "training_steps": training_steps,
            "eval_interval": eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "context_size": CONTEXT_SIZE,
            "prediction_loss_weight": PREDICTION_LOSS_WEIGHT,
            "compile_model": args.compile_model,
            "device": str(device),
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
            "vocab_size": corpus.vocab_size,
            "train_dataset_size": len(corpus.train_dataset),
            "val_examples": int(corpus.val_inputs.shape[0]),
        },
    )

    overall_started_at = perf_counter()
    per_seed_results: list[dict[str, object]] = []
    for seed in args.seeds:
        append_log(args.log_path, {"stage": "seed_started", "seed": seed})
        for variant_key, spec in selected_specs.items():
            per_seed_results.append(
                train_single_variant(
                    seed=seed,
                    variant_key=variant_key,
                    spec=spec,
                    args=args,
                    corpus=corpus,
                    device=device,
                    val_inputs=val_inputs,
                    val_targets=val_targets,
                    training_steps=training_steps,
                    eval_interval=eval_interval,
                )
            )
        append_log(args.log_path, {"stage": "seed_done", "seed": seed})

    wall_seconds = perf_counter() - overall_started_at
    git_status_short = current_git_status_short()
    summary_by_variant = summarize_results(per_seed_results=per_seed_results, specs=selected_specs)
    report = {
        "config": {
            "training_steps": training_steps,
            "eval_interval": eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "eval_samples": args.eval_samples,
            "seeds": args.seeds,
            "context_size": CONTEXT_SIZE,
            "d_model": D_MODEL,
            "feedforward_dim": FEEDFORWARD_DIM,
            "prediction_loss_weight": PREDICTION_LOSS_WEIGHT,
            "compile_model": args.compile_model,
            "dataset": "wikitext-103-raw",
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
            "sanity_check": args.sanity_check,
        },
        "environment": {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": git_status_short == [],
            "git_status_short": git_status_short,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
        },
        "dataset": {
            "train_examples": len(corpus.train_dataset),
            "val_examples": int(corpus.val_inputs.shape[0]),
            "vocab_size": corpus.vocab_size,
        },
        "timing": {
            "overall_wall_seconds": round(wall_seconds, 6),
        },
        "variants": {key: asdict(spec) for key, spec in selected_specs.items()},
        "per_seed_results": per_seed_results,
        "summary_by_variant": summary_by_variant,
        "comparison": comparison(summary_by_variant),
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "done",
            "report_path": str(args.report_path),
            "summary_by_variant": {
                key: {
                    "mean_final_val_loss": value["mean_final_val_loss"],
                    "mean_final_pred_loss": value["mean_final_pred_loss"],
                    "mean_wall_seconds": value["mean_wall_seconds"],
                }
                for key, value in summary_by_variant.items()
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
