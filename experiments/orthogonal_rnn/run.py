from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from base_experiments.transformer import build_fixed_length_split
from core.fixed_window_char import set_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_FILE = REPO_ROOT / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
ARTIFACTS_DIR = REPO_ROOT / "experiments" / "orthogonal_rnn" / "artifacts"
OUTPUT_FILE = ARTIFACTS_DIR / "results.json"
TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

DEFAULT_CONTEXT_SIZE = 64
DEFAULT_TRAIN_CHARS = 900_000
DEFAULT_VAL_CHARS = 100_000
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 256
DEFAULT_EVAL_BATCH_SIZE = 512
DEFAULT_GRADIENT_CLIP = 1.0
DEFAULT_SEED = 42
DEFAULT_D_MODEL = 216
DEFAULT_NUM_HEADS = 4
DEFAULT_TEMPORAL_WINDOW = 8
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_MIN_LEARNING_RATE = 3e-5
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MATRIX_EXP_BENCH_REPEATS = 20
DEFAULT_FORWARD_BENCH_REPEATS = 5
TRANSFORMER_TARGET_VAL_LOSS = 1.535
TRANSFORMER_TARGET_PARAMS = 190_000
MUON_TARGET_VAL_LOSS = 1.606
MUON_TARGET_PARAMS = 185_000


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError(f"Mix probability must be between 0 and 1, got {probability}.")
    return math.log(probability / (1.0 - probability))


def round_metric(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def current_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def batched_pairs(inputs: Tensor, targets: Tensor, *, batch_size: int):
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        yield inputs[start:stop], targets[start:stop]


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def ensure_corpus_file() -> Path:
    if TEXT_FILE.exists():
        return TEXT_FILE
    TEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(TINY_SHAKESPEARE_URL) as response:
        TEXT_FILE.write_bytes(response.read())
    return TEXT_FILE


@dataclass(frozen=True)
class SplitConfig:
    context_size: int = DEFAULT_CONTEXT_SIZE
    train_characters: int = DEFAULT_TRAIN_CHARS
    val_characters: int = DEFAULT_VAL_CHARS


@dataclass(frozen=True)
class OrthogonalResidualConfig:
    context_size: int
    d_model: int
    temporal_window: int
    num_heads: int = DEFAULT_NUM_HEADS
    token_mix_init: float = 0.5
    block_mix_init: float = 0.9
    time_mix_init: float = 0.9


@dataclass(frozen=True)
class AttentionRuntimeWeights:
    query: Tensor
    key: Tensor
    value: Tensor
    output: Tensor


@dataclass(frozen=True)
class BlockRuntimeWeights:
    proj_in: Tensor
    proj_out: Tensor


class MixAdd(nn.Module):
    def __init__(self, *, init: float) -> None:
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.tensor(_logit(init), dtype=torch.float32))

    def coefficient(self) -> Tensor:
        return torch.sigmoid(self.alpha_logit)

    def coefficient_value(self) -> float:
        return self.coefficient().detach().item()

    def forward(self, stream: Tensor, delta: Tensor) -> Tensor:
        mix = self.coefficient().to(device=stream.device, dtype=stream.dtype)
        return (mix * stream) + ((1.0 - mix) * delta)


class ExpMapOrthogonalLinear(nn.Module):
    def __init__(self, size: int, *, bias: bool = True, init_std: float = 0.02) -> None:
        super().__init__()
        self.size = size
        parameter_count = size * (size - 1) // 2
        self.upper_triangle_params = nn.Parameter(torch.empty(parameter_count))
        nn.init.normal_(self.upper_triangle_params, mean=0.0, std=init_std)
        if bias:
            self.bias = nn.Parameter(torch.zeros(size))
        else:
            self.register_parameter("bias", None)
        indices = torch.triu_indices(size, size, offset=1)
        self.register_buffer("row_indices", indices[0], persistent=False)
        self.register_buffer("col_indices", indices[1], persistent=False)

    def orthogonal_weight(self) -> Tensor:
        skew = self.upper_triangle_params.new_zeros((self.size, self.size))
        skew[self.row_indices, self.col_indices] = self.upper_triangle_params
        skew = skew - skew.transpose(0, 1)
        return torch.linalg.matrix_exp(skew)

    def apply(self, inputs: Tensor, *, weight: Tensor) -> Tensor:
        return F.linear(inputs, weight, self.bias)

    def orthogonality_error(self) -> float:
        with torch.no_grad():
            weight = self.orthogonal_weight().float()
            identity = torch.eye(self.size, device=weight.device, dtype=weight.dtype)
            error = torch.linalg.matrix_norm(weight.transpose(0, 1) @ weight - identity)
        return error.item()


class OrthogonalTemporalWindowAttention(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = ExpMapOrthogonalLinear(d_model)
        self.key = ExpMapOrthogonalLinear(d_model)
        self.value = ExpMapOrthogonalLinear(d_model)
        self.output = ExpMapOrthogonalLinear(d_model)

    def runtime_weights(self) -> AttentionRuntimeWeights:
        return AttentionRuntimeWeights(
            query=self.query.orthogonal_weight(),
            key=self.key.orthogonal_weight(),
            value=self.value.orthogonal_weight(),
            output=self.output.orthogonal_weight(),
        )

    def forward(
        self,
        query_source: Tensor,
        past_states: Tensor,
        *,
        runtime_weights: AttentionRuntimeWeights,
        capture_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if past_states.shape[1] == 0:
            return torch.zeros_like(query_source), None

        batch_size, _, d_model = past_states.shape
        query = self.query.apply(query_source, weight=runtime_weights.query).reshape(
            batch_size,
            self.num_heads,
            self.head_dim,
        )
        keys = self.key.apply(past_states, weight=runtime_weights.key).reshape(
            batch_size,
            past_states.shape[1],
            self.num_heads,
            self.head_dim,
        )
        values = self.value.apply(past_states, weight=runtime_weights.value).reshape(
            batch_size,
            past_states.shape[1],
            self.num_heads,
            self.head_dim,
        )
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        logits = torch.einsum("bhd,bhwd->bhw", query, keys) * (self.head_dim ** -0.5)
        weights = torch.softmax(logits, dim=-1)
        attended = torch.einsum("bhw,bhwd->bhd", weights, values).reshape(batch_size, d_model)
        context = self.output.apply(attended, weight=runtime_weights.output)
        if not capture_weights:
            return context, None
        return context, weights


class OrthogonalResidualFeedForwardBlock(nn.Module):
    def __init__(self, *, d_model: int) -> None:
        super().__init__()
        self.proj_in = ExpMapOrthogonalLinear(d_model)
        self.activation = nn.GELU()
        self.proj_out = ExpMapOrthogonalLinear(d_model)

    def runtime_weights(self) -> BlockRuntimeWeights:
        return BlockRuntimeWeights(
            proj_in=self.proj_in.orthogonal_weight(),
            proj_out=self.proj_out.orthogonal_weight(),
        )

    def forward(self, stream: Tensor, *, runtime_weights: BlockRuntimeWeights) -> Tensor:
        hidden = self.proj_in.apply(stream, weight=runtime_weights.proj_in)
        activated = self.activation(hidden)
        return self.proj_out.apply(activated, weight=runtime_weights.proj_out)


class OrthogonalResidualStreamTimeMixAddCharModel(nn.Module):
    def __init__(self, *, vocab_size: int, config: OrthogonalResidualConfig) -> None:
        super().__init__()
        self.config = config
        self.context_size = config.context_size
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_size, config.d_model)
        self.mix_token = MixAdd(init=config.token_mix_init)
        self.mix_block = MixAdd(init=config.block_mix_init)
        self.mix_time = MixAdd(init=config.time_mix_init)
        self.temporal_attention = OrthogonalTemporalWindowAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
        )
        self.block = OrthogonalResidualFeedForwardBlock(d_model=config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size)

    def embedded_tokens(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.shape[1]
        if sequence_length != self.context_size:
            raise ValueError(f"Expected context length {self.context_size}, got {sequence_length}.")
        positions = torch.arange(sequence_length, device=tokens.device)
        return self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)

    def mix_coefficients(self) -> dict[str, float]:
        return {
            "token": self.mix_token.coefficient_value(),
            "block": self.mix_block.coefficient_value(),
            "time": self.mix_time.coefficient_value(),
        }

    def runtime_weights(self) -> tuple[AttentionRuntimeWeights, BlockRuntimeWeights]:
        return self.temporal_attention.runtime_weights(), self.block.runtime_weights()

    def monitored_orthogonality_error(self) -> float:
        return self.block.proj_in.orthogonality_error()

    def forward(self, tokens: Tensor) -> Tensor:
        embeddings = self.embedded_tokens(tokens)
        batch_size = tokens.shape[0]
        stream = torch.zeros(batch_size, self.config.d_model, device=tokens.device, dtype=embeddings.dtype)
        history: list[Tensor] = []
        attention_runtime_weights, block_runtime_weights = self.runtime_weights()

        for time_index in range(self.context_size):
            block_input = self.mix_token(stream, embeddings[:, time_index, :])
            past_states = history[-self.config.temporal_window :]
            if past_states:
                stacked_past = torch.stack(past_states, dim=1)
            else:
                stacked_past = torch.empty(
                    batch_size,
                    0,
                    self.config.d_model,
                    device=tokens.device,
                    dtype=embeddings.dtype,
                )
            temporal_context, _ = self.temporal_attention(
                block_input,
                stacked_past,
                runtime_weights=attention_runtime_weights,
                capture_weights=False,
            )
            block_delta = self.block(block_input, runtime_weights=block_runtime_weights)
            post_block = self.mix_block(block_input, block_delta)
            stream = self.mix_time(post_block, temporal_context)
            history.append(stream)

        return self.output(stream)


def benchmark_runtime_weight_refresh(
    model: OrthogonalResidualStreamTimeMixAddCharModel,
    *,
    device: torch.device,
    repeats: int,
) -> float:
    maybe_synchronize(device)
    started_at = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            attention_weights, block_weights = model.runtime_weights()
            del attention_weights
            del block_weights
    maybe_synchronize(device)
    return (time.perf_counter() - started_at) * 1000.0 / repeats


def benchmark_forward_pass(
    model: OrthogonalResidualStreamTimeMixAddCharModel,
    sample_inputs: Tensor,
    *,
    device: torch.device,
    repeats: int,
) -> float:
    was_training = model.training
    model.eval()
    maybe_synchronize(device)
    started_at = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(sample_inputs)
    maybe_synchronize(device)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0 / repeats
    if was_training:
        model.train()
    return elapsed_ms


def evaluate_model(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in batched_pairs(inputs, targets, batch_size=batch_size):
            logits = model(batch_inputs)
            total_loss += F.cross_entropy(logits, batch_targets, reduction="sum").item()
            total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
            total_examples += batch_targets.shape[0]
    if was_training:
        model.train()
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_inputs: Tensor,
    train_targets: Tensor,
    *,
    batch_size: int,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    permutation = torch.randperm(train_inputs.shape[0], device=train_inputs.device)
    total_examples = 0
    total_loss = 0.0
    total_correct = 0

    for start in range(0, permutation.shape[0], batch_size):
        batch_indices = permutation[start : start + batch_size]
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Encountered non-finite training loss: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        batch_examples = batch_targets.shape[0]
        total_examples += batch_examples
        total_loss += loss.item() * batch_examples
        total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    parser.add_argument("--train-chars", type=int, default=DEFAULT_TRAIN_CHARS)
    parser.add_argument("--val-chars", type=int, default=DEFAULT_VAL_CHARS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--gradient-clip", type=float, default=DEFAULT_GRADIENT_CLIP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--temporal-window", type=int, default=DEFAULT_TEMPORAL_WINDOW)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--min-learning-rate", type=float, default=DEFAULT_MIN_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--matrix-exp-bench-repeats", type=int, default=DEFAULT_MATRIX_EXP_BENCH_REPEATS)
    parser.add_argument("--forward-bench-repeats", type=int, default=DEFAULT_FORWARD_BENCH_REPEATS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    corpus_file = ensure_corpus_file()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    set_seed(args.seed)

    raw_text = corpus_file.read_text(encoding="utf-8")
    split_config = SplitConfig(
        context_size=args.context_size,
        train_characters=args.train_chars,
        val_characters=args.val_chars,
    )
    split = build_fixed_length_split(raw_text, config=split_config)
    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)
    vocab_size = split.train_dataset.vocab_size

    config = OrthogonalResidualConfig(
        context_size=args.context_size,
        d_model=args.d_model,
        temporal_window=args.temporal_window,
        num_heads=args.num_heads,
    )
    model = OrthogonalResidualStreamTimeMixAddCharModel(vocab_size=vocab_size, config=config).to(device)
    parameter_count = count_parameters(model)
    print(
        f"model=orthogonal_residual_stream_time d_model={args.d_model} params={parameter_count} "
        f"target_muon_params={MUON_TARGET_PARAMS} target_transformer_params={TRANSFORMER_TARGET_PARAMS}",
        flush=True,
    )

    sample_batch = train_inputs[: min(args.batch_size, 32)]
    matrix_exp_refresh_ms = benchmark_runtime_weight_refresh(
        model,
        device=device,
        repeats=args.matrix_exp_bench_repeats,
    )
    forward_ms = benchmark_forward_pass(
        model,
        sample_batch,
        device=device,
        repeats=args.forward_bench_repeats,
    )
    print(
        f"benchmark matrix_exp_refresh_ms={matrix_exp_refresh_ms:.3f} "
        f"forward_ms={forward_ms:.3f} sample_batch={sample_batch.shape[0]}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_learning_rate,
    )

    history: list[dict[str, float | int]] = []
    started_at = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started_at = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_inputs,
            train_targets,
            batch_size=args.batch_size,
            gradient_clip_norm=args.gradient_clip,
        )
        val_metrics = evaluate_model(
            model,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
        )
        orthogonality_error = model.monitored_orthogonality_error()
        epoch_seconds = time.perf_counter() - epoch_started_at
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": round_metric(train_metrics["loss"]),
            "train_accuracy": round_metric(train_metrics["accuracy"]),
            "val_loss": round_metric(val_metrics["loss"]),
            "val_accuracy": round_metric(val_metrics["accuracy"]),
            "learning_rate": round_metric(optimizer.param_groups[0]["lr"], digits=8),
            "orthogonality_error": round_metric(orthogonality_error, digits=8),
            "epoch_seconds": round_metric(epoch_seconds, digits=2),
        }
        history.append(record)
        print(
            f"epoch={epoch}/{args.epochs} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} orth_error={orthogonality_error:.6e} "
            f"targets(muon={MUON_TARGET_VAL_LOSS:.3f},transformer={TRANSFORMER_TARGET_VAL_LOSS:.3f})",
            flush=True,
        )
        scheduler.step()

    runtime_seconds = time.perf_counter() - started_at
    best_record = min(history, key=lambda item: float(item["val_loss"]))
    final_record = history[-1]
    final_val_loss = float(final_record["val_loss"])
    comparison = {
        "final_val_minus_muon": round_metric(final_val_loss - MUON_TARGET_VAL_LOSS),
        "final_val_minus_transformer": round_metric(final_val_loss - TRANSFORMER_TARGET_VAL_LOSS),
        "muon_target_val_loss": MUON_TARGET_VAL_LOSS,
        "transformer_target_val_loss": TRANSFORMER_TARGET_VAL_LOSS,
        "muon_target_parameter_count": MUON_TARGET_PARAMS,
        "transformer_target_parameter_count": TRANSFORMER_TARGET_PARAMS,
    }
    print(
        f"final_comparison final_val_loss={final_val_loss:.6f} "
        f"vs_muon_delta={comparison['final_val_minus_muon']:.6f} "
        f"vs_transformer_delta={comparison['final_val_minus_transformer']:.6f}",
        flush=True,
    )

    results = {
        "git_sha": current_git_sha(),
        "seed": args.seed,
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_is_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "dataset": {
            "text_file": str(corpus_file),
            "context_size": args.context_size,
            "train_characters": args.train_chars,
            "val_characters": args.val_chars,
            "train_windows": int(split.train_inputs.shape[0]),
            "val_windows": int(split.val_inputs.shape[0]),
            "vocab_size": vocab_size,
            "val_start": int(split.val_start),
        },
        "model": {
            "name": "orthogonal_residual_stream_time",
            "parameterization": "exp_map_skew_symmetric",
            "parameter_count": parameter_count,
            "architecture": {
                "d_model": args.d_model,
                "feedforward_dim": args.d_model,
                "num_heads": args.num_heads,
                "temporal_window": args.temporal_window,
                "context_size": args.context_size,
                "orthogonal_layers": [
                    "block.proj_in",
                    "block.proj_out",
                    "temporal_attention.query",
                    "temporal_attention.key",
                    "temporal_attention.value",
                    "temporal_attention.output",
                ],
                "regular_layers": [
                    "token_embedding",
                    "position_embedding",
                    "mix_token.alpha_logit",
                    "mix_block.alpha_logit",
                    "mix_time.alpha_logit",
                    "output",
                ],
            },
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
        },
        "benchmarks": {
            "matrix_exp_refresh_ms": round_metric(matrix_exp_refresh_ms, digits=4),
            "forward_ms": round_metric(forward_ms, digits=4),
            "sample_batch_size": int(sample_batch.shape[0]),
        },
        "runtime_seconds": round_metric(runtime_seconds, digits=1),
        "best_epoch": int(best_record["epoch"]),
        "best_val_loss": float(best_record["val_loss"]),
        "best_val_accuracy": float(best_record["val_accuracy"]),
        "final_val_loss": final_val_loss,
        "final_val_accuracy": float(final_record["val_accuracy"]),
        "orthogonality_monitor_layer": "block.proj_in",
        "history": history,
        "comparison_targets": {
            "transformer": {
                "parameter_count": TRANSFORMER_TARGET_PARAMS,
                "val_loss": TRANSFORMER_TARGET_VAL_LOSS,
            },
            "residual_stream_time_muon": {
                "parameter_count": MUON_TARGET_PARAMS,
                "val_loss": MUON_TARGET_VAL_LOSS,
            },
        },
        "comparison": comparison,
    }
    OUTPUT_FILE.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
