from __future__ import annotations

import argparse
import gc
import json
import subprocess
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


VOCAB_SIZE = 65
DEVICE = torch.device("cuda")


@dataclass(frozen=True)
class FamilyConfig:
    family: str
    hidden_size: int
    num_layers: int
    num_heads: int | None = None
    feedforward_mult: int = 4


@dataclass(frozen=True)
class TrainMetrics:
    batch_size: int
    tokens_per_second: float
    milliseconds_per_step: float
    peak_vram_bytes: int
    loss_mean: float
    loss_stdev: float


@dataclass(frozen=True)
class DecodeMetrics:
    tokens_per_second: float
    milliseconds_per_token: float


@dataclass(frozen=True)
class BenchmarkResult:
    family: str
    size_label: str
    context_size: int
    parameter_count: int
    config: FamilyConfig
    train: TrainMetrics
    decode: DecodeMetrics


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def transformer_parameter_count(
    hidden_size: int,
    num_layers: int,
    context_size: int,
    *,
    feedforward_mult: int = 4,
    vocab_size: int = VOCAB_SIZE,
) -> int:
    feedforward_dim = hidden_size * feedforward_mult
    embedding = hidden_size * vocab_size + hidden_size * context_size
    block = (
        4 * hidden_size * hidden_size
        + 9 * hidden_size
        + 2 * hidden_size * feedforward_dim
        + feedforward_dim
    )
    final_norm = 2 * hidden_size
    lm_head = hidden_size * vocab_size + vocab_size
    return embedding + num_layers * block + final_norm + lm_head


def recurrent_parameter_count(
    family: str,
    hidden_size: int,
    num_layers: int,
    *,
    vocab_size: int = VOCAB_SIZE,
) -> int:
    embedding = hidden_size * vocab_size
    if family == "lstm":
        block = 8 * hidden_size * hidden_size + 8 * hidden_size
    elif family == "gru":
        block = 6 * hidden_size * hidden_size + 6 * hidden_size
    else:
        raise ValueError(f"Unsupported recurrent family: {family}")
    lm_head = hidden_size * vocab_size + vocab_size
    return embedding + num_layers * block + lm_head


def choose_num_heads(hidden_size: int) -> int:
    candidates = [16, 12, 8, 6, 4, 2, 1]
    for num_heads in candidates:
        if hidden_size % num_heads != 0:
            continue
        head_dim = hidden_size // num_heads
        if 32 <= head_dim <= 128:
            return num_heads
    for num_heads in candidates:
        if hidden_size % num_heads == 0:
            return num_heads
    raise ValueError(f"Could not choose a head count for hidden size {hidden_size}.")


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}."
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.output = nn.Linear(hidden_size, hidden_size)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, _ = x.shape
        return x.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch_size, _, sequence_length, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.hidden_size)

    def forward(
        self,
        x: Tensor,
        *,
        cache: tuple[Tensor, Tensor] | None = None,
        return_cache: bool = False,
        max_cache_length: int | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)

        if cache is not None:
            cached_k, cached_v = cache
            k_heads = torch.cat((cached_k, k_heads), dim=2)
            v_heads = torch.cat((cached_v, v_heads), dim=2)
            if max_cache_length is not None and k_heads.shape[2] > max_cache_length:
                k_heads = k_heads[:, :, -max_cache_length:, :]
                v_heads = v_heads[:, :, -max_cache_length:, :]
            attention = F.scaled_dot_product_attention(
                q_heads,
                k_heads,
                v_heads,
                dropout_p=0.0,
                is_causal=False,
            )
        else:
            attention = F.scaled_dot_product_attention(
                q_heads,
                k_heads,
                v_heads,
                dropout_p=0.0,
                is_causal=True,
            )

        output = self.output(self._merge_heads(attention))
        next_cache = (k_heads, v_heads) if return_cache else None
        return output, next_cache


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, feedforward_mult: int) -> None:
        super().__init__()
        feedforward_dim = hidden_size * feedforward_mult
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = CausalSelfAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.feedforward_norm = nn.LayerNorm(hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, hidden_size),
        )

    def forward(
        self,
        x: Tensor,
        *,
        cache: tuple[Tensor, Tensor] | None = None,
        return_cache: bool = False,
        max_cache_length: int | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        attention_output, next_cache = self.attention(
            self.attention_norm(x),
            cache=cache,
            return_cache=return_cache,
            max_cache_length=max_cache_length,
        )
        x = x + attention_output
        x = x + self.feedforward(self.feedforward_norm(x))
        return x, next_cache


class CausalTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        context_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        feedforward_mult: int,
    ) -> None:
        super().__init__()
        self.context_size = context_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(context_size, hidden_size)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    feedforward_mult=feedforward_mult,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.shape[1]
        if sequence_length > self.context_size:
            raise ValueError(
                f"Transformer context overflow: got {sequence_length}, max {self.context_size}."
            )
        positions = torch.arange(sequence_length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)
        for block in self.blocks:
            x, _ = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    def prefill(self, tokens: Tensor) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        sequence_length = tokens.shape[1]
        if sequence_length > self.context_size:
            tokens = tokens[:, -self.context_size :]
            sequence_length = tokens.shape[1]
        positions = torch.arange(sequence_length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)
        caches: list[tuple[Tensor, Tensor]] = []
        for block in self.blocks:
            x, cache = block(x, return_cache=True, max_cache_length=self.context_size)
            if cache is None:
                raise RuntimeError("Transformer block did not return a decode cache.")
            caches.append(cache)
        x = self.final_norm(x)
        logits = self.lm_head(x[:, -1, :])
        return logits, caches

    def decode_step(
        self,
        token: Tensor,
        caches: list[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        if token.shape != (1, 1):
            raise ValueError(f"Expected single-token batch with shape (1, 1), got {tuple(token.shape)}.")
        next_position = caches[0][0].shape[2] if caches else 0
        position = torch.tensor(
            [min(next_position, self.context_size - 1)],
            device=token.device,
        )
        x = self.token_embedding(token) + self.position_embedding(position).view(1, 1, -1)
        next_caches: list[tuple[Tensor, Tensor]] = []
        for block, cache in zip(self.blocks, caches, strict=True):
            x, next_cache = block(
                x,
                cache=cache,
                return_cache=True,
                max_cache_length=self.context_size,
            )
            if next_cache is None:
                raise RuntimeError("Transformer block did not return an updated decode cache.")
            next_caches.append(next_cache)
        x = self.final_norm(x)
        return self.lm_head(x[:, -1, :]), next_caches


class RecurrentLM(nn.Module):
    def __init__(
        self,
        *,
        cell_type: str,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.cell_type = cell_type
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if cell_type == "lstm":
            self.rnn: nn.Module = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        elif cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported recurrent cell type: {cell_type}")
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        embedded = self.embedding(tokens)
        hidden_states, _ = self.rnn(embedded)
        return self.lm_head(hidden_states)

    def prefill(self, tokens: Tensor) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor]]:
        hidden_states, state = self.rnn(self.embedding(tokens))
        logits = self.lm_head(hidden_states[:, -1, :])
        return logits, state

    def decode_step(
        self,
        token: Tensor,
        state: Tensor | tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor]]:
        hidden_states, next_state = self.rnn(self.embedding(token), state)
        return self.lm_head(hidden_states[:, -1, :]), next_state


def build_model(family: str, context_size: int, config: FamilyConfig) -> nn.Module:
    if family == "transformer":
        if config.num_heads is None:
            raise ValueError("Transformer config must specify num_heads.")
        return CausalTransformerLM(
            vocab_size=VOCAB_SIZE,
            context_size=context_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            feedforward_mult=config.feedforward_mult,
        )
    if family in {"lstm", "gru"}:
        return RecurrentLM(
            cell_type=family,
            vocab_size=VOCAB_SIZE,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        )
    raise ValueError(f"Unsupported family: {family}")


def clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def parameter_count_for_config(family: str, context_size: int, config: FamilyConfig) -> int:
    if family == "transformer":
        return transformer_parameter_count(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            context_size=context_size,
            feedforward_mult=config.feedforward_mult,
        )
    if family in {"lstm", "gru"}:
        return recurrent_parameter_count(
            family=family,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        )
    raise ValueError(f"Unsupported family: {family}")


def search_transformer_config(target_parameters: int, context_size: int) -> FamilyConfig:
    best_config: FamilyConfig | None = None
    best_error = float("inf")
    for num_layers in range(1, 17):
        for hidden_size in range(64, 1537, 16):
            num_heads = choose_num_heads(hidden_size)
            config = FamilyConfig(
                family="transformer",
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
            )
            parameter_count = parameter_count_for_config("transformer", context_size, config)
            error = abs(parameter_count - target_parameters)
            if error < best_error:
                best_error = error
                best_config = config
    if best_config is None:
        raise RuntimeError("Failed to find a transformer configuration.")
    return best_config


def search_recurrent_config(
    family: str,
    target_parameters: int,
    context_size: int,
) -> FamilyConfig:
    best_config: FamilyConfig | None = None
    best_error = float("inf")
    for num_layers in range(1, 13):
        for hidden_size in range(32, 2049, 8):
            config = FamilyConfig(
                family=family,
                hidden_size=hidden_size,
                num_layers=num_layers,
            )
            parameter_count = parameter_count_for_config(family, context_size, config)
            error = abs(parameter_count - target_parameters)
            if error < best_error:
                best_error = error
                best_config = config
    if best_config is None:
        raise RuntimeError(f"Failed to find a {family} configuration.")
    return best_config


def search_family_config(family: str, target_parameters: int, context_size: int) -> FamilyConfig:
    if family == "transformer":
        return search_transformer_config(target_parameters=target_parameters, context_size=context_size)
    if family in {"lstm", "gru"}:
        return search_recurrent_config(
            family=family,
            target_parameters=target_parameters,
            context_size=context_size,
        )
    raise ValueError(f"Unsupported family: {family}")


def training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
    loss.backward()
    optimizer.step()
    return float(loss.item())


def is_oom_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def try_training_batch_size(
    family: str,
    config: FamilyConfig,
    context_size: int,
    batch_size: int,
    *,
    warmup_steps: int,
    timed_steps: int,
) -> tuple[bool, float]:
    clear_cuda_memory()
    model = build_model(family=family, context_size=context_size, config=config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    inputs = torch.randint(
        VOCAB_SIZE,
        (batch_size, context_size),
        device=DEVICE,
        dtype=torch.long,
    )
    targets = torch.randint(
        VOCAB_SIZE,
        (batch_size, context_size),
        device=DEVICE,
        dtype=torch.long,
    )
    try:
        for _ in range(warmup_steps):
            training_step(model, optimizer, inputs, targets)
        torch.cuda.synchronize()
        started_at = time.perf_counter()
        for _ in range(timed_steps):
            training_step(model, optimizer, inputs, targets)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started_at
    except RuntimeError as error:
        if not is_oom_error(error):
            raise
        del model
        del optimizer
        clear_cuda_memory()
        return False, 0.0
    tokens_per_second = (batch_size * context_size * timed_steps) / elapsed
    del model
    del optimizer
    clear_cuda_memory()
    return True, tokens_per_second


def tune_batch_size(family: str, config: FamilyConfig, context_size: int) -> int:
    best_batch_size = 8
    best_tokens_per_second = 0.0
    worse_streak = 0
    # Cap at 512 to avoid driver crashes from aggressive OOM during backward pass
    for batch_size in (8, 16, 32, 64, 128, 256, 512):
        fits, tokens_per_second = try_training_batch_size(
            family=family,
            config=config,
            context_size=context_size,
            batch_size=batch_size,
            warmup_steps=2,
            timed_steps=3,
        )
        if not fits:
            break
        if tokens_per_second > best_tokens_per_second:
            best_batch_size = batch_size
            best_tokens_per_second = tokens_per_second
            worse_streak = 0
            continue
        if tokens_per_second < best_tokens_per_second * 0.97:
            worse_streak += 1
            if worse_streak >= 2 and batch_size >= best_batch_size * 2:
                break
    return best_batch_size


def measure_training(
    family: str,
    config: FamilyConfig,
    context_size: int,
    *,
    batch_size: int,
    warmup_steps: int,
    timed_steps: int,
) -> TrainMetrics:
    clear_cuda_memory()
    model = build_model(family=family, context_size=context_size, config=config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    inputs = torch.randint(
        VOCAB_SIZE,
        (batch_size, context_size),
        device=DEVICE,
        dtype=torch.long,
    )
    targets = torch.randint(
        VOCAB_SIZE,
        (batch_size, context_size),
        device=DEVICE,
        dtype=torch.long,
    )
    losses: list[float] = []

    for _ in range(warmup_steps):
        training_step(model, optimizer, inputs, targets)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    started_at = time.perf_counter()
    for _ in range(timed_steps):
        losses.append(training_step(model, optimizer, inputs, targets))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started_at
    peak_vram_bytes = torch.cuda.max_memory_allocated(DEVICE)

    del model
    del optimizer
    clear_cuda_memory()

    return TrainMetrics(
        batch_size=batch_size,
        tokens_per_second=(batch_size * context_size * timed_steps) / elapsed,
        milliseconds_per_step=(elapsed * 1000.0) / timed_steps,
        peak_vram_bytes=peak_vram_bytes,
        loss_mean=statistics.fmean(losses),
        loss_stdev=statistics.pstdev(losses) if len(losses) > 1 else 0.0,
    )


def prime_decode_state(
    model: nn.Module,
    family: str,
    context_size: int,
) -> tuple[Tensor, object]:
    prompt = torch.randint(
        VOCAB_SIZE,
        (1, context_size),
        device=DEVICE,
        dtype=torch.long,
    )
    if family == "transformer":
        logits, state = model.prefill(prompt)
        next_token = logits.argmax(dim=-1, keepdim=True)
        return next_token, state
    logits, state = model.prefill(prompt)
    next_token = logits.argmax(dim=-1, keepdim=True)
    return next_token, state


def decode_tokens(
    model: nn.Module,
    family: str,
    start_token: Tensor,
    state: object,
    token_count: int,
) -> Tensor:
    token = start_token
    current_state = state
    for _ in range(token_count):
        if family == "transformer":
            logits, current_state = model.decode_step(token, current_state)
        else:
            logits, current_state = model.decode_step(token, current_state)
        token = logits.argmax(dim=-1, keepdim=True)
    return token


def measure_decode(
    family: str,
    config: FamilyConfig,
    context_size: int,
    *,
    warmup_tokens: int,
    timed_tokens: int,
) -> DecodeMetrics:
    clear_cuda_memory()
    model = build_model(family=family, context_size=context_size, config=config).to(DEVICE)
    model.eval()
    with torch.inference_mode():
        warmup_token, warmup_state = prime_decode_state(model, family, context_size)
        decode_tokens(model, family, warmup_token, warmup_state, warmup_tokens)
        timed_token, timed_state = prime_decode_state(model, family, context_size)
        torch.cuda.synchronize()
        started_at = time.perf_counter()
        decode_tokens(model, family, timed_token, timed_state, timed_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started_at
    del model
    clear_cuda_memory()
    return DecodeMetrics(
        tokens_per_second=timed_tokens / elapsed,
        milliseconds_per_token=(elapsed * 1000.0) / timed_tokens,
    )


def format_integer(value: int) -> str:
    return f"{value:,}"


def format_float(value: float) -> str:
    return f"{value:,.1f}"


def render_summary_table(results: list[BenchmarkResult]) -> str:
    lines = [
        "family       size   ctx  params      batch  train tok/s  ms/step  peak VRAM  decode tok/s",
        "-----------  -----  ---  ----------  -----  -----------  -------  ---------  ------------",
    ]
    for result in results:
        lines.append(
            f"{result.family:<11}  "
            f"{result.size_label:<5}  "
            f"{result.context_size:>3}  "
            f"{format_integer(result.parameter_count):>10}  "
            f"{result.train.batch_size:>5}  "
            f"{format_float(result.train.tokens_per_second):>11}  "
            f"{result.train.milliseconds_per_step:>7.2f}  "
            f"{result.train.peak_vram_bytes / (1024**3):>9.2f}G  "
            f"{format_float(result.decode.tokens_per_second):>12}"
        )
    return "\n".join(lines)


def result_to_dict(result: BenchmarkResult) -> dict[str, object]:
    payload = asdict(result)
    payload["config"] = {key: value for key, value in payload["config"].items() if value is not None}
    return payload


def current_git_sha(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_path = args.output or repo_root / "experiments" / "gpu_utilization" / "artifacts" / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_sizes = [("~1M", 1_000_000), ("~10M", 10_000_000)]
    contexts = [32, 256]
    families = ["transformer", "lstm", "gru"]
    train_warmup_steps = 10
    train_timed_steps = 50
    decode_warmup_tokens = 10
    decode_timed_tokens = 100
    if args.smoke:
        target_sizes = [("~1M", 1_000_000)]
        contexts = [32]
        train_warmup_steps = 2
        train_timed_steps = 5
        decode_warmup_tokens = 4
        decode_timed_tokens = 16

    family_configs = {
        size_label: {
            family: search_family_config(
                family=family,
                target_parameters=target_parameters,
                context_size=max(contexts),
            )
            for family in families
        }
        for size_label, target_parameters in target_sizes
    }

    results: list[BenchmarkResult] = []
    for size_label, _ in target_sizes:
        for family in families:
            config = family_configs[size_label][family]
            for context_size in contexts:
                parameter_count = parameter_count_for_config(family, context_size, config)
                batch_size = tune_batch_size(
                    family=family,
                    config=config,
                    context_size=context_size,
                )
                train = measure_training(
                    family=family,
                    config=config,
                    context_size=context_size,
                    batch_size=batch_size,
                    warmup_steps=train_warmup_steps,
                    timed_steps=train_timed_steps,
                )
                decode = measure_decode(
                    family=family,
                    config=config,
                    context_size=context_size,
                    warmup_tokens=decode_warmup_tokens,
                    timed_tokens=decode_timed_tokens,
                )
                results.append(
                    BenchmarkResult(
                        family=family,
                        size_label=size_label,
                        context_size=context_size,
                        parameter_count=parameter_count,
                        config=config,
                        train=train,
                        decode=decode,
                    )
                )

    summary_table = render_summary_table(results)
    print(summary_table)
    output_path.write_text(
        json.dumps(
            {
                "git_sha": current_git_sha(repo_root),
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_device_name": torch.cuda.get_device_name(0),
                "train_warmup_steps": train_warmup_steps,
                "train_timed_steps": train_timed_steps,
                "decode_warmup_tokens": decode_warmup_tokens,
                "decode_timed_tokens": decode_timed_tokens,
                "results": [result_to_dict(result) for result in results],
                "summary_table": summary_table,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
