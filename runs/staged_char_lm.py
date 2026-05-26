from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from typing import Callable
from torch.nn import functional as F

from core.fixed_window_char import load_dataset, set_seed

DEFAULT_SEED = 42
SHORT_CONTEXT = 8
LONG_CONTEXT = 64
TRAIN_CHARACTERS = 80_000
VAL_CHARACTERS = 20_000
PHASE1_STEPS = 500
PHASE2_STEPS = 300
TRAINING_STEPS = PHASE1_STEPS + PHASE2_STEPS
BATCH_SIZE = 256
LEARNING_RATE = 3e-3
LATERAL_SCALE = 0.2
PRINT_INTERVAL = 100
ALL_CONDITIONS = ("block0_alone", "staged_lateral", "cotrained_lateral", "shuffled_staged")


@dataclass(frozen=True)
class WindowDataset:
    long_inputs: Int[Tensor, "examples long_context"]
    short_inputs: Int[Tensor, "examples short_context"]
    targets: Int[Tensor, "examples"]
    vocab_size: int


@dataclass(frozen=True)
class EvalResult:
    val_loss: float
    val_accuracy: float


@dataclass(frozen=True)
class ConditionResult:
    name: str
    val_loss: float
    val_accuracy: float
    delta_from_baseline: float
    wall_seconds: float


def parse_condition_names(raw_conditions: str) -> tuple[str, ...]:
    requested_conditions = tuple(condition.strip() for condition in raw_conditions.split(",") if condition.strip())
    if len(requested_conditions) == 0:
        raise ValueError("conditions must contain at least one condition name")

    invalid_conditions = [condition for condition in requested_conditions if condition not in ALL_CONDITIONS]
    if invalid_conditions:
        raise ValueError(
            "Unknown conditions requested: "
            + ", ".join(invalid_conditions)
            + ". Valid conditions are: "
            + ", ".join(ALL_CONDITIONS)
        )
    return requested_conditions


class CausalSelfAttention(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, max_context: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} and {n_heads}.")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        causal_mask = torch.tril(torch.ones(max_context, max_context, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: Float[Tensor, "batch context d_model"]) -> Float[Tensor, "batch context d_model"]:
        batch_size, context_size, d_model = x.shape
        queries, keys, values = self.qkv(x).chunk(3, dim=-1)

        def reshape_heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch_size, context_size, self.n_heads, self.head_dim).transpose(1, 2)

        queries = reshape_heads(queries)
        keys = reshape_heads(keys)
        values = reshape_heads(values)
        attention_scores = (queries @ keys.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_scores = attention_scores.masked_fill(
            ~self.causal_mask[:context_size, :context_size],
            float("-inf"),
        )
        attention_weights = attention_scores.softmax(dim=-1)
        attended = attention_weights @ values
        attended = attended.transpose(1, 2).contiguous().view(batch_size, context_size, d_model)
        return self.out_proj(attended)


class FeedForward(nn.Module):
    def __init__(self, *, d_model: int, ff_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, ff_dim)
        self.out_proj = nn.Linear(ff_dim, d_model)

    def forward(self, x: Float[Tensor, "batch context d_model"]) -> Float[Tensor, "batch context d_model"]:
        return self.out_proj(F.gelu(self.in_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, ff_dim: int, max_context: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, max_context=max_context)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, ff_dim=ff_dim)

    def forward(self, x: Float[Tensor, "batch context d_model"]) -> Float[Tensor, "batch context d_model"]:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Block0Model(nn.Module):
    def __init__(self, *, vocab_size: int, d_model: int, short_ctx: int, n_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.short_ctx = short_ctx
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(short_ctx, d_model)
        self.block = TransformerBlock(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim, max_context=short_ctx)
        self.output_head = nn.Linear(d_model, vocab_size)

    def encode(self, short_inputs: Int[Tensor, "batch short_context"]) -> Float[Tensor, "batch short_context d_model"]:
        positions = torch.arange(self.short_ctx, device=short_inputs.device)
        hidden = self.token_embedding(short_inputs) + self.position_embedding(positions)
        return self.block(hidden)

    def logits(
        self,
        short_inputs: Int[Tensor, "batch short_context"],
        *,
        lateral: Float[Tensor, "batch 1 d_model"] | None = None,
    ) -> Float[Tensor, "batch vocab"]:
        hidden = self.encode(short_inputs)
        last_hidden = hidden[:, -1:, :]
        if lateral is not None:
            last_hidden = last_hidden + lateral
        return self.output_head(last_hidden).squeeze(1)


class Block1Model(nn.Module):
    def __init__(self, *, vocab_size: int, d_model: int, long_ctx: int, n_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.long_ctx = long_ctx
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(long_ctx, d_model)
        self.block = TransformerBlock(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim, max_context=long_ctx)
        self.local_head = nn.Linear(d_model, vocab_size)

    def encode(self, long_inputs: Int[Tensor, "batch long_context"]) -> Float[Tensor, "batch long_context d_model"]:
        positions = torch.arange(self.long_ctx, device=long_inputs.device)
        hidden = self.token_embedding(long_inputs) + self.position_embedding(positions)
        return self.block(hidden)

    def local_logits_from_hidden(
        self,
        hidden: Float[Tensor, "batch long_context d_model"],
    ) -> Float[Tensor, "batch vocab"]:
        return self.local_head(hidden[:, -1, :])

    def local_logits(self, long_inputs: Int[Tensor, "batch long_context"]) -> Float[Tensor, "batch vocab"]:
        return self.local_logits_from_hidden(self.encode(long_inputs))

    def last_hidden(self, long_inputs: Int[Tensor, "batch long_context"]) -> Float[Tensor, "batch 1 d_model"]:
        return self.encode(long_inputs)[:, -1:, :]


class StagedLateralModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        short_ctx: int,
        long_ctx: int,
        n_heads: int,
        ff_dim: int,
        lateral_scale: float,
    ) -> None:
        super().__init__()
        self.block0 = Block0Model(
            vocab_size=vocab_size,
            d_model=d_model,
            short_ctx=short_ctx,
            n_heads=n_heads,
            ff_dim=ff_dim,
        )
        self.block1 = Block1Model(
            vocab_size=vocab_size,
            d_model=d_model,
            long_ctx=long_ctx,
            n_heads=n_heads,
            ff_dim=ff_dim,
        )
        self.lateral_proj = nn.Linear(d_model, d_model)
        self.lateral_scale = lateral_scale

    def lateral_from_long(
        self,
        long_inputs: Int[Tensor, "batch long_context"],
        *,
        shuffle: bool,
        detach_block1: bool,
    ) -> Float[Tensor, "batch 1 d_model"]:
        if detach_block1:
            with torch.no_grad():
                long_last_hidden = self.block1.last_hidden(long_inputs)
        else:
            long_last_hidden = self.block1.last_hidden(long_inputs)

        lateral_source = long_last_hidden
        if shuffle and lateral_source.shape[0] > 1:
            permutation = torch.randperm(lateral_source.shape[0], device=lateral_source.device)
            lateral_source = lateral_source[permutation]
        return self.lateral_scale * self.lateral_proj(lateral_source)

    def block0_logits(
        self,
        short_inputs: Int[Tensor, "batch short_context"],
        *,
        lateral: Float[Tensor, "batch 1 d_model"] | None,
    ) -> Float[Tensor, "batch vocab"]:
        return self.block0.logits(short_inputs, lateral=lateral)

    def block1_features_and_local_logits(
        self,
        long_inputs: Int[Tensor, "batch long_context"],
    ) -> tuple[Float[Tensor, "batch 1 d_model"], Float[Tensor, "batch vocab"]]:
        hidden = self.block1.encode(long_inputs)
        return hidden[:, -1:, :], self.block1.local_logits_from_hidden(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--phase1-steps", type=int, default=PHASE1_STEPS)
    parser.add_argument("--phase2-steps", type=int, default=PHASE2_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--short-ctx", type=int, default=SHORT_CONTEXT)
    parser.add_argument("--long-ctx", type=int, default=LONG_CONTEXT)
    parser.add_argument("--train-characters", type=int, default=TRAIN_CHARACTERS)
    parser.add_argument("--val-characters", type=int, default=VAL_CHARACTERS)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--lateral-scale", type=float, default=LATERAL_SCALE)
    parser.add_argument("--print-interval", type=int, default=PRINT_INTERVAL)
    parser.add_argument("--conditions", type=str, default=",".join(ALL_CONDITIONS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    errors: list[str] = []
    if args.seed < 0:
        errors.append(f"seed must be non-negative, got {args.seed}")
    if args.phase1_steps <= 0:
        errors.append(f"phase1_steps must be positive, got {args.phase1_steps}")
    if args.phase2_steps <= 0:
        errors.append(f"phase2_steps must be positive, got {args.phase2_steps}")
    if args.batch_size <= 0:
        errors.append(f"batch_size must be positive, got {args.batch_size}")
    if args.learning_rate <= 0:
        errors.append(f"learning_rate must be positive, got {args.learning_rate}")
    if args.short_ctx <= 0:
        errors.append(f"short_ctx must be positive, got {args.short_ctx}")
    if args.long_ctx <= args.short_ctx:
        errors.append(f"long_ctx must exceed short_ctx, got short_ctx={args.short_ctx} long_ctx={args.long_ctx}")
    if args.train_characters <= args.long_ctx:
        errors.append(
            f"train_characters must exceed long_ctx so training windows exist, got train_characters={args.train_characters} long_ctx={args.long_ctx}"
        )
    if args.val_characters <= args.long_ctx:
        errors.append(
            f"val_characters must exceed long_ctx so validation windows exist, got val_characters={args.val_characters} long_ctx={args.long_ctx}"
        )
    if args.d_model <= 0:
        errors.append(f"d_model must be positive, got {args.d_model}")
    if args.ff_dim <= 0:
        errors.append(f"ff_dim must be positive, got {args.ff_dim}")
    if args.n_heads <= 0:
        errors.append(f"n_heads must be positive, got {args.n_heads}")
    if args.d_model % args.n_heads != 0:
        errors.append(f"d_model must be divisible by n_heads, got {args.d_model} and {args.n_heads}")
    if not (0.0 < args.lateral_scale <= 1.0):
        errors.append(f"lateral_scale must be in (0, 1], got {args.lateral_scale}")
    if args.print_interval <= 0:
        errors.append(f"print_interval must be positive, got {args.print_interval}")
    if errors:
        raise ValueError("Argument validation failed:\n- " + "\n- ".join(errors))

    parse_condition_names(args.conditions)


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device is not None:
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_window_dataset(
    *,
    long_ctx: int,
    short_ctx: int,
    train_characters: int,
    val_characters: int,
) -> tuple[WindowDataset, WindowDataset]:
    (train_long_inputs, train_targets), (val_long_inputs, val_targets), vocab_size = load_dataset(
        context_size=long_ctx,
        train_characters=train_characters,
        val_characters=val_characters,
    )
    return (
        WindowDataset(
            long_inputs=train_long_inputs,
            short_inputs=train_long_inputs[:, -short_ctx:],
            targets=train_targets,
            vocab_size=vocab_size,
        ),
        WindowDataset(
            long_inputs=val_long_inputs,
            short_inputs=val_long_inputs[:, -short_ctx:],
            targets=val_targets,
            vocab_size=vocab_size,
        ),
    )


def sample_batch(dataset: WindowDataset, *, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    indices = torch.randint(0, dataset.targets.shape[0], (batch_size,))
    long_inputs = dataset.long_inputs[indices].to(device)
    short_inputs = dataset.short_inputs[indices].to(device)
    targets = dataset.targets[indices].to(device)
    return long_inputs, short_inputs, targets


def next_char_metrics(
    logits: Float[Tensor, "batch vocab"],
    targets: Int[Tensor, "batch"],
) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
    loss = F.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=-1) == targets).float().mean()
    return loss, accuracy


@torch.no_grad()
def evaluate_block0(
    model: Block0Model,
    dataset: WindowDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = dataset.targets.shape[0]

    for start in range(0, total_examples, batch_size):
        stop = min(start + batch_size, total_examples)
        short_inputs = dataset.short_inputs[start:stop].to(device)
        targets = dataset.targets[start:stop].to(device)
        logits = model.logits(short_inputs)
        loss, _ = next_char_metrics(logits, targets)
        total_loss += loss.item() * (stop - start)
        total_correct += (logits.argmax(dim=-1) == targets).sum().item()

    return EvalResult(
        val_loss=total_loss / total_examples,
        val_accuracy=total_correct / total_examples,
    )


@torch.no_grad()
def evaluate_lateral(
    model: StagedLateralModel,
    dataset: WindowDataset,
    *,
    batch_size: int,
    device: torch.device,
    shuffle_lateral: bool,
    detach_block1: bool,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = dataset.targets.shape[0]

    for start in range(0, total_examples, batch_size):
        stop = min(start + batch_size, total_examples)
        long_inputs = dataset.long_inputs[start:stop].to(device)
        short_inputs = dataset.short_inputs[start:stop].to(device)
        targets = dataset.targets[start:stop].to(device)
        if detach_block1:
            with torch.no_grad():
                long_last_hidden, _ = model.block1_features_and_local_logits(long_inputs)
        else:
            long_last_hidden, _ = model.block1_features_and_local_logits(long_inputs)
        lateral_source = long_last_hidden
        if shuffle_lateral and lateral_source.shape[0] > 1:
            permutation = torch.randperm(lateral_source.shape[0], device=lateral_source.device)
            lateral_source = lateral_source[permutation]
        lateral = model.lateral_scale * model.lateral_proj(lateral_source)
        logits = model.block0_logits(short_inputs, lateral=lateral)
        loss, _ = next_char_metrics(logits, targets)
        total_loss += loss.item() * (stop - start)
        total_correct += (logits.argmax(dim=-1) == targets).sum().item()

    return EvalResult(
        val_loss=total_loss / total_examples,
        val_accuracy=total_correct / total_examples,
    )


def train_block0_alone(dataset: WindowDataset, val_dataset: WindowDataset, args: argparse.Namespace, device: torch.device) -> ConditionResult:
    set_seed(args.seed)
    model = Block0Model(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        short_ctx=args.short_ctx,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = args.phase1_steps + args.phase2_steps
    started_at = perf_counter()

    for step in range(1, total_steps + 1):
        model.train()
        _, short_inputs, targets = sample_batch(dataset, batch_size=args.batch_size, device=device)
        logits = model.logits(short_inputs)
        loss, accuracy = next_char_metrics(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.print_interval == 0 or step == total_steps:
            print(
                f"[block0_alone] step={step:03d}/{total_steps} ce={loss.item():.4f} acc={accuracy.item():.4%}",
                flush=True,
            )

    evaluation = evaluate_block0(model, val_dataset, batch_size=args.batch_size, device=device)
    wall_seconds = perf_counter() - started_at
    return ConditionResult(
        name="block0_alone",
        val_loss=evaluation.val_loss,
        val_accuracy=evaluation.val_accuracy,
        delta_from_baseline=0.0,
        wall_seconds=wall_seconds,
    )


def train_cotrained_lateral(
    dataset: WindowDataset,
    val_dataset: WindowDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> ConditionResult:
    set_seed(args.seed)
    model = StagedLateralModel(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        short_ctx=args.short_ctx,
        long_ctx=args.long_ctx,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        lateral_scale=args.lateral_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = args.phase1_steps + args.phase2_steps
    started_at = perf_counter()

    for step in range(1, total_steps + 1):
        model.train()
        long_inputs, short_inputs, targets = sample_batch(dataset, batch_size=args.batch_size, device=device)
        long_last_hidden, block1_local_logits = model.block1_features_and_local_logits(long_inputs)
        lateral = model.lateral_scale * model.lateral_proj(long_last_hidden.detach())
        logits = model.block0_logits(short_inputs, lateral=lateral)
        ce_loss, accuracy = next_char_metrics(logits, targets)
        local_loss, local_accuracy = next_char_metrics(block1_local_logits, targets)
        total_loss = ce_loss + local_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step % args.print_interval == 0 or step == total_steps:
            print(
                f"[cotrained_lateral] step={step:03d}/{total_steps} ce={ce_loss.item():.4f} local_ce={local_loss.item():.4f} block0_acc={accuracy.item():.4%} block1_acc={local_accuracy.item():.4%}",
                flush=True,
            )

    evaluation = evaluate_lateral(
        model,
        val_dataset,
        batch_size=args.batch_size,
        device=device,
        shuffle_lateral=False,
        detach_block1=False,
    )
    wall_seconds = perf_counter() - started_at
    return ConditionResult(
        name="cotrained_lateral",
        val_loss=evaluation.val_loss,
        val_accuracy=evaluation.val_accuracy,
        delta_from_baseline=0.0,
        wall_seconds=wall_seconds,
    )


def run_stage1(
    model: StagedLateralModel,
    dataset: WindowDataset,
    *,
    args: argparse.Namespace,
    device: torch.device,
    label: str,
) -> None:
    block0_optimizer = torch.optim.AdamW(model.block0.parameters(), lr=args.learning_rate)
    block1_optimizer = torch.optim.AdamW(model.block1.parameters(), lr=args.learning_rate)

    for step in range(1, args.phase1_steps + 1):
        model.train()
        long_inputs, short_inputs, targets = sample_batch(dataset, batch_size=args.batch_size, device=device)

        block0_logits = model.block0_logits(short_inputs, lateral=None)
        block0_loss, block0_accuracy = next_char_metrics(block0_logits, targets)
        block0_optimizer.zero_grad(set_to_none=True)
        block0_loss.backward()
        block0_optimizer.step()

        block1_logits = model.block1.local_logits(long_inputs)
        local_loss, local_accuracy = next_char_metrics(block1_logits, targets)
        block1_optimizer.zero_grad(set_to_none=True)
        local_loss.backward()
        block1_optimizer.step()

        if step % args.print_interval == 0 or step == args.phase1_steps:
            print(
                f"[{label}] phase1 step={step:03d}/{args.phase1_steps} ce={block0_loss.item():.4f} local_ce={local_loss.item():.4f} block0_acc={block0_accuracy.item():.4%} block1_acc={local_accuracy.item():.4%}",
                flush=True,
            )


def run_stage2(
    model: StagedLateralModel,
    dataset: WindowDataset,
    *,
    args: argparse.Namespace,
    device: torch.device,
    label: str,
    shuffle_lateral: bool,
) -> None:
    for parameter in model.block1.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        list(model.block0.parameters()) + list(model.lateral_proj.parameters()),
        lr=args.learning_rate,
    )

    for step in range(1, args.phase2_steps + 1):
        model.train()
        long_inputs, short_inputs, targets = sample_batch(dataset, batch_size=args.batch_size, device=device)
        lateral = model.lateral_from_long(long_inputs, shuffle=shuffle_lateral, detach_block1=True)
        logits = model.block0_logits(short_inputs, lateral=lateral)
        loss, accuracy = next_char_metrics(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.print_interval == 0 or step == args.phase2_steps:
            print(
                f"[{label}] phase2 step={step:03d}/{args.phase2_steps} ce={loss.item():.4f} acc={accuracy.item():.4%}",
                flush=True,
            )


def train_staged_condition(
    *,
    condition_name: str,
    dataset: WindowDataset,
    val_dataset: WindowDataset,
    args: argparse.Namespace,
    device: torch.device,
    shuffle_lateral: bool,
) -> ConditionResult:
    set_seed(args.seed)
    model = StagedLateralModel(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        short_ctx=args.short_ctx,
        long_ctx=args.long_ctx,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        lateral_scale=args.lateral_scale,
    ).to(device)
    started_at = perf_counter()

    run_stage1(model, dataset, args=args, device=device, label=condition_name)
    run_stage2(
        model,
        dataset,
        args=args,
        device=device,
        label=condition_name,
        shuffle_lateral=shuffle_lateral,
    )
    evaluation = evaluate_lateral(
        model,
        val_dataset,
        batch_size=args.batch_size,
        device=device,
        shuffle_lateral=shuffle_lateral,
        detach_block1=True,
    )
    wall_seconds = perf_counter() - started_at
    return ConditionResult(
        name=condition_name,
        val_loss=evaluation.val_loss,
        val_accuracy=evaluation.val_accuracy,
        delta_from_baseline=0.0,
        wall_seconds=wall_seconds,
    )


def print_results_table(results: list[ConditionResult]) -> None:
    baseline_loss = next(result.val_loss for result in results if result.name == "block0_alone")
    header = (
        "condition".ljust(20)
        + "val_loss".rjust(12)
        + "val_acc".rjust(12)
        + "delta".rjust(12)
        + "wall_s".rjust(12)
    )
    print("\nresults", flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for result in results:
        delta = result.val_loss - baseline_loss
        print(
            result.name.ljust(20)
            + f"{result.val_loss:12.4f}"
            + f"{result.val_accuracy:12.4%}"
            + f"{delta:12.4f}"
            + f"{result.wall_seconds:12.2f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    requested_conditions = parse_condition_names(args.conditions)
    device = resolve_device(args.device)
    set_seed(args.seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    train_dataset, val_dataset = load_window_dataset(
        long_ctx=args.long_ctx,
        short_ctx=args.short_ctx,
        train_characters=args.train_characters,
        val_characters=args.val_characters,
    )

    print(
        "staged char lm "
        f"device={device.type} seed={args.seed} train_examples={train_dataset.targets.shape[0]} "
        f"val_examples={val_dataset.targets.shape[0]} vocab={train_dataset.vocab_size} "
        f"short_ctx={args.short_ctx} long_ctx={args.long_ctx} "
        f"phase1_steps={args.phase1_steps} phase2_steps={args.phase2_steps} batch={args.batch_size} "
        f"lateral_scale={args.lateral_scale} conditions={','.join(requested_conditions)}",
        flush=True,
    )

    condition_runners: dict[str, Callable[[], ConditionResult]] = {
        "block0_alone": lambda: train_block0_alone(train_dataset, val_dataset, args, device),
        "staged_lateral": lambda: train_staged_condition(
            condition_name="staged_lateral",
            dataset=train_dataset,
            val_dataset=val_dataset,
            args=args,
            device=device,
            shuffle_lateral=False,
        ),
        "cotrained_lateral": lambda: train_cotrained_lateral(train_dataset, val_dataset, args, device),
        "shuffled_staged": lambda: train_staged_condition(
            condition_name="shuffled_staged",
            dataset=train_dataset,
            val_dataset=val_dataset,
            args=args,
            device=device,
            shuffle_lateral=True,
        ),
    }
    results = [condition_runners[condition_name]() for condition_name in requested_conditions]

    baseline_loss = next(result.val_loss for result in results if result.name == "block0_alone")
    adjusted_results = [
        ConditionResult(
            name=result.name,
            val_loss=result.val_loss,
            val_accuracy=result.val_accuracy,
            delta_from_baseline=result.val_loss - baseline_loss,
            wall_seconds=result.wall_seconds,
        )
        for result in results
    ]
    print_results_table(adjusted_results)


if __name__ == "__main__":
    main()
