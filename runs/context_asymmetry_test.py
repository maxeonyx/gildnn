from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from time import perf_counter

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import set_seed

SEED = 42
VOCAB_SIZE = 8
SHORT_CONTEXT = 4
LONG_CONTEXT = 32
DELAY = 16
TRAIN_SEQUENCES = 4_096
VAL_SEQUENCES = 1_024
TRAINING_STEPS = 500
BATCH_SIZE = 64
LEARNING_RATE = 3e-3
LAMBDA_LOCAL = 1.0
LATERAL_SCALE = 0.1
PRINT_INTERVAL = 100


@dataclass(frozen=True)
class Dataset:
    inputs: Int[Tensor, "examples long_context"]
    targets: Int[Tensor, "examples"]


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    use_lateral: bool
    shuffle_lateral: bool
    use_local_loss: bool
    oracle_long_readout: bool


@dataclass(frozen=True)
class ConditionResult:
    condition: str
    train_ce_loss: float
    train_local_loss: float
    val_total_loss: float
    val_ce_loss: float
    val_local_loss: float
    val_accuracy: float
    wall_seconds: float


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


class ContextAsymmetryModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        short_ctx: int,
        long_ctx: int,
        n_heads: int,
        ff_dim: int,
    ) -> None:
        super().__init__()
        self.short_ctx = short_ctx
        self.long_ctx = long_ctx
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.short_position_embedding = nn.Embedding(short_ctx, d_model)
        self.long_position_embedding = nn.Embedding(long_ctx, d_model)
        self.block0 = TransformerBlock(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim, max_context=short_ctx)
        self.block1 = TransformerBlock(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim, max_context=long_ctx)
        self.lateral_proj = nn.Linear(d_model, d_model)
        self.predictor = nn.Linear(d_model, d_model)
        self.output_head = nn.Linear(d_model, vocab_size)
        self.oracle_head = nn.Linear(d_model, vocab_size)

    def encode_short(self, tokens: Int[Tensor, "batch short_context"]) -> Float[Tensor, "batch short_context d_model"]:
        short_positions = torch.arange(self.short_ctx, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.short_position_embedding(short_positions)
        return self.block0(hidden)

    def encode_long(self, tokens: Int[Tensor, "batch long_context"]) -> Float[Tensor, "batch long_context d_model"]:
        long_positions = torch.arange(self.long_ctx, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.long_position_embedding(long_positions)
        return self.block1(hidden)

    def forward(
        self,
        tokens: Int[Tensor, "batch long_context"],
        *,
        use_lateral: bool,
        shuffle_lateral: bool,
    ) -> tuple[Float[Tensor, "batch vocab"], Float[Tensor, "batch 1 d_model"], Float[Tensor, "batch 1 d_model"]]:
        short_tokens = tokens[:, -self.short_ctx :]
        h0 = self.encode_short(short_tokens)
        h1 = self.encode_long(tokens)
        h0_last = h0[:, -1:, :]
        h1_last = h1[:, -1:, :]

        combined_last = h0_last
        if use_lateral:
            lateral_source = h1_last.detach()
            if shuffle_lateral:
                permutation = torch.randperm(lateral_source.shape[0], device=lateral_source.device)
                lateral_source = lateral_source[permutation]
            combined_last = combined_last + (LATERAL_SCALE * self.lateral_proj(lateral_source))

        logits = self.output_head(combined_last).squeeze(1)
        predicted_hidden = self.predictor(h1_last)
        target_hidden = h0_last.detach()
        return logits, predicted_hidden, target_hidden

    def oracle_logits(self, tokens: Int[Tensor, "batch long_context"]) -> Float[Tensor, "batch vocab"]:
        h1 = self.encode_long(tokens)
        return self.oracle_head(h1[:, -1, :])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--train-sequences", type=int, default=TRAIN_SEQUENCES)
    parser.add_argument("--val-sequences", type=int, default=VAL_SEQUENCES)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--short-ctx", type=int, default=SHORT_CONTEXT)
    parser.add_argument("--long-ctx", type=int, default=LONG_CONTEXT)
    parser.add_argument("--delay", type=int, default=DELAY)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--ff-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--lambda-local", type=float, default=LAMBDA_LOCAL)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    errors: list[str] = []
    if args.steps <= 0:
        errors.append(f"steps must be positive, got {args.steps}")
    if args.batch_size <= 0:
        errors.append(f"batch_size must be positive, got {args.batch_size}")
    if args.learning_rate <= 0:
        errors.append(f"learning_rate must be positive, got {args.learning_rate}")
    if args.train_sequences <= 0:
        errors.append(f"train_sequences must be positive, got {args.train_sequences}")
    if args.val_sequences <= 0:
        errors.append(f"val_sequences must be positive, got {args.val_sequences}")
    if args.vocab_size <= 1:
        errors.append(f"vocab_size must be > 1, got {args.vocab_size}")
    if args.short_ctx <= 0:
        errors.append(f"short_ctx must be positive, got {args.short_ctx}")
    if args.long_ctx <= 0:
        errors.append(f"long_ctx must be positive, got {args.long_ctx}")
    if args.short_ctx >= args.long_ctx:
        errors.append(f"short_ctx must be < long_ctx, got {args.short_ctx} and {args.long_ctx}")
    if args.delay <= args.short_ctx:
        errors.append(
            f"delay must exceed short_ctx so the short block cannot see the source token, got delay={args.delay} short_ctx={args.short_ctx}"
        )
    if args.delay >= args.long_ctx:
        errors.append(f"delay must be < long_ctx, got delay={args.delay} long_ctx={args.long_ctx}")
    if args.d_model <= 0:
        errors.append(f"d_model must be positive, got {args.d_model}")
    if args.ff_dim <= 0:
        errors.append(f"ff_dim must be positive, got {args.ff_dim}")
    if args.n_heads <= 0:
        errors.append(f"n_heads must be positive, got {args.n_heads}")
    if args.d_model % args.n_heads != 0:
        errors.append(f"d_model must be divisible by n_heads, got {args.d_model} and {args.n_heads}")
    if args.lambda_local < 0:
        errors.append(f"lambda_local must be non-negative, got {args.lambda_local}")
    if errors:
        raise ValueError("Argument validation failed:\n- " + "\n- ".join(errors))


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device is not None:
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_dataset(
    *,
    num_sequences: int,
    long_ctx: int,
    vocab_size: int,
    delay: int,
    generator: torch.Generator,
) -> Dataset:
    inputs = torch.randint(0, vocab_size, (num_sequences, long_ctx), generator=generator)
    targets = inputs[:, long_ctx - delay]
    return Dataset(inputs=inputs, targets=targets)


def sample_batch(dataset: Dataset, *, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    indices = torch.randint(0, dataset.inputs.shape[0], (batch_size,))
    inputs = dataset.inputs[indices].to(device)
    targets = dataset.targets[indices].to(device)
    return inputs, targets


def compute_losses(
    model: ContextAsymmetryModel,
    inputs: Int[Tensor, "batch long_context"],
    targets: Int[Tensor, "batch"],
    *,
    condition: Condition,
    lambda_local: float,
) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
    if condition.oracle_long_readout:
        logits = model.oracle_logits(inputs)
        ce_loss = F.cross_entropy(logits, targets)
        local_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        return ce_loss, ce_loss.detach(), local_loss

    logits, predicted_hidden, target_hidden = model(
        inputs,
        use_lateral=condition.use_lateral,
        shuffle_lateral=condition.shuffle_lateral,
    )
    ce_loss = F.cross_entropy(logits, targets)
    local_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
    if condition.use_local_loss:
        local_loss = F.mse_loss(predicted_hidden, target_hidden)
    total_loss = ce_loss + (lambda_local * local_loss)
    return total_loss, ce_loss.detach(), local_loss.detach()


@torch.no_grad()
def evaluate(
    model: ContextAsymmetryModel,
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device,
    condition: Condition,
    lambda_local: float,
) -> tuple[float, float, float, float]:
    model.eval()
    total_total_loss = 0.0
    total_ce_loss = 0.0
    total_local_loss = 0.0
    total_correct = 0

    for start in range(0, dataset.inputs.shape[0], batch_size):
        stop = min(start + batch_size, dataset.inputs.shape[0])
        inputs = dataset.inputs[start:stop].to(device)
        targets = dataset.targets[start:stop].to(device)
        total_loss, ce_loss, local_loss = compute_losses(
            model,
            inputs,
            targets,
            condition=condition,
            lambda_local=lambda_local,
        )
        if condition.oracle_long_readout:
            logits = model.oracle_logits(inputs)
        else:
            logits, _, _ = model(
                inputs,
                use_lateral=condition.use_lateral,
                shuffle_lateral=condition.shuffle_lateral,
            )
        predictions = logits.argmax(dim=-1)
        batch_examples = stop - start
        total_total_loss += total_loss.item() * batch_examples
        total_ce_loss += ce_loss.item() * batch_examples
        total_local_loss += local_loss.item() * batch_examples
        total_correct += (predictions == targets).sum().item()

    total_examples = dataset.inputs.shape[0]
    return (
        total_total_loss / total_examples,
        total_ce_loss / total_examples,
        total_local_loss / total_examples,
        total_correct / total_examples,
    )


def conditions() -> tuple[Condition, ...]:
    return (
        Condition(
            key="block0_alone",
            label="block0_alone",
            use_lateral=False,
            shuffle_lateral=False,
            use_local_loss=False,
            oracle_long_readout=False,
        ),
        Condition(
            key="block0_plus_block1",
            label="block0_plus_block1",
            use_lateral=True,
            shuffle_lateral=False,
            use_local_loss=True,
            oracle_long_readout=False,
        ),
        Condition(
            key="block1_oracle",
            label="block1_oracle",
            use_lateral=False,
            shuffle_lateral=False,
            use_local_loss=False,
            oracle_long_readout=True,
        ),
        Condition(
            key="shuffled_lateral",
            label="block0_plus_shuffled_lateral",
            use_lateral=True,
            shuffle_lateral=True,
            use_local_loss=True,
            oracle_long_readout=False,
        ),
    )


def train_condition(
    *,
    condition: Condition,
    train_dataset: Dataset,
    val_dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> ConditionResult:
    set_seed(SEED)
    model = ContextAsymmetryModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        short_ctx=args.short_ctx,
        long_ctx=args.long_ctx,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    latest_ce_loss = 0.0
    latest_local_loss = 0.0
    started_at = perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        inputs, targets = sample_batch(train_dataset, batch_size=args.batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        total_loss, ce_loss, local_loss = compute_losses(
            model,
            inputs,
            targets,
            condition=condition,
            lambda_local=args.lambda_local,
        )
        total_loss.backward()
        optimizer.step()

        latest_ce_loss = ce_loss.item()
        latest_local_loss = local_loss.item()
        if step % PRINT_INTERVAL == 0 or step == args.steps:
            print(
                f"[{condition.label}] step={step:03d}/{args.steps} "
                f"ce={latest_ce_loss:.4f} local={latest_local_loss:.4f} total={total_loss.item():.4f}",
                flush=True,
            )

    wall_seconds = perf_counter() - started_at
    val_total_loss, val_ce_loss, val_local_loss, val_accuracy = evaluate(
        model,
        val_dataset,
        batch_size=args.batch_size,
        device=device,
        condition=condition,
        lambda_local=args.lambda_local,
    )
    print(
        f"[{condition.label}] done wall_seconds={wall_seconds:.2f} "
        f"val_total={val_total_loss:.4f} val_ce={val_ce_loss:.4f} val_local={val_local_loss:.4f} val_acc={val_accuracy:.4%}",
        flush=True,
    )
    return ConditionResult(
        condition=condition.label,
        train_ce_loss=latest_ce_loss,
        train_local_loss=latest_local_loss,
        val_total_loss=val_total_loss,
        val_ce_loss=val_ce_loss,
        val_local_loss=val_local_loss,
        val_accuracy=val_accuracy,
        wall_seconds=wall_seconds,
    )


def verdicts(results: dict[str, ConditionResult], *, vocab_size: int) -> dict[str, bool]:
    chance = 1.0 / vocab_size
    return {
        "block0_near_chance": results["block0_alone"].val_accuracy <= max(chance + 0.10, 0.25),
        "lateral_above_50pct": results["block0_plus_block1"].val_accuracy >= 0.50,
        "oracle_strong": results["block1_oracle"].val_accuracy >= 0.95,
        "shuffled_near_chance": results["shuffled_lateral"].val_accuracy <= max(chance + 0.10, 0.25),
        "lateral_beats_controls": results["block0_plus_block1"].val_accuracy
        >= max(results["block0_alone"].val_accuracy, results["shuffled_lateral"].val_accuracy) + 0.30,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    set_seed(SEED)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    generator = torch.Generator().manual_seed(SEED)
    train_dataset = generate_dataset(
        num_sequences=args.train_sequences,
        long_ctx=args.long_ctx,
        vocab_size=args.vocab_size,
        delay=args.delay,
        generator=generator,
    )
    val_dataset = generate_dataset(
        num_sequences=args.val_sequences,
        long_ctx=args.long_ctx,
        vocab_size=args.vocab_size,
        delay=args.delay,
        generator=generator,
    )

    print(
        "context-asymmetry test "
        f"device={device.type} seed={SEED} vocab={args.vocab_size} short_ctx={args.short_ctx} "
        f"long_ctx={args.long_ctx} delay={args.delay} steps={args.steps} batch={args.batch_size}",
        flush=True,
    )

    results_by_key: dict[str, ConditionResult] = {}
    for condition in conditions():
        result = train_condition(
            condition=condition,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=args,
            device=device,
        )
        results_by_key[condition.key] = result

    checks = verdicts(results_by_key, vocab_size=args.vocab_size)
    overall_pass = all(checks.values())

    print("\nresults", flush=True)
    for condition in conditions():
        print(asdict(results_by_key[condition.key]), flush=True)

    print("\nverdict", flush=True)
    for key, passed in checks.items():
        print(f"{key}: {'PASS' if passed else 'FAIL'}", flush=True)
    print(f"overall: {'PASS' if overall_pass else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
