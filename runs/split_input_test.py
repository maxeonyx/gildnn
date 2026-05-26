from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from time import perf_counter

import torch
from torch import Tensor, nn
from torch.nn import functional as F

SEED = 42
VOCAB_SIZE = 8
SEQUENCE_LENGTH = 32
CONTEXT_STRIDE = 8
TRAIN_SEQUENCES = 4096
VAL_SEQUENCES = 1024
TRAINING_STEPS = 400
BATCH_SIZE = 64
LEARNING_RATE = 3e-3
LAMBDA_LOCAL = 1.0
PRINT_INTERVAL = 100
LATERAL_SCALE = 1.0


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    use_lateral: bool
    shuffle_lateral: bool
    use_local_loss: bool


@dataclass(frozen=True)
class Dataset:
    tokens_a: Tensor
    tokens_b: Tensor
    targets: Tensor


@dataclass(frozen=True)
class ConditionResult:
    condition: str
    train_ce_loss: float
    train_local_loss: float
    val_loss: float
    val_accuracy: float


class Block(nn.Module):
    def __init__(self, d_model: int, feedforward_dim: int) -> None:
        super().__init__()
        self.proj_in = nn.Linear(d_model, feedforward_dim)
        self.proj_out = nn.Linear(feedforward_dim, d_model)

    def forward(self, hidden: Tensor) -> Tensor:
        residual = hidden
        hidden = self.proj_in(hidden)
        hidden = F.gelu(hidden)
        hidden = self.proj_out(hidden)
        return residual + hidden


class SplitInputModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32, feedforward_dim: int = 64) -> None:
        super().__init__()
        self.embed_a = nn.Embedding(vocab_size, d_model)
        self.embed_b = nn.Embedding(vocab_size, d_model)
        self.block0 = Block(d_model=d_model, feedforward_dim=feedforward_dim)
        self.block1 = Block(d_model=d_model, feedforward_dim=feedforward_dim)
        self.lateral_proj = nn.Linear(d_model, d_model)
        self.local_head = nn.Linear(d_model, d_model)
        self.combiner = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tokens_a: Tensor,
        tokens_b: Tensor,
        *,
        use_lateral: bool,
        shuffle_lateral: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        embedded_a = self.embed_a(tokens_a)
        embedded_b = self.embed_b(tokens_b)
        h0 = self.block0(embedded_a)
        h1 = self.block1(embedded_b)

        lateral = torch.zeros_like(h0)
        if use_lateral:
            lateral_source = h1.detach()
            if shuffle_lateral:
                permutation = torch.randperm(lateral_source.shape[0], device=lateral_source.device)
                lateral_source = lateral_source[permutation]
            lateral = LATERAL_SCALE * self.lateral_proj(lateral_source)

        combined = self.combiner(torch.cat([h0, lateral], dim=-1))
        logits = self.output_head(combined)
        local_prediction = self.local_head(h1)
        local_target = embedded_b.detach()
        return logits, local_prediction, local_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--train-sequences", type=int, default=TRAIN_SEQUENCES)
    parser.add_argument("--val-sequences", type=int, default=VAL_SEQUENCES)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--context-stride", type=int, default=CONTEXT_STRIDE)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--feedforward-dim", type=int, default=64)
    parser.add_argument("--lambda-local", type=float, default=LAMBDA_LOCAL)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device is not None:
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    if args.sequence_length <= 0:
        errors.append(f"sequence_length must be positive, got {args.sequence_length}")
    if args.vocab_size <= 1:
        errors.append(f"vocab_size must be > 1, got {args.vocab_size}")
    if args.context_stride <= 0:
        errors.append(f"context_stride must be positive, got {args.context_stride}")
    if args.d_model <= 0:
        errors.append(f"d_model must be positive, got {args.d_model}")
    if args.feedforward_dim <= 0:
        errors.append(f"feedforward_dim must be positive, got {args.feedforward_dim}")
    if args.lambda_local < 0:
        errors.append(f"lambda_local must be non-negative, got {args.lambda_local}")
    if errors:
        raise ValueError("Argument validation failed:\n- " + "\n- ".join(errors))


def make_dataset(
    *,
    num_sequences: int,
    sequence_length: int,
    vocab_size: int,
    context_stride: int,
    generator: torch.Generator,
) -> Dataset:
    tokens_a = torch.randint(0, vocab_size, (num_sequences, sequence_length), generator=generator)
    context_segments = math.ceil(sequence_length / context_stride)
    slow_context = torch.randint(0, vocab_size, (num_sequences, context_segments), generator=generator)
    tokens_b = slow_context.repeat_interleave(context_stride, dim=1)[:, :sequence_length]
    targets = (tokens_a + tokens_b) % vocab_size
    return Dataset(tokens_a=tokens_a, tokens_b=tokens_b, targets=targets)


def sample_batch(dataset: Dataset, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    indices = torch.randint(0, dataset.tokens_a.shape[0], (batch_size,))
    tokens_a = dataset.tokens_a[indices].to(device)
    tokens_b = dataset.tokens_b[indices].to(device)
    targets = dataset.targets[indices].to(device)
    return tokens_a, tokens_b, targets


def compute_losses(
    model: SplitInputModel,
    tokens_a: Tensor,
    tokens_b: Tensor,
    targets: Tensor,
    *,
    condition: Condition,
    lambda_local: float,
) -> tuple[Tensor, Tensor, Tensor]:
    logits, local_prediction, local_target = model(
        tokens_a,
        tokens_b,
        use_lateral=condition.use_lateral,
        shuffle_lateral=condition.shuffle_lateral,
    )
    ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    local_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
    if condition.use_local_loss:
        local_loss = F.mse_loss(local_prediction, local_target)
    total_loss = ce_loss + (lambda_local * local_loss)
    return total_loss, ce_loss.detach(), local_loss.detach()


@torch.no_grad()
def evaluate(
    model: SplitInputModel,
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device,
    condition: Condition,
    lambda_local: float,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    for start in range(0, dataset.tokens_a.shape[0], batch_size):
        end = min(start + batch_size, dataset.tokens_a.shape[0])
        tokens_a = dataset.tokens_a[start:end].to(device)
        tokens_b = dataset.tokens_b[start:end].to(device)
        targets = dataset.targets[start:end].to(device)
        loss, _, _ = compute_losses(
            model,
            tokens_a,
            tokens_b,
            targets,
            condition=condition,
            lambda_local=lambda_local,
        )
        logits, _, _ = model(
            tokens_a,
            tokens_b,
            use_lateral=condition.use_lateral,
            shuffle_lateral=condition.shuffle_lateral,
        )
        predictions = logits.argmax(dim=-1)
        total_loss += loss.item() * (end - start)
        total_correct += (predictions == targets).sum().item()
        total_tokens += targets.numel()
    return total_loss / dataset.tokens_a.shape[0], total_correct / total_tokens


def train_condition(
    *,
    condition: Condition,
    train_dataset: Dataset,
    val_dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> ConditionResult:
    set_seed(SEED)
    model = SplitInputModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        feedforward_dim=args.feedforward_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    latest_ce_loss = 0.0
    latest_local_loss = 0.0
    start_time = perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        tokens_a, tokens_b, targets = sample_batch(train_dataset, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        total_loss, ce_loss, local_loss = compute_losses(
            model,
            tokens_a,
            tokens_b,
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
                f"ce={latest_ce_loss:.4f} local={latest_local_loss:.4f} total={total_loss.item():.4f}"
            )

    elapsed = perf_counter() - start_time
    val_loss, val_accuracy = evaluate(
        model,
        val_dataset,
        batch_size=args.batch_size,
        device=device,
        condition=condition,
        lambda_local=args.lambda_local,
    )
    print(
        f"[{condition.label}] done in {elapsed:.2f}s "
        f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4%}"
    )
    return ConditionResult(
        condition=condition.label,
        train_ce_loss=latest_ce_loss,
        train_local_loss=latest_local_loss,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
    )


def conditions() -> tuple[Condition, ...]:
    return (
        Condition(
            key="no_lateral",
            label="block0_alone",
            use_lateral=False,
            shuffle_lateral=False,
            use_local_loss=False,
        ),
        Condition(
            key="lateral",
            label="block0_plus_block1",
            use_lateral=True,
            shuffle_lateral=False,
            use_local_loss=True,
        ),
        Condition(
            key="shuffled",
            label="block0_plus_shuffled_lateral",
            use_lateral=True,
            shuffle_lateral=True,
            use_local_loss=True,
        ),
    )


def verdicts(results: dict[str, ConditionResult], *, vocab_size: int) -> dict[str, bool]:
    chance = 1.0 / vocab_size
    no_lateral = results["no_lateral"].val_accuracy
    with_lateral = results["lateral"].val_accuracy
    shuffled = results["shuffled"].val_accuracy
    return {
        "no_lateral_near_chance": no_lateral <= max(chance + 0.10, 0.25),
        "lateral_beats_50_percent": with_lateral >= 0.50,
        "shuffled_near_chance": shuffled <= max(chance + 0.10, 0.25),
        "lateral_beats_controls": with_lateral >= max(no_lateral, shuffled) + 0.30,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    set_seed(SEED)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    data_generator = torch.Generator().manual_seed(SEED)
    train_dataset = make_dataset(
        num_sequences=args.train_sequences,
        sequence_length=args.sequence_length,
        vocab_size=args.vocab_size,
        context_stride=args.context_stride,
        generator=data_generator,
    )
    val_dataset = make_dataset(
        num_sequences=args.val_sequences,
        sequence_length=args.sequence_length,
        vocab_size=args.vocab_size,
        context_stride=args.context_stride,
        generator=data_generator,
    )

    print(
        "split-input test "
        f"device={device.type} seed={SEED} vocab={args.vocab_size} seq={args.sequence_length} "
        f"stride={args.context_stride} steps={args.steps} batch={args.batch_size}"
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

    print("\nresults")
    for condition in conditions():
        result = results_by_key[condition.key]
        print(asdict(result))

    print("\nverdict")
    for key, passed in checks.items():
        print(f"{key}: {'PASS' if passed else 'FAIL'}")
    print(f"overall: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
