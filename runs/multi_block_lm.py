from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from contextlib import nullcontext

# Ensure repo root is importable regardless of how this script is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import load_dataset, set_seed

DEFAULT_SEED = 42
SHORT_CONTEXT = 4
MID_CONTEXT = 32
LONG_CONTEXT = 128
TRAIN_CHARACTERS = 80_000
VAL_CHARACTERS = 20_000
TRAINING_STEPS = 800
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 8192
LEARNING_RATE = 3e-3
LATERAL_SCALE = 0.2
PRINT_INTERVAL = 100
ALL_CONDITIONS = ("block0_alone", "one_block_mid", "one_block_long", "two_blocks")


@dataclass(frozen=True)
class WindowDataset:
    short_inputs: Int[Tensor, "examples short_context"]
    mid_inputs: Int[Tensor, "examples mid_context"]
    long_inputs: Int[Tensor, "examples long_context"]
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


def dataset_to_device(dataset: WindowDataset, device: torch.device) -> WindowDataset:
    return WindowDataset(
        short_inputs=dataset.short_inputs.to(device),
        mid_inputs=dataset.mid_inputs.to(device),
        long_inputs=dataset.long_inputs.to(device),
        targets=dataset.targets.to(device),
        vocab_size=dataset.vocab_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--conditions", type=str, default=",".join(ALL_CONDITIONS))
    return parser.parse_args()


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


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device is not None:
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_window_dataset() -> tuple[WindowDataset, WindowDataset]:
    (train_long_inputs, train_targets), (val_long_inputs, val_targets), vocab_size = load_dataset(
        context_size=LONG_CONTEXT,
        train_characters=TRAIN_CHARACTERS,
        val_characters=VAL_CHARACTERS,
    )
    return (
        WindowDataset(
            short_inputs=train_long_inputs[:, -SHORT_CONTEXT:],
            mid_inputs=train_long_inputs[:, -MID_CONTEXT:],
            long_inputs=train_long_inputs,
            targets=train_targets,
            vocab_size=vocab_size,
        ),
        WindowDataset(
            short_inputs=val_long_inputs[:, -SHORT_CONTEXT:],
            mid_inputs=val_long_inputs[:, -MID_CONTEXT:],
            long_inputs=val_long_inputs,
            targets=val_targets,
            vocab_size=vocab_size,
        ),
    )


def sample_batch(
    dataset: WindowDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    indices = torch.randint(0, dataset.targets.shape[0], (batch_size,), device=device)
    short_inputs = dataset.short_inputs[indices]
    mid_inputs = dataset.mid_inputs[indices]
    long_inputs = dataset.long_inputs[indices]
    targets = dataset.targets[indices]
    return short_inputs, mid_inputs, long_inputs, targets


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def next_char_metrics(
    logits: Float[Tensor, "batch vocab"],
    targets: Int[Tensor, "batch"],
) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
    loss = F.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=-1) == targets).float().mean()
    return loss, accuracy


class CausalSelfAttention(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, max_context: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} and {n_heads}.")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Float[Tensor, "batch context d_model"]) -> Float[Tensor, "batch context d_model"]:
        batch_size, context_size, d_model = x.shape
        queries, keys, values = self.qkv(x).chunk(3, dim=-1)

        def reshape_heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch_size, context_size, self.n_heads, self.head_dim).transpose(1, 2)

        queries = reshape_heads(queries)
        keys = reshape_heads(keys)
        values = reshape_heads(values)
        attended = F.scaled_dot_product_attention(queries, keys, values, is_causal=True)
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


class SequenceEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, context_size: int, d_model: int, n_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.context_size = context_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.block = TransformerBlock(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim, max_context=context_size)

    def forward(self, inputs: Int[Tensor, "batch context"]) -> Float[Tensor, "batch context d_model"]:
        positions = torch.arange(self.context_size, device=inputs.device)
        hidden = self.token_embedding(inputs) + self.position_embedding(positions)
        return self.block(hidden)


class OutputBlock(nn.Module):
    def __init__(self, *, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.encoder = SequenceEncoder(
            vocab_size=vocab_size,
            context_size=SHORT_CONTEXT,
            d_model=d_model,
            n_heads=2,
            ff_dim=128,
        )
        self.output_head = nn.Linear(d_model, vocab_size)

    def encode_last_hidden(self, short_inputs: Int[Tensor, "batch short_context"]) -> Float[Tensor, "batch 1 d_model"]:
        return self.encoder(short_inputs)[:, -1:, :]

    def logits(
        self,
        short_inputs: Int[Tensor, "batch short_context"],
        *,
        lateral_sum: Float[Tensor, "batch 1 d_model"] | None = None,
    ) -> Float[Tensor, "batch vocab"]:
        last_hidden = self.encode_last_hidden(short_inputs)
        if lateral_sum is not None:
            last_hidden = last_hidden + lateral_sum
        return self.output_head(last_hidden).squeeze(1)


class InteriorBlock(nn.Module):
    def __init__(self, *, vocab_size: int, context_size: int, d_model: int, n_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.encoder = SequenceEncoder(
            vocab_size=vocab_size,
            context_size=context_size,
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
        )
        self.local_head = nn.Linear(d_model, vocab_size)
        self.lateral_proj = nn.Linear(d_model, d_model)

    def encode_last_hidden(self, inputs: Int[Tensor, "batch context"]) -> Float[Tensor, "batch 1 d_model"]:
        return self.encoder(inputs)[:, -1:, :]

    def last_hidden_and_local_logits(
        self,
        inputs: Int[Tensor, "batch context"],
    ) -> tuple[Float[Tensor, "batch 1 d_model"], Float[Tensor, "batch vocab"]]:
        hidden = self.encoder(inputs)
        last_hidden = hidden[:, -1:, :]
        local_logits = self.local_head(last_hidden.squeeze(1))
        return last_hidden, local_logits

    def lateral(self, last_hidden: Float[Tensor, "batch 1 d_model"]) -> Float[Tensor, "batch 1 d_model"]:
        return LATERAL_SCALE * self.lateral_proj(last_hidden.detach())


class MultiBlockModel(nn.Module):
    def __init__(self, *, vocab_size: int, use_mid: bool, use_long: bool) -> None:
        super().__init__()
        d_model = 64
        n_heads = 2
        ff_dim = 128
        self.output_block = OutputBlock(vocab_size=vocab_size, d_model=d_model)
        self.mid_block = (
            InteriorBlock(
                vocab_size=vocab_size,
                context_size=MID_CONTEXT,
                d_model=d_model,
                n_heads=n_heads,
                ff_dim=ff_dim,
            )
            if use_mid
            else None
        )
        self.long_block = (
            InteriorBlock(
                vocab_size=vocab_size,
                context_size=LONG_CONTEXT,
                d_model=d_model,
                n_heads=n_heads,
                ff_dim=ff_dim,
            )
            if use_long
            else None
        )

    def output_logits(
        self,
        short_inputs: Int[Tensor, "batch short_context"],
        *,
        mid_last_hidden: Float[Tensor, "batch 1 d_model"] | None = None,
        long_last_hidden: Float[Tensor, "batch 1 d_model"] | None = None,
    ) -> Float[Tensor, "batch vocab"]:
        lateral_terms: list[Tensor] = []
        if self.mid_block is not None and mid_last_hidden is not None:
            lateral_terms.append(self.mid_block.lateral(mid_last_hidden))
        if self.long_block is not None and long_last_hidden is not None:
            lateral_terms.append(self.long_block.lateral(long_last_hidden))
        lateral_sum = sum(lateral_terms) if lateral_terms else None
        return self.output_block.logits(short_inputs, lateral_sum=lateral_sum)


def build_model(condition_name: str, vocab_size: int) -> MultiBlockModel:
    match condition_name:
        case "block0_alone":
            return MultiBlockModel(vocab_size=vocab_size, use_mid=False, use_long=False)
        case "one_block_mid":
            return MultiBlockModel(vocab_size=vocab_size, use_mid=True, use_long=False)
        case "one_block_long":
            return MultiBlockModel(vocab_size=vocab_size, use_mid=False, use_long=True)
        case "two_blocks":
            return MultiBlockModel(vocab_size=vocab_size, use_mid=True, use_long=True)
        case _:
            raise ValueError(f"Unsupported condition: {condition_name}")


def make_optimizer(model: nn.Module, device: torch.device) -> torch.optim.Optimizer:
    fused = device.type == "cuda"
    return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, fused=fused)


def warm_up_cuda(train_dataset: WindowDataset, device: torch.device) -> None:
    if device.type != "cuda":
        return

    set_seed(DEFAULT_SEED)
    model = build_model("two_blocks", train_dataset.vocab_size).to(device)
    optimizer = make_optimizer(model, device)
    short_inputs, mid_inputs, long_inputs, targets = sample_batch(
        train_dataset,
        batch_size=BATCH_SIZE,
        device=device,
    )

    model.train()
    with autocast_context(device):
        mid_last_hidden, mid_local_logits = model.mid_block.last_hidden_and_local_logits(mid_inputs)
        long_last_hidden, long_local_logits = model.long_block.last_hidden_and_local_logits(long_inputs)
        output_logits = model.output_logits(
            short_inputs,
            mid_last_hidden=mid_last_hidden,
            long_last_hidden=long_last_hidden,
        )
        output_loss, _ = next_char_metrics(output_logits, targets)
        mid_loss, _ = next_char_metrics(mid_local_logits, targets)
        long_loss, _ = next_char_metrics(long_local_logits, targets)
        total_loss = output_loss + mid_loss + long_loss

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode(), autocast_context(device):
        mid_last_hidden = model.mid_block.encode_last_hidden(mid_inputs)
        long_last_hidden = model.long_block.encode_last_hidden(long_inputs)
        model.output_logits(
            short_inputs,
            mid_last_hidden=mid_last_hidden,
            long_last_hidden=long_last_hidden,
        )

    torch.cuda.synchronize()
    del optimizer
    del model


def train_condition(
    condition_name: str,
    train_dataset: WindowDataset,
    val_dataset: WindowDataset,
    *,
    seed: int,
    device: torch.device,
) -> ConditionResult:
    set_seed(seed)
    model = build_model(condition_name, train_dataset.vocab_size).to(device)
    optimizer = make_optimizer(model, device)
    started_at = perf_counter()

    for step in range(1, TRAINING_STEPS + 1):
        model.train()
        short_inputs, mid_inputs, long_inputs, targets = sample_batch(
            train_dataset,
            batch_size=BATCH_SIZE,
            device=device,
        )

        mid_loss_value = 0.0
        long_loss_value = 0.0
        mid_acc_value = 0.0
        long_acc_value = 0.0
        total_loss_terms: list[Tensor] = []

        mid_last_hidden = None
        long_last_hidden = None

        with autocast_context(device):
            if model.mid_block is not None:
                mid_last_hidden, mid_local_logits = model.mid_block.last_hidden_and_local_logits(mid_inputs)
                mid_loss, mid_accuracy = next_char_metrics(mid_local_logits, targets)
                total_loss_terms.append(mid_loss)
                mid_loss_value = mid_loss.item()
                mid_acc_value = mid_accuracy.item()

            if model.long_block is not None:
                long_last_hidden, long_local_logits = model.long_block.last_hidden_and_local_logits(long_inputs)
                long_loss, long_accuracy = next_char_metrics(long_local_logits, targets)
                total_loss_terms.append(long_loss)
                long_loss_value = long_loss.item()
                long_acc_value = long_accuracy.item()

            output_logits = model.output_logits(
                short_inputs,
                mid_last_hidden=mid_last_hidden,
                long_last_hidden=long_last_hidden,
            )
            output_loss, output_accuracy = next_char_metrics(output_logits, targets)
            total_loss_terms.insert(0, output_loss)
            total_loss = sum(total_loss_terms)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step % PRINT_INTERVAL == 0 or step == TRAINING_STEPS:
            print(
                f"[{condition_name}] step={step:03d}/{TRAINING_STEPS} "
                f"block0_ce={output_loss.item():.4f} block0_acc={output_accuracy.item():.4%} "
                f"mid_ce={mid_loss_value:.4f} mid_acc={mid_acc_value:.4%} "
                f"long_ce={long_loss_value:.4f} long_acc={long_acc_value:.4%}",
                flush=True,
            )

    evaluation = evaluate_condition(model, val_dataset, batch_size=EVAL_BATCH_SIZE, device=device)
    wall_seconds = perf_counter() - started_at
    return ConditionResult(
        name=condition_name,
        val_loss=evaluation.val_loss,
        val_accuracy=evaluation.val_accuracy,
        delta_from_baseline=0.0,
        wall_seconds=wall_seconds,
    )


@torch.no_grad()
def evaluate_condition(
    model: MultiBlockModel,
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
        short_inputs = dataset.short_inputs[start:stop]
        mid_inputs = dataset.mid_inputs[start:stop]
        long_inputs = dataset.long_inputs[start:stop]
        targets = dataset.targets[start:stop]

        with autocast_context(device):
            mid_last_hidden = model.mid_block.encode_last_hidden(mid_inputs) if model.mid_block is not None else None
            long_last_hidden = model.long_block.encode_last_hidden(long_inputs) if model.long_block is not None else None
            logits = model.output_logits(
                short_inputs,
                mid_last_hidden=mid_last_hidden,
                long_last_hidden=long_last_hidden,
            )
            loss, _ = next_char_metrics(logits, targets)
        total_loss += loss.item() * (stop - start)
        total_correct += (logits.argmax(dim=-1) == targets).sum().item()

    return EvalResult(
        val_loss=total_loss / total_examples,
        val_accuracy=total_correct / total_examples,
    )


def print_results_table(results: list[ConditionResult]) -> None:
    baseline_loss = next((result.val_loss for result in results if result.name == "block0_alone"), None)
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
        delta = result.delta_from_baseline if baseline_loss is not None else float("nan")
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
    requested_conditions = parse_condition_names(args.conditions)
    device = resolve_device(args.device)
    set_seed(args.seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_dataset, val_dataset = load_window_dataset()
    train_dataset = dataset_to_device(train_dataset, device)
    val_dataset = dataset_to_device(val_dataset, device)
    warm_up_cuda(train_dataset, device)
    print(
        "multi block lm "
        f"device={device.type} seed={args.seed} train_examples={train_dataset.targets.shape[0]} "
        f"val_examples={val_dataset.targets.shape[0]} vocab={train_dataset.vocab_size} "
        f"short_ctx={SHORT_CONTEXT} mid_ctx={MID_CONTEXT} long_ctx={LONG_CONTEXT} "
        f"steps={TRAINING_STEPS} batch={BATCH_SIZE} lateral_scale={LATERAL_SCALE} "
        f"conditions={','.join(requested_conditions)}",
        flush=True,
    )

    results = [
        train_condition(
            condition_name,
            train_dataset,
            val_dataset,
            seed=args.seed,
            device=device,
        )
        for condition_name in requested_conditions
    ]

    baseline_loss = next((result.val_loss for result in results if result.name == "block0_alone"), None)
    adjusted_results = [
        ConditionResult(
            name=result.name,
            val_loss=result.val_loss,
            val_accuracy=result.val_accuracy,
            delta_from_baseline=(result.val_loss - baseline_loss) if baseline_loss is not None else float("nan"),
            wall_seconds=result.wall_seconds,
        )
        for result in results
    ]
    print_results_table(adjusted_results)


if __name__ == "__main__":
    main()
