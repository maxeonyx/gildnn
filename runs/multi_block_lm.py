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

from core.fixed_window_char import set_seed

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
    targets_multi: Int[Tensor, "examples target_offsets"]
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


@dataclass(frozen=True)
class LongBlockConfig:
    d_model: int
    n_layers: int


@dataclass(frozen=True)
class PretrainConfig:
    mid_steps: int
    long_steps: int


def dataset_to_device(dataset: WindowDataset, device: torch.device) -> WindowDataset:
    return WindowDataset(
        short_inputs=dataset.short_inputs.to(device),
        mid_inputs=dataset.mid_inputs.to(device),
        long_inputs=dataset.long_inputs.to(device),
        targets_multi=dataset.targets_multi.to(device),
        vocab_size=dataset.vocab_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--conditions", type=str, default=",".join(ALL_CONDITIONS))
    parser.add_argument("--long-d-model", type=int, default=64)
    parser.add_argument("--long-n-layers", type=int, default=1)
    parser.add_argument("--long-target-offset", type=int, default=0)
    parser.add_argument("--pretrain-mid-steps", type=int, default=0)
    parser.add_argument("--pretrain-long-steps", type=int, default=0)
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS)
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


def resolve_long_block_config(args: argparse.Namespace) -> LongBlockConfig:
    if args.long_d_model <= 0:
        raise ValueError(f"long_d_model must be positive, got {args.long_d_model}")
    if args.long_n_layers <= 0:
        raise ValueError(f"long_n_layers must be positive, got {args.long_n_layers}")
    return LongBlockConfig(d_model=args.long_d_model, n_layers=args.long_n_layers)


def resolve_long_target_offset(args: argparse.Namespace) -> int:
    if args.long_target_offset < 0:
        raise ValueError(f"long_target_offset must be non-negative, got {args.long_target_offset}")
    return args.long_target_offset


def resolve_pretrain_config(args: argparse.Namespace) -> PretrainConfig:
    if args.pretrain_mid_steps < 0:
        raise ValueError(f"pretrain_mid_steps must be non-negative, got {args.pretrain_mid_steps}")
    if args.pretrain_long_steps < 0:
        raise ValueError(f"pretrain_long_steps must be non-negative, got {args.pretrain_long_steps}")
    return PretrainConfig(mid_steps=args.pretrain_mid_steps, long_steps=args.pretrain_long_steps)


def _resolve_text_file() -> Path:
    return Path(__file__).resolve().parent.parent / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"


def _choose_validation_text(
    text: str,
    *,
    train_text: str,
    val_characters: int,
) -> str:
    start = len(train_text)
    stop = start + val_characters
    candidate = text[start:stop]
    if len(candidate) < val_characters:
        raise ValueError(
            f"Need validation slice of {val_characters} characters, got {len(candidate)}."
        )
    if set(candidate).issubset(set(train_text)):
        return candidate

    max_start = len(text) - val_characters
    for candidate_start in range(start + 1, max_start + 1):
        candidate_stop = candidate_start + val_characters
        candidate = text[candidate_start:candidate_stop]
        if set(candidate).issubset(set(train_text)):
            return candidate

    missing = sorted(set(text[start : max_start + val_characters]) - set(train_text))
    raise ValueError(
        "Could not find a validation slice whose characters are all present in the training slice. "
        f"Missing training vocabulary coverage for: {missing}"
    )


def _encode_window_dataset(
    text: str,
    *,
    max_offset: int,
    stoi: dict[str, int] | None = None,
) -> tuple[Tensor, Tensor, int]:
    required_characters = LONG_CONTEXT + max_offset + 1
    if len(text) < required_characters:
        raise ValueError(
            f"Text split must be at least {required_characters} characters long, got {len(text)}."
        )

    if stoi is None:
        vocab = sorted(set(text))
        stoi = {char: index for index, char in enumerate(vocab)}
    encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)

    inputs = []
    targets_multi = []
    for start in range(len(encoded) - LONG_CONTEXT - max_offset):
        stop = start + LONG_CONTEXT
        target_stop = stop + max_offset + 1
        inputs.append(encoded[start:stop])
        targets_multi.append(encoded[stop:target_stop])

    input_tensor = torch.stack(inputs)
    targets_multi_tensor = torch.stack(targets_multi)
    return input_tensor, targets_multi_tensor, len(stoi)


def load_window_dataset(*, max_offset: int) -> tuple[WindowDataset, WindowDataset]:
    raw_text = _resolve_text_file().read_text(encoding="utf-8")
    required_characters = TRAIN_CHARACTERS + VAL_CHARACTERS
    if len(raw_text) < required_characters:
        raise ValueError(
            f"Need at least {required_characters} characters, got {len(raw_text)}."
        )

    train_text = raw_text[:TRAIN_CHARACTERS]
    val_text = _choose_validation_text(
        raw_text,
        train_text=train_text,
        val_characters=VAL_CHARACTERS,
    )
    train_long_inputs, train_targets_multi, vocab_size = _encode_window_dataset(train_text, max_offset=max_offset)
    train_vocab = sorted(set(train_text))
    stoi = {char: index for index, char in enumerate(train_vocab)}
    missing_val_chars = sorted(set(val_text) - set(train_text))
    if missing_val_chars:
        raise ValueError(
            f"Validation text contains characters absent from training text: {missing_val_chars}"
        )
    val_long_inputs, val_targets_multi, _ = _encode_window_dataset(val_text, max_offset=max_offset, stoi=stoi)

    return (
        WindowDataset(
            short_inputs=train_long_inputs[:, -SHORT_CONTEXT:],
            mid_inputs=train_long_inputs[:, -MID_CONTEXT:],
            long_inputs=train_long_inputs,
            targets_multi=train_targets_multi,
            vocab_size=vocab_size,
        ),
        WindowDataset(
            short_inputs=val_long_inputs[:, -SHORT_CONTEXT:],
            mid_inputs=val_long_inputs[:, -MID_CONTEXT:],
            long_inputs=val_long_inputs,
            targets_multi=val_targets_multi,
            vocab_size=vocab_size,
        ),
    )


def sample_batch(
    dataset: WindowDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    indices = torch.randint(0, dataset.targets_multi.shape[0], (batch_size,), device=device)
    short_inputs = dataset.short_inputs[indices]
    mid_inputs = dataset.mid_inputs[indices]
    long_inputs = dataset.long_inputs[indices]
    targets_multi = dataset.targets_multi[indices]
    return short_inputs, mid_inputs, long_inputs, targets_multi


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
    def __init__(
        self,
        *,
        vocab_size: int,
        context_size: int,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        self.context_size = context_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim, max_context=context_size)
            for _ in range(n_layers)
        )

    def forward(self, inputs: Int[Tensor, "batch context"]) -> Float[Tensor, "batch context d_model"]:
        positions = torch.arange(self.context_size, device=inputs.device)
        hidden = self.token_embedding(inputs) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class OutputBlock(nn.Module):
    def __init__(self, *, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.encoder = SequenceEncoder(
            vocab_size=vocab_size,
            context_size=SHORT_CONTEXT,
            d_model=d_model,
            n_heads=2,
            ff_dim=128,
            n_layers=1,
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
    def __init__(
        self,
        *,
        vocab_size: int,
        context_size: int,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        n_layers: int,
        lateral_out_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = SequenceEncoder(
            vocab_size=vocab_size,
            context_size=context_size,
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            n_layers=n_layers,
        )
        self.local_head = nn.Linear(d_model, vocab_size)
        self.lateral_proj = nn.Linear(d_model, lateral_out_dim)

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

    def lateral(self, last_hidden: Float[Tensor, "batch 1 d_model"]) -> Float[Tensor, "batch 1 lateral_d_model"]:
        return LATERAL_SCALE * self.lateral_proj(last_hidden.detach())

    def pretrain_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters()) + list(self.local_head.parameters())

    def freeze_pretrained_representation(self) -> None:
        self.encoder.requires_grad_(False)
        self.local_head.requires_grad_(False)


class MultiBlockModel(nn.Module):
    def __init__(self, *, vocab_size: int, use_mid: bool, use_long: bool, long_block_config: LongBlockConfig) -> None:
        super().__init__()
        output_d_model = 64
        n_heads = 2
        ff_dim = 128
        self.output_block = OutputBlock(vocab_size=vocab_size, d_model=output_d_model)
        self.mid_block = (
            InteriorBlock(
                vocab_size=vocab_size,
                context_size=MID_CONTEXT,
                d_model=output_d_model,
                n_heads=n_heads,
                ff_dim=ff_dim,
                n_layers=1,
                lateral_out_dim=output_d_model,
            )
            if use_mid
            else None
        )
        self.long_block = (
            InteriorBlock(
                vocab_size=vocab_size,
                context_size=LONG_CONTEXT,
                d_model=long_block_config.d_model,
                n_heads=n_heads,
                ff_dim=ff_dim,
                n_layers=long_block_config.n_layers,
                lateral_out_dim=output_d_model,
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


def build_model(condition_name: str, vocab_size: int, long_block_config: LongBlockConfig) -> MultiBlockModel:
    match condition_name:
        case "block0_alone":
            return MultiBlockModel(vocab_size=vocab_size, use_mid=False, use_long=False, long_block_config=long_block_config)
        case "one_block_mid":
            return MultiBlockModel(vocab_size=vocab_size, use_mid=True, use_long=False, long_block_config=long_block_config)
        case "one_block_long":
            return MultiBlockModel(vocab_size=vocab_size, use_mid=False, use_long=True, long_block_config=long_block_config)
        case "two_blocks":
            return MultiBlockModel(vocab_size=vocab_size, use_mid=True, use_long=True, long_block_config=long_block_config)
        case _:
            raise ValueError(f"Unsupported condition: {condition_name}")


def make_optimizer(model: nn.Module, device: torch.device) -> torch.optim.Optimizer:
    fused = device.type == "cuda"
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(trainable_parameters) == 0:
        raise ValueError("Model has no trainable parameters.")
    return torch.optim.AdamW(trainable_parameters, lr=LEARNING_RATE, fused=fused)


def make_parameter_optimizer(parameters: list[nn.Parameter], device: torch.device) -> torch.optim.Optimizer:
    if len(parameters) == 0:
        raise ValueError("Need at least one trainable parameter.")
    fused = device.type == "cuda"
    return torch.optim.AdamW(parameters, lr=LEARNING_RATE, fused=fused)


def warm_up_cuda(
    train_dataset: WindowDataset,
    device: torch.device,
    long_block_config: LongBlockConfig,
    *,
    long_target_offset: int,
) -> None:
    if device.type != "cuda":
        return

    set_seed(DEFAULT_SEED)
    model = build_model("two_blocks", train_dataset.vocab_size, long_block_config).to(device)
    optimizer = make_optimizer(model, device)
    short_inputs, mid_inputs, long_inputs, targets_multi = sample_batch(
        train_dataset,
        batch_size=BATCH_SIZE,
        device=device,
    )
    next_targets = targets_multi[:, 0]
    long_targets = targets_multi[:, long_target_offset]

    model.train()
    with autocast_context(device):
        mid_last_hidden, mid_local_logits = model.mid_block.last_hidden_and_local_logits(mid_inputs)
        long_last_hidden, long_local_logits = model.long_block.last_hidden_and_local_logits(long_inputs)
        output_logits = model.output_logits(
            short_inputs,
            mid_last_hidden=mid_last_hidden,
            long_last_hidden=long_last_hidden,
        )
        output_loss, _ = next_char_metrics(output_logits, next_targets)
        mid_loss, _ = next_char_metrics(mid_local_logits, next_targets)
        long_loss, _ = next_char_metrics(long_local_logits, long_targets)
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


def pretrain_interior_block(
    block: InteriorBlock,
    *,
    block_name: str,
    train_inputs: Int[Tensor, "examples context"],
    local_targets: Int[Tensor, "examples"],
    steps: int,
    device: torch.device,
) -> None:
    if steps == 0:
        return

    optimizer = make_parameter_optimizer(block.pretrain_parameters(), device)
    block.train()
    for step in range(1, steps + 1):
        indices = torch.randint(0, train_inputs.shape[0], (BATCH_SIZE,), device=device)
        batch_inputs = train_inputs[indices]
        batch_targets = local_targets[indices]

        with autocast_context(device):
            _, local_logits = block.last_hidden_and_local_logits(batch_inputs)
            local_loss, local_accuracy = next_char_metrics(local_logits, batch_targets)

        optimizer.zero_grad(set_to_none=True)
        local_loss.backward()
        optimizer.step()

        if step % 500 == 0 or step == steps:
            print(
                f"[pretrain_{block_name}] step={step:04d}/{steps} "
                f"local_ce={local_loss.item():.4f} local_acc={local_accuracy.item():.4%}",
                flush=True,
            )


def maybe_pretrain_blocks(
    model: MultiBlockModel,
    train_dataset: WindowDataset,
    *,
    pretrain_config: PretrainConfig,
    long_target_offset: int,
    device: torch.device,
) -> None:
    next_targets = train_dataset.targets_multi[:, 0]
    if model.mid_block is not None and pretrain_config.mid_steps > 0:
        pretrain_interior_block(
            model.mid_block,
            block_name="mid",
            train_inputs=train_dataset.mid_inputs,
            local_targets=next_targets,
            steps=pretrain_config.mid_steps,
            device=device,
        )
        model.mid_block.freeze_pretrained_representation()

    if model.long_block is not None and pretrain_config.long_steps > 0:
        pretrain_interior_block(
            model.long_block,
            block_name="long",
            train_inputs=train_dataset.long_inputs,
            local_targets=train_dataset.targets_multi[:, long_target_offset],
            steps=pretrain_config.long_steps,
            device=device,
        )
        model.long_block.freeze_pretrained_representation()


def train_condition(
    condition_name: str,
    train_dataset: WindowDataset,
    val_dataset: WindowDataset,
    *,
    seed: int,
    device: torch.device,
    long_block_config: LongBlockConfig,
    long_target_offset: int,
    pretrain_config: PretrainConfig,
    training_steps: int,
) -> ConditionResult:
    set_seed(seed)
    model = build_model(condition_name, train_dataset.vocab_size, long_block_config).to(device)
    maybe_pretrain_blocks(
        model,
        train_dataset,
        pretrain_config=pretrain_config,
        long_target_offset=long_target_offset,
        device=device,
    )
    optimizer = make_optimizer(model, device)
    started_at = perf_counter()

    for step in range(1, training_steps + 1):
        model.train()
        short_inputs, mid_inputs, long_inputs, targets_multi = sample_batch(
            train_dataset,
            batch_size=BATCH_SIZE,
            device=device,
        )
        next_targets = targets_multi[:, 0]
        long_targets = targets_multi[:, long_target_offset]

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
                mid_loss, mid_accuracy = next_char_metrics(mid_local_logits, next_targets)
                total_loss_terms.append(mid_loss)
                mid_loss_value = mid_loss.item()
                mid_acc_value = mid_accuracy.item()

            if model.long_block is not None:
                long_last_hidden, long_local_logits = model.long_block.last_hidden_and_local_logits(long_inputs)
                long_loss, long_accuracy = next_char_metrics(long_local_logits, long_targets)
                total_loss_terms.append(long_loss)
                long_loss_value = long_loss.item()
                long_acc_value = long_accuracy.item()

            output_logits = model.output_logits(
                short_inputs,
                mid_last_hidden=mid_last_hidden,
                long_last_hidden=long_last_hidden,
            )
            output_loss, output_accuracy = next_char_metrics(output_logits, next_targets)
            total_loss_terms.insert(0, output_loss)
            total_loss = sum(total_loss_terms)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if step % PRINT_INTERVAL == 0 or step == training_steps:
            print(
                f"[{condition_name}] step={step:03d}/{training_steps} "
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
    total_examples = dataset.targets_multi.shape[0]

    for start in range(0, total_examples, batch_size):
        stop = min(start + batch_size, total_examples)
        short_inputs = dataset.short_inputs[start:stop]
        mid_inputs = dataset.mid_inputs[start:stop]
        long_inputs = dataset.long_inputs[start:stop]
        targets = dataset.targets_multi[start:stop, 0]

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
    long_block_config = resolve_long_block_config(args)
    long_target_offset = resolve_long_target_offset(args)
    pretrain_config = resolve_pretrain_config(args)
    set_seed(args.seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_dataset, val_dataset = load_window_dataset(max_offset=long_target_offset)
    train_dataset = dataset_to_device(train_dataset, device)
    val_dataset = dataset_to_device(val_dataset, device)
    warm_up_cuda(
        train_dataset,
        device,
        long_block_config,
        long_target_offset=long_target_offset,
    )
    print(
        "multi block lm "
        f"device={device.type} seed={args.seed} train_examples={train_dataset.targets_multi.shape[0]} "
        f"val_examples={val_dataset.targets_multi.shape[0]} vocab={train_dataset.vocab_size} "
        f"short_ctx={SHORT_CONTEXT} mid_ctx={MID_CONTEXT} long_ctx={LONG_CONTEXT} "
        f"long_d_model={long_block_config.d_model} long_n_layers={long_block_config.n_layers} "
        f"long_target_offset={long_target_offset} "
        f"pretrain_mid_steps={pretrain_config.mid_steps} pretrain_long_steps={pretrain_config.long_steps} "
        f"steps={args.steps} batch={BATCH_SIZE} lateral_scale={LATERAL_SCALE} "
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
            long_block_config=long_block_config,
            long_target_offset=long_target_offset,
            pretrain_config=pretrain_config,
            training_steps=args.steps,
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
