from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tied_readout_lm import (
    DEFAULT_SEED,
    D_MODEL,
    FF_DIM,
    LEARNING_RATE,
    N_HEADS,
    N_LAYERS,
    PRINT_INTERVAL,
    SHORT_CONTEXT,
    TRAIN_CHARACTERS,
    VAL_CHARACTERS,
    Corpus,
    ModelSize,
    SequenceEncoder,
    autocast_context,
    choose_validation_text,
    load_corpus_text,
    normalize_hidden,
    positive_int,
    resolve_device,
    set_seed,
    tied_logits,
)

DEFAULT_SLOW_CONTEXT = 32
TRAINING_STEPS = 4_800
SANITY_CHECK_STEPS = 100
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
SEQ_LENGTH = 128
TEMPERATURE = 0.07
LATERAL_SCALE = 0.2
ALL_CONDITIONS = (
    "block0_alone",
    "direct_ctx32_refresh1",
    "buffered_ctx32_refresh8",
)


@dataclass(frozen=True)
class SequentialDataset:
    sequences: Tensor
    vocab_size: int


@dataclass(frozen=True)
class AgeBucketResult:
    age: int
    val_loss: float
    count: int


@dataclass(frozen=True)
class EvalResult:
    val_loss: float
    val_accuracy: float
    ablated_val_loss: float
    ablation_delta: float
    age_buckets: tuple[AgeBucketResult, ...]


@dataclass(frozen=True)
class ConditionResult:
    name: str
    val_loss: float
    val_accuracy: float
    ablated_val_loss: float
    ablation_delta: float
    delta_from_baseline: float
    wall_seconds: float
    age_buckets: tuple[AgeBucketResult, ...]


@dataclass(frozen=True)
class TrainedCondition:
    result: ConditionResult
    model: "MultiRateBufferModel"


@dataclass(frozen=True)
class TrainMetrics:
    total_loss: Tensor
    mean_block0_loss: float
    mean_block0_accuracy: float
    mean_slow_loss: float
    fired_positions: int


@dataclass(frozen=True)
class RefreshSchedule:
    output_position_ends: Tensor
    eval_mask: Tensor
    fire_output_indices: Tensor
    fire_long_window_indices: Tensor
    valid_output_indices: Tensor
    valid_to_fire_slot: Tensor
    ages: Tensor


@dataclass(frozen=True)
class ExperimentConfig:
    slow_context: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--conditions", type=str, default=",".join(ALL_CONDITIONS))
    parser.add_argument("--steps", type=positive_int, default=TRAINING_STEPS)
    parser.add_argument("--sanity-check-only", action="store_true")
    parser.add_argument("--batch-size", type=positive_int, default=BATCH_SIZE)
    parser.add_argument("--seq-length", type=positive_int, default=SEQ_LENGTH)
    parser.add_argument("--eval-batch-size", type=positive_int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--lateral-scale", type=float, default=LATERAL_SCALE)
    parser.add_argument("--slow-context", type=positive_int, default=DEFAULT_SLOW_CONTEXT)
    parser.add_argument("--d-model", type=positive_int, default=D_MODEL)
    parser.add_argument("--n-heads", type=positive_int, default=N_HEADS)
    parser.add_argument("--ff-dim", type=positive_int, default=FF_DIM)
    parser.add_argument("--n-layers", type=positive_int, default=N_LAYERS)
    return parser.parse_args()


def resolve_training_steps(args: argparse.Namespace) -> int:
    return min(args.steps, SANITY_CHECK_STEPS) if args.sanity_check_only else args.steps


def resolve_model_size(args: argparse.Namespace) -> ModelSize:
    return ModelSize(
        d_model=args.d_model,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        n_layers=args.n_layers,
    )


def resolve_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(slow_context=args.slow_context)


def parse_condition_names(raw_conditions: str) -> tuple[str, ...]:
    requested_conditions = tuple(condition.strip() for condition in raw_conditions.split(",") if condition.strip())
    if len(requested_conditions) == 0:
        raise ValueError("conditions must contain at least one condition name")

    invalid_conditions = [condition for condition in requested_conditions if condition not in ALL_CONDITIONS]
    if len(invalid_conditions) > 0:
        raise ValueError(
            "Unknown conditions requested: "
            + ", ".join(invalid_conditions)
            + ". Valid conditions are: "
            + ", ".join(ALL_CONDITIONS)
        )
    return requested_conditions


def encode_text(text: str, *, stoi: dict[str, int] | None = None) -> tuple[Tensor, dict[str, int], tuple[str, ...]]:
    if stoi is None:
        vocab = sorted(set(text))
        stoi = {char: index for index, char in enumerate(vocab)}
        itos = tuple(vocab)
    else:
        missing_characters = sorted(set(text) - set(stoi))
        if len(missing_characters) > 0:
            raise ValueError(f"Text contains characters absent from training vocabulary: {missing_characters}")
        sorted_pairs = sorted(stoi.items(), key=lambda item: item[1])
        itos = tuple(char for char, _ in sorted_pairs)
    encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    return encoded, stoi, itos


def split_into_sequences(encoded_text: Tensor, *, seq_length: int) -> Tensor:
    minimum_seq_length = SHORT_CONTEXT + 1
    if seq_length <= minimum_seq_length:
        raise ValueError(f"seq_length must be greater than {minimum_seq_length}, got {seq_length}")
    usable_characters = (encoded_text.shape[0] // seq_length) * seq_length
    if usable_characters < seq_length:
        raise ValueError(f"Need at least {seq_length} encoded characters, got {encoded_text.shape[0]}")
    return encoded_text[:usable_characters].view(-1, seq_length)


def load_sequential_dataset(seq_length: int) -> tuple[SequentialDataset, SequentialDataset, Corpus]:
    raw_text = load_corpus_text()
    required_characters = TRAIN_CHARACTERS + VAL_CHARACTERS
    if len(raw_text) < required_characters:
        raise ValueError(f"Need at least {required_characters} characters, got {len(raw_text)}.")

    train_text = raw_text[:TRAIN_CHARACTERS]
    val_text = choose_validation_text(raw_text, train_text=train_text, val_characters=VAL_CHARACTERS)
    train_encoded, stoi, itos = encode_text(train_text)
    val_encoded, _, _ = encode_text(val_text, stoi=stoi)

    return (
        SequentialDataset(sequences=split_into_sequences(train_encoded, seq_length=seq_length), vocab_size=len(stoi)),
        SequentialDataset(sequences=split_into_sequences(val_encoded, seq_length=seq_length), vocab_size=len(stoi)),
        Corpus(train_text=train_text, val_text=val_text, stoi=stoi, itos=itos),
    )


def dataset_to_device(dataset: SequentialDataset, device: torch.device) -> SequentialDataset:
    return SequentialDataset(sequences=dataset.sequences.to(device), vocab_size=dataset.vocab_size)


def summed_cross_entropy(logits: Tensor, targets: Tensor, *, batch_size: int) -> Tensor:
    return F.cross_entropy(logits, targets, reduction="sum") / batch_size


def mean_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(logits, targets)


def extract_output_windows_and_targets(sequence_batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    output_windows = sequence_batch.unfold(1, SHORT_CONTEXT, 1)[:, :-1, :].contiguous()
    targets = sequence_batch[:, SHORT_CONTEXT:].contiguous()
    output_position_ends = torch.arange(SHORT_CONTEXT - 1, sequence_batch.shape[1] - 1, device=sequence_batch.device)
    return output_windows, targets, output_position_ends


def extract_slow_windows(sequence_batch: Tensor, *, slow_context: int) -> Tensor:
    return sequence_batch.unfold(1, slow_context, 1)[:, :-1, :].contiguous()


def refresh_interval_for_condition(condition_name: str) -> int | None:
    match condition_name:
        case "block0_alone":
            return None
        case "direct_ctx32_refresh1":
            return 1
        case "buffered_ctx32_refresh8":
            return 8
        case _:
            raise ValueError(f"Unsupported condition: {condition_name}")


def build_refresh_schedule(
    output_position_ends: Tensor,
    *,
    slow_context: int,
    refresh_interval: int | None,
) -> RefreshSchedule:
    eval_mask = output_position_ends >= (slow_context - 1)
    if refresh_interval is None:
        empty = torch.empty(0, dtype=torch.long, device=output_position_ends.device)
        return RefreshSchedule(
            output_position_ends=output_position_ends,
            eval_mask=eval_mask,
            fire_output_indices=empty,
            fire_long_window_indices=empty,
            valid_output_indices=empty,
            valid_to_fire_slot=empty,
            ages=empty,
        )

    valid_output_indices = torch.nonzero(eval_mask, as_tuple=False).squeeze(-1)
    valid_position_ends = output_position_ends[valid_output_indices]
    relative_positions = valid_position_ends - (slow_context - 1)
    fire_mask = relative_positions.remainder(refresh_interval) == 0
    fire_output_indices = valid_output_indices[fire_mask]
    fire_position_ends = output_position_ends[fire_output_indices]
    fire_long_window_indices = fire_position_ends - (slow_context - 1)
    valid_to_fire_slot = torch.div(relative_positions, refresh_interval, rounding_mode="floor")
    ages = relative_positions.remainder(refresh_interval)
    return RefreshSchedule(
        output_position_ends=output_position_ends,
        eval_mask=eval_mask,
        fire_output_indices=fire_output_indices,
        fire_long_window_indices=fire_long_window_indices,
        valid_output_indices=valid_output_indices,
        valid_to_fire_slot=valid_to_fire_slot,
        ages=ages,
    )


class MultiRateBufferModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        model_size: ModelSize,
        slow_context: int,
        use_slow_block: bool,
        temperature: float,
        lateral_scale: float,
    ) -> None:
        super().__init__()
        self.model_size = model_size
        self.slow_context = slow_context
        self.temperature = temperature
        self.lateral_scale = lateral_scale
        self.token_embedding = nn.Embedding(vocab_size, model_size.d_model)
        self.output_block = SequenceEncoder(
            context_size=SHORT_CONTEXT,
            d_model=model_size.d_model,
            n_heads=model_size.n_heads,
            ff_dim=model_size.ff_dim,
            n_layers=model_size.n_layers,
        )
        self.slow_block = (
            SequenceEncoder(
                context_size=slow_context,
                d_model=model_size.d_model,
                n_heads=model_size.n_heads,
                ff_dim=model_size.ff_dim,
                n_layers=model_size.n_layers,
            )
            if use_slow_block
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.model_size.d_model**-0.5)

    def output_hidden(self, inputs: Tensor) -> Tensor:
        hidden = self.output_block(inputs, self.token_embedding)[:, -1, :]
        return normalize_hidden(hidden)

    def slow_hidden(self, inputs: Tensor) -> Tensor:
        if self.slow_block is None:
            raise ValueError("slow_block is not enabled for this condition")
        hidden = self.slow_block(inputs, self.token_embedding)[:, -1, :]
        return normalize_hidden(hidden)

    def output_logits(self, output_hidden: Tensor, slow_lateral: Tensor | None = None) -> Tensor:
        combined_hidden = output_hidden
        if slow_lateral is not None:
            combined_hidden = combined_hidden + self.lateral_scale * slow_lateral.detach()
        return tied_logits(
            combined_hidden,
            self.token_embedding,
            temperature=self.temperature,
            normalize=True,
        )


def build_model(
    condition_name: str,
    *,
    vocab_size: int,
    model_size: ModelSize,
    slow_context: int,
    temperature: float,
    lateral_scale: float,
) -> MultiRateBufferModel:
    return MultiRateBufferModel(
        vocab_size=vocab_size,
        model_size=model_size,
        slow_context=slow_context,
        use_slow_block=condition_name != "block0_alone",
        temperature=temperature,
        lateral_scale=lateral_scale,
    )


def make_optimizer(model: nn.Module, device: torch.device) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, fused=device.type == "cuda")


def next_sequence_batch(sequence_order: Tensor, *, cursor: int, batch_size: int) -> tuple[Tensor, int]:
    num_sequences = sequence_order.shape[0]
    if batch_size > num_sequences:
        raise ValueError(f"batch_size {batch_size} exceeds number of sequences {num_sequences}")
    if cursor + batch_size <= num_sequences:
        return sequence_order[cursor : cursor + batch_size], cursor + batch_size

    remaining = sequence_order[cursor:]
    reshuffled = torch.randperm(num_sequences, device=sequence_order.device)
    needed = batch_size - remaining.shape[0]
    return torch.cat((remaining, reshuffled[:needed])), needed


def build_slow_lateral(
    model: MultiRateBufferModel,
    slow_windows: Tensor,
    targets: Tensor,
    schedule: RefreshSchedule,
) -> tuple[Tensor | None, Tensor | None, int]:
    if model.slow_block is None:
        return None, None, 0

    batch_size = slow_windows.shape[0]
    num_positions = schedule.output_position_ends.shape[0]
    device = slow_windows.device
    dtype = model.token_embedding.weight.dtype
    lateral = torch.zeros((batch_size, num_positions, model.model_size.d_model), device=device, dtype=dtype)

    if schedule.fire_long_window_indices.numel() == 0:
        return lateral.view(batch_size * num_positions, -1), None, 0

    fire_windows = slow_windows[:, schedule.fire_long_window_indices, :]
    flat_fire_windows = fire_windows.reshape(-1, model.slow_context)
    fire_hidden = model.slow_hidden(flat_fire_windows)
    fire_targets = targets[:, schedule.fire_output_indices].reshape(-1)
    fire_logits = tied_logits(fire_hidden, model.token_embedding, temperature=model.temperature, normalize=True)
    slow_loss = summed_cross_entropy(fire_logits, fire_targets, batch_size=batch_size)

    fire_hidden_by_position = fire_hidden.view(batch_size, schedule.fire_output_indices.shape[0], -1)
    held_hidden = fire_hidden_by_position[:, schedule.valid_to_fire_slot, :]
    lateral[:, schedule.valid_output_indices, :] = held_hidden
    return lateral.view(batch_size * num_positions, -1), slow_loss, schedule.fire_output_indices.shape[0]


def run_sequence_batch_train(
    model: MultiRateBufferModel,
    sequence_batch: Tensor,
    *,
    slow_context: int,
    refresh_interval: int | None,
) -> TrainMetrics:
    batch_size = sequence_batch.shape[0]
    output_windows, targets, output_position_ends = extract_output_windows_and_targets(sequence_batch)
    schedule = build_refresh_schedule(output_position_ends, slow_context=slow_context, refresh_interval=refresh_interval)
    slow_windows = extract_slow_windows(sequence_batch, slow_context=slow_context)
    num_positions = output_windows.shape[1]
    flat_output_windows = output_windows.reshape(batch_size * num_positions, SHORT_CONTEXT)
    flat_targets = targets.reshape(batch_size * num_positions)

    with autocast_context(sequence_batch.device):
        output_hidden = model.output_hidden(flat_output_windows)
        slow_lateral, slow_loss, fired_positions = build_slow_lateral(model, slow_windows, targets, schedule)
        logits = model.output_logits(output_hidden, slow_lateral)
        block0_loss = summed_cross_entropy(logits, flat_targets, batch_size=batch_size)
        block0_accuracy = (logits.argmax(dim=-1) == flat_targets).float().mean()
        total_loss = block0_loss if slow_loss is None else block0_loss + slow_loss

    return TrainMetrics(
        total_loss=total_loss,
        mean_block0_loss=block0_loss.item() / num_positions,
        mean_block0_accuracy=block0_accuracy.item(),
        mean_slow_loss=0.0 if slow_loss is None else slow_loss.item() / max(fired_positions, 1),
        fired_positions=fired_positions,
    )


@torch.no_grad()
def evaluate_condition(
    model: MultiRateBufferModel,
    dataset: SequentialDataset,
    *,
    batch_size: int,
    slow_context: int,
    refresh_interval: int | None,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_ablated_loss = 0.0
    bucket_loss_sums = [0.0 for _ in range(8)]
    bucket_counts = [0 for _ in range(8)]

    for batch_start in range(0, dataset.sequences.shape[0], batch_size):
        sequence_batch = dataset.sequences[batch_start : batch_start + batch_size]
        output_windows, targets, output_position_ends = extract_output_windows_and_targets(sequence_batch)
        schedule = build_refresh_schedule(output_position_ends, slow_context=slow_context, refresh_interval=refresh_interval)
        eval_mask = schedule.eval_mask
        slow_windows = extract_slow_windows(sequence_batch, slow_context=slow_context)

        batch_sequences = output_windows.shape[0]
        num_positions = output_windows.shape[1]
        flat_output_windows = output_windows.reshape(batch_sequences * num_positions, SHORT_CONTEXT)
        flat_targets = targets.reshape(batch_sequences * num_positions)

        with autocast_context(sequence_batch.device):
            output_hidden = model.output_hidden(flat_output_windows)
            slow_lateral, _, _ = build_slow_lateral(model, slow_windows, targets, schedule)
            logits = model.output_logits(output_hidden, slow_lateral)
            ablated_logits = model.output_logits(output_hidden, None)

        losses = F.cross_entropy(logits, flat_targets, reduction="none").view(batch_sequences, num_positions)
        ablated_losses = F.cross_entropy(ablated_logits, flat_targets, reduction="none").view(batch_sequences, num_positions)
        correct = (logits.argmax(dim=-1) == flat_targets).view(batch_sequences, num_positions)

        eval_losses = losses[:, eval_mask]
        eval_ablated_losses = ablated_losses[:, eval_mask]
        eval_correct = correct[:, eval_mask]

        total_loss += eval_losses.sum().item()
        total_ablated_loss += eval_ablated_losses.sum().item()
        total_correct += eval_correct.sum().item()
        total_examples += eval_losses.numel()

        if refresh_interval == 8:
            for age in range(8):
                age_indices = schedule.valid_output_indices[schedule.ages == age]
                if age_indices.numel() == 0:
                    continue
                age_losses = losses[:, age_indices]
                bucket_loss_sums[age] += age_losses.sum().item()
                bucket_counts[age] += age_losses.numel()

    if total_examples == 0:
        raise ValueError("No evaluation positions available. Increase --seq-length.")

    age_buckets = tuple(
        AgeBucketResult(
            age=age,
            val_loss=(bucket_loss_sums[age] / bucket_counts[age]) if bucket_counts[age] > 0 else float("nan"),
            count=bucket_counts[age],
        )
        for age in range(8)
        if refresh_interval == 8
    )
    val_loss = total_loss / total_examples
    ablated_val_loss = total_ablated_loss / total_examples
    return EvalResult(
        val_loss=val_loss,
        val_accuracy=total_correct / total_examples,
        ablated_val_loss=ablated_val_loss,
        ablation_delta=ablated_val_loss - val_loss,
        age_buckets=age_buckets,
    )


def warm_up_cuda(
    train_dataset: SequentialDataset,
    *,
    device: torch.device,
    model_size: ModelSize,
    slow_context: int,
    temperature: float,
    lateral_scale: float,
    batch_size: int,
) -> None:
    if device.type != "cuda":
        return

    set_seed(DEFAULT_SEED)
    model = build_model(
        "buffered_ctx32_refresh8",
        vocab_size=train_dataset.vocab_size,
        model_size=model_size,
        slow_context=slow_context,
        temperature=temperature,
        lateral_scale=lateral_scale,
    ).to(device)
    optimizer = make_optimizer(model, device)
    sequence_batch = train_dataset.sequences[:batch_size]

    model.train()
    metrics = run_sequence_batch_train(model, sequence_batch, slow_context=slow_context, refresh_interval=8)
    optimizer.zero_grad(set_to_none=True)
    metrics.total_loss.backward()
    optimizer.step()

    model.eval()
    evaluate_condition(
        model,
        SequentialDataset(sequences=sequence_batch, vocab_size=train_dataset.vocab_size),
        batch_size=batch_size,
        slow_context=slow_context,
        refresh_interval=8,
    )
    torch.cuda.synchronize()


def train_condition(
    condition_name: str,
    train_dataset: SequentialDataset,
    val_dataset: SequentialDataset,
    *,
    seed: int,
    device: torch.device,
    model_size: ModelSize,
    slow_context: int,
    temperature: float,
    lateral_scale: float,
    training_steps: int,
    batch_size: int,
    eval_batch_size: int,
) -> TrainedCondition:
    set_seed(seed)
    model = build_model(
        condition_name,
        vocab_size=train_dataset.vocab_size,
        model_size=model_size,
        slow_context=slow_context,
        temperature=temperature,
        lateral_scale=lateral_scale,
    ).to(device)
    optimizer = make_optimizer(model, device)
    refresh_interval = refresh_interval_for_condition(condition_name)
    started_at = perf_counter()

    sequence_order = torch.randperm(train_dataset.sequences.shape[0], device=device)
    cursor = 0

    for step in range(1, training_steps + 1):
        model.train()
        batch_indices, cursor = next_sequence_batch(sequence_order, cursor=cursor, batch_size=batch_size)
        if cursor == train_dataset.sequences.shape[0]:
            sequence_order = torch.randperm(train_dataset.sequences.shape[0], device=device)
            cursor = 0
        elif cursor < batch_size:
            sequence_order = torch.randperm(train_dataset.sequences.shape[0], device=device)

        sequence_batch = train_dataset.sequences[batch_indices]
        metrics = run_sequence_batch_train(
            model,
            sequence_batch,
            slow_context=slow_context,
            refresh_interval=refresh_interval,
        )
        optimizer.zero_grad(set_to_none=True)
        metrics.total_loss.backward()
        optimizer.step()

        if step == 1 or step % PRINT_INTERVAL == 0 or step == training_steps:
            print(
                f"[{condition_name}] step={step:04d}/{training_steps} "
                f"block0_ce={metrics.mean_block0_loss:.4f} block0_acc={metrics.mean_block0_accuracy:.4%} "
                f"slow_ce={metrics.mean_slow_loss:.4f} fired_positions={metrics.fired_positions}",
                flush=True,
            )

    evaluation = evaluate_condition(
        model,
        val_dataset,
        batch_size=eval_batch_size,
        slow_context=slow_context,
        refresh_interval=refresh_interval,
    )
    wall_seconds = perf_counter() - started_at
    print(
        f"[{condition_name}] final val_loss={evaluation.val_loss:.4f} "
        f"val_accuracy={evaluation.val_accuracy:.4%} "
        f"ablated_val_loss={evaluation.ablated_val_loss:.4f} "
        f"ablation_delta={evaluation.ablation_delta:.4f} "
        f"wall_seconds={wall_seconds:.2f}",
        flush=True,
    )
    return TrainedCondition(
        result=ConditionResult(
            name=condition_name,
            val_loss=evaluation.val_loss,
            val_accuracy=evaluation.val_accuracy,
            ablated_val_loss=evaluation.ablated_val_loss,
            ablation_delta=evaluation.ablation_delta,
            delta_from_baseline=0.0,
            wall_seconds=wall_seconds,
            age_buckets=evaluation.age_buckets,
        ),
        model=model,
    )


def print_results_table(results: list[ConditionResult]) -> None:
    header = (
        "condition".ljust(26)
        + "val_loss".rjust(12)
        + "val_acc".rjust(12)
        + "ablated".rjust(12)
        + "ablate_d".rjust(12)
        + "delta".rjust(12)
        + "wall_s".rjust(12)
    )
    print("\nresults", flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for result in results:
        print(
            result.name.ljust(26)
            + f"{result.val_loss:12.4f}"
            + f"{result.val_accuracy:12.4%}"
            + f"{result.ablated_val_loss:12.4f}"
            + f"{result.ablation_delta:12.4f}"
            + f"{result.delta_from_baseline:12.4f}"
            + f"{result.wall_seconds:12.2f}",
            flush=True,
        )


def print_age_buckets(results: list[ConditionResult]) -> None:
    for result in results:
        if len(result.age_buckets) == 0:
            continue
        print(f"\n[{result.name}] age_bucket_val_loss", flush=True)
        print("age".ljust(8) + "val_loss".rjust(12) + "count".rjust(12), flush=True)
        print("-" * 32, flush=True)
        for bucket in result.age_buckets:
            print(
                str(bucket.age).ljust(8) + f"{bucket.val_loss:12.4f}" + f"{bucket.count:12d}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    if args.temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {args.temperature}")
    if args.seq_length <= args.slow_context + 1:
        raise ValueError(f"seq_length must be greater than slow_context + 1 ({args.slow_context + 1}), got {args.seq_length}")

    requested_conditions = parse_condition_names(args.conditions)
    training_steps = resolve_training_steps(args)
    model_size = resolve_model_size(args)
    config = resolve_experiment_config(args)
    device = resolve_device(args.device)
    set_seed(args.seed)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_dataset, val_dataset, _ = load_sequential_dataset(args.seq_length)
    train_dataset = dataset_to_device(train_dataset, device)
    val_dataset = dataset_to_device(val_dataset, device)

    warm_up_cuda(
        train_dataset,
        device=device,
        model_size=model_size,
        slow_context=config.slow_context,
        temperature=args.temperature,
        lateral_scale=args.lateral_scale,
        batch_size=min(args.batch_size, train_dataset.sequences.shape[0]),
    )

    print(
        "multirate buffer lm "
        f"device={device.type} seed={args.seed} train_sequences={train_dataset.sequences.shape[0]} "
        f"val_sequences={val_dataset.sequences.shape[0]} vocab={train_dataset.vocab_size} "
        f"short_ctx={SHORT_CONTEXT} slow_ctx={config.slow_context} seq_length={args.seq_length} "
        f"d_model={model_size.d_model} n_heads={model_size.n_heads} ff_dim={model_size.ff_dim} n_layers={model_size.n_layers} "
        f"steps={training_steps} requested_steps={args.steps} sanity_check_only={args.sanity_check_only} "
        f"batch={args.batch_size} eval_batch={args.eval_batch_size} "
        f"temperature={args.temperature:.4f} lateral_scale={args.lateral_scale:.4f} "
        f"refresh_alignment=first_fire_at_end_index_{config.slow_context - 1} "
        f"conditions={','.join(requested_conditions)}",
        flush=True,
    )

    trained_conditions = [
        train_condition(
            condition_name,
            train_dataset,
            val_dataset,
            seed=args.seed,
            device=device,
            model_size=model_size,
            slow_context=config.slow_context,
            temperature=args.temperature,
            lateral_scale=args.lateral_scale,
            training_steps=training_steps,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
        )
        for condition_name in requested_conditions
    ]

    results = [trained_condition.result for trained_condition in trained_conditions]
    baseline_loss = next((result.val_loss for result in results if result.name == "block0_alone"), None)
    adjusted_results = [
        ConditionResult(
            name=result.name,
            val_loss=result.val_loss,
            val_accuracy=result.val_accuracy,
            ablated_val_loss=result.ablated_val_loss,
            ablation_delta=result.ablation_delta,
            delta_from_baseline=(result.val_loss - baseline_loss) if baseline_loss is not None else float("nan"),
            wall_seconds=result.wall_seconds,
            age_buckets=result.age_buckets,
        )
        for result in results
    ]
    print_results_table(adjusted_results)
    print_age_buckets(adjusted_results)


if __name__ == "__main__":
    main()
