from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tied_readout_lm import (
    D_MODEL,
    DEFAULT_SEED,
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
    interior_local_loss,
    load_corpus_text,
    normalize_hidden,
    positive_int,
    resolve_device,
    set_seed,
    tied_logits,
)

TRAINING_STEPS = 2_000
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 256
SEQ_LENGTH = 128
BURN_IN = 32
TEMPERATURE = 1.0
ALL_CONDITIONS = ("block0_alone", "no_persistence", "recurrent_lateral", "reset_32", "reset_128")


@dataclass(frozen=True)
class SequentialDataset:
    sequences: Tensor
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
class TrainedCondition:
    result: ConditionResult
    model: "RecurrentLateralModel"


@dataclass(frozen=True)
class TrainMetrics:
    total_loss: Tensor
    mean_block0_loss: float
    mean_block0_accuracy: float
    mean_recurrent_loss: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--conditions", type=str, default=",".join(ALL_CONDITIONS))
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--batch-size", type=positive_int, default=BATCH_SIZE)
    parser.add_argument("--seq-length", type=positive_int, default=SEQ_LENGTH)
    parser.add_argument("--eval-batch-size", type=positive_int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--burn-in", type=int, default=BURN_IN)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--lateral-scale", type=float, default=1.0)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-loss", choices=("ce", "cosine", "l2"), default="ce")
    parser.add_argument("--d-model", type=positive_int, default=D_MODEL)
    parser.add_argument("--n-heads", type=positive_int, default=N_HEADS)
    parser.add_argument("--ff-dim", type=positive_int, default=FF_DIM)
    parser.add_argument("--n-layers", type=positive_int, default=N_LAYERS)
    return parser.parse_args()


def resolve_model_size(args: argparse.Namespace) -> ModelSize:
    return ModelSize(
        d_model=args.d_model,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        n_layers=args.n_layers,
    )


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
    if seq_length <= SHORT_CONTEXT:
        raise ValueError(f"seq_length must be greater than {SHORT_CONTEXT}, got {seq_length}")
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


def next_char_metrics(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    loss = F.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=-1) == targets).float().mean()
    return loss, accuracy


def summed_cross_entropy(logits: Tensor, targets: Tensor, *, batch_size: int) -> Tensor:
    return F.cross_entropy(logits, targets, reduction="sum") / batch_size


def aggregated_interior_local_loss(
    hidden: Tensor,
    *,
    embedding: nn.Embedding,
    targets: Tensor,
    temperature: float,
    normalize: bool,
    local_loss: str,
    batch_size: int,
    num_positions: int,
) -> Tensor:
    if local_loss == "ce":
        logits = tied_logits(hidden, embedding, temperature=temperature, normalize=normalize)
        return F.cross_entropy(logits, targets, reduction="sum") / batch_size

    target_embeddings = F.normalize(embedding.weight[targets].float(), dim=-1).to(embedding.weight.dtype)
    if local_loss == "cosine":
        cosine_terms = F.cosine_similarity(hidden.float(), target_embeddings.float(), dim=-1)
        return hidden.new_tensor(float(num_positions)) - cosine_terms.sum().to(hidden.dtype) / batch_size
    if local_loss == "l2":
        squared_error = (hidden.float() - target_embeddings.float()).pow(2).sum()
        return squared_error.to(hidden.dtype) / (batch_size * hidden.shape[-1])
    raise ValueError(f"Unsupported local_loss: {local_loss}")


def sequence_encoder_hidden(
    encoder: SequenceEncoder,
    token_embedding: nn.Embedding,
    inputs: Tensor,
    *,
    additive_bias: Tensor | None = None,
) -> Tensor:
    if inputs.shape[1] != encoder.context_size:
        raise ValueError(f"Expected context {encoder.context_size}, got {inputs.shape[1]}")
    positions = torch.arange(encoder.context_size, device=inputs.device)
    hidden = token_embedding(inputs) + encoder.position_embedding(positions)
    if additive_bias is not None:
        hidden = hidden + additive_bias.to(hidden.dtype).unsqueeze(1)
    for block in encoder.blocks:
        hidden = block(hidden)
    return hidden


def reset_interval_for_condition(condition_name: str) -> int | None:
    match condition_name:
        case "block0_alone":
            return None
        case "no_persistence":
            return 1
        case "recurrent_lateral":
            return None
        case "reset_32":
            return 32
        case "reset_128":
            return 128
        case _:
            raise ValueError(f"Unsupported condition: {condition_name}")


def uses_recurrent_block(condition_name: str) -> bool:
    return condition_name != "block0_alone"


class RecurrentLateralModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        model_size: ModelSize,
        use_recurrent_block: bool,
        temperature: float,
        lateral_scale: float,
        normalize: bool,
        local_loss: str,
    ) -> None:
        super().__init__()
        self.model_size = model_size
        self.temperature = temperature
        self.lateral_scale = lateral_scale
        self.normalize = normalize
        self.local_loss = local_loss
        self.token_embedding = nn.Embedding(vocab_size, model_size.d_model)
        self.output_block = SequenceEncoder(
            context_size=SHORT_CONTEXT,
            d_model=model_size.d_model,
            n_heads=model_size.n_heads,
            ff_dim=model_size.ff_dim,
            n_layers=model_size.n_layers,
        )
        self.recurrent_block = (
            SequenceEncoder(
                context_size=SHORT_CONTEXT,
                d_model=model_size.d_model,
                n_heads=model_size.n_heads,
                ff_dim=model_size.ff_dim,
                n_layers=model_size.n_layers,
            )
            if use_recurrent_block
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.model_size.d_model**-0.5)

    def output_hidden(self, inputs: Tensor) -> Tensor:
        hidden = sequence_encoder_hidden(self.output_block, self.token_embedding, inputs)[:, -1, :]
        if self.normalize:
            return normalize_hidden(hidden)
        return hidden

    def recurrent_raw_hidden(self, inputs: Tensor, state_bias: Tensor) -> Tensor:
        if self.recurrent_block is None:
            raise ValueError("recurrent_block is not enabled for this model")
        return sequence_encoder_hidden(
            self.recurrent_block,
            self.token_embedding,
            inputs,
            additive_bias=state_bias,
        )[:, -1, :]

    def recurrent_hidden(self, inputs: Tensor, state_bias: Tensor) -> tuple[Tensor, Tensor]:
        raw_hidden = self.recurrent_raw_hidden(inputs, state_bias)
        recurrent_hidden = normalize_hidden(raw_hidden) if self.normalize else raw_hidden
        next_state = normalize_hidden(raw_hidden).detach()
        return recurrent_hidden, next_state

    def output_logits(self, output_hidden: Tensor, recurrent_hidden: Tensor | None = None) -> Tensor:
        combined_hidden = output_hidden
        if recurrent_hidden is not None:
            combined_hidden = combined_hidden + self.lateral_scale * recurrent_hidden.detach()
        return tied_logits(
            combined_hidden,
            self.token_embedding,
            temperature=self.temperature,
            normalize=self.normalize,
        )


def build_model(
    condition_name: str,
    *,
    vocab_size: int,
    model_size: ModelSize,
    temperature: float,
    lateral_scale: float,
    normalize: bool,
    local_loss: str,
) -> RecurrentLateralModel:
    return RecurrentLateralModel(
        vocab_size=vocab_size,
        model_size=model_size,
        use_recurrent_block=uses_recurrent_block(condition_name),
        temperature=temperature,
        lateral_scale=lateral_scale,
        normalize=normalize,
        local_loss=local_loss,
    )


def make_optimizer(model: nn.Module, device: torch.device) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, fused=device.type == "cuda")


def zero_state(*, batch_size: int, d_model: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.zeros((batch_size, d_model), device=device, dtype=dtype)


def should_reset_state(position_index: int, reset_interval: int | None) -> bool:
    return reset_interval is not None and position_index > 0 and position_index % reset_interval == 0


def iter_positions(seq_length: int) -> range:
    return range(SHORT_CONTEXT - 1, seq_length - 1)


def extract_windows_and_targets(sequence_batch: Tensor) -> tuple[Tensor, Tensor]:
    windows = sequence_batch.unfold(1, SHORT_CONTEXT, 1)[:, :-1, :].contiguous()
    targets = sequence_batch[:, SHORT_CONTEXT:].contiguous()
    return windows, targets


@torch.no_grad()
def precompute_state_biases(
    model: RecurrentLateralModel,
    windows: Tensor,
    *,
    reset_interval: int | None,
    device: torch.device,
) -> Tensor | None:
    if model.recurrent_block is None:
        return None

    batch_size, num_positions, _ = windows.shape
    state = zero_state(
        batch_size=batch_size,
        d_model=model.model_size.d_model,
        device=device,
        dtype=model.token_embedding.weight.dtype,
    )
    state_biases: list[Tensor] = []

    with autocast_context(device):
        for position_index in range(num_positions):
            if should_reset_state(position_index, reset_interval):
                state = torch.zeros_like(state)
            state_biases.append(state)
            _, state = model.recurrent_hidden(windows[:, position_index, :], state)

    return torch.stack(state_biases, dim=1)


def run_sequence_batch_train(
    model: RecurrentLateralModel,
    sequence_batch: Tensor,
    *,
    reset_interval: int | None,
    device: torch.device,
) -> TrainMetrics:
    batch_size = sequence_batch.shape[0]
    windows, targets = extract_windows_and_targets(sequence_batch)
    num_positions = windows.shape[1]
    flat_windows = windows.view(batch_size * num_positions, SHORT_CONTEXT)
    flat_targets = targets.view(batch_size * num_positions)
    state_biases = precompute_state_biases(model, windows, reset_interval=reset_interval, device=device)

    with autocast_context(device):
        output_hidden = model.output_hidden(flat_windows)
        recurrent_hidden = None
        recurrent_loss: Tensor | None = None
        if model.recurrent_block is not None:
            if state_biases is None:
                raise ValueError("state_biases missing for recurrent condition")
            flat_state_biases = state_biases.view(batch_size * num_positions, model.model_size.d_model)
            recurrent_hidden, _ = model.recurrent_hidden(flat_windows, flat_state_biases)
            recurrent_loss = aggregated_interior_local_loss(
                recurrent_hidden,
                embedding=model.token_embedding,
                targets=flat_targets,
                temperature=model.temperature,
                normalize=model.normalize,
                local_loss=model.local_loss,
                batch_size=batch_size,
                num_positions=num_positions,
            )
        logits = model.output_logits(output_hidden, recurrent_hidden)
        block0_loss = summed_cross_entropy(logits, flat_targets, batch_size=batch_size)
        block0_accuracy = (logits.argmax(dim=-1) == flat_targets).float().mean()
        total_loss = block0_loss if recurrent_loss is None else block0_loss + recurrent_loss

    return TrainMetrics(
        total_loss=total_loss,
        mean_block0_loss=block0_loss.item() / num_positions,
        mean_block0_accuracy=block0_accuracy.item(),
        mean_recurrent_loss=0.0 if recurrent_loss is None else recurrent_loss.item() / num_positions,
    )


@torch.no_grad()
def evaluate_condition(
    model: RecurrentLateralModel,
    dataset: SequentialDataset,
    *,
    batch_size: int,
    device: torch.device,
    burn_in: int,
    reset_interval: int | None,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_start in range(0, dataset.sequences.shape[0], batch_size):
        sequence_batch = dataset.sequences[batch_start : batch_start + batch_size]
        windows, targets = extract_windows_and_targets(sequence_batch)
        num_positions = windows.shape[1]
        state_biases = precompute_state_biases(model, windows, reset_interval=reset_interval, device=device)
        valid_position_mask = torch.arange(num_positions, device=device) >= burn_in
        if not valid_position_mask.any():
            continue

        valid_windows = windows[:, valid_position_mask, :]
        valid_targets = targets[:, valid_position_mask]
        flat_windows = valid_windows.reshape(-1, SHORT_CONTEXT)
        flat_targets = valid_targets.reshape(-1)

        with autocast_context(device):
            output_hidden = model.output_hidden(flat_windows)
            recurrent_hidden = None
            if model.recurrent_block is not None:
                if state_biases is None:
                    raise ValueError("state_biases missing for recurrent condition")
                valid_state_biases = state_biases[:, valid_position_mask, :]
                flat_state_biases = valid_state_biases.reshape(-1, model.model_size.d_model)
                recurrent_hidden, _ = model.recurrent_hidden(flat_windows, flat_state_biases)

            logits = model.output_logits(output_hidden, recurrent_hidden)
            loss, _ = next_char_metrics(logits, flat_targets)
            total_loss += loss.item() * flat_targets.shape[0]
            total_correct += (logits.argmax(dim=-1) == flat_targets).sum().item()
            total_examples += flat_targets.shape[0]

    if total_examples == 0:
        raise ValueError("Burn-in masked every evaluation position. Reduce --burn-in or increase --seq-length.")
    return EvalResult(val_loss=total_loss / total_examples, val_accuracy=total_correct / total_examples)


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


def warm_up_cuda(
    train_dataset: SequentialDataset,
    *,
    device: torch.device,
    model_size: ModelSize,
    temperature: float,
    lateral_scale: float,
    normalize: bool,
    local_loss: str,
    batch_size: int,
) -> None:
    if device.type != "cuda":
        return

    set_seed(DEFAULT_SEED)
    model = build_model(
        "recurrent_lateral",
        vocab_size=train_dataset.vocab_size,
        model_size=model_size,
        temperature=temperature,
        lateral_scale=lateral_scale,
        normalize=normalize,
        local_loss=local_loss,
    ).to(device)
    optimizer = make_optimizer(model, device)
    sequence_batch = train_dataset.sequences[:batch_size]

    model.train()
    metrics = run_sequence_batch_train(model, sequence_batch, reset_interval=None, device=device)
    optimizer.zero_grad(set_to_none=True)
    metrics.total_loss.backward()
    optimizer.step()

    model.eval()
    evaluate_condition(
        model,
        SequentialDataset(sequences=sequence_batch, vocab_size=train_dataset.vocab_size),
        batch_size=batch_size,
        device=device,
        burn_in=min(BURN_IN, sequence_batch.shape[1] - SHORT_CONTEXT - 1),
        reset_interval=None,
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
    temperature: float,
    lateral_scale: float,
    normalize: bool,
    local_loss: str,
    training_steps: int,
    batch_size: int,
    eval_batch_size: int,
    burn_in: int,
) -> TrainedCondition:
    set_seed(seed)
    model = build_model(
        condition_name,
        vocab_size=train_dataset.vocab_size,
        model_size=model_size,
        temperature=temperature,
        lateral_scale=lateral_scale,
        normalize=normalize,
        local_loss=local_loss,
    ).to(device)
    optimizer = make_optimizer(model, device)
    reset_interval = reset_interval_for_condition(condition_name)
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
        metrics = run_sequence_batch_train(model, sequence_batch, reset_interval=reset_interval, device=device)
        optimizer.zero_grad(set_to_none=True)
        metrics.total_loss.backward()
        optimizer.step()

        if step % PRINT_INTERVAL == 0 or step == training_steps:
            print(
                f"[{condition_name}] step={step:04d}/{training_steps} "
                f"block0_ce={metrics.mean_block0_loss:.4f} block0_acc={metrics.mean_block0_accuracy:.4%} "
                f"recurrent_loss={metrics.mean_recurrent_loss:.4f}",
                flush=True,
            )

    evaluation = evaluate_condition(
        model,
        val_dataset,
        batch_size=eval_batch_size,
        device=device,
        burn_in=burn_in,
        reset_interval=reset_interval,
    )
    wall_seconds = perf_counter() - started_at
    print(
        f"[{condition_name}] final val_loss={evaluation.val_loss:.4f} "
        f"val_accuracy={evaluation.val_accuracy:.4%} wall_seconds={wall_seconds:.2f}",
        flush=True,
    )
    return TrainedCondition(
        result=ConditionResult(
            name=condition_name,
            val_loss=evaluation.val_loss,
            val_accuracy=evaluation.val_accuracy,
            delta_from_baseline=0.0,
            wall_seconds=wall_seconds,
        ),
        model=model,
    )


def print_results_table(results: list[ConditionResult]) -> None:
    header = "condition".ljust(20) + "val_loss".rjust(12) + "val_acc".rjust(12) + "delta".rjust(12) + "wall_s".rjust(12)
    print("\nresults", flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for result in results:
        print(
            result.name.ljust(20)
            + f"{result.val_loss:12.4f}"
            + f"{result.val_accuracy:12.4%}"
            + f"{result.delta_from_baseline:12.4f}"
            + f"{result.wall_seconds:12.2f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {args.temperature}")
    if args.burn_in < 0:
        raise ValueError(f"burn_in must be non-negative, got {args.burn_in}")

    requested_conditions = parse_condition_names(args.conditions)
    model_size = resolve_model_size(args)
    device = resolve_device(args.device)
    set_seed(args.seed)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_dataset, val_dataset, _ = load_sequential_dataset(args.seq_length)
    train_dataset = dataset_to_device(train_dataset, device)
    val_dataset = dataset_to_device(val_dataset, device)

    valid_positions = train_dataset.sequences.shape[1] - SHORT_CONTEXT
    if args.burn_in >= valid_positions:
        raise ValueError(
            f"burn_in {args.burn_in} masks all {valid_positions} positions; reduce --burn-in or increase --seq-length"
        )

    warm_up_cuda(
        train_dataset,
        device=device,
        model_size=model_size,
        temperature=args.temperature,
        lateral_scale=args.lateral_scale,
        normalize=args.normalize,
        local_loss=args.local_loss,
        batch_size=min(args.batch_size, train_dataset.sequences.shape[0]),
    )

    print(
        "recurrent lateral lm "
        f"device={device.type} seed={args.seed} train_sequences={train_dataset.sequences.shape[0]} "
        f"val_sequences={val_dataset.sequences.shape[0]} vocab={train_dataset.vocab_size} "
        f"short_ctx={SHORT_CONTEXT} seq_length={args.seq_length} burn_in={args.burn_in} "
        f"d_model={model_size.d_model} n_heads={model_size.n_heads} ff_dim={model_size.ff_dim} n_layers={model_size.n_layers} "
        f"steps={args.steps} batch={args.batch_size} eval_batch={args.eval_batch_size} "
        f"temperature={args.temperature:.4f} lateral_scale={args.lateral_scale:.4f} normalize={args.normalize} local_loss={args.local_loss} "
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
            temperature=args.temperature,
            lateral_scale=args.lateral_scale,
            normalize=args.normalize,
            local_loss=args.local_loss,
            training_steps=args.steps,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            burn_in=args.burn_in,
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
            delta_from_baseline=(result.val_loss - baseline_loss) if baseline_loss is not None else float("nan"),
            wall_seconds=result.wall_seconds,
        )
        for result in results
    ]
    print_results_table(adjusted_results)


if __name__ == "__main__":
    main()
