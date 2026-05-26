from __future__ import annotations

from dataclasses import dataclass
import random
from time import perf_counter

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import set_seed
from core.model import ParallelDiagonalModel
from core.run_utils import random_batches

TRAIN_CHARACTERS = 50_000
VAL_CHARACTERS = 10_000
CONTEXT_SIZE = 32
TRAINING_STEPS = 500
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 1e-3
SEED = 42
LAMBDA_LOCAL = 1.0
LOCAL_MIX_ALPHA = 0.1
D_MODEL = 64
FEEDFORWARD_DIM = 128
NUM_HEADS = 4
VAL_EVAL_SAMPLES = 4_096


@dataclass(frozen=True)
class SyntheticDataset:
    train_tokens: Int[Tensor, "tokens"]
    val_inputs: Int[Tensor, "examples context"]
    val_targets: Int[Tensor, "examples"]
    vocab_size: int


@dataclass(frozen=True)
class Condition:
    name: str
    model: ParallelDiagonalModel
    predictor: nn.Linear | None


@dataclass(frozen=True)
class EvalMetrics:
    ce_loss: float
    accuracy: float
    local_mse: float | None


@dataclass(frozen=True)
class ConditionResult:
    name: str
    initial_eval: EvalMetrics
    final_eval: EvalMetrics
    wall_seconds: float


def generate_balanced_bracket_stream(*, total_characters: int, seed: int) -> str:
    if total_characters % 2 != 0:
        raise ValueError(f"Balanced bracket stream length must be even, got {total_characters}.")

    rng = random.Random(seed)
    chunks: list[str] = []
    generated = 0
    while generated < total_characters:
        remaining_characters = total_characters - generated
        max_pairs = min(48, remaining_characters // 2)
        min_pairs = min(4, max_pairs)
        pair_count = max_pairs if max_pairs <= min_pairs else rng.randint(min_pairs, max_pairs)
        chunks.append(generate_dyck1_word(pair_count=pair_count, rng=rng))
        generated += pair_count * 2
    return "".join(chunks)


def generate_dyck1_word(*, pair_count: int, rng: random.Random) -> str:
    if pair_count <= 0:
        raise ValueError(f"pair_count must be positive, got {pair_count}.")

    depth = 0
    opens_used = 0
    pieces: list[str] = []
    for _ in range(pair_count * 2):
        opens_remaining = pair_count - opens_used
        if depth == 0:
            pieces.append("(")
            depth += 1
            opens_used += 1
            continue
        if opens_remaining == 0:
            pieces.append(")")
            depth -= 1
            continue
        if rng.random() < 0.55:
            pieces.append("(")
            depth += 1
            opens_used += 1
        else:
            pieces.append(")")
            depth -= 1

    if depth != 0:
        raise RuntimeError(f"Generated invalid Dyck-1 word with final depth {depth}.")
    return "".join(pieces)


def encode_text(text: str) -> Int[Tensor, "tokens"]:
    stoi = {"(": 0, ")": 1}
    unknown = sorted(set(text) - set(stoi))
    if len(unknown) > 0:
        raise ValueError(f"Unexpected characters in synthetic text: {unknown}")
    return torch.tensor([stoi[character] for character in text], dtype=torch.long)


def make_windows(
    encoded: Int[Tensor, "tokens"],
    *,
    context_size: int,
) -> tuple[Int[Tensor, "examples context"], Int[Tensor, "examples"]]:
    if encoded.numel() <= context_size:
        raise ValueError(
            f"Need more encoded tokens than context_size, got {encoded.numel()} tokens and context {context_size}."
        )
    windows = encoded.unfold(0, context_size + 1, 1)
    return windows[:, :-1].contiguous(), windows[:, -1].contiguous()


def build_dataset() -> SyntheticDataset:
    train_text = generate_balanced_bracket_stream(total_characters=TRAIN_CHARACTERS, seed=SEED)
    val_text = generate_balanced_bracket_stream(total_characters=VAL_CHARACTERS, seed=SEED + 1)
    train_tokens = encode_text(train_text)
    val_tokens = encode_text(val_text)
    val_inputs, val_targets = make_windows(val_tokens, context_size=CONTEXT_SIZE)
    val_inputs = val_inputs[:VAL_EVAL_SAMPLES]
    val_targets = val_targets[:VAL_EVAL_SAMPLES]
    return SyntheticDataset(
        train_tokens=train_tokens,
        val_inputs=val_inputs,
        val_targets=val_targets,
        vocab_size=2,
    )


def resolve_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_baseline(*, vocab_size: int, device: torch.device) -> Condition:
    model = ParallelDiagonalModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        context_size=CONTEXT_SIZE,
        num_blocks=1,
        readout_mode="first",
        token_injection="block0",
    ).to(device)
    return Condition(name="baseline", model=model, predictor=None)


def build_treatment(*, vocab_size: int, device: torch.device) -> Condition:
    model = ParallelDiagonalModel(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        context_size=CONTEXT_SIZE,
        num_blocks=2,
        lateral_mix_init=LOCAL_MIX_ALPHA,
        detach_lateral=True,
        topology="top_down_to_first",
        readout_mode="first",
        token_injection="block0",
    ).to(device)
    predictor = nn.Linear(D_MODEL, D_MODEL).to(device)
    return Condition(name="treatment", model=model, predictor=predictor)


def combined_parameters(condition: Condition) -> list[nn.Parameter]:
    parameters = list(condition.model.parameters())
    if condition.predictor is not None:
        parameters.extend(condition.predictor.parameters())
    return parameters


def forward_losses(
    condition: Condition,
    batch_inputs: Int[Tensor, "batch context"],
    batch_targets: Int[Tensor, "batch"],
) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""] | None]:
    logits, state = condition.model.forward_with_state(batch_inputs)
    ce_loss = F.cross_entropy(logits, batch_targets)
    local_loss = None
    if condition.predictor is not None:
        if len(state.block_outputs) < 2:
            raise RuntimeError("Treatment state is missing block 1 outputs.")
        predictions = condition.predictor(state.block_outputs[1])
        targets = state.block_outputs[0].detach()
        local_loss = F.mse_loss(predictions, targets)
    total_loss = ce_loss if local_loss is None else ce_loss + (LAMBDA_LOCAL * local_loss)
    return total_loss, ce_loss, local_loss


@torch.inference_mode()
def evaluate_condition(
    condition: Condition,
    *,
    inputs: Int[Tensor, "examples context"],
    targets: Int[Tensor, "examples"],
    batch_size: int,
) -> EvalMetrics:
    was_training = condition.model.training
    predictor_was_training = condition.predictor.training if condition.predictor is not None else None
    condition.model.eval()
    if condition.predictor is not None:
        condition.predictor.eval()

    total_examples = 0
    total_ce = 0.0
    total_correct = 0
    total_local = 0.0
    total_local_elements = 0
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        batch_inputs = inputs[start:stop]
        batch_targets = targets[start:stop]
        logits, state = condition.model.forward_with_state(batch_inputs)
        batch_examples = batch_targets.shape[0]
        total_examples += batch_examples
        total_ce += F.cross_entropy(logits, batch_targets, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()

        if condition.predictor is not None:
            predictions = condition.predictor(state.block_outputs[1])
            detached_targets = state.block_outputs[0].detach()
            total_local += F.mse_loss(predictions, detached_targets, reduction="sum").item()
            total_local_elements += detached_targets.numel()

    if was_training:
        condition.model.train()
    if condition.predictor is not None and predictor_was_training:
        condition.predictor.train()

    local_mse = None if total_local_elements == 0 else total_local / total_local_elements
    return EvalMetrics(
        ce_loss=total_ce / total_examples,
        accuracy=total_correct / total_examples,
        local_mse=local_mse,
    )


def print_eval(name: str, step: int, metrics: EvalMetrics) -> None:
    local = "n/a" if metrics.local_mse is None else f"{metrics.local_mse:.6f}"
    print(
        f"[{name}] step {step:>3} val_ce={metrics.ce_loss:.6f} "
        f"val_acc={metrics.accuracy:.4f} local_mse={local}",
        flush=True,
    )


def train_condition(
    condition: Condition,
    *,
    dataset: SyntheticDataset,
    device: torch.device,
) -> ConditionResult:
    set_seed(SEED)
    optimizer = torch.optim.AdamW(combined_parameters(condition), lr=LEARNING_RATE)
    batch_rng = torch.Generator(device="cpu")
    batch_rng.manual_seed(SEED)
    batch_iterator = random_batches(
        dataset.train_tokens,
        context_size=CONTEXT_SIZE,
        batch_size=BATCH_SIZE,
        device=device,
        rng=batch_rng,
    )

    initial_eval = evaluate_condition(
        condition,
        inputs=dataset.val_inputs.to(device),
        targets=dataset.val_targets.to(device),
        batch_size=EVAL_BATCH_SIZE,
    )
    print_eval(condition.name, 0, initial_eval)

    started_at = perf_counter()
    for step in range(1, TRAINING_STEPS + 1):
        batch_inputs, batch_targets = next(batch_iterator)
        optimizer.zero_grad(set_to_none=True)
        total_loss, ce_loss, local_loss = forward_losses(condition, batch_inputs, batch_targets)
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"{condition.name} diverged at step {step}: loss is not finite.")
        total_loss.backward()
        optimizer.step()

        if step % 100 != 0:
            continue

        local = "n/a" if local_loss is None else f"{local_loss.detach().item():.6f}"
        print(
            f"[{condition.name}] step {step:>3} train_ce={ce_loss.detach().item():.6f} local_mse={local}",
            flush=True,
        )
        checkpoint = evaluate_condition(
            condition,
            inputs=dataset.val_inputs.to(device),
            targets=dataset.val_targets.to(device),
            batch_size=EVAL_BATCH_SIZE,
        )
        print_eval(condition.name, step, checkpoint)

    final_eval = evaluate_condition(
        condition,
        inputs=dataset.val_inputs.to(device),
        targets=dataset.val_targets.to(device),
        batch_size=EVAL_BATCH_SIZE,
    )
    return ConditionResult(
        name=condition.name,
        initial_eval=initial_eval,
        final_eval=final_eval,
        wall_seconds=perf_counter() - started_at,
    )


def main() -> int:
    if D_MODEL % NUM_HEADS != 0:
        raise ValueError(f"d_model must be divisible by num_heads, got {D_MODEL} and {NUM_HEADS}.")

    set_seed(SEED)
    device = resolve_device()
    dataset = build_dataset()

    print(
        f"device={device} train_chars={TRAIN_CHARACTERS} val_chars={VAL_CHARACTERS} "
        f"context={CONTEXT_SIZE} steps={TRAINING_STEPS} batch_size={BATCH_SIZE}",
        flush=True,
    )

    baseline = train_condition(build_baseline(vocab_size=dataset.vocab_size, device=device), dataset=dataset, device=device)
    treatment = train_condition(build_treatment(vocab_size=dataset.vocab_size, device=device), dataset=dataset, device=device)

    delta = treatment.final_eval.ce_loss - baseline.final_eval.ce_loss
    success = treatment.final_eval.ce_loss <= baseline.final_eval.ce_loss
    local_start = treatment.initial_eval.local_mse
    local_end = treatment.final_eval.local_mse
    if local_start is None or local_end is None:
        raise RuntimeError("Treatment local MSE metrics are missing.")

    print("=== final comparison ===", flush=True)
    print(
        f"baseline  val_ce={baseline.final_eval.ce_loss:.6f} val_acc={baseline.final_eval.accuracy:.4f} "
        f"wall_seconds={baseline.wall_seconds:.2f}",
        flush=True,
    )
    print(
        f"treatment val_ce={treatment.final_eval.ce_loss:.6f} val_acc={treatment.final_eval.accuracy:.4f} "
        f"local_mse_start={local_start:.6f} local_mse_end={local_end:.6f} "
        f"wall_seconds={treatment.wall_seconds:.2f}",
        flush=True,
    )
    print(f"delta_treatment_minus_baseline={delta:.6f}", flush=True)
    print(f"success={success}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
