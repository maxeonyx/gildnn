from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    raise RuntimeError(
        "Run from the repo root with `.\\.venv\\Scripts\\python.exe -m runs.assembled_architecture` so `core` imports resolve cleanly."
    )

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from core.dataset import CorpusData, RandomWindowCharDataset, load_corpus
from core.fixed_window_char import set_seed
from core.model import ParallelDiagonalCarryState, ParallelDiagonalModel, count_parameters
from core.run_utils import (
    append_log,
    log_run_restarted,
    prepare_output_paths,
    redirect_sanity_check_paths,
    register_active_lock,
    resolve_device,
)
from core.tied_readout import tied_logits
from core.training import current_git_sha, current_git_status_short, write_json

TRAIN_CHARACTERS = 100_000
VAL_CHARACTERS = 20_000
DEFAULT_STEPS = 200
SANITY_CHECK_STEPS = 60
DEFAULT_BATCH_SIZE = 8
SANITY_CHECK_BATCH_SIZE = 1
DEFAULT_SEQ_LEN = 2048
SANITY_CHECK_SEQ_LEN = 1024
DEFAULT_BPTT_CHUNK = 128
DEFAULT_D_MODEL = 96
DEFAULT_FEEDFORWARD_DIM = 192
DEFAULT_LEARNING_RATE = 3e-4
SANITY_CHECK_LEARNING_RATE = 1e-3
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_EVAL_INTERVAL = 20
SANITY_CHECK_EVAL_INTERVAL = 10
DEFAULT_SEED = 42
TEMPERATURE = 0.07
NORMALIZE = True
LATERAL_SCALE = 0.2
BLOCK_RATES = (1, 2, 4)


@dataclass(frozen=True)
class TinyShakespeareData:
    corpus: CorpusData
    train_encoded: Tensor
    val_encoded: Tensor
    source_path: Path


@dataclass(frozen=True)
class BatchLossSummary:
    total_loss: float
    block_losses: tuple[float, ...]
    valid_positions: tuple[int, ...]


@dataclass(frozen=True)
class CrossHorizonAnalysisSummary:
    horizons: tuple[int, ...]
    loss_matrix: tuple[tuple[float, ...], ...]
    valid_positions: tuple[tuple[int, ...], ...]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive float, got {value}")
    return parsed


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "tinyshakespeare" / "artifacts" / "assembled_architecture"
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=positive_int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq-len", type=positive_int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--bptt-chunk", type=positive_int, default=DEFAULT_BPTT_CHUNK)
    parser.add_argument("--d-model", type=positive_int, default=DEFAULT_D_MODEL)
    parser.add_argument("--feedforward-dim", type=positive_int, default=DEFAULT_FEEDFORWARD_DIM)
    parser.add_argument("--lr", type=positive_float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sanity-check-only", action="store_true")
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.jsonl")
    return parser.parse_args()


def resolve_steps(args: argparse.Namespace) -> int:
    return SANITY_CHECK_STEPS if args.sanity_check_only else args.steps


def resolve_batch_size(args: argparse.Namespace) -> int:
    return SANITY_CHECK_BATCH_SIZE if args.sanity_check_only else args.batch_size


def resolve_seq_len(args: argparse.Namespace) -> int:
    return SANITY_CHECK_SEQ_LEN if args.sanity_check_only else args.seq_len


def resolve_eval_interval(args: argparse.Namespace) -> int:
    return SANITY_CHECK_EVAL_INTERVAL if args.sanity_check_only else DEFAULT_EVAL_INTERVAL


def resolve_learning_rate(args: argparse.Namespace) -> float:
    return SANITY_CHECK_LEARNING_RATE if args.sanity_check_only else args.lr


def validate_args(args: argparse.Namespace) -> None:
    resolved_seq_len = resolve_seq_len(args)
    errors: list[str] = []
    if args.bptt_chunk > resolved_seq_len:
        errors.append(f"bptt_chunk must be <= seq_len, got bptt_chunk={args.bptt_chunk}, seq_len={resolved_seq_len}")
    if resolved_seq_len <= max(BLOCK_RATES):
        errors.append(f"seq_len must be > max block rate {max(BLOCK_RATES)}, got {resolved_seq_len}")
    if errors:
        raise ValueError("Argument validation failed:\n- " + "\n- ".join(errors))


def prepare_tinyshakespeare_data(*, repo_root: Path, seq_len: int) -> TinyShakespeareData:
    source_path = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    raw_text = source_path.read_text(encoding="utf-8")
    required_characters = TRAIN_CHARACTERS + VAL_CHARACTERS
    if len(raw_text) < required_characters:
        raise ValueError(f"Need at least {required_characters} characters, got {len(raw_text)}.")

    train_text = raw_text[:TRAIN_CHARACTERS]
    val_text = raw_text[TRAIN_CHARACTERS : TRAIN_CHARACTERS + VAL_CHARACTERS]
    missing_val_characters = sorted(set(val_text) - set(train_text))
    if len(missing_val_characters) > 0:
        raise ValueError(
            "Validation slice contains characters absent from the training slice: "
            f"{missing_val_characters}"
        )

    with tempfile.TemporaryDirectory(prefix="gildnn_assembled_architecture_corpus_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        train_path = tmp_path / "tinyshakespeare_train.txt"
        val_path = tmp_path / "tinyshakespeare_val.txt"
        train_path.write_text(train_text, encoding="utf-8")
        val_path.write_text(val_text, encoding="utf-8")
        corpus = load_corpus(
            train_path=train_path,
            val_path=val_path,
            context_size=seq_len,
            eval_samples=1,
        )
        train_dataset = corpus.train_dataset
        if not isinstance(train_dataset, RandomWindowCharDataset):
            raise TypeError("Expected load_corpus() to return RandomWindowCharDataset.")
        train_encoded = train_dataset.encoded_corpus.clone()
        val_encoded = corpus.encode_text(val_text)
        return TinyShakespeareData(
            corpus=corpus,
            train_encoded=train_encoded,
            val_encoded=torch.tensor(val_encoded, dtype=torch.long),
            source_path=source_path,
        )


def build_tbptt_sequences(encoded: Tensor, *, seq_len: int) -> Tensor:
    num_sequences = encoded.numel() // seq_len
    if num_sequences == 0:
        raise ValueError(
            "Training corpus is too short for TBPTT sequence construction. "
            f"Got {encoded.numel()} tokens and seq_len={seq_len}."
        )
    usable = num_sequences * seq_len
    return encoded[:usable].view(num_sequences, seq_len).clone()


def sample_tbptt_batch(
    sequences: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    rng: torch.Generator,
) -> Tensor:
    indices = torch.randint(0, sequences.shape[0], (batch_size,), generator=rng)
    pin_memory = device.type == "cuda"
    batch_inputs = sequences[indices]
    if pin_memory:
        batch_inputs = batch_inputs.pin_memory()
    return batch_inputs.to(device=device, dtype=torch.long, non_blocking=pin_memory)


def fixed_tbptt_batch(
    sequences: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    indices = torch.arange(batch_size, dtype=torch.long) % sequences.shape[0]
    pin_memory = device.type == "cuda"
    batch_inputs = sequences[indices]
    if pin_memory:
        batch_inputs = batch_inputs.pin_memory()
    return batch_inputs.to(device=device, dtype=torch.long, non_blocking=pin_memory)


def compute_chunk_loss(
    model: ParallelDiagonalModel,
    chunk_tokens: Int[Tensor, "batch chunk"],
    *,
    carry_state: ParallelDiagonalCarryState | None,
) -> tuple[Float[Tensor, ""], list[Float[Tensor, ""]], list[int], ParallelDiagonalCarryState]:
    _, state, next_carry_state = model.forward_with_state_and_carry(chunk_tokens, carry_state=carry_state)
    block_losses: list[Float[Tensor, ""]] = []
    valid_positions: list[int] = []
    for block_hidden, rate in zip(state.block_outputs, model.rates, strict=True):
        if block_hidden.shape[1] <= rate:
            raise RuntimeError(
                f"Chunk length {block_hidden.shape[1]} must be greater than block rate {rate} to compute local loss."
            )
        hidden = block_hidden[:, :-rate, :]
        targets = chunk_tokens[:, rate:]
        logits = tied_logits(hidden, model.token_embedding, temperature=TEMPERATURE, normalize=NORMALIZE)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        block_losses.append(loss)
        valid_positions.append(int(targets.numel()))
    total_loss = torch.stack(block_losses).sum()
    return total_loss, block_losses, valid_positions, next_carry_state


@torch.inference_mode()
def cross_horizon_analysis(
    model: ParallelDiagonalModel,
    batch_tokens: Int[Tensor, "batch seq"],
    bptt_chunk: int,
) -> CrossHorizonAnalysisSummary:
    was_training = model.training
    model.eval()
    horizons = tuple(int(rate) for rate in model.rates)
    carry_state: ParallelDiagonalCarryState | None = None
    loss_sums = [[0.0 for _ in horizons] for _ in model.rates]
    valid_position_sums = [[0 for _ in horizons] for _ in model.rates]
    for start in range(0, batch_tokens.shape[1], bptt_chunk):
        stop = min(start + bptt_chunk, batch_tokens.shape[1])
        chunk_tokens = batch_tokens[:, start:stop]
        if chunk_tokens.shape[1] <= max(horizons):
            continue
        _, state, carry_state = model.forward_with_state_and_carry(chunk_tokens, carry_state=carry_state)
        for block_index, block_hidden in enumerate(state.block_outputs):
            for horizon_index, horizon in enumerate(horizons):
                hidden = block_hidden[:, :-horizon, :]
                targets = chunk_tokens[:, horizon:]
                logits = tied_logits(hidden, model.token_embedding, temperature=TEMPERATURE, normalize=NORMALIZE)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                valid_positions = int(targets.numel())
                loss_sums[block_index][horizon_index] += loss.item() * valid_positions
                valid_position_sums[block_index][horizon_index] += valid_positions
        carry_state = carry_state.detach()
    if was_training:
        model.train()
    if any(valid_positions == 0 for row in valid_position_sums for valid_positions in row):
        raise RuntimeError(f"Cross-horizon analysis had zero valid positions: {valid_position_sums}")
    return CrossHorizonAnalysisSummary(
        horizons=horizons,
        loss_matrix=tuple(
            tuple(loss_sum / valid_positions for loss_sum, valid_positions in zip(loss_row, valid_row, strict=True))
            for loss_row, valid_row in zip(loss_sums, valid_position_sums, strict=True)
        ),
        valid_positions=tuple(tuple(row) for row in valid_position_sums),
    )


def format_cross_horizon_matrix(summary: CrossHorizonAnalysisSummary) -> str:
    column_width = 10
    header = "block/h".ljust(column_width) + " ".join(
        f"{horizon}-ahead".rjust(column_width) for horizon in summary.horizons
    )
    rows = [header]
    for block_index, losses in enumerate(summary.loss_matrix):
        row = f"block {block_index}".ljust(column_width) + " ".join(
            f"{loss:.4f}".rjust(column_width) for loss in losses
        )
        rows.append(row)
    return "\n".join(rows)


def block_parameter_groups(model: ParallelDiagonalModel) -> list[tuple[nn.Parameter, ...]]:
    if model.window_proj is not None:
        raise ValueError("Separate block optimizers do not support shared window_proj parameters.")
    if model.token_injection != "block0":
        raise ValueError(
            "Separate block optimizers in this script require token_injection='block0' so shared token inputs are owned by block 0."
        )

    groups: list[tuple[nn.Parameter, ...]] = []
    seen_parameter_ids: dict[int, int] = {}
    for block_index, (block, token_mix, block_mix, lateral_mix) in enumerate(
        zip(model.blocks, model.token_mixes, model.block_mixes, model.lateral_mixes, strict=True)
    ):
        modules: list[nn.Module] = [block, token_mix, block_mix, lateral_mix]
        if block_index == 0:
            modules.append(model.position_embedding)

        group_parameters: list[nn.Parameter] = []
        local_seen: set[int] = set()
        for module in modules:
            for parameter in module.parameters():
                if not parameter.requires_grad:
                    continue
                parameter_id = id(parameter)
                if parameter_id in local_seen:
                    continue
                owner = seen_parameter_ids.get(parameter_id)
                if owner is not None:
                    raise ValueError(
                        f"Parameter sharing across block optimizer groups is unsupported: parameter already assigned to block {owner}, encountered again in block {block_index}."
                    )
                seen_parameter_ids[parameter_id] = block_index
                local_seen.add(parameter_id)
                group_parameters.append(parameter)
        groups.append(tuple(group_parameters))

    unassigned_trainable_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in seen_parameter_ids
    ]
    if len(unassigned_trainable_parameters) > 0:
        raise ValueError(
            "Separate block optimizers require all trainable parameters to belong to exactly one block group. "
            f"Unassigned trainable parameters: {unassigned_trainable_parameters}"
        )

    return groups


def build_block_optimizers(
    model: ParallelDiagonalModel,
    *,
    device: torch.device,
    learning_rate: float,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> tuple[list[torch.optim.AdamW], list[tuple[nn.Parameter, ...]]]:
    parameter_groups = block_parameter_groups(model)
    optimizers: list[torch.optim.AdamW] = []
    for block_index, parameters in enumerate(parameter_groups):
        if len(parameters) == 0:
            raise ValueError(f"Block {block_index} has no trainable parameters for its optimizer.")
        optimizer_kwargs = {
            "lr": learning_rate,
            "betas": (0.9, 0.999),
            "weight_decay": weight_decay,
        }
        if device.type == "cuda":
            optimizer_kwargs["capturable"] = True
        optimizers.append(torch.optim.AdamW(parameters, **optimizer_kwargs))
    return optimizers, parameter_groups


def tbptt_training_step(
    model: ParallelDiagonalModel,
    block_optimizers: list[torch.optim.Optimizer],
    *,
    block_parameter_groups: list[tuple[nn.Parameter, ...]],
    batch_tokens: Int[Tensor, "batch seq"],
    bptt_chunk: int,
    grad_clip_norm: float,
) -> BatchLossSummary:
    if len(block_optimizers) != len(model.rates) or len(block_parameter_groups) != len(model.rates):
        raise ValueError(
            f"Expected one optimizer and one parameter group per block, got optimizers={len(block_optimizers)}, groups={len(block_parameter_groups)}, blocks={len(model.rates)}."
        )
    for optimizer in block_optimizers:
        optimizer.zero_grad(set_to_none=True)
    carry_state: ParallelDiagonalCarryState | None = None
    total_loss = torch.zeros((), device=batch_tokens.device)
    block_loss_sums = [torch.zeros((), device=batch_tokens.device) for _ in model.rates]
    valid_position_sums = [0 for _ in model.rates]
    # Pre-compute total chunks so we can average gradient across them.
    # This makes gradient magnitude independent of seq_len/bptt_chunk.
    total_chunks = sum(
        1
        for start in range(0, batch_tokens.shape[1], bptt_chunk)
        if min(start + bptt_chunk, batch_tokens.shape[1]) - start > max(model.rates)
    )
    for start in range(0, batch_tokens.shape[1], bptt_chunk):
        stop = min(start + bptt_chunk, batch_tokens.shape[1])
        chunk_tokens = batch_tokens[:, start:stop]
        if chunk_tokens.shape[1] <= max(model.rates):
            continue
        chunk_loss, chunk_block_losses, chunk_valid_positions, carry_state = compute_chunk_loss(
            model,
            chunk_tokens,
            carry_state=carry_state,
        )
        if not torch.isfinite(chunk_loss):
            # Diagnostic: what does the state look like at crash?
            diag_parts = [f"chunk_start={start}", f"chunk_size={chunk_tokens.shape[1]}"]
            if carry_state is not None:
                for bi, ps in enumerate(carry_state.previous_states):
                    diag_parts.append(f"block{bi}_state_norm={ps.norm().item():.4f}")
            for bi, bl in enumerate(chunk_block_losses):
                diag_parts.append(f"block{bi}_loss={bl.item() if torch.isfinite(bl) else 'NaN/Inf'}")
            raise RuntimeError(
                f"Assembled architecture training diverged: loss is NaN or Inf. "
                f"Diagnostics: {', '.join(diag_parts)}"
            )
        # Check carry state health after each chunk — detect drift before it becomes NaN.
        if carry_state is not None:
            max_norm = max(ps.norm().item() for ps in carry_state.previous_states)
            if max_norm > 1e4 or not torch.isfinite(torch.tensor(max_norm)):
                diag = [f"chunk_start={start}", f"max_carry_norm={max_norm:.2f}"]
                for bi, ps in enumerate(carry_state.previous_states):
                    diag.append(f"block{bi}_norm={ps.norm().item():.4f}")
                raise RuntimeError(
                    f"Carry state norm exploding. "
                    f"Diagnostics: {', '.join(diag)}"
                )
        (chunk_loss / total_chunks).backward()
        total_loss = total_loss + chunk_loss.detach()
        for index, (block_loss, valid_positions) in enumerate(zip(chunk_block_losses, chunk_valid_positions, strict=True)):
            block_loss_sums[index] = block_loss_sums[index] + block_loss.detach() * valid_positions
            valid_position_sums[index] += valid_positions
        carry_state = carry_state.detach()
    for parameters in block_parameter_groups:
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)
    for optimizer in block_optimizers:
        optimizer.step()
    if any(valid_positions == 0 for valid_positions in valid_position_sums):
        raise RuntimeError(f"Some block losses had zero valid positions: {valid_position_sums}")
    mean_block_losses = tuple(
        (block_loss_sum / valid_positions).item()
        for block_loss_sum, valid_positions in zip(block_loss_sums, valid_position_sums, strict=True)
    )
    return BatchLossSummary(
        total_loss=float(sum(mean_block_losses)),
        block_losses=mean_block_losses,
        valid_positions=tuple(valid_position_sums),
    )


@torch.inference_mode()
def evaluate_tbptt_batch_loss(
    model: ParallelDiagonalModel,
    *,
    batch_tokens: Int[Tensor, "batch seq"],
    bptt_chunk: int,
) -> BatchLossSummary:
    was_training = model.training
    model.eval()
    carry_state: ParallelDiagonalCarryState | None = None
    block_loss_sums = [0.0 for _ in model.rates]
    valid_position_sums = [0 for _ in model.rates]
    for start in range(0, batch_tokens.shape[1], bptt_chunk):
        stop = min(start + bptt_chunk, batch_tokens.shape[1])
        chunk_tokens = batch_tokens[:, start:stop]
        if chunk_tokens.shape[1] <= max(model.rates):
            continue
        _, chunk_block_losses, chunk_valid_positions, carry_state = compute_chunk_loss(
            model,
            chunk_tokens,
            carry_state=carry_state,
        )
        for index, (block_loss, valid_positions) in enumerate(zip(chunk_block_losses, chunk_valid_positions, strict=True)):
            block_loss_sums[index] += block_loss.item() * valid_positions
            valid_position_sums[index] += valid_positions
        carry_state = carry_state.detach()
    if was_training:
        model.train()
    if any(valid_positions == 0 for valid_positions in valid_position_sums):
        raise RuntimeError(f"Some block losses had zero valid positions during evaluation: {valid_position_sums}")
    mean_block_losses = tuple(
        block_loss_sum / valid_positions
        for block_loss_sum, valid_positions in zip(block_loss_sums, valid_position_sums, strict=True)
    )
    return BatchLossSummary(
        total_loss=float(sum(mean_block_losses)),
        block_losses=mean_block_losses,
        valid_positions=tuple(valid_position_sums),
    )


def checkpoint_payload(*, step: int, train_summary: BatchLossSummary, sanity_summary: BatchLossSummary) -> dict[str, object]:
    payload: dict[str, object] = {
        "step": step,
        "train_total_loss": round(train_summary.total_loss, 6),
        "sanity_total_loss": round(sanity_summary.total_loss, 6),
    }
    for block_index, (train_loss, sanity_loss, rate) in enumerate(
        zip(train_summary.block_losses, sanity_summary.block_losses, BLOCK_RATES, strict=True)
    ):
        payload[f"train_block_{block_index}_loss"] = round(train_loss, 6)
        payload[f"sanity_block_{block_index}_loss"] = round(sanity_loss, 6)
        payload[f"block_{block_index}_rate"] = rate
    return payload


def losses_went_down(initial: BatchLossSummary, current: BatchLossSummary) -> list[bool]:
    return [
        current_loss < initial_loss
        for initial_loss, current_loss in zip(initial.block_losses, current.block_losses, strict=True)
    ]


def main() -> int:
    args = parse_args()
    validate_args(args)

    if args.sanity_check_only:
        redirect_sanity_check_paths(args)

    steps = resolve_steps(args)
    batch_size = resolve_batch_size(args)
    seq_len = resolve_seq_len(args)
    eval_interval = resolve_eval_interval(args)
    learning_rate = resolve_learning_rate(args)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    prepare_output_paths(report_path=args.report_path, log_path=args.log_path)
    log_run_restarted(args.log_path)
    register_active_lock(
        experiment_name="assembled_architecture",
        variants=[f"dm{args.d_model}", f"ff{args.feedforward_dim}", f"seq{seq_len}", f"chunk{args.bptt_chunk}"],
        enabled=not args.no_lock,
    )

    repo_root = Path(__file__).resolve().parents[1]
    set_seed(args.seed)
    data = prepare_tinyshakespeare_data(repo_root=repo_root, seq_len=seq_len)
    tbptt_sequences = build_tbptt_sequences(data.train_encoded, seq_len=seq_len)
    val_tbptt_sequences = build_tbptt_sequences(data.val_encoded, seq_len=seq_len)

    model = ParallelDiagonalModel(
        vocab_size=data.corpus.vocab_size,
        context_size=args.bptt_chunk,
        d_model=args.d_model,
        feedforward_dim=args.feedforward_dim,
        num_blocks=3,
        rates=BLOCK_RATES,
        topology="upward",
        detach_lateral=True,
        lateral_scale=LATERAL_SCALE,
    ).to(device)
    with torch.no_grad():
        nn.init.normal_(model.token_embedding.weight)
        model.token_embedding.weight.copy_(F.normalize(model.token_embedding.weight, dim=-1))
    model.token_embedding.weight.requires_grad_(False)
    model.output.requires_grad_(False)
    if model.readout_logits is not None:
        model.readout_logits.requires_grad_(False)
    parameter_count = count_parameters(model)
    block_optimizers, block_optimizer_parameter_groups = build_block_optimizers(
        model,
        device=device,
        learning_rate=learning_rate,
    )

    append_log(
        args.log_path,
        {
            "stage": "run_started",
            "device": str(device),
            "sanity_check_only": args.sanity_check_only,
            "steps": steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "bptt_chunk": args.bptt_chunk,
            "d_model": args.d_model,
            "feedforward_dim": args.feedforward_dim,
            "lr": learning_rate,
            "seed": args.seed,
            "temperature": TEMPERATURE,
            "normalize": NORMALIZE,
            "lateral_scale": LATERAL_SCALE,
            "rates": list(BLOCK_RATES),
            "fixed_embeddings": True,
            "separate_optimizers": True,
            "frozen_unused_readout": True,
        },
    )
    append_log(
        args.log_path,
        {
            "stage": "dataset_loaded",
            "source_path": str(data.source_path),
            "train_tokens": int(data.train_encoded.numel()),
            "val_tokens": int(data.val_encoded.numel()),
            "vocab_size": data.corpus.vocab_size,
            "tbptt_sequences": int(tbptt_sequences.shape[0]),
        },
    )
    append_log(
        args.log_path,
        {
            "stage": "model_built",
            "parameter_count": parameter_count,
            "mix_coefficients": model.mix_coefficients(),
            "block_optimizer_parameter_counts": [sum(parameter.numel() for parameter in parameters) for parameters in block_optimizer_parameter_groups],
        },
    )

    tbptt_rng = torch.Generator(device="cpu")
    tbptt_rng.manual_seed(args.seed)
    fixed_batch_tokens = fixed_tbptt_batch(tbptt_sequences, batch_size=batch_size, device=device)
    fixed_val_batch_tokens = fixed_tbptt_batch(val_tbptt_sequences, batch_size=batch_size, device=device)
    initial_sanity_summary = evaluate_tbptt_batch_loss(model, batch_tokens=fixed_batch_tokens, bptt_chunk=args.bptt_chunk)
    append_log(
        args.log_path,
        {
            "stage": "initial_sanity",
            "total_loss": round(initial_sanity_summary.total_loss, 6),
            **{f"block_{index}_loss": round(loss, 6) for index, loss in enumerate(initial_sanity_summary.block_losses)},
        },
    )

    checkpoints: list[dict[str, object]] = []
    started_at = perf_counter()
    final_train_summary = initial_sanity_summary
    final_sanity_summary = initial_sanity_summary
    final_step = 0
    early_stopped = False

    for step in range(1, steps + 1):
        if args.sanity_check_only:
            batch_tokens = fixed_batch_tokens
        else:
            batch_tokens = sample_tbptt_batch(tbptt_sequences, batch_size=batch_size, device=device, rng=tbptt_rng)
        final_train_summary = tbptt_training_step(
            model,
            block_optimizers,
            block_parameter_groups=block_optimizer_parameter_groups,
            batch_tokens=batch_tokens,
            bptt_chunk=args.bptt_chunk,
            grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
        )

        should_eval = step % eval_interval == 0 or step == steps
        if not should_eval:
            continue

        final_sanity_summary = evaluate_tbptt_batch_loss(model, batch_tokens=fixed_batch_tokens, bptt_chunk=args.bptt_chunk)
        final_step = step
        checkpoint = checkpoint_payload(step=step, train_summary=final_train_summary, sanity_summary=final_sanity_summary)
        checkpoints.append(checkpoint)
        append_log(args.log_path, {"stage": "checkpoint", **checkpoint})
        if args.sanity_check_only and step >= eval_interval:
            success_by_block = losses_went_down(initial_sanity_summary, final_sanity_summary)
            if all(success_by_block):
                early_stopped = True
                break

    wall_seconds = round(perf_counter() - started_at, 6)
    git_status_short = current_git_status_short()
    success_by_block = losses_went_down(initial_sanity_summary, final_sanity_summary)
    cross_horizon_summary = cross_horizon_analysis(model, fixed_val_batch_tokens, args.bptt_chunk)
    cross_horizon_matrix_text = format_cross_horizon_matrix(cross_horizon_summary)
    print("Cross-horizon loss matrix:")
    print(cross_horizon_matrix_text)
    report = {
        "config": {
            "sanity_check_only": args.sanity_check_only,
            "steps": steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "bptt_chunk": args.bptt_chunk,
            "d_model": args.d_model,
            "feedforward_dim": args.feedforward_dim,
            "learning_rate": args.lr,
            "effective_learning_rate": learning_rate,
            "seed": args.seed,
            "fixed_embeddings": True,
            "separate_optimizers": True,
            "train_characters": TRAIN_CHARACTERS,
            "val_characters": VAL_CHARACTERS,
            "temperature": TEMPERATURE,
            "normalize": NORMALIZE,
            "lateral_scale": LATERAL_SCALE,
            "rates": list(BLOCK_RATES),
        },
        "environment": {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": git_status_short == [],
            "git_status_short": git_status_short,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_device_name": None if device.type != "cuda" else torch.cuda.get_device_name(device),
        },
        "dataset": {
            "source_path": str(data.source_path),
            "train_tokens": int(data.train_encoded.numel()),
            "val_tokens": int(data.val_encoded.numel()),
            "vocab_size": data.corpus.vocab_size,
            "tbptt_sequences": int(tbptt_sequences.shape[0]),
            "val_tbptt_sequences": int(val_tbptt_sequences.shape[0]),
        },
        "model": {
            "parameter_count": parameter_count,
            "mix_coefficients": model.mix_coefficients(),
        },
        "results": {
            "final_step": final_step,
            "initial_total_loss": round(initial_sanity_summary.total_loss, 6),
            "final_train_total_loss": round(final_train_summary.total_loss, 6),
            "final_sanity_total_loss": round(final_sanity_summary.total_loss, 6),
            "initial_block_losses": [round(loss, 6) for loss in initial_sanity_summary.block_losses],
            "final_block_losses": [round(loss, 6) for loss in final_sanity_summary.block_losses],
            "loss_went_down_by_block": success_by_block,
            "early_stopped": early_stopped,
            "wall_seconds": wall_seconds,
            "cross_horizon_analysis": {
                "horizons": list(cross_horizon_summary.horizons),
                "loss_matrix": [
                    [round(loss, 6) for loss in row]
                    for row in cross_horizon_summary.loss_matrix
                ],
                "valid_positions": [list(row) for row in cross_horizon_summary.valid_positions],
                "best_horizon_by_block": [
                    cross_horizon_summary.horizons[min(range(len(row)), key=row.__getitem__)]
                    for row in cross_horizon_summary.loss_matrix
                ],
                "best_block_by_horizon": [
                    min(range(len(cross_horizon_summary.loss_matrix)), key=lambda block_index: cross_horizon_summary.loss_matrix[block_index][horizon_index])
                    for horizon_index in range(len(cross_horizon_summary.horizons))
                ],
                "matrix_text": cross_horizon_matrix_text,
            },
        },
        "checkpoints": checkpoints,
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "run_completed",
            "report_path": str(args.report_path),
            "final_step": final_step,
            "final_sanity_total_loss": round(final_sanity_summary.total_loss, 6),
            "loss_went_down_by_block": success_by_block,
            "cross_horizon_horizons": list(cross_horizon_summary.horizons),
            "cross_horizon_loss_matrix": [
                [round(loss, 6) for loss in row]
                for row in cross_horizon_summary.loss_matrix
            ],
            "early_stopped": early_stopped,
            "wall_seconds": wall_seconds,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
