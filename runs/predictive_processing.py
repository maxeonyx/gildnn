from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    raise RuntimeError(
        "Run from the repo root with `.\\.venv\\Scripts\\python.exe -m runs.predictive_processing` so `core` imports resolve cleanly."
    )

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from core.dataset import CorpusData, RandomWindowCharDataset, load_corpus
from core.fixed_window_char import set_seed
from core.model import count_parameters
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
DEFAULT_BLOCK0_STEPS = 200
DEFAULT_BLOCK1_STEPS = 800
SANITY_BLOCK0_STEPS = 30
SANITY_BLOCK1_STEPS = 60
DEFAULT_BATCH_SIZE = 8
SANITY_BATCH_SIZE = 2
DEFAULT_SEQ_LEN = 2048
SANITY_SEQ_LEN = 128
DEFAULT_BPTT_CHUNK = 128
DEFAULT_D_MODEL = 96
DEFAULT_BLOCK0_LR = 3e-4
DEFAULT_BLOCK1_LR = 3e-4
SANITY_BLOCK0_LR = 1e-3
SANITY_BLOCK1_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_BLOCK0_EVAL_INTERVAL = 20
DEFAULT_BLOCK1_EVAL_INTERVAL = 50
SANITY_BLOCK0_EVAL_INTERVAL = 20
SANITY_BLOCK1_EVAL_INTERVAL = 40
DEFAULT_SEED = 42
TEMPERATURE = 0.07
NORMALIZE = True
RANDOM_BASELINE_SEED = 7


@dataclass(frozen=True)
class TinyShakespeareData:
    corpus: CorpusData
    train_encoded: Tensor
    val_encoded: Tensor
    source_path: Path


@dataclass(frozen=True)
class PhaseCheckpoint:
    step: int
    train_loss: float
    eval_loss: float
    copy_baseline: float | None = None
    random_baseline: float | None = None
    beats_copy: bool | None = None


@dataclass(frozen=True)
class PhaseSummary:
    initial_eval_loss: float
    final_train_loss: float
    final_eval_loss: float
    checkpoints: list[PhaseCheckpoint]


@dataclass(frozen=True)
class Block1Metrics:
    mse: float
    copy_baseline: float
    random_baseline: float
    beats_copy: bool


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
    artifact_dir = repo_root / "experiments" / "tinyshakespeare" / "artifacts" / "predictive_processing"
    parser = argparse.ArgumentParser()
    parser.add_argument("--block0-steps", type=positive_int, default=DEFAULT_BLOCK0_STEPS)
    parser.add_argument("--block1-steps", type=positive_int, default=DEFAULT_BLOCK1_STEPS)
    parser.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq-len", type=positive_int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--bptt-chunk", type=positive_int, default=DEFAULT_BPTT_CHUNK)
    parser.add_argument("--d-model", type=positive_int, default=DEFAULT_D_MODEL)
    parser.add_argument("--block0-lr", type=positive_float, default=DEFAULT_BLOCK0_LR)
    parser.add_argument("--block1-lr", type=positive_float, default=DEFAULT_BLOCK1_LR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sanity-check-only", action="store_true")
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.jsonl")
    return parser.parse_args()


def resolve_block0_steps(args: argparse.Namespace) -> int:
    return SANITY_BLOCK0_STEPS if args.sanity_check_only else args.block0_steps


def resolve_block1_steps(args: argparse.Namespace) -> int:
    return SANITY_BLOCK1_STEPS if args.sanity_check_only else args.block1_steps


def resolve_batch_size(args: argparse.Namespace) -> int:
    return SANITY_BATCH_SIZE if args.sanity_check_only else args.batch_size


def resolve_seq_len(args: argparse.Namespace) -> int:
    return SANITY_SEQ_LEN if args.sanity_check_only else args.seq_len


def resolve_block0_lr(args: argparse.Namespace) -> float:
    return SANITY_BLOCK0_LR if args.sanity_check_only else args.block0_lr


def resolve_block1_lr(args: argparse.Namespace) -> float:
    return SANITY_BLOCK1_LR if args.sanity_check_only else args.block1_lr


def resolve_block0_eval_interval(args: argparse.Namespace) -> int:
    return SANITY_BLOCK0_EVAL_INTERVAL if args.sanity_check_only else DEFAULT_BLOCK0_EVAL_INTERVAL


def resolve_block1_eval_interval(args: argparse.Namespace) -> int:
    return SANITY_BLOCK1_EVAL_INTERVAL if args.sanity_check_only else DEFAULT_BLOCK1_EVAL_INTERVAL


def validate_args(args: argparse.Namespace) -> None:
    resolved_seq_len = resolve_seq_len(args)
    errors: list[str] = []
    if args.bptt_chunk > resolved_seq_len:
        errors.append(f"bptt_chunk must be <= seq_len, got bptt_chunk={args.bptt_chunk}, seq_len={resolved_seq_len}")
    if resolved_seq_len < 2:
        errors.append(f"seq_len must be at least 2, got {resolved_seq_len}")
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

    with tempfile.TemporaryDirectory(prefix="gildnn_predictive_processing_corpus_") as tmp_dir:
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
    batch_tokens = sequences[indices]
    if pin_memory:
        batch_tokens = batch_tokens.pin_memory()
    return batch_tokens.to(device=device, dtype=torch.long, non_blocking=pin_memory)


def fixed_tbptt_batch(
    sequences: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    indices = torch.arange(batch_size, dtype=torch.long) % sequences.shape[0]
    pin_memory = device.type == "cuda"
    batch_tokens = sequences[indices]
    if pin_memory:
        batch_tokens = batch_tokens.pin_memory()
    return batch_tokens.to(device=device, dtype=torch.long, non_blocking=pin_memory)


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


class TokenPredictorBlock(nn.Module):
    def __init__(self, *, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.cell = nn.GRUCell(d_model, d_model)
        self.reset_fixed_embeddings()

    @torch.no_grad()
    def reset_fixed_embeddings(self) -> None:
        nn.init.normal_(self.token_embedding.weight)
        self.token_embedding.weight.copy_(F.normalize(self.token_embedding.weight.float(), dim=-1))
        self.token_embedding.weight.requires_grad_(False)

    def initial_hidden(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> Float[Tensor, "batch d_model"]:
        return torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)

    def forward_chunk(
        self,
        tokens: Int[Tensor, "batch chunk"],
        hidden: Float[Tensor, "batch d_model"] | None = None,
    ) -> tuple[Float[Tensor, "batch chunk d_model"], Float[Tensor, "batch d_model"]]:
        embeddings = self.token_embedding(tokens)
        if hidden is None:
            hidden = self.initial_hidden(batch_size=tokens.shape[0], device=tokens.device, dtype=embeddings.dtype)

        outputs: list[Tensor] = []
        current_hidden = hidden
        for time_index in range(tokens.shape[1]):
            current_hidden = self.cell(embeddings[:, time_index, :], current_hidden)
            current_hidden = F.normalize(current_hidden.float(), dim=-1).to(current_hidden.dtype)
            outputs.append(current_hidden)
        return torch.stack(outputs, dim=1), current_hidden


class RepresentationPredictorBlock(nn.Module):
    def __init__(self, *, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.cell = nn.GRUCell(d_model, d_model)

    def initial_hidden(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> Float[Tensor, "batch d_model"]:
        return torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)

    def forward_chunk(
        self,
        previous_mu: Float[Tensor, "batch chunk d_model"],
        hidden: Float[Tensor, "batch d_model"] | None = None,
        *,
        use_recurrence: bool,
    ) -> tuple[Float[Tensor, "batch chunk d_model"], Float[Tensor, "batch d_model"]]:
        if hidden is None:
            hidden = self.initial_hidden(batch_size=previous_mu.shape[0], device=previous_mu.device, dtype=previous_mu.dtype)

        outputs: list[Tensor] = []
        current_hidden = hidden
        for time_index in range(previous_mu.shape[1]):
            recurrent_hidden = current_hidden if use_recurrence else torch.zeros_like(current_hidden)
            next_hidden = self.cell(previous_mu[:, time_index, :], recurrent_hidden)
            next_hidden = F.normalize(next_hidden.float(), dim=-1).to(next_hidden.dtype)
            outputs.append(next_hidden)
            current_hidden = next_hidden if use_recurrence else torch.zeros_like(next_hidden)
        return torch.stack(outputs, dim=1), current_hidden


def chunk_ranges(seq_len: int, bptt_chunk: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + bptt_chunk, seq_len))
        for start in range(0, seq_len, bptt_chunk)
        if min(start + bptt_chunk, seq_len) > start
    ]


def block0_loss_from_mu(
    block0: TokenPredictorBlock,
    mu_chunk: Float[Tensor, "batch chunk d_model"],
    chunk_tokens: Int[Tensor, "batch chunk"],
) -> tuple[Float[Tensor, ""], int]:
    if chunk_tokens.shape[1] <= 1:
        raise ValueError(f"Need chunk length > 1 for block 0 CE loss, got {chunk_tokens.shape[1]}.")
    logits = tied_logits(mu_chunk[:, :-1, :], block0.token_embedding, temperature=TEMPERATURE, normalize=NORMALIZE)
    targets = chunk_tokens[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return loss, int(targets.numel())


def block1_loss_from_predictions(
    predicted_mu: Float[Tensor, "batch chunk d_model"],
    target_mu: Float[Tensor, "batch chunk d_model"],
    *,
    skip_first_position: bool,
) -> tuple[Float[Tensor, ""], int]:
    start = 1 if skip_first_position else 0
    valid_predictions = predicted_mu[:, start:, :]
    valid_targets = target_mu[:, start:, :]
    if valid_predictions.shape[1] == 0:
        raise ValueError("Need at least one valid position for block 1 MSE loss.")
    loss = F.mse_loss(valid_predictions.float(), valid_targets.float())
    return loss, int(valid_predictions.shape[0] * valid_predictions.shape[1])


def random_baseline_vector(*, d_model: int, device: torch.device, dtype: torch.dtype) -> Float[Tensor, "d_model"]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(RANDOM_BASELINE_SEED)
    vector = torch.randn(d_model, generator=generator, dtype=torch.float32)
    vector = F.normalize(vector, dim=0)
    return vector.to(device=device, dtype=dtype)


def collect_block0_representations(
    block0: TokenPredictorBlock,
    batch_tokens: Int[Tensor, "batch seq"],
    *,
    bptt_chunk: int,
) -> Float[Tensor, "batch seq d_model"]:
    was_training = block0.training
    block0.eval()
    hidden: Tensor | None = None
    chunks: list[Tensor] = []
    with torch.no_grad():
        for start, stop in chunk_ranges(batch_tokens.shape[1], bptt_chunk):
            mu_chunk, hidden = block0.forward_chunk(batch_tokens[:, start:stop], hidden)
            chunks.append(mu_chunk)
            hidden = hidden.detach()
    if was_training:
        block0.train()
    return torch.cat(chunks, dim=1)


@torch.inference_mode()
def evaluate_block0_loss(
    block0: TokenPredictorBlock,
    *,
    batch_tokens: Int[Tensor, "batch seq"],
    bptt_chunk: int,
) -> float:
    was_training = block0.training
    block0.eval()
    hidden: Tensor | None = None
    loss_sum = 0.0
    valid_positions = 0
    for start, stop in chunk_ranges(batch_tokens.shape[1], bptt_chunk):
        chunk_tokens = batch_tokens[:, start:stop]
        if chunk_tokens.shape[1] <= 1:
            continue
        mu_chunk, hidden = block0.forward_chunk(chunk_tokens, hidden)
        chunk_loss, chunk_valid_positions = block0_loss_from_mu(block0, mu_chunk, chunk_tokens)
        loss_sum += chunk_loss.item() * chunk_valid_positions
        valid_positions += chunk_valid_positions
        hidden = hidden.detach()
    if was_training:
        block0.train()
    if valid_positions == 0:
        raise RuntimeError("Block 0 evaluation had zero valid positions.")
    return loss_sum / valid_positions


@torch.inference_mode()
def evaluate_block1_metrics(
    block1: RepresentationPredictorBlock,
    *,
    mu_sequence: Float[Tensor, "batch seq d_model"],
    bptt_chunk: int,
    use_recurrence: bool,
) -> Block1Metrics:
    was_training = block1.training
    block1.eval()
    hidden: Tensor | None = None
    previous_mu = torch.zeros(mu_sequence.shape[0], mu_sequence.shape[2], device=mu_sequence.device, dtype=mu_sequence.dtype)
    predicted_chunks: list[Tensor] = []
    for start, stop in chunk_ranges(mu_sequence.shape[1], bptt_chunk):
        target_chunk = mu_sequence[:, start:stop, :]
        predictor_input = torch.cat([previous_mu.unsqueeze(1), target_chunk[:, :-1, :]], dim=1)
        predicted_chunk, hidden = block1.forward_chunk(predictor_input, hidden, use_recurrence=use_recurrence)
        predicted_chunks.append(predicted_chunk)
        previous_mu = target_chunk[:, -1, :].detach()
        hidden = hidden.detach()

    predicted_mu = torch.cat(predicted_chunks, dim=1)
    valid_target = mu_sequence[:, 1:, :]
    valid_prediction = predicted_mu[:, 1:, :]
    copy_baseline = mu_sequence[:, :-1, :]
    random_vector = random_baseline_vector(d_model=mu_sequence.shape[2], device=mu_sequence.device, dtype=mu_sequence.dtype)
    random_baseline = random_vector.view(1, 1, -1).expand_as(valid_target)
    if was_training:
        block1.train()
    mse = F.mse_loss(valid_prediction.float(), valid_target.float()).item()
    copy_mse = F.mse_loss(copy_baseline.float(), valid_target.float()).item()
    random_mse = F.mse_loss(random_baseline.float(), valid_target.float()).item()
    return Block1Metrics(
        mse=mse,
        copy_baseline=copy_mse,
        random_baseline=random_mse,
        beats_copy=mse < copy_mse,
    )


def train_block0(
    block0: TokenPredictorBlock,
    optimizer: torch.optim.Optimizer,
    *,
    train_sequences: Tensor,
    eval_batch_tokens: Tensor,
    steps: int,
    batch_size: int,
    bptt_chunk: int,
    device: torch.device,
    eval_interval: int,
    grad_clip_norm: float,
    sanity_check_only: bool,
    seed: int,
    log_path: Path,
) -> PhaseSummary:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    initial_eval_loss = evaluate_block0_loss(block0, batch_tokens=eval_batch_tokens, bptt_chunk=bptt_chunk)
    checkpoints: list[PhaseCheckpoint] = [
        PhaseCheckpoint(step=0, train_loss=initial_eval_loss, eval_loss=initial_eval_loss)
    ]
    append_log(
        log_path,
        {
            "stage": "phase_a_initial",
            "step": 0,
            "train_loss": round(initial_eval_loss, 6),
            "eval_loss": round(initial_eval_loss, 6),
        },
    )

    final_train_loss = initial_eval_loss
    final_eval_loss = initial_eval_loss
    ranges = chunk_ranges(eval_batch_tokens.shape[1], bptt_chunk)
    total_chunks = sum(1 for start, stop in ranges if stop - start > 1)
    if total_chunks == 0:
        raise RuntimeError("Block 0 training requires at least one chunk with a next-token target.")

    for step in range(1, steps + 1):
        batch_tokens = (
            fixed_tbptt_batch(train_sequences, batch_size=batch_size, device=device)
            if sanity_check_only
            else sample_tbptt_batch(train_sequences, batch_size=batch_size, device=device, rng=rng)
        )
        optimizer.zero_grad(set_to_none=True)
        hidden: Tensor | None = None
        loss_sum = 0.0
        valid_positions = 0
        for start, stop in ranges:
            chunk_tokens = batch_tokens[:, start:stop]
            if chunk_tokens.shape[1] <= 1:
                continue
            mu_chunk, hidden = block0.forward_chunk(chunk_tokens, hidden)
            chunk_loss, chunk_valid_positions = block0_loss_from_mu(block0, mu_chunk, chunk_tokens)
            if not torch.isfinite(chunk_loss):
                raise RuntimeError(f"Block 0 training diverged at step {step}.")
            (chunk_loss / total_chunks).backward()
            loss_sum += chunk_loss.detach().item() * chunk_valid_positions
            valid_positions += chunk_valid_positions
            hidden = hidden.detach()

        torch.nn.utils.clip_grad_norm_(trainable_parameters(block0), grad_clip_norm)
        optimizer.step()
        final_train_loss = loss_sum / valid_positions

        if step % eval_interval != 0 and step != steps:
            continue

        final_eval_loss = evaluate_block0_loss(block0, batch_tokens=eval_batch_tokens, bptt_chunk=bptt_chunk)
        checkpoint = PhaseCheckpoint(step=step, train_loss=final_train_loss, eval_loss=final_eval_loss)
        checkpoints.append(checkpoint)
        append_log(
            log_path,
            {
                "stage": "phase_a_checkpoint",
                "step": step,
                "train_loss": round(final_train_loss, 6),
                "eval_loss": round(final_eval_loss, 6),
            },
        )

    return PhaseSummary(
        initial_eval_loss=initial_eval_loss,
        final_train_loss=final_train_loss,
        final_eval_loss=final_eval_loss,
        checkpoints=checkpoints,
    )


def train_block1(
    block0: TokenPredictorBlock,
    block1: RepresentationPredictorBlock,
    optimizer: torch.optim.Optimizer,
    *,
    train_sequences: Tensor,
    eval_batch_tokens: Tensor,
    steps: int,
    batch_size: int,
    bptt_chunk: int,
    device: torch.device,
    eval_interval: int,
    grad_clip_norm: float,
    sanity_check_only: bool,
    seed: int,
    log_path: Path,
    checkpoint_stage: str,
    use_recurrence: bool,
) -> tuple[PhaseSummary, Block1Metrics]:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    eval_mu = collect_block0_representations(block0, eval_batch_tokens, bptt_chunk=bptt_chunk)
    initial_metrics = evaluate_block1_metrics(
        block1,
        mu_sequence=eval_mu,
        bptt_chunk=bptt_chunk,
        use_recurrence=use_recurrence,
    )
    checkpoints: list[PhaseCheckpoint] = [
        PhaseCheckpoint(
            step=0,
            train_loss=initial_metrics.mse,
            eval_loss=initial_metrics.mse,
            copy_baseline=initial_metrics.copy_baseline,
            random_baseline=initial_metrics.random_baseline,
            beats_copy=initial_metrics.beats_copy,
        )
    ]
    append_log(
        log_path,
        {
            "stage": f"{checkpoint_stage}_initial",
            "step": 0,
            "train_loss": round(initial_metrics.mse, 6),
            "eval_loss": round(initial_metrics.mse, 6),
            "copy_baseline": round(initial_metrics.copy_baseline, 6),
            "random_baseline": round(initial_metrics.random_baseline, 6),
            "beats_copy": initial_metrics.beats_copy,
        },
    )

    final_train_loss = initial_metrics.mse
    final_eval_metrics = initial_metrics
    total_chunks = 0
    for start, stop in chunk_ranges(eval_batch_tokens.shape[1], bptt_chunk):
        chunk_length = stop - start
        valid_length = chunk_length - 1 if start == 0 else chunk_length
        if valid_length > 0:
            total_chunks += 1
    if total_chunks == 0:
        raise RuntimeError("Block 1 training requires at least one valid target chunk.")

    for step in range(1, steps + 1):
        batch_tokens = (
            fixed_tbptt_batch(train_sequences, batch_size=batch_size, device=device)
            if sanity_check_only
            else sample_tbptt_batch(train_sequences, batch_size=batch_size, device=device, rng=rng)
        )
        with torch.no_grad():
            mu_sequence = collect_block0_representations(block0, batch_tokens, bptt_chunk=bptt_chunk)

        optimizer.zero_grad(set_to_none=True)
        hidden: Tensor | None = None
        previous_mu = torch.zeros(mu_sequence.shape[0], mu_sequence.shape[2], device=device, dtype=mu_sequence.dtype)
        loss_sum = 0.0
        valid_positions = 0
        for start, stop in chunk_ranges(mu_sequence.shape[1], bptt_chunk):
            target_chunk = mu_sequence[:, start:stop, :]
            predictor_input = torch.cat([previous_mu.unsqueeze(1), target_chunk[:, :-1, :]], dim=1)
            predicted_chunk, hidden = block1.forward_chunk(predictor_input, hidden, use_recurrence=use_recurrence)
            chunk_loss, chunk_valid_positions = block1_loss_from_predictions(
                predicted_chunk,
                target_chunk,
                skip_first_position=start == 0,
            )
            if not torch.isfinite(chunk_loss):
                raise RuntimeError(f"Block 1 training diverged at step {step}.")
            (chunk_loss / total_chunks).backward()
            loss_sum += chunk_loss.detach().item() * chunk_valid_positions
            valid_positions += chunk_valid_positions
            previous_mu = target_chunk[:, -1, :].detach()
            hidden = hidden.detach()

        torch.nn.utils.clip_grad_norm_(trainable_parameters(block1), grad_clip_norm)
        optimizer.step()
        final_train_loss = loss_sum / valid_positions

        if step % eval_interval != 0 and step != steps:
            continue

        final_eval_metrics = evaluate_block1_metrics(
            block1,
            mu_sequence=eval_mu,
            bptt_chunk=bptt_chunk,
            use_recurrence=use_recurrence,
        )
        checkpoint = PhaseCheckpoint(
            step=step,
            train_loss=final_train_loss,
            eval_loss=final_eval_metrics.mse,
            copy_baseline=final_eval_metrics.copy_baseline,
            random_baseline=final_eval_metrics.random_baseline,
            beats_copy=final_eval_metrics.beats_copy,
        )
        checkpoints.append(checkpoint)
        append_log(
            log_path,
            {
                "stage": checkpoint_stage,
                "step": step,
                "train_loss": round(final_train_loss, 6),
                "eval_loss": round(final_eval_metrics.mse, 6),
                "copy_baseline": round(final_eval_metrics.copy_baseline, 6),
                "random_baseline": round(final_eval_metrics.random_baseline, 6),
                "beats_copy": final_eval_metrics.beats_copy,
            },
        )

    return (
        PhaseSummary(
            initial_eval_loss=initial_metrics.mse,
            final_train_loss=final_train_loss,
            final_eval_loss=final_eval_metrics.mse,
            checkpoints=checkpoints,
        ),
        final_eval_metrics,
    )


def phase_checkpoints_to_payload(checkpoints: list[PhaseCheckpoint]) -> list[dict[str, object]]:
    return [asdict(checkpoint) for checkpoint in checkpoints]


def main() -> int:
    args = parse_args()
    validate_args(args)

    if args.sanity_check_only:
        redirect_sanity_check_paths(args)

    block0_steps = resolve_block0_steps(args)
    block1_steps = resolve_block1_steps(args)
    batch_size = resolve_batch_size(args)
    seq_len = resolve_seq_len(args)
    block0_lr = resolve_block0_lr(args)
    block1_lr = resolve_block1_lr(args)
    block0_eval_interval = resolve_block0_eval_interval(args)
    block1_eval_interval = resolve_block1_eval_interval(args)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    prepare_output_paths(report_path=args.report_path, log_path=args.log_path)
    log_run_restarted(args.log_path)
    register_active_lock(
        experiment_name="predictive_processing",
        variants=[f"dm{args.d_model}", f"seq{seq_len}", f"chunk{args.bptt_chunk}"],
        enabled=not args.no_lock and not args.sanity_check_only,
    )

    repo_root = Path(__file__).resolve().parents[1]
    set_seed(args.seed)
    data = prepare_tinyshakespeare_data(repo_root=repo_root, seq_len=seq_len)
    train_sequences = build_tbptt_sequences(data.train_encoded, seq_len=seq_len)
    val_sequences = build_tbptt_sequences(data.val_encoded, seq_len=seq_len)
    eval_batch_tokens = fixed_tbptt_batch(val_sequences, batch_size=batch_size, device=device)

    block0 = TokenPredictorBlock(vocab_size=data.corpus.vocab_size, d_model=args.d_model).to(device)
    block1 = RepresentationPredictorBlock(d_model=args.d_model).to(device)
    block1_no_recurrence = RepresentationPredictorBlock(d_model=args.d_model).to(device)
    block1_no_recurrence.load_state_dict(copy.deepcopy(block1.state_dict()))

    block0_optimizer = torch.optim.AdamW(
        trainable_parameters(block0),
        lr=block0_lr,
        betas=(0.9, 0.999),
        weight_decay=DEFAULT_WEIGHT_DECAY,
        capturable=device.type == "cuda",
    )
    block1_optimizer = torch.optim.AdamW(
        trainable_parameters(block1),
        lr=block1_lr,
        betas=(0.9, 0.999),
        weight_decay=DEFAULT_WEIGHT_DECAY,
        capturable=device.type == "cuda",
    )
    block1_no_recurrence_optimizer = torch.optim.AdamW(
        trainable_parameters(block1_no_recurrence),
        lr=block1_lr,
        betas=(0.9, 0.999),
        weight_decay=DEFAULT_WEIGHT_DECAY,
        capturable=device.type == "cuda",
    )

    append_log(
        args.log_path,
        {
            "stage": "run_started",
            "device": str(device),
            "sanity_check_only": args.sanity_check_only,
            "block0_steps": block0_steps,
            "block1_steps": block1_steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "bptt_chunk": args.bptt_chunk,
            "d_model": args.d_model,
            "block0_lr": block0_lr,
            "block1_lr": block1_lr,
            "seed": args.seed,
            "temperature": TEMPERATURE,
            "normalize": NORMALIZE,
            "fixed_embeddings": True,
            "separate_optimizers": True,
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
            "train_sequences": int(train_sequences.shape[0]),
            "val_sequences": int(val_sequences.shape[0]),
        },
    )
    append_log(
        args.log_path,
        {
            "stage": "models_built",
            "block0_parameters": count_parameters(block0),
            "block1_parameters": count_parameters(block1),
            "block1_no_recurrence_parameters": count_parameters(block1_no_recurrence),
        },
    )

    phase_started_at = perf_counter()
    phase_a_summary = train_block0(
        block0,
        block0_optimizer,
        train_sequences=train_sequences,
        eval_batch_tokens=eval_batch_tokens,
        steps=block0_steps,
        batch_size=batch_size,
        bptt_chunk=args.bptt_chunk,
        device=device,
        eval_interval=block0_eval_interval,
        grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
        sanity_check_only=args.sanity_check_only,
        seed=args.seed,
        log_path=args.log_path,
    )
    phase_a_wall_seconds = round(perf_counter() - phase_started_at, 6)

    for parameter in block0.parameters():
        parameter.requires_grad_(False)

    phase_started_at = perf_counter()
    phase_b_summary, phase_b_metrics = train_block1(
        block0,
        block1,
        block1_optimizer,
        train_sequences=train_sequences,
        eval_batch_tokens=eval_batch_tokens,
        steps=block1_steps,
        batch_size=batch_size,
        bptt_chunk=args.bptt_chunk,
        device=device,
        eval_interval=block1_eval_interval,
        grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
        sanity_check_only=args.sanity_check_only,
        seed=args.seed + 1,
        log_path=args.log_path,
        checkpoint_stage="phase_b_checkpoint",
        use_recurrence=True,
    )
    phase_b_wall_seconds = round(perf_counter() - phase_started_at, 6)

    phase_started_at = perf_counter()
    phase_b_no_recurrence_summary, phase_b_no_recurrence_metrics = train_block1(
        block0,
        block1_no_recurrence,
        block1_no_recurrence_optimizer,
        train_sequences=train_sequences,
        eval_batch_tokens=eval_batch_tokens,
        steps=block1_steps,
        batch_size=batch_size,
        bptt_chunk=args.bptt_chunk,
        device=device,
        eval_interval=block1_eval_interval,
        grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
        sanity_check_only=args.sanity_check_only,
        seed=args.seed + 1,
        log_path=args.log_path,
        checkpoint_stage="phase_b_no_recurrence_checkpoint",
        use_recurrence=False,
    )
    phase_b_no_recurrence_wall_seconds = round(perf_counter() - phase_started_at, 6)

    summary_lines = [
        "Predictive processing results:",
        f"  Phase A Block 0 CE: {phase_a_summary.final_eval_loss:.6f} (initial {phase_a_summary.initial_eval_loss:.6f})",
        f"  Phase B Block 1 MSE: {phase_b_metrics.mse:.6f}",
        f"  Copy baseline MSE: {phase_b_metrics.copy_baseline:.6f}",
        f"  Random baseline MSE: {phase_b_metrics.random_baseline:.6f}",
        f"  No-recurrence MSE: {phase_b_no_recurrence_metrics.mse:.6f}",
        f"  Block 1 beats copy baseline: {phase_b_metrics.beats_copy}",
    ]
    print("\n".join(summary_lines))

    git_status_short = current_git_status_short()
    report = {
        "config": {
            "sanity_check_only": args.sanity_check_only,
            "block0_steps": block0_steps,
            "block1_steps": block1_steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "bptt_chunk": args.bptt_chunk,
            "d_model": args.d_model,
            "block0_lr": block0_lr,
            "block1_lr": block1_lr,
            "seed": args.seed,
            "temperature": TEMPERATURE,
            "normalize": NORMALIZE,
            "fixed_embeddings": True,
            "train_characters": TRAIN_CHARACTERS,
            "val_characters": VAL_CHARACTERS,
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
            "train_sequences": int(train_sequences.shape[0]),
            "val_sequences": int(val_sequences.shape[0]),
        },
        "model": {
            "block0_parameters": count_parameters(block0),
            "block1_parameters": count_parameters(block1),
            "block1_no_recurrence_parameters": count_parameters(block1_no_recurrence),
        },
        "results": {
            "phase_a": {
                "initial_eval_ce": phase_a_summary.initial_eval_loss,
                "final_train_ce": phase_a_summary.final_train_loss,
                "final_eval_ce": phase_a_summary.final_eval_loss,
                "loss_went_down": phase_a_summary.final_eval_loss < phase_a_summary.initial_eval_loss,
                "wall_seconds": phase_a_wall_seconds,
                "checkpoints": phase_checkpoints_to_payload(phase_a_summary.checkpoints),
            },
            "phase_b": {
                "initial_eval_mse": phase_b_summary.initial_eval_loss,
                "final_train_mse": phase_b_summary.final_train_loss,
                "final_eval_mse": phase_b_metrics.mse,
                "copy_baseline": phase_b_metrics.copy_baseline,
                "random_baseline": phase_b_metrics.random_baseline,
                "beats_copy": phase_b_metrics.beats_copy,
                "loss_went_down": phase_b_metrics.mse < phase_b_summary.initial_eval_loss,
                "wall_seconds": phase_b_wall_seconds,
                "checkpoints": phase_checkpoints_to_payload(phase_b_summary.checkpoints),
            },
            "phase_b_no_recurrence": {
                "initial_eval_mse": phase_b_no_recurrence_summary.initial_eval_loss,
                "final_train_mse": phase_b_no_recurrence_summary.final_train_loss,
                "final_eval_mse": phase_b_no_recurrence_metrics.mse,
                "copy_baseline": phase_b_no_recurrence_metrics.copy_baseline,
                "random_baseline": phase_b_no_recurrence_metrics.random_baseline,
                "beats_copy": phase_b_no_recurrence_metrics.beats_copy,
                "loss_went_down": phase_b_no_recurrence_metrics.mse < phase_b_no_recurrence_summary.initial_eval_loss,
                "wall_seconds": phase_b_no_recurrence_wall_seconds,
                "checkpoints": phase_checkpoints_to_payload(phase_b_no_recurrence_summary.checkpoints),
            },
        },
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "run_completed",
            "report_path": str(args.report_path),
            "phase_a_final_eval_ce": round(phase_a_summary.final_eval_loss, 6),
            "phase_b_final_eval_mse": round(phase_b_metrics.mse, 6),
            "phase_b_copy_baseline": round(phase_b_metrics.copy_baseline, 6),
            "phase_b_random_baseline": round(phase_b_metrics.random_baseline, 6),
            "phase_b_beats_copy": phase_b_metrics.beats_copy,
            "phase_b_no_recurrence_eval_mse": round(phase_b_no_recurrence_metrics.mse, 6),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
