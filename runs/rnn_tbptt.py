from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    raise RuntimeError(
        "Run from the repo root with `.\\.venv\\Scripts\\python.exe -m runs.rnn_tbptt ...` so `core` imports resolve cleanly."
    )

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.dataset import CorpusData, RandomWindowCharDataset, load_corpus
from core.fixed_window_char import set_seed
from core.model import count_parameters
from core.run_utils import (
    append_log,
    build_optimizer,
    log_run_restarted,
    prepare_output_paths,
    random_batches,
    redirect_sanity_check_paths,
    register_active_lock,
    resolve_device,
)
from core.training import current_git_sha, current_git_status_short, write_json

TRAIN_CHARACTERS = 100_000
VAL_CHARACTERS = 20_000
DEFAULT_STEPS = 400
SANITY_CHECK_STEPS = 120
DEFAULT_BATCH_SIZE = 16
SANITY_CHECK_BATCH_SIZE = 1
DEFAULT_SEQ_LEN = 2048
DEFAULT_BPTT_CHUNK = 128
DEFAULT_EMBED_DIM = 64
DEFAULT_HIDDEN_DIM = 208
DEFAULT_LEARNING_RATE = 3e-3
SANITY_CHECK_LEARNING_RATE = 2e-2
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_EVAL_INTERVAL = 25
SANITY_CHECK_EVAL_INTERVAL = 10
DEFAULT_SEED = 42


@dataclass(frozen=True)
class TinyShakespeareData:
    corpus: CorpusData
    train_encoded: Tensor
    val_encoded: Tensor
    source_path: Path


class GruLanguageModel(nn.Module):
    def __init__(self, *, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        tokens: Tensor,
        hidden_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        embedded = self.embedding(tokens)
        outputs, next_hidden_state = self.gru(embedded, hidden_state)
        return self.readout(outputs), next_hidden_state


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
    artifact_dir = repo_root / "experiments" / "tinyshakespeare" / "artifacts" / "rnn_tbptt"
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tbptt", "window"), required=True)
    parser.add_argument("--steps", type=positive_int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq-len", type=positive_int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--bptt-chunk", type=positive_int, default=DEFAULT_BPTT_CHUNK)
    parser.add_argument("--embed-dim", type=positive_int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--hidden-dim", type=positive_int, default=DEFAULT_HIDDEN_DIM)
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


def resolve_eval_interval(args: argparse.Namespace) -> int:
    return SANITY_CHECK_EVAL_INTERVAL if args.sanity_check_only else DEFAULT_EVAL_INTERVAL


def resolve_learning_rate(args: argparse.Namespace) -> float:
    return SANITY_CHECK_LEARNING_RATE if args.sanity_check_only else args.lr


def validate_args(args: argparse.Namespace) -> None:
    if args.bptt_chunk > args.seq_len:
        raise ValueError(f"bptt_chunk must be <= seq_len, got bptt_chunk={args.bptt_chunk}, seq_len={args.seq_len}.")


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

    with tempfile.TemporaryDirectory(prefix="gildnn_rnn_tbptt_corpus_") as tmp_dir:
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


def build_tbptt_sequences(encoded: Tensor, *, seq_len: int) -> tuple[Tensor, Tensor]:
    tokens_per_sequence = seq_len + 1
    num_sequences = encoded.numel() // tokens_per_sequence
    if num_sequences == 0:
        raise ValueError(
            "Training corpus is too short for TBPTT sequence construction. "
            f"Got {encoded.numel()} tokens and seq_len={seq_len}."
        )
    usable = num_sequences * tokens_per_sequence
    sequences = encoded[:usable].view(num_sequences, tokens_per_sequence)
    return sequences[:, :-1].clone(), sequences[:, 1:].clone()


def sample_tbptt_batch(
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    rng: torch.Generator,
) -> tuple[Tensor, Tensor]:
    indices = torch.randint(0, inputs.shape[0], (batch_size,), generator=rng)
    pin_memory = device.type == "cuda"
    batch_inputs = inputs[indices]
    batch_targets = targets[indices]
    if pin_memory:
        batch_inputs = batch_inputs.pin_memory()
        batch_targets = batch_targets.pin_memory()
    return (
        batch_inputs.to(device=device, dtype=torch.long, non_blocking=pin_memory),
        batch_targets.to(device=device, dtype=torch.long, non_blocking=pin_memory),
    )


def fixed_tbptt_batch(
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    indices = torch.arange(batch_size, dtype=torch.long) % inputs.shape[0]
    pin_memory = device.type == "cuda"
    batch_inputs = inputs[indices]
    batch_targets = targets[indices]
    if pin_memory:
        batch_inputs = batch_inputs.pin_memory()
        batch_targets = batch_targets.pin_memory()
    return (
        batch_inputs.to(device=device, dtype=torch.long, non_blocking=pin_memory),
        batch_targets.to(device=device, dtype=torch.long, non_blocking=pin_memory),
    )


def expand_window_targets(window_inputs: Tensor, next_targets: Tensor) -> Tensor:
    return torch.cat((window_inputs[:, 1:], next_targets[:, None]), dim=1)


def cross_entropy_all_positions(logits: Tensor, targets: Tensor, *, reduction: str) -> Tensor:
    vocab_size = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction=reduction)


def tbptt_training_step(
    model: GruLanguageModel,
    optimizer: torch.optim.Optimizer,
    *,
    batch_inputs: Tensor,
    batch_targets: Tensor,
    bptt_chunk: int,
    grad_clip_norm: float,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    hidden_state: Tensor | None = None
    total_loss = torch.zeros((), device=batch_inputs.device)
    total_tokens = 0
    for start in range(0, batch_inputs.shape[1], bptt_chunk):
        stop = min(start + bptt_chunk, batch_inputs.shape[1])
        logits, hidden_state = model(batch_inputs[:, start:stop], hidden_state)
        chunk_targets = batch_targets[:, start:stop]
        chunk_loss = cross_entropy_all_positions(logits, chunk_targets, reduction="sum")
        if not torch.isfinite(chunk_loss):
            raise RuntimeError("TBPTT training diverged: loss is NaN or Inf.")
        chunk_loss.backward()
        total_loss = total_loss + chunk_loss.detach()
        total_tokens += chunk_targets.numel()
        hidden_state = hidden_state.detach()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return (total_loss / total_tokens).item()


def window_training_step(
    model: GruLanguageModel,
    optimizer: torch.optim.Optimizer,
    *,
    window_inputs: Tensor,
    next_targets: Tensor,
    grad_clip_norm: float,
) -> float:
    full_targets = expand_window_targets(window_inputs, next_targets)
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model(window_inputs)
    loss = cross_entropy_all_positions(logits, full_targets, reduction="mean")
    if not torch.isfinite(loss):
        raise RuntimeError("Window training diverged: loss is NaN or Inf.")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return loss.item()


@torch.inference_mode()
def evaluate_tbptt_batch_loss(
    model: GruLanguageModel,
    *,
    batch_inputs: Tensor,
    batch_targets: Tensor,
    bptt_chunk: int,
) -> float:
    was_training = model.training
    model.eval()
    hidden_state: Tensor | None = None
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, batch_inputs.shape[1], bptt_chunk):
        stop = min(start + bptt_chunk, batch_inputs.shape[1])
        logits, hidden_state = model(batch_inputs[:, start:stop], hidden_state)
        chunk_targets = batch_targets[:, start:stop]
        total_loss += cross_entropy_all_positions(logits, chunk_targets, reduction="sum").item()
        total_tokens += chunk_targets.numel()
    if was_training:
        model.train()
    return total_loss / total_tokens


@torch.inference_mode()
def evaluate_window_batch_loss(
    model: GruLanguageModel,
    *,
    window_inputs: Tensor,
    next_targets: Tensor,
) -> float:
    was_training = model.training
    model.eval()
    logits, _ = model(window_inputs)
    full_targets = expand_window_targets(window_inputs, next_targets)
    loss = cross_entropy_all_positions(logits, full_targets, reduction="mean").item()
    if was_training:
        model.train()
    return loss


@torch.inference_mode()
def evaluate_sequential_loss(
    model: GruLanguageModel,
    *,
    encoded_tokens: Tensor,
    device: torch.device,
    chunk_size: int,
) -> float:
    if encoded_tokens.numel() < 2:
        raise ValueError("Sequential evaluation requires at least 2 tokens.")
    was_training = model.training
    model.eval()
    inputs = encoded_tokens[:-1]
    targets = encoded_tokens[1:]
    hidden_state: Tensor | None = None
    total_loss = 0.0
    total_tokens = 0
    pin_memory = device.type == "cuda"
    for start in range(0, inputs.numel(), chunk_size):
        stop = min(start + chunk_size, inputs.numel())
        chunk_inputs = inputs[start:stop].unsqueeze(0)
        chunk_targets = targets[start:stop].unsqueeze(0)
        if pin_memory:
            chunk_inputs = chunk_inputs.pin_memory()
            chunk_targets = chunk_targets.pin_memory()
        chunk_inputs = chunk_inputs.to(device=device, dtype=torch.long, non_blocking=pin_memory)
        chunk_targets = chunk_targets.to(device=device, dtype=torch.long, non_blocking=pin_memory)
        logits, hidden_state = model(chunk_inputs, hidden_state)
        total_loss += cross_entropy_all_positions(logits, chunk_targets, reduction="sum").item()
        total_tokens += chunk_targets.numel()
    if was_training:
        model.train()
    return total_loss / total_tokens


def checkpoint_payload(
    *,
    step: int,
    train_loss: float,
    sanity_batch_loss: float,
    val_loss: float,
) -> dict[str, float | int]:
    return {
        "step": step,
        "train_loss": round(train_loss, 6),
        "sanity_batch_loss": round(sanity_batch_loss, 6),
        "val_loss": round(val_loss, 6),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)

    if args.sanity_check_only:
        redirect_sanity_check_paths(args)

    steps = resolve_steps(args)
    batch_size = resolve_batch_size(args)
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
        experiment_name="rnn_tbptt",
        variants=[args.mode, f"ed{args.embed_dim}", f"hd{args.hidden_dim}", f"seq{args.seq_len}", f"chunk{args.bptt_chunk}"],
        enabled=not args.no_lock,
    )

    repo_root = Path(__file__).resolve().parents[1]
    set_seed(args.seed)
    data = prepare_tinyshakespeare_data(repo_root=repo_root, seq_len=args.seq_len)
    tbptt_inputs, tbptt_targets = build_tbptt_sequences(data.train_encoded, seq_len=args.seq_len)

    model = GruLanguageModel(
        vocab_size=data.corpus.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    parameter_count = count_parameters(model)
    optimizer = build_optimizer(
        model,
        device=device,
        compile_model=False,
        learning_rate=learning_rate,
    )

    append_log(
        args.log_path,
        {
            "stage": "run_started",
            "mode": args.mode,
            "device": str(device),
            "sanity_check_only": args.sanity_check_only,
            "steps": steps,
            "batch_size": batch_size,
            "seq_len": args.seq_len,
            "bptt_chunk": args.bptt_chunk,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "lr": learning_rate,
            "seed": args.seed,
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
            "tbptt_sequences": int(tbptt_inputs.shape[0]),
        },
    )
    append_log(
        args.log_path,
        {
            "stage": "model_built",
            "parameter_count": parameter_count,
        },
    )

    window_rng = torch.Generator(device="cpu")
    window_rng.manual_seed(args.seed)
    window_iterator = random_batches(
        data.train_encoded,
        context_size=args.bptt_chunk,
        batch_size=batch_size,
        device=device,
        rng=window_rng,
    )
    tbptt_rng = torch.Generator(device="cpu")
    tbptt_rng.manual_seed(args.seed)

    if args.mode == "tbptt":
        fixed_batch_inputs, fixed_batch_targets = fixed_tbptt_batch(
            tbptt_inputs,
            tbptt_targets,
            batch_size=batch_size,
            device=device,
        )
    else:
        fixed_batch_inputs, fixed_batch_next_targets = next(window_iterator)

    checkpoints: list[dict[str, float | int]] = []
    started_at = perf_counter()
    early_stopped = False
    final_train_loss = float("nan")
    final_sanity_batch_loss = float("nan")
    final_val_loss = float("nan")
    final_step = 0

    for step in range(1, steps + 1):
        if args.mode == "tbptt":
            if args.sanity_check_only:
                batch_inputs = fixed_batch_inputs
                batch_targets = fixed_batch_targets
            else:
                batch_inputs, batch_targets = sample_tbptt_batch(
                    tbptt_inputs,
                    tbptt_targets,
                    batch_size=batch_size,
                    device=device,
                    rng=tbptt_rng,
                )
            final_train_loss = tbptt_training_step(
                model,
                optimizer,
                batch_inputs=batch_inputs,
                batch_targets=batch_targets,
                bptt_chunk=args.bptt_chunk,
                grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
            )
        else:
            if args.sanity_check_only:
                batch_inputs = fixed_batch_inputs
                batch_next_targets = fixed_batch_next_targets
            else:
                batch_inputs, batch_next_targets = next(window_iterator)
            final_train_loss = window_training_step(
                model,
                optimizer,
                window_inputs=batch_inputs,
                next_targets=batch_next_targets,
                grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
            )

        should_eval = step % eval_interval == 0 or step == steps
        if not should_eval:
            continue

        if args.mode == "tbptt":
            final_sanity_batch_loss = evaluate_tbptt_batch_loss(
                model,
                batch_inputs=fixed_batch_inputs,
                batch_targets=fixed_batch_targets,
                bptt_chunk=args.bptt_chunk,
            )
        else:
            final_sanity_batch_loss = evaluate_window_batch_loss(
                model,
                window_inputs=fixed_batch_inputs,
                next_targets=fixed_batch_next_targets,
            )
        final_val_loss = evaluate_sequential_loss(
            model,
            encoded_tokens=data.val_encoded,
            device=device,
            chunk_size=args.bptt_chunk,
        )
        final_step = step
        checkpoint = checkpoint_payload(
            step=step,
            train_loss=final_train_loss,
            sanity_batch_loss=final_sanity_batch_loss,
            val_loss=final_val_loss,
        )
        checkpoints.append(checkpoint)
        append_log(
            args.log_path,
            {
                "stage": "checkpoint",
                "mode": args.mode,
                **checkpoint,
            },
        )
        if args.sanity_check_only and final_sanity_batch_loss < 0.05:
            early_stopped = True
            break

    wall_seconds = round(perf_counter() - started_at, 6)
    git_status_short = current_git_status_short()
    report = {
        "config": {
            "mode": args.mode,
            "sanity_check_only": args.sanity_check_only,
            "steps": steps,
            "batch_size": batch_size,
            "seq_len": args.seq_len,
            "bptt_chunk": args.bptt_chunk,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.lr,
            "effective_learning_rate": learning_rate,
            "seed": args.seed,
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
            "tbptt_sequences": int(tbptt_inputs.shape[0]),
        },
        "model": {
            "parameter_count": parameter_count,
        },
        "results": {
            "final_step": final_step,
            "final_train_loss": round(final_train_loss, 6),
            "final_sanity_batch_loss": round(final_sanity_batch_loss, 6),
            "final_val_loss": round(final_val_loss, 6),
            "early_stopped": early_stopped,
            "wall_seconds": wall_seconds,
        },
        "checkpoints": checkpoints,
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "run_completed",
            "mode": args.mode,
            "report_path": str(args.report_path),
            "final_step": final_step,
            "final_train_loss": round(final_train_loss, 6),
            "final_sanity_batch_loss": round(final_sanity_batch_loss, 6),
            "final_val_loss": round(final_val_loss, 6),
            "early_stopped": early_stopped,
            "wall_seconds": wall_seconds,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
