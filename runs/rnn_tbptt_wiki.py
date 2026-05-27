from __future__ import annotations

import argparse
import atexit
import shutil
import sys
import tempfile
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    raise RuntimeError(
        "Run from the repo root with `.\\.venv\\Scripts\\python.exe -m runs.rnn_tbptt_wiki ...` so `core` imports resolve cleanly."
    )

import torch
from torch import Tensor
from torch.nn import functional as F

from core.dataset import UNKNOWN_CHAR_TOKEN, download_wikitext_103_raw
from core.fixed_window_char import set_seed
from core.model import count_parameters
from core.run_utils import (
    append_log,
    build_optimizer,
    log_run_restarted,
    prepare_output_paths,
    redirect_sanity_check_paths,
    register_active_lock,
    release_memory,
    resolve_device,
)
from core.training import current_git_sha, current_git_status_short, write_json
from runs.rnn_tbptt import (
    GruLanguageModel,
    build_tbptt_sequences,
    cross_entropy_all_positions,
    evaluate_sequential_loss,
    fixed_tbptt_batch,
    positive_float,
    positive_int,
    sample_tbptt_batch,
    tbptt_training_step,
    validate_args as validate_tbptt_args,
)

DEFAULT_TRAIN_CHARACTERS = 10_000_000
DEFAULT_VAL_CHARACTERS = 50_000
SANITY_TRAIN_CHARACTERS = 10_000
SANITY_VAL_CHARACTERS = 2_000

DEFAULT_STEPS = 2_000
SANITY_STEPS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEQ_LEN = 2_048
DEFAULT_BPTT_CHUNK = 256
DEFAULT_EMBED_DIM = 128
DEFAULT_HIDDEN_DIM = 864
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_EVAL_INTERVAL = 100
SANITY_EVAL_INTERVAL = 10
DEFAULT_SEED = 42

FULL_RESET_HORIZONS = (1, 8, 32, 128, 256, 512, 2_048, 8_192, "full")
SANITY_RESET_HORIZONS = (1, 128, "full")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "wikitext" / "artifacts" / "rnn_tbptt_wiki"
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tbptt", "chunk-reset", "both"), required=True)
    parser.add_argument("--steps", type=positive_int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq-len", type=positive_int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--bptt-chunk", type=positive_int, default=DEFAULT_BPTT_CHUNK)
    parser.add_argument("--embed-dim", type=positive_int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--hidden-dim", type=positive_int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--lr", type=positive_float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--eval-interval", type=positive_int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sanity-check-only", action="store_true")
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.jsonl")
    parser.set_defaults(no_lock=True)
    parser.add_argument("--lock", dest="no_lock", action="store_false")
    parser.add_argument("--no-lock", dest="no_lock", action="store_true")
    return parser.parse_args()


def resolve_steps(args: argparse.Namespace) -> int:
    return SANITY_STEPS if args.sanity_check_only else args.steps


def resolve_eval_interval(args: argparse.Namespace) -> int:
    return SANITY_EVAL_INTERVAL if args.sanity_check_only else args.eval_interval


def resolve_train_characters(args: argparse.Namespace) -> int:
    return SANITY_TRAIN_CHARACTERS if args.sanity_check_only else DEFAULT_TRAIN_CHARACTERS


def resolve_val_characters(args: argparse.Namespace) -> int:
    return SANITY_VAL_CHARACTERS if args.sanity_check_only else DEFAULT_VAL_CHARACTERS


def resolve_reset_horizons(args: argparse.Namespace) -> tuple[int | str, ...]:
    return SANITY_RESET_HORIZONS if args.sanity_check_only else FULL_RESET_HORIZONS


def apply_sanity_checkpoint_redirection(args: argparse.Namespace) -> None:
    if not args.sanity_check_only:
        return
    checkpoint_dir = Path(tempfile.mkdtemp(prefix="gildnn_rnn_tbptt_wiki_ckpt_"))

    def _cleanup() -> None:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    atexit.register(_cleanup)
    args.checkpoint_dir = checkpoint_dir


def build_char_vocab(train_text: str) -> tuple[dict[str, int], dict[int, str]]:
    vocabulary = sorted(set(train_text))
    if UNKNOWN_CHAR_TOKEN in vocabulary:
        raise ValueError(
            f"Training corpus contains reserved token {UNKNOWN_CHAR_TOKEN!r}; choose a different unknown token strategy."
        )
    vocabulary.append(UNKNOWN_CHAR_TOKEN)
    char_to_idx = {char: index for index, char in enumerate(vocabulary)}
    idx_to_char = {index: char for char, index in char_to_idx.items()}
    return char_to_idx, idx_to_char


def encode_text(text: str, *, char_to_idx: dict[str, int]) -> tuple[Tensor, int]:
    unknown_index = char_to_idx[UNKNOWN_CHAR_TOKEN]
    unknown_count = 0
    encoded: list[int] = []
    for char in text:
        token = char_to_idx.get(char)
        if token is None:
            token = unknown_index
            unknown_count += 1
        encoded.append(token)
    return torch.tensor(encoded, dtype=torch.long), unknown_count


def load_wikitext_subset(
    *,
    train_characters: int,
    val_characters: int,
) -> dict[str, object]:
    corpus_paths = download_wikitext_103_raw()
    train_path = corpus_paths["wiki.train.raw"]
    valid_path = corpus_paths["wiki.valid.raw"]

    train_text = train_path.read_text(encoding="utf-8")
    valid_text = valid_path.read_text(encoding="utf-8")
    if len(train_text) < train_characters:
        raise ValueError(f"Need at least {train_characters} training characters, got {len(train_text)}.")
    if len(valid_text) < val_characters:
        raise ValueError(f"Need at least {val_characters} validation characters, got {len(valid_text)}.")

    train_slice = train_text[:train_characters]
    val_slice = valid_text[:val_characters]
    char_to_idx, idx_to_char = build_char_vocab(train_slice)
    train_encoded, train_unknowns = encode_text(train_slice, char_to_idx=char_to_idx)
    val_encoded, val_unknowns = encode_text(val_slice, char_to_idx=char_to_idx)

    return {
        "corpus_paths": {name: str(path) for name, path in corpus_paths.items()},
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "train_characters": len(train_slice),
        "val_characters": len(val_slice),
        "train_encoded": train_encoded,
        "val_encoded": val_encoded,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "vocab_size": len(char_to_idx),
        "train_unknowns": train_unknowns,
        "val_unknowns": val_unknowns,
    }


def cpu_state_dict(model: GruLanguageModel) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def checkpoint_path_for_mode(args: argparse.Namespace, mode: str) -> Path:
    checkpoint_dir: Path = args.checkpoint_dir
    return checkpoint_dir / f"best_model_{mode}.pt"


def save_best_checkpoint(
    checkpoint_path: Path,
    *,
    model: GruLanguageModel,
    mode: str,
    vocab_size: int,
    embed_dim: int,
    hidden_dim: int,
    parameter_count: int,
    best_step: int,
    best_val_loss: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mode": mode,
            "model_state_dict": cpu_state_dict(model),
            "model_config": {
                "vocab_size": vocab_size,
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
            },
            "parameter_count": parameter_count,
            "best_step": best_step,
            "best_val_loss": round(best_val_loss, 6),
        },
        checkpoint_path,
    )


@torch.inference_mode()
def evaluate_reset_horizon_loss(
    model: GruLanguageModel,
    *,
    encoded_tokens: Tensor,
    device: torch.device,
    reset_horizon: int,
    eval_chunk_size: int,
) -> tuple[float, int, int]:
    if encoded_tokens.numel() < 2:
        raise ValueError("Reset-sweep evaluation requires at least 2 tokens.")
    if reset_horizon <= 0:
        raise ValueError(f"reset_horizon must be positive, got {reset_horizon}.")

    was_training = model.training
    model.eval()
    inputs = encoded_tokens[:-1]
    targets = encoded_tokens[1:]
    hidden_state: Tensor | None = None
    total_loss = 0.0
    total_tokens = 0
    ignored_tokens = 0
    span_progress = 0
    pin_memory = device.type == "cuda"

    start = 0
    while start < inputs.numel():
        remaining_in_span = reset_horizon - span_progress
        chunk_stop = min(start + eval_chunk_size, start + remaining_in_span, inputs.numel())
        chunk_inputs = inputs[start:chunk_stop].unsqueeze(0)
        chunk_targets = targets[start:chunk_stop].unsqueeze(0)
        if pin_memory:
            chunk_inputs = chunk_inputs.pin_memory()
            chunk_targets = chunk_targets.pin_memory()
        chunk_inputs = chunk_inputs.to(device=device, dtype=torch.long, non_blocking=pin_memory)
        chunk_targets = chunk_targets.to(device=device, dtype=torch.long, non_blocking=pin_memory)

        if span_progress == 0:
            hidden_state = None
        logits, hidden_state = model(chunk_inputs, hidden_state)
        flat_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            chunk_targets.reshape(-1),
            reduction="none",
        )

        if reset_horizon == 1:
            total_loss += flat_losses.sum().item()
            total_tokens += flat_losses.numel()
        else:
            keep_mask = torch.ones(flat_losses.shape[0], dtype=torch.bool, device=flat_losses.device)
            if span_progress == 0:
                keep_mask[0] = False
                ignored_tokens += 1
            total_loss += flat_losses[keep_mask].sum().item()
            total_tokens += int(keep_mask.sum().item())

        span_progress += chunk_stop - start
        if span_progress == reset_horizon:
            span_progress = 0
        start = chunk_stop

    if total_tokens == 0:
        raise RuntimeError(f"Reset horizon {reset_horizon} left no evaluable tokens.")
    if was_training:
        model.train()
    return total_loss / total_tokens, total_tokens, ignored_tokens


def sweep_reset_horizons(
    *,
    model: GruLanguageModel,
    encoded_tokens: Tensor,
    device: torch.device,
    bptt_chunk: int,
    reset_horizons: tuple[int | str, ...],
    log_path: Path,
    mode: str,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    results: dict[str, float] = {}
    details: list[dict[str, object]] = []
    full_horizon = int(encoded_tokens.numel())
    for requested_horizon in reset_horizons:
        if requested_horizon == "full":
            effective_horizon = full_horizon
            label = "full"
        else:
            effective_horizon = min(int(requested_horizon), full_horizon)
            label = str(effective_horizon)
        loss, tokens_evaluated, ignored_tokens = evaluate_reset_horizon_loss(
            model,
            encoded_tokens=encoded_tokens,
            device=device,
            reset_horizon=effective_horizon,
            eval_chunk_size=bptt_chunk,
        )
        rounded_loss = round(loss, 6)
        results[label] = rounded_loss
        payload = {
            "mode": mode,
            "reset_horizon": label,
            "val_loss": rounded_loss,
            "tokens_evaluated": tokens_evaluated,
            "ignored_tokens": ignored_tokens,
        }
        details.append(payload)
        append_log(log_path, {"stage": "reset_sweep_result", **payload})
    return results, details


def train_mode(
    *,
    args: argparse.Namespace,
    mode: str,
    device: torch.device,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_encoded: Tensor,
    steps: int,
    eval_interval: int,
    vocab_size: int,
) -> dict[str, object]:
    set_seed(args.seed)
    model = GruLanguageModel(vocab_size=vocab_size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim).to(device)
    parameter_count = count_parameters(model)
    optimizer = build_optimizer(
        model,
        device=device,
        compile_model=False,
        learning_rate=args.lr,
    )

    fixed_batch_inputs, fixed_batch_targets = fixed_tbptt_batch(
        train_inputs,
        train_targets,
        batch_size=args.batch_size,
        device=device,
    )
    tbptt_rng = torch.Generator(device="cpu")
    tbptt_rng.manual_seed(args.seed)
    carry_state = mode == "tbptt"

    append_log(
        args.log_path,
        {
            "stage": "mode_started",
            "mode": mode,
            "steps": steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "bptt_chunk": args.bptt_chunk,
            "lr": args.lr,
            "carry_state": carry_state,
            "parameter_count": parameter_count,
        },
    )

    checkpoints: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_step = 0
    final_train_loss = float("nan")
    final_val_loss = float("nan")
    started_at = perf_counter()

    for step in range(1, steps + 1):
        if args.sanity_check_only:
            batch_inputs = fixed_batch_inputs
            batch_targets = fixed_batch_targets
        else:
            batch_inputs, batch_targets = sample_tbptt_batch(
                train_inputs,
                train_targets,
                batch_size=args.batch_size,
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
            carry_state=carry_state,
        )

        if step % eval_interval != 0 and step != steps:
            continue

        sanity_batch_logits, _ = model(fixed_batch_inputs)
        sanity_batch_loss = cross_entropy_all_positions(
            sanity_batch_logits,
            fixed_batch_targets,
            reduction="mean",
        ).item()
        final_val_loss = evaluate_sequential_loss(
            model,
            encoded_tokens=val_encoded,
            device=device,
            chunk_size=args.bptt_chunk,
        )
        checkpoint = {
            "step": step,
            "train_loss": round(final_train_loss, 6),
            "sanity_batch_loss": round(sanity_batch_loss, 6),
            "val_loss": round(final_val_loss, 6),
        }
        checkpoints.append(checkpoint)
        append_log(args.log_path, {"stage": "checkpoint", "mode": mode, **checkpoint})

        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_step = step
            save_best_checkpoint(
                checkpoint_path_for_mode(args, mode),
                model=model,
                mode=mode,
                vocab_size=vocab_size,
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                parameter_count=parameter_count,
                best_step=best_step,
                best_val_loss=best_val_loss,
            )

    best_checkpoint = torch.load(checkpoint_path_for_mode(args, mode), map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    wall_seconds = round(perf_counter() - started_at, 6)
    append_log(
        args.log_path,
        {
            "stage": "mode_completed",
            "mode": mode,
            "best_step": best_step,
            "best_val_loss": round(best_val_loss, 6),
            "final_train_loss": round(final_train_loss, 6),
            "final_val_loss": round(final_val_loss, 6),
            "wall_seconds": wall_seconds,
            "checkpoint_path": str(checkpoint_path_for_mode(args, mode)),
        },
    )

    return {
        "mode": mode,
        "parameter_count": parameter_count,
        "best_step": best_step,
        "best_val_loss": round(best_val_loss, 6),
        "final_train_loss": round(final_train_loss, 6),
        "final_val_loss": round(final_val_loss, 6),
        "wall_seconds": wall_seconds,
        "checkpoint_path": str(checkpoint_path_for_mode(args, mode)),
        "checkpoints": checkpoints,
        "best_state_dict": best_checkpoint["model_state_dict"],
    }


def comparison_summary(mode_reports: dict[str, dict[str, object]], reset_sweeps: dict[str, dict[str, float]]) -> dict[str, object] | None:
    if "tbptt" not in mode_reports or "chunk-reset" not in mode_reports:
        return None
    sweep_deltas = {
        horizon: round(reset_sweeps["tbptt"][horizon] - reset_sweeps["chunk-reset"][horizon], 6)
        for horizon in reset_sweeps["tbptt"]
        if horizon in reset_sweeps["chunk-reset"]
    }
    return {
        "best_val_loss_delta_tbptt_minus_chunk_reset": round(
            float(mode_reports["tbptt"]["best_val_loss"]) - float(mode_reports["chunk-reset"]["best_val_loss"]),
            6,
        ),
        "reset_sweep_delta_tbptt_minus_chunk_reset": sweep_deltas,
    }


def main() -> int:
    args = parse_args()
    validate_tbptt_args(args)

    args.checkpoint_dir = args.report_path.parent
    if args.sanity_check_only:
        redirect_sanity_check_paths(args)
    apply_sanity_checkpoint_redirection(args)

    steps = resolve_steps(args)
    eval_interval = resolve_eval_interval(args)
    reset_horizons = resolve_reset_horizons(args)
    train_characters = resolve_train_characters(args)
    val_characters = resolve_val_characters(args)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    prepare_output_paths(report_path=args.report_path, log_path=args.log_path)
    log_run_restarted(args.log_path)
    register_active_lock(
        experiment_name="rnn_tbptt_wiki",
        variants=[args.mode, f"ed{args.embed_dim}", f"hd{args.hidden_dim}", f"seq{args.seq_len}", f"chunk{args.bptt_chunk}"],
        enabled=not args.no_lock,
    )

    data = load_wikitext_subset(train_characters=train_characters, val_characters=val_characters)
    train_inputs, train_targets = build_tbptt_sequences(data["train_encoded"], seq_len=args.seq_len)
    val_eval_tokens = data["val_encoded"]
    modes = ("tbptt", "chunk-reset") if args.mode == "both" else (args.mode,)

    append_log(
        args.log_path,
        {
            "stage": "run_started",
            "mode": args.mode,
            "device": str(device),
            "sanity_check_only": args.sanity_check_only,
            "steps": steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "bptt_chunk": args.bptt_chunk,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "eval_interval": eval_interval,
            "seed": args.seed,
            "reset_horizons": list(reset_horizons),
            "train_characters": train_characters,
            "val_characters": val_characters,
            "no_lock": args.no_lock,
        },
    )
    append_log(
        args.log_path,
        {
            "stage": "dataset_loaded",
            "train_path": data["train_path"],
            "valid_path": data["valid_path"],
            "train_tokens": int(data["train_encoded"].numel()),
            "val_tokens": int(data["val_encoded"].numel()),
            "vocab_size": int(data["vocab_size"]),
            "tbptt_sequences": int(train_inputs.shape[0]),
            "train_unknowns": int(data["train_unknowns"]),
            "val_unknowns": int(data["val_unknowns"]),
        },
    )

    mode_reports: dict[str, dict[str, object]] = {}
    reset_sweeps: dict[str, dict[str, float]] = {}
    reset_sweep_details: list[dict[str, object]] = []
    started_at = perf_counter()

    for mode in modes:
        mode_report = train_mode(
            args=args,
            mode=mode,
            device=device,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_encoded=val_eval_tokens,
            steps=steps,
            eval_interval=eval_interval,
            vocab_size=int(data["vocab_size"]),
        )
        best_state_dict = mode_report.pop("best_state_dict")
        model = GruLanguageModel(
            vocab_size=int(data["vocab_size"]),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        model.load_state_dict(best_state_dict)
        reset_results, reset_details = sweep_reset_horizons(
            model=model,
            encoded_tokens=val_eval_tokens,
            device=device,
            bptt_chunk=args.bptt_chunk,
            reset_horizons=reset_horizons,
            log_path=args.log_path,
            mode=mode,
        )
        mode_reports[mode] = mode_report
        reset_sweeps[mode] = reset_results
        reset_sweep_details.extend(reset_details)

        del model
        release_memory(device=device)

    wall_seconds = round(perf_counter() - started_at, 6)
    git_status_short = current_git_status_short()
    report = {
        "config": {
            "mode": args.mode,
            "sanity_check_only": args.sanity_check_only,
            "steps": steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "bptt_chunk": args.bptt_chunk,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.lr,
            "eval_interval": eval_interval,
            "seed": args.seed,
            "reset_horizons": ["full" if horizon == "full" else int(horizon) for horizon in reset_horizons],
            "no_lock": args.no_lock,
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
            "corpus_paths": data["corpus_paths"],
            "train_path": data["train_path"],
            "valid_path": data["valid_path"],
            "train_characters": int(data["train_characters"]),
            "val_characters": int(data["val_characters"]),
            "train_tokens": int(data["train_encoded"].numel()),
            "val_tokens": int(data["val_encoded"].numel()),
            "vocab_size": int(data["vocab_size"]),
            "tbptt_sequences": int(train_inputs.shape[0]),
            "train_unknowns": int(data["train_unknowns"]),
            "val_unknowns": int(data["val_unknowns"]),
        },
        "model": {
            "parameter_count": int(next(iter(mode_reports.values()))["parameter_count"]),
        },
        "training": mode_reports,
        "reset_sweep": reset_sweeps,
        "reset_sweep_details": reset_sweep_details,
        "comparison": comparison_summary(mode_reports, reset_sweeps),
        "wall_seconds": wall_seconds,
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "run_completed",
            "report_path": str(args.report_path),
            "wall_seconds": wall_seconds,
            "comparison": report["comparison"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
