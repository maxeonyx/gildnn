from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    raise RuntimeError(
        "Run from the repo root with `.\\.venv\\Scripts\\python.exe -m runs.rnn_reset_sweep ...` so `core` imports resolve cleanly."
    )

import torch
from torch import Tensor
from torch.nn import functional as F

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
    release_memory,
    resolve_device,
)
from core.training import current_git_sha, current_git_status_short, write_json
from runs.rnn_tbptt import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BPTT_CHUNK,
    DEFAULT_EMBED_DIM,
    DEFAULT_EVAL_INTERVAL,
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_SEQ_LEN,
    GruLanguageModel,
    SANITY_CHECK_BATCH_SIZE,
    SANITY_CHECK_EVAL_INTERVAL,
    SANITY_CHECK_LEARNING_RATE,
    build_tbptt_sequences,
    evaluate_sequential_loss,
    fixed_tbptt_batch,
    positive_float,
    positive_int,
    prepare_tinyshakespeare_data,
    sample_tbptt_batch,
    tbptt_training_step,
    validate_args as validate_tbptt_args,
    window_training_step,
)

TRAINING_MODES = ("tbptt", "chunk-reset", "window")
FULL_RESET_HORIZONS = (1, 8, 32, 128, 512, 2048, 20_000)
SANITY_RESET_HORIZONS = (1, 8, 32)
DEFAULT_TRAINING_STEPS = 200
SANITY_CHECK_STEPS = 8
SANITY_CHECK_VAL_TOKENS = 4096


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "tinyshakespeare" / "artifacts" / "rnn_reset_sweep"
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=positive_int, default=DEFAULT_TRAINING_STEPS)
    parser.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq-len", type=positive_int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--bptt-chunk", type=positive_int, default=DEFAULT_BPTT_CHUNK)
    parser.add_argument("--embed-dim", type=positive_int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--hidden-dim", type=positive_int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--lr", type=positive_float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sanity-check-only", action="store_true")
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.jsonl")
    parser.add_argument("--checkpoints-dir", type=Path, default=artifact_dir / "checkpoints")
    parser.set_defaults(no_lock=True)
    parser.add_argument("--lock", dest="no_lock", action="store_false")
    parser.add_argument("--no-lock", dest="no_lock", action="store_true")
    return parser.parse_args()


def resolve_steps(args: argparse.Namespace) -> int:
    return SANITY_CHECK_STEPS if args.sanity_check_only else args.steps


def resolve_batch_size(args: argparse.Namespace) -> int:
    return SANITY_CHECK_BATCH_SIZE if args.sanity_check_only else args.batch_size


def resolve_eval_interval(args: argparse.Namespace) -> int:
    return SANITY_CHECK_EVAL_INTERVAL if args.sanity_check_only else DEFAULT_EVAL_INTERVAL


def resolve_learning_rate(args: argparse.Namespace) -> float:
    return SANITY_CHECK_LEARNING_RATE if args.sanity_check_only else args.lr


def resolve_reset_horizons(args: argparse.Namespace) -> tuple[int, ...]:
    return SANITY_RESET_HORIZONS if args.sanity_check_only else FULL_RESET_HORIZONS


def resolve_eval_tokens(args: argparse.Namespace, *, encoded_tokens: Tensor) -> Tensor:
    if not args.sanity_check_only:
        return encoded_tokens
    if encoded_tokens.numel() < SANITY_CHECK_VAL_TOKENS:
        return encoded_tokens
    return encoded_tokens[:SANITY_CHECK_VAL_TOKENS].clone()


def horizon_label(reset_horizon: int, *, full_stream_tokens: int) -> str:
    return "full_stream" if reset_horizon >= full_stream_tokens else str(reset_horizon)


def cpu_state_dict(model: GruLanguageModel) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


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
        raise RuntimeError(
            f"Reset horizon {reset_horizon} left no evaluable tokens. This should only happen for broken masking logic."
        )
    if was_training:
        model.train()
    return total_loss / total_tokens, total_tokens, ignored_tokens


def train_mode(
    *,
    mode: str,
    data: object,
    tbptt_inputs: Tensor,
    tbptt_targets: Tensor,
    device: torch.device,
    steps: int,
    batch_size: int,
    eval_interval: int,
    learning_rate: float,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, Tensor], int]:
    set_seed(args.seed)
    model = GruLanguageModel(
        vocab_size=data.corpus.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = build_optimizer(
        model,
        device=device,
        compile_model=False,
        learning_rate=learning_rate,
    )
    parameter_count = count_parameters(model)
    tbptt_like_mode = mode in ("tbptt", "chunk-reset")
    carry_state = mode == "tbptt"

    append_log(
        args.log_path,
        {
            "stage": "mode_started",
            "mode": mode,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "parameter_count": parameter_count,
        },
    )

    if tbptt_like_mode:
        fixed_batch_inputs, fixed_batch_targets = fixed_tbptt_batch(
            tbptt_inputs,
            tbptt_targets,
            batch_size=batch_size,
            device=device,
        )
        tbptt_rng = torch.Generator(device="cpu")
        tbptt_rng.manual_seed(args.seed)
    else:
        window_rng = torch.Generator(device="cpu")
        window_rng.manual_seed(args.seed)
        window_iterator = random_batches(
            data.train_encoded,
            context_size=args.bptt_chunk,
            batch_size=batch_size,
            device=device,
            rng=window_rng,
        )
        fixed_batch_inputs, fixed_batch_next_targets = next(window_iterator)

    best_val_loss = float("inf")
    best_state_dict = cpu_state_dict(model)
    best_step = 0
    checkpoints: list[dict[str, object]] = []
    final_train_loss = float("nan")
    final_val_loss = float("nan")
    started_at = perf_counter()

    for step in range(1, steps + 1):
        if tbptt_like_mode:
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
                carry_state=carry_state,
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

        if step % eval_interval != 0 and step != steps:
            continue

        final_val_loss = evaluate_sequential_loss(
            model,
            encoded_tokens=data.val_encoded,
            device=device,
            chunk_size=args.bptt_chunk,
        )
        checkpoint = {
            "step": step,
            "train_loss": round(final_train_loss, 6),
            "val_loss": round(final_val_loss, 6),
            "best_so_far": round(min(best_val_loss, final_val_loss), 6),
        }
        checkpoints.append(checkpoint)
        append_log(
            args.log_path,
            {
                "stage": "checkpoint",
                "mode": mode,
                **checkpoint,
            },
        )
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_state_dict = cpu_state_dict(model)
            best_step = step

    wall_seconds = round(perf_counter() - started_at, 6)
    checkpoint_path = args.checkpoints_dir / f"{mode}-best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state_dict, checkpoint_path)
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
            "checkpoint_path": str(checkpoint_path),
        },
    )
    return (
        {
            "mode": mode,
            "parameter_count": parameter_count,
            "best_step": best_step,
            "best_val_loss": round(best_val_loss, 6),
            "final_train_loss": round(final_train_loss, 6),
            "final_val_loss": round(final_val_loss, 6),
            "wall_seconds": wall_seconds,
            "checkpoint_path": str(checkpoint_path),
            "checkpoints": checkpoints,
        },
        best_state_dict,
        parameter_count,
    )


def main() -> int:
    args = parse_args()
    validate_tbptt_args(args)

    if args.sanity_check_only:
        redirect_sanity_check_paths(args)

    steps = resolve_steps(args)
    batch_size = resolve_batch_size(args)
    eval_interval = resolve_eval_interval(args)
    learning_rate = resolve_learning_rate(args)
    reset_horizons = resolve_reset_horizons(args)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    prepare_output_paths(report_path=args.report_path, log_path=args.log_path)
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_run_restarted(args.log_path)
    register_active_lock(
        experiment_name="rnn_reset_sweep",
        variants=[f"ed{args.embed_dim}", f"hd{args.hidden_dim}", f"seq{args.seq_len}", f"chunk{args.bptt_chunk}"],
        enabled=not args.no_lock,
    )

    repo_root = Path(__file__).resolve().parents[1]
    set_seed(args.seed)
    data = prepare_tinyshakespeare_data(repo_root=repo_root, seq_len=args.seq_len)
    tbptt_inputs, tbptt_targets = build_tbptt_sequences(data.train_encoded, seq_len=args.seq_len)

    val_eval_tokens = resolve_eval_tokens(args, encoded_tokens=data.val_encoded)

    append_log(
        args.log_path,
        {
            "stage": "run_started",
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
            "reset_horizons": list(reset_horizons),
            "eval_val_tokens": int(val_eval_tokens.numel()),
            "no_lock": args.no_lock,
        },
    )
    append_log(
        args.log_path,
        {
            "stage": "dataset_loaded",
            "source_path": str(data.source_path),
            "train_tokens": int(data.train_encoded.numel()),
            "val_tokens": int(data.val_encoded.numel()),
            "eval_val_tokens": int(val_eval_tokens.numel()),
            "vocab_size": data.corpus.vocab_size,
            "tbptt_sequences": int(tbptt_inputs.shape[0]),
        },
    )

    mode_reports: dict[str, dict[str, object]] = {}
    results_matrix: dict[str, dict[str, float]] = {}
    detailed_results: list[dict[str, object]] = []
    parameter_count = 0
    started_at = perf_counter()

    for mode in TRAINING_MODES:
        mode_report, best_state_dict, parameter_count = train_mode(
            mode=mode,
            data=data,
            tbptt_inputs=tbptt_inputs,
            tbptt_targets=tbptt_targets,
            device=device,
            steps=steps,
            batch_size=batch_size,
            eval_interval=eval_interval,
            learning_rate=learning_rate,
            args=args,
        )
        mode_reports[mode] = mode_report

        model = GruLanguageModel(
            vocab_size=data.corpus.vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        model.load_state_dict(best_state_dict)

        mode_results: dict[str, float] = {}
        for reset_horizon in reset_horizons:
            effective_horizon = min(reset_horizon, int(data.val_encoded.numel()))
            reset_loss, tokens_evaluated, ignored_tokens = evaluate_reset_horizon_loss(
                model,
                encoded_tokens=val_eval_tokens,
                device=device,
                reset_horizon=effective_horizon,
                eval_chunk_size=args.bptt_chunk,
            )
            label = horizon_label(effective_horizon, full_stream_tokens=int(data.val_encoded.numel()))
            rounded_loss = round(reset_loss, 6)
            mode_results[label] = rounded_loss
            detailed_result = {
                "mode": mode,
                "reset_horizon": label,
                "val_loss": rounded_loss,
                "tokens_evaluated": tokens_evaluated,
                "ignored_tokens": ignored_tokens,
            }
            detailed_results.append(detailed_result)
            append_log(
                args.log_path,
                {
                    "stage": "reset_sweep_result",
                    **detailed_result,
                },
            )
        results_matrix[mode] = mode_results

        del model
        release_memory(device=device)

    wall_seconds = round(perf_counter() - started_at, 6)
    git_status_short = current_git_status_short()
    report = {
        "config": {
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
            "reset_horizons": [
                horizon_label(min(horizon, int(data.val_encoded.numel())), full_stream_tokens=int(data.val_encoded.numel()))
                for horizon in reset_horizons
            ],
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
            "source_path": str(data.source_path),
            "train_tokens": int(data.train_encoded.numel()),
            "val_tokens": int(data.val_encoded.numel()),
            "eval_val_tokens": int(val_eval_tokens.numel()),
            "vocab_size": data.corpus.vocab_size,
            "tbptt_sequences": int(tbptt_inputs.shape[0]),
        },
        "model": {
            "parameter_count": parameter_count,
        },
        "training": mode_reports,
        "results": results_matrix,
        "detailed_results": detailed_results,
        "wall_seconds": wall_seconds,
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "run_completed",
            "report_path": str(args.report_path),
            "wall_seconds": wall_seconds,
            "results": results_matrix,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
