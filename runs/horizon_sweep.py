from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    raise RuntimeError(
        "Run from the repo root with `.\\.venv\\Scripts\\python.exe -m runs.horizon_sweep` so `core` imports resolve cleanly."
    )

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from core.model import count_parameters
from core.run_utils import (
    append_log,
    log_run_restarted,
    prepare_output_paths,
    redirect_sanity_check_paths,
    register_active_lock,
    resolve_device,
)
from core.training import current_git_sha, current_git_status_short, write_json
from runs.predictive_processing import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BLOCK0_EVAL_INTERVAL,
    DEFAULT_BLOCK0_LR,
    DEFAULT_BLOCK0_STEPS,
    DEFAULT_BLOCK1_EVAL_INTERVAL,
    DEFAULT_BLOCK1_LR,
    DEFAULT_BLOCK1_STEPS,
    DEFAULT_BPTT_CHUNK,
    DEFAULT_D_MODEL,
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_SEED,
    DEFAULT_SEQ_LEN,
    DEFAULT_WEIGHT_DECAY,
    NORMALIZE,
    RANDOM_BASELINE_SEED,
    SANITY_BATCH_SIZE,
    SANITY_BLOCK0_EVAL_INTERVAL,
    SANITY_BLOCK0_LR,
    SANITY_BLOCK0_STEPS,
    SANITY_BLOCK1_EVAL_INTERVAL,
    SANITY_BLOCK1_LR,
    SANITY_BLOCK1_STEPS,
    SANITY_SEQ_LEN,
    TEMPERATURE,
    TRAIN_CHARACTERS,
    VAL_CHARACTERS,
    Block1Metrics,
    PhaseCheckpoint,
    PhaseSummary,
    RepresentationPredictorBlock,
    TokenPredictorBlock,
    build_tbptt_sequences,
    chunk_ranges,
    collect_block0_representations,
    fixed_tbptt_batch,
    phase_checkpoints_to_payload,
    positive_float,
    positive_int,
    prepare_tinyshakespeare_data,
    sample_tbptt_batch,
    set_seed,
    train_block0,
    trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "tinyshakespeare" / "artifacts" / "horizon_sweep"
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=positive_int, default=1)
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
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    args = parser.parse_args()
    if args.report_path is None:
        args.report_path = artifact_dir / f"report_h{args.horizon}.json"
    if args.log_path is None:
        args.log_path = artifact_dir / f"run_h{args.horizon}.jsonl"
    return args


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


def resolve_execution_device(args: argparse.Namespace) -> torch.device:
    if args.sanity_check_only and args.device is None:
        return torch.device("cpu")
    return resolve_device(args.device)


def validate_args(args: argparse.Namespace) -> None:
    resolved_seq_len = resolve_seq_len(args)
    errors: list[str] = []
    if args.bptt_chunk > resolved_seq_len:
        errors.append(f"bptt_chunk must be <= seq_len, got bptt_chunk={args.bptt_chunk}, seq_len={resolved_seq_len}")
    if resolved_seq_len < 2:
        errors.append(f"seq_len must be at least 2, got {resolved_seq_len}")
    if args.horizon >= resolved_seq_len:
        errors.append(f"horizon must be < seq_len, got horizon={args.horizon}, seq_len={resolved_seq_len}")
    if errors:
        raise ValueError("Argument validation failed:\n- " + "\n- ".join(errors))


def initial_horizon_history(
    *,
    batch_size: int,
    horizon: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[Tensor, "batch horizon d_model"]:
    return torch.zeros(batch_size, horizon, d_model, device=device, dtype=dtype)


def horizon_predictor_input(
    target_chunk: Float[Tensor, "batch chunk d_model"],
    history: Float[Tensor, "batch horizon d_model"],
) -> tuple[Float[Tensor, "batch chunk d_model"], Float[Tensor, "batch horizon d_model"]]:
    if history.shape[1] == 0:
        raise ValueError("horizon history must have positive length.")
    context = torch.cat([history, target_chunk], dim=1)
    predictor_input = context[:, : target_chunk.shape[1], :]
    next_history = context[:, -history.shape[1] :, :].detach()
    return predictor_input, next_history


def valid_horizon_start(*, start: int, horizon: int) -> int:
    return max(0, horizon - start)


def valid_horizon_length(*, start: int, stop: int, horizon: int) -> int:
    return max(0, stop - max(start, horizon))


def block1_horizon_loss_from_predictions(
    predicted_mu: Float[Tensor, "batch chunk d_model"],
    target_mu: Float[Tensor, "batch chunk d_model"],
    *,
    start: int,
    horizon: int,
) -> tuple[Float[Tensor, ""], int]:
    valid_start = valid_horizon_start(start=start, horizon=horizon)
    valid_predictions = predicted_mu[:, valid_start:, :]
    valid_targets = target_mu[:, valid_start:, :]
    if valid_predictions.shape[1] == 0:
        raise ValueError("Need at least one valid position for block 1 MSE loss.")
    loss = F.mse_loss(valid_predictions.float(), valid_targets.float())
    return loss, int(valid_predictions.shape[0] * valid_predictions.shape[1])


def random_baseline_tensor(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[Tensor, "batch seq d_model"]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(RANDOM_BASELINE_SEED)
    vector = torch.randn(d_model, generator=generator, dtype=torch.float32)
    vector = F.normalize(vector, dim=0).to(device=device, dtype=dtype)
    return vector.view(1, 1, -1).expand(batch_size, seq_len, -1)


@torch.inference_mode()
def evaluate_block1_metrics(
    block1: RepresentationPredictorBlock,
    *,
    mu_sequence: Float[Tensor, "batch seq d_model"],
    bptt_chunk: int,
    use_recurrence: bool,
    horizon: int,
) -> Block1Metrics:
    was_training = block1.training
    block1.eval()
    hidden: Tensor | None = None
    history = initial_horizon_history(
        batch_size=mu_sequence.shape[0],
        horizon=horizon,
        d_model=mu_sequence.shape[2],
        device=mu_sequence.device,
        dtype=mu_sequence.dtype,
    )
    predicted_chunks: list[Tensor] = []
    for start, stop in chunk_ranges(mu_sequence.shape[1], bptt_chunk):
        target_chunk = mu_sequence[:, start:stop, :]
        predictor_input, history = horizon_predictor_input(target_chunk, history)
        predicted_chunk, hidden = block1.forward_chunk(predictor_input, hidden, use_recurrence=use_recurrence)
        predicted_chunks.append(predicted_chunk)
        hidden = hidden.detach()

    predicted_mu = torch.cat(predicted_chunks, dim=1)
    valid_target = mu_sequence[:, horizon:, :]
    valid_prediction = predicted_mu[:, horizon:, :]
    copy_baseline = mu_sequence[:, :-horizon, :]
    random_baseline = random_baseline_tensor(
        batch_size=valid_target.shape[0],
        seq_len=valid_target.shape[1],
        d_model=mu_sequence.shape[2],
        device=mu_sequence.device,
        dtype=mu_sequence.dtype,
    )
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
    horizon: int,
) -> tuple[PhaseSummary, Block1Metrics]:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    eval_mu = collect_block0_representations(block0, eval_batch_tokens, bptt_chunk=bptt_chunk)
    initial_metrics = evaluate_block1_metrics(
        block1,
        mu_sequence=eval_mu,
        bptt_chunk=bptt_chunk,
        use_recurrence=use_recurrence,
        horizon=horizon,
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
            "horizon": horizon,
            "use_recurrence": use_recurrence,
        },
    )

    final_train_loss = initial_metrics.mse
    final_eval_metrics = initial_metrics
    total_chunks = sum(
        1
        for start, stop in chunk_ranges(eval_batch_tokens.shape[1], bptt_chunk)
        if valid_horizon_length(start=start, stop=stop, horizon=horizon) > 0
    )
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
        history = initial_horizon_history(
            batch_size=mu_sequence.shape[0],
            horizon=horizon,
            d_model=mu_sequence.shape[2],
            device=device,
            dtype=mu_sequence.dtype,
        )
        loss_sum = 0.0
        valid_positions = 0
        for start, stop in chunk_ranges(mu_sequence.shape[1], bptt_chunk):
            target_chunk = mu_sequence[:, start:stop, :]
            predictor_input, history = horizon_predictor_input(target_chunk, history)
            predicted_chunk, hidden = block1.forward_chunk(predictor_input, hidden, use_recurrence=use_recurrence)
            chunk_valid_positions = valid_horizon_length(start=start, stop=stop, horizon=horizon)
            if chunk_valid_positions > 0:
                chunk_loss, _ = block1_horizon_loss_from_predictions(
                    predicted_chunk,
                    target_chunk,
                    start=start,
                    horizon=horizon,
                )
                if not torch.isfinite(chunk_loss):
                    raise RuntimeError(f"Block 1 training diverged at step {step}.")
                (chunk_loss / total_chunks).backward()
                loss_sum += chunk_loss.detach().item() * chunk_valid_positions
                valid_positions += chunk_valid_positions
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
            horizon=horizon,
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
                "horizon": horizon,
                "use_recurrence": use_recurrence,
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
    device = resolve_execution_device(args)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    prepare_output_paths(report_path=args.report_path, log_path=args.log_path)
    log_run_restarted(args.log_path)
    register_active_lock(
        experiment_name="horizon_sweep",
        variants=[f"h{args.horizon}", f"dm{args.d_model}", f"seq{seq_len}", f"chunk{args.bptt_chunk}"],
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
            "horizon": args.horizon,
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
    block1_recurrent_summary, block1_recurrent_metrics = train_block1(
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
        checkpoint_stage="block1_recurrent_checkpoint",
        use_recurrence=True,
        horizon=args.horizon,
    )
    block1_recurrent_wall_seconds = round(perf_counter() - phase_started_at, 6)

    phase_started_at = perf_counter()
    block1_no_recurrence_summary, block1_no_recurrence_metrics = train_block1(
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
        checkpoint_stage="block1_no_recurrence_checkpoint",
        use_recurrence=False,
        horizon=args.horizon,
    )
    block1_no_recurrence_wall_seconds = round(perf_counter() - phase_started_at, 6)

    recurrence_gap = block1_no_recurrence_metrics.mse - block1_recurrent_metrics.mse
    summary_lines = [
        f"Horizon sweep results (h={args.horizon}):",
        f"  Phase A Block 0 CE: {phase_a_summary.final_eval_loss:.6f} (initial {phase_a_summary.initial_eval_loss:.6f})",
        f"  Recurrent Block 1 MSE: {block1_recurrent_metrics.mse:.6f}",
        f"  No-recurrence Block 1 MSE: {block1_no_recurrence_metrics.mse:.6f}",
        f"  Recurrence gap: {recurrence_gap:.6f}",
        f"  Horizon-{args.horizon} copy baseline MSE: {block1_recurrent_metrics.copy_baseline:.6f}",
        f"  Random baseline MSE: {block1_recurrent_metrics.random_baseline:.6f}",
    ]
    print("\n".join(summary_lines))

    git_status_short = current_git_status_short()
    report = {
        "config": {
            "sanity_check_only": args.sanity_check_only,
            "horizon": args.horizon,
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
            "block1_recurrent": {
                "initial_eval_mse": block1_recurrent_summary.initial_eval_loss,
                "final_train_mse": block1_recurrent_summary.final_train_loss,
                "final_eval_mse": block1_recurrent_metrics.mse,
                "copy_baseline": block1_recurrent_metrics.copy_baseline,
                "random_baseline": block1_recurrent_metrics.random_baseline,
                "beats_copy": block1_recurrent_metrics.beats_copy,
                "loss_went_down": block1_recurrent_metrics.mse < block1_recurrent_summary.initial_eval_loss,
                "wall_seconds": block1_recurrent_wall_seconds,
                "checkpoints": phase_checkpoints_to_payload(block1_recurrent_summary.checkpoints),
            },
            "block1_no_recurrence": {
                "initial_eval_mse": block1_no_recurrence_summary.initial_eval_loss,
                "final_train_mse": block1_no_recurrence_summary.final_train_loss,
                "final_eval_mse": block1_no_recurrence_metrics.mse,
                "copy_baseline": block1_no_recurrence_metrics.copy_baseline,
                "random_baseline": block1_no_recurrence_metrics.random_baseline,
                "beats_copy": block1_no_recurrence_metrics.beats_copy,
                "loss_went_down": block1_no_recurrence_metrics.mse < block1_no_recurrence_summary.initial_eval_loss,
                "wall_seconds": block1_no_recurrence_wall_seconds,
                "checkpoints": phase_checkpoints_to_payload(block1_no_recurrence_summary.checkpoints),
            },
            "recurrence_gap": recurrence_gap,
        },
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "run_completed",
            "report_path": str(args.report_path),
            "horizon": args.horizon,
            "phase_a_final_eval_ce": round(phase_a_summary.final_eval_loss, 6),
            "block1_recurrent_final_eval_mse": round(block1_recurrent_metrics.mse, 6),
            "block1_no_recurrence_final_eval_mse": round(block1_no_recurrence_metrics.mse, 6),
            "copy_baseline": round(block1_recurrent_metrics.copy_baseline, 6),
            "random_baseline": round(block1_recurrent_metrics.random_baseline, 6),
            "recurrence_gap": round(recurrence_gap, 6),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
