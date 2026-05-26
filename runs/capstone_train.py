from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    raise RuntimeError(
        "Run from the repo root with `.\\.venv\\Scripts\\python.exe -m runs.capstone_train ...` so `core` imports resolve cleanly."
    )

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional as F

from core.dataset import CorpusData, RandomWindowCharDataset, download_wikitext_103_raw, load_corpus
from core.fixed_window_char import set_seed
from core.model import count_parameters
from core.recurrent_depth import RecurrentDepthConfig, RecurrentDepthLM
from core.run_utils import (
    append_log,
    log_run_restarted,
    prepare_output_paths,
    random_batches,
    redirect_sanity_check_paths,
    register_active_lock,
    resolve_device,
)


DEFAULT_STEPS = 20_000
SANITY_CHECK_STEPS = 10
DEFAULT_CONTEXT_SIZE = 256
DEFAULT_D_MODEL = 256
DEFAULT_N_HEADS = 4
DEFAULT_FF_DIM = 1_024
DEFAULT_ITERATIONS = 8
DEFAULT_TEMPERATURE = 0.07
DEFAULT_BATCH_SIZE = 128
SANITY_CHECK_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_EVAL_INTERVAL = 1_000
SANITY_CHECK_EVAL_INTERVAL = 5
DEFAULT_EVAL_BATCH_SIZE = 2_048
DEFAULT_EVAL_SAMPLES = 2_048
DEFAULT_SEED = 42
DEFAULT_HALT_WEIGHT = 0.1
DEFAULT_HALT_EPSILON = 0.0
DEFAULT_DROPOUT = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    context_size: int = DEFAULT_CONTEXT_SIZE
    d_model: int = DEFAULT_D_MODEL
    n_heads: int = DEFAULT_N_HEADS
    ff_dim: int = DEFAULT_FF_DIM
    iterations: int = DEFAULT_ITERATIONS
    temperature: float = DEFAULT_TEMPERATURE
    normalize: bool = True
    dropout: float = DEFAULT_DROPOUT
    halt_weight: float = DEFAULT_HALT_WEIGHT
    halt_epsilon: float = DEFAULT_HALT_EPSILON
    eval_interval: int = DEFAULT_EVAL_INTERVAL
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    eval_samples: int = DEFAULT_EVAL_SAMPLES


@dataclass(frozen=True)
class TrainingSummary:
    final_step: int
    final_train_loss: float
    final_ce_loss: float
    final_halt_loss: float
    wall_seconds: float


@dataclass(frozen=True)
class EvalSummary:
    step: int
    train_loss: float
    ce_loss: float
    halt_loss: float
    val_loss: float
    val_halt_loss: float
    avg_depth: float
    learning_rate: float
    wall_seconds: float


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
    artifact_dir = repo_root / "experiments" / "capstone_generation" / "artifacts" / "train"
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=positive_int, default=DEFAULT_STEPS)
    parser.add_argument("--d-model", type=positive_int, default=DEFAULT_D_MODEL)
    parser.add_argument("--ff-dim", type=positive_int, default=DEFAULT_FF_DIM)
    parser.add_argument("--n-heads", type=positive_int, default=DEFAULT_N_HEADS)
    parser.add_argument("--iterations", type=positive_int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--context-size", type=positive_int, default=DEFAULT_CONTEXT_SIZE)
    parser.add_argument("--batch-size", type=positive_int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=positive_float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sanity-check-only", action="store_true")
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.jsonl")
    return parser.parse_args()


def resolve_training_steps(args: argparse.Namespace) -> int:
    return SANITY_CHECK_STEPS if args.sanity_check_only else args.steps


def resolve_batch_size(args: argparse.Namespace) -> int:
    return SANITY_CHECK_BATCH_SIZE if args.sanity_check_only else args.batch_size


def resolve_eval_interval(args: argparse.Namespace) -> int:
    return SANITY_CHECK_EVAL_INTERVAL if args.sanity_check_only else DEFAULT_EVAL_INTERVAL


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def ensure_wikitext_corpus(*, config: ExperimentConfig) -> tuple[CorpusData, dict[str, str]]:
    paths = download_wikitext_103_raw()
    corpus = load_corpus(
        train_path=paths["wiki.train.raw"],
        val_path=paths["wiki.valid.raw"],
        context_size=config.context_size,
        eval_samples=config.eval_samples,
    )
    return corpus, {name: str(path) for name, path in paths.items()}


def build_model(*, vocab_size: int, config: ExperimentConfig) -> RecurrentDepthLM:
    return RecurrentDepthLM(
        vocab_size=vocab_size,
        config=RecurrentDepthConfig(
            context_size=config.context_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            ff_dim=config.ff_dim,
            iterations=config.iterations,
            temperature=config.temperature,
            dropout=config.dropout,
            normalize=config.normalize,
        ),
    )


def choose_depth_indices(
    predicted_gains: Float[Tensor, "batch depth"],
    *,
    epsilon: float,
    full_depth_index: int,
) -> Int[Tensor, "batch"]:
    early_halt_mask = predicted_gains < epsilon
    any_halt = early_halt_mask.any(dim=1)
    return torch.where(
        any_halt,
        early_halt_mask.float().argmax(dim=1),
        torch.full((predicted_gains.shape[0],), full_depth_index, dtype=torch.int64, device=predicted_gains.device),
    )


def compute_actual_gains(per_depth_losses: Float[Tensor, "batch depth"]) -> Float[Tensor, "batch depth_minus_one"]:
    return per_depth_losses[:, :-1] - per_depth_losses[:, -1:].expand(-1, per_depth_losses.shape[1] - 1)


def compute_per_depth_losses_and_predictions(
    model: RecurrentDepthLM,
    batch_inputs: Int[Tensor, "batch context"],
    batch_targets: Int[Tensor, "batch"],
) -> tuple[
    Float[Tensor, "batch depth"],
    Float[Tensor, "batch depth_minus_one"],
]:
    _, iteration_states = model.iteration_states(batch_inputs, collect_iteration_states=True)
    per_depth_losses: list[Tensor] = []
    halt_predictions: list[Tensor] = []
    for depth_index, iteration_hidden in enumerate(iteration_states, start=1):
        last_hidden = iteration_hidden[:, -1, :]
        logits = model.lm_logits_from_hidden(last_hidden)
        per_depth_losses.append(F.cross_entropy(logits.float(), batch_targets, reduction="none"))
        if depth_index < model.config.iterations:
            halt_predictions.append(model.predicted_gain_from_hidden(last_hidden, depth_index=depth_index))
    return torch.stack(per_depth_losses, dim=1), torch.stack(halt_predictions, dim=1)


@torch.inference_mode()
def evaluate_model(
    model: RecurrentDepthLM,
    *,
    val_inputs: Int[Tensor, "examples context"],
    val_targets: Int[Tensor, "examples"],
    eval_batch_size: int,
    halt_epsilon: float,
    device: torch.device,
) -> tuple[float, float, float, float]:
    del device
    started_at = perf_counter()
    model.eval()
    total_examples = 0
    total_val_loss = 0.0
    total_val_halt_loss = 0.0
    total_depth = 0.0
    full_depth_index = model.config.iterations - 1

    for start in range(0, val_targets.shape[0], eval_batch_size):
        stop = min(start + eval_batch_size, val_targets.shape[0])
        batch_inputs = val_inputs[start:stop]
        batch_targets = val_targets[start:stop]
        with autocast_context(val_inputs.device):
            per_depth_losses, halt_predictions = compute_per_depth_losses_and_predictions(model, batch_inputs, batch_targets)
        actual_gains = compute_actual_gains(per_depth_losses)
        depth_indices = choose_depth_indices(halt_predictions.float(), epsilon=halt_epsilon, full_depth_index=full_depth_index)
        total_examples += batch_targets.shape[0]
        total_val_loss += per_depth_losses[:, -1].sum().item()
        total_val_halt_loss += F.mse_loss(halt_predictions.float(), actual_gains.float(), reduction="sum").item()
        total_depth += (depth_indices + 1).float().sum().item()

    if total_examples == 0:
        raise RuntimeError("validation set was empty")
    halting_predictions = model.halting_depth_count * total_examples
    mean_halt_loss = 0.0 if halting_predictions == 0 else total_val_halt_loss / halting_predictions
    return (
        total_val_loss / total_examples,
        mean_halt_loss,
        total_depth / total_examples,
        perf_counter() - started_at,
    )


def save_checkpoint(
    checkpoint_path: Path,
    *,
    model: RecurrentDepthLM,
    config: ExperimentConfig,
    corpus: CorpusData,
    corpus_paths: dict[str, str],
    device: torch.device,
    steps: int,
    batch_size: int,
    learning_rate: float,
    training: TrainingSummary,
    final_eval: EvalSummary,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "experiment_config": asdict(config),
        "char_to_idx": corpus.char_to_idx,
        "idx_to_char": {str(index): char for index, char in corpus.idx_to_char.items()},
        "vocab_size": corpus.vocab_size,
        "training_metadata": {
            "device": device.type,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "final_step": training.final_step,
            "final_train_loss": round(training.final_train_loss, 6),
            "final_ce_loss": round(training.final_ce_loss, 6),
            "final_halt_loss": round(training.final_halt_loss, 6),
            "final_val_loss": round(final_eval.val_loss, 6),
            "final_val_halt_loss": round(final_eval.val_halt_loss, 6),
            "final_avg_depth": round(final_eval.avg_depth, 6),
            "training_wall_seconds": round(training.wall_seconds, 6),
            "final_eval_wall_seconds": round(final_eval.wall_seconds, 6),
        },
        "corpus_paths": corpus_paths,
    }
    torch.save(checkpoint, checkpoint_path)


def write_report(
    report_path: Path,
    *,
    config: ExperimentConfig,
    corpus: CorpusData,
    corpus_paths: dict[str, str],
    parameter_count: int,
    device: torch.device,
    sanity_check_only: bool,
    steps: int,
    batch_size: int,
    learning_rate: float,
    training: TrainingSummary,
    checkpoints: list[EvalSummary],
    checkpoint_path: Path,
) -> None:
    payload = {
        "device": device.type,
        "sanity_check_only": sanity_check_only,
        "parameter_count": parameter_count,
        "experiment_config": asdict(config),
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "vocab_size": corpus.vocab_size,
        "corpus_paths": corpus_paths,
        "checkpoint_path": str(checkpoint_path),
        "training": {
            "final_step": training.final_step,
            "final_train_loss": round(training.final_train_loss, 6),
            "final_ce_loss": round(training.final_ce_loss, 6),
            "final_halt_loss": round(training.final_halt_loss, 6),
            "wall_seconds": round(training.wall_seconds, 6),
        },
        "eval_checkpoints": [
            {
                "step": checkpoint.step,
                "train_loss": round(checkpoint.train_loss, 6),
                "ce_loss": round(checkpoint.ce_loss, 6),
                "halt_loss": round(checkpoint.halt_loss, 6),
                "val_loss": round(checkpoint.val_loss, 6),
                "val_halt_loss": round(checkpoint.val_halt_loss, 6),
                "avg_depth": round(checkpoint.avg_depth, 6),
                "learning_rate": round(checkpoint.learning_rate, 8),
                "wall_seconds": round(checkpoint.wall_seconds, 6),
            }
            for checkpoint in checkpoints
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_report_path(path: Path) -> Path:
    """If path is a directory, append report.json. Otherwise use as-is."""
    if path.is_dir():
        return path / "report.json"
    return path


def main() -> None:
    args = parse_args()
    if args.iterations < 2:
        raise ValueError(f"iterations must be at least 2, got {args.iterations}")
    if args.d_model % args.n_heads != 0:
        raise ValueError(f"d_model must be divisible by n_heads, got d_model={args.d_model}, n_heads={args.n_heads}")

    steps = resolve_training_steps(args)
    batch_size = resolve_batch_size(args)
    config = ExperimentConfig(
        context_size=args.context_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        iterations=args.iterations,
        eval_interval=resolve_eval_interval(args),
    )
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.sanity_check_only:
        redirect_sanity_check_paths(args)

    # Resolve report_path: if it's a directory, use report.json inside it
    args.report_path = resolve_report_path(args.report_path)
    prepare_output_paths(report_path=args.report_path, log_path=args.log_path)
    log_run_restarted(args.log_path)
    if not args.sanity_check_only:
        register_active_lock(
            experiment_name="capstone_train",
            variants=[f"d{config.d_model}", f"ctx{config.context_size}", f"iter{config.iterations}"],
            enabled=not args.no_lock,
        )

    append_log(
        args.log_path,
        {
            "stage": "run_started",
            "device": device.type,
            "sanity_check_only": args.sanity_check_only,
            "steps": steps,
            "batch_size": batch_size,
            "lr": args.lr,
            "config": asdict(config),
        },
    )

    set_seed(args.seed)
    corpus, corpus_paths = ensure_wikitext_corpus(config=config)
    train_dataset = corpus.train_dataset
    if not isinstance(train_dataset, RandomWindowCharDataset):
        raise TypeError("Expected load_corpus() to return RandomWindowCharDataset for WikiText-103 training.")

    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(args.seed)
    train_iterator = random_batches(
        train_dataset.encoded_corpus,
        context_size=config.context_size,
        batch_size=batch_size,
        device=device,
        rng=train_generator,
    )
    val_inputs = corpus.val_inputs.to(device=device, dtype=torch.long)
    val_targets = corpus.val_targets.to(device=device, dtype=torch.long)

    append_log(
        args.log_path,
        {
            "stage": "dataset_loaded",
            "train_windows": len(train_dataset),
            "val_examples": int(val_targets.shape[0]),
            "vocab_size": corpus.vocab_size,
            "corpus_paths": corpus_paths,
        },
    )

    model = build_model(vocab_size=corpus.vocab_size, config=config).to(device)
    optimizer_kwargs: dict[str, object] = {"lr": args.lr}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    parameter_count = count_parameters(model)

    append_log(
        args.log_path,
        {
            "stage": "model_built",
            "parameter_count": parameter_count,
            "halting_depth_count": model.halting_depth_count,
        },
    )

    checkpoints: list[EvalSummary] = []
    started_at = perf_counter()
    last_train_loss = float("nan")
    last_ce_loss = float("nan")
    last_halt_loss = float("nan")

    for step in range(1, steps + 1):
        model.train()
        batch_inputs, batch_targets = next(train_iterator)
        with autocast_context(device):
            per_depth_losses, halt_predictions = compute_per_depth_losses_and_predictions(model, batch_inputs, batch_targets)
            actual_gains = compute_actual_gains(per_depth_losses.detach())
            ce_loss = per_depth_losses[:, -1].mean()
            halt_loss = F.mse_loss(halt_predictions.float(), actual_gains.float())
            train_loss = ce_loss + (config.halt_weight * halt_loss)

        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()

        last_train_loss = float(train_loss.item())
        last_ce_loss = float(ce_loss.item())
        last_halt_loss = float(halt_loss.item())

        if step % config.eval_interval != 0 and step != steps:
            continue

        val_loss, val_halt_loss, avg_depth, eval_wall_seconds = evaluate_model(
            model,
            val_inputs=val_inputs,
            val_targets=val_targets,
            eval_batch_size=config.eval_batch_size,
            halt_epsilon=config.halt_epsilon,
            device=device,
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        summary = EvalSummary(
            step=step,
            train_loss=last_train_loss,
            ce_loss=last_ce_loss,
            halt_loss=last_halt_loss,
            val_loss=val_loss,
            val_halt_loss=val_halt_loss,
            avg_depth=avg_depth,
            learning_rate=learning_rate,
            wall_seconds=eval_wall_seconds,
        )
        checkpoints.append(summary)
        append_log(
            args.log_path,
            {
                "stage": "train_progress",
                "step": summary.step,
                "train_loss": round(summary.train_loss, 6),
                "val_loss": round(summary.val_loss, 6),
                "avg_depth": round(summary.avg_depth, 6),
                "learning_rate": round(summary.learning_rate, 8),
                "ce_loss": round(summary.ce_loss, 6),
                "halt_loss": round(summary.halt_loss, 6),
                "val_halt_loss": round(summary.val_halt_loss, 6),
                "eval_wall_seconds": round(summary.wall_seconds, 6),
            },
        )
        print(
            f"step={step:05d}/{steps} train_loss={summary.train_loss:.4f} ce_loss={summary.ce_loss:.4f} "
            f"halt_loss={summary.halt_loss:.4f} val_loss={summary.val_loss:.4f} avg_depth={summary.avg_depth:.3f}",
            flush=True,
        )

    if len(checkpoints) == 0:
        raise RuntimeError("training loop completed without producing any evaluation checkpoint")

    training = TrainingSummary(
        final_step=steps,
        final_train_loss=last_train_loss,
        final_ce_loss=last_ce_loss,
        final_halt_loss=last_halt_loss,
        wall_seconds=perf_counter() - started_at,
    )
    checkpoint_path = args.report_path.parent / "checkpoint.pt"
    final_eval = checkpoints[-1]
    save_checkpoint(
        checkpoint_path,
        model=model,
        config=config,
        corpus=corpus,
        corpus_paths=corpus_paths,
        device=device,
        steps=steps,
        batch_size=batch_size,
        learning_rate=args.lr,
        training=training,
        final_eval=final_eval,
    )
    write_report(
        args.report_path,
        config=config,
        corpus=corpus,
        corpus_paths=corpus_paths,
        parameter_count=parameter_count,
        device=device,
        sanity_check_only=args.sanity_check_only,
        steps=steps,
        batch_size=batch_size,
        learning_rate=args.lr,
        training=training,
        checkpoints=checkpoints,
        checkpoint_path=checkpoint_path,
    )
    append_log(
        args.log_path,
        {
            "stage": "run_completed",
            "report_path": str(args.report_path),
            "checkpoint_path": str(checkpoint_path),
            "training_wall_seconds": round(training.wall_seconds, 6),
            "final_step": final_eval.step,
            "final_train_loss": round(final_eval.train_loss, 6),
            "final_val_loss": round(final_eval.val_loss, 6),
            "final_avg_depth": round(final_eval.avg_depth, 6),
            "final_eval_wall_seconds": round(final_eval.wall_seconds, 6),
        },
    )


if __name__ == "__main__":
    main()
