from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from core.dataset import CorpusData, load_corpus
from core.fixed_window_char import set_seed
from core.model import ParallelDiagonalModel, count_parameters
from core.training import (
    GraphTrainer,
    capturable_adamw,
    current_git_sha,
    current_git_status_short,
    evaluate_model,
    write_json,
)

CONTEXT_SIZE = 32
TRAINING_STEPS = 20_000
EVAL_INTERVAL = 1_000
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 3e-4
DEFAULT_SEEDS = (42, 43)
WARMUP_STEPS = 3
D_MODEL = 256
FEEDFORWARD_DIM = 512
FOUR_BLOCK_RATES = (1, 1, 1, 1)


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    d_model: int
    feedforward_dim: int
    num_blocks: int
    rates: tuple[int, ...]
    readout_mode: str
    token_injection: str
    topology: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "wikitext_103" / "artifacts" / "baseline"
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--train-path",
        type=Path,
        default=repo_root / "data" / "wikitext-103-raw" / "wiki.train.raw",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=repo_root / "data" / "wikitext-103-raw" / "wiki.valid.raw",
    )
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.jsonl")
    return parser.parse_args()


def variant_specs() -> dict[str, VariantSpec]:
    return {
        "A_single_block": VariantSpec(
            key="A_single_block",
            label="wikitext_baseline_A_single_block",
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=1,
            rates=(1,),
            readout_mode="last",
            token_injection="block0",
            topology="upward",
        ),
        "B_4block_old": VariantSpec(
            key="B_4block_old",
            label="wikitext_baseline_B_4block_old",
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=4,
            rates=FOUR_BLOCK_RATES,
            readout_mode="all",
            token_injection="all",
            topology="upward",
        ),
        "C_4block_corrected": VariantSpec(
            key="C_4block_corrected",
            label="wikitext_baseline_C_4block_corrected",
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=4,
            rates=FOUR_BLOCK_RATES,
            readout_mode="all",
            token_injection="block0",
            topology="upward",
        ),
        "D_4block_bidirectional": VariantSpec(
            key="D_4block_bidirectional",
            label="wikitext_baseline_D_4block_bidirectional",
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=4,
            rates=FOUR_BLOCK_RATES,
            readout_mode="first",
            token_injection="block0",
            topology="top_down_to_first",
        ),
    }


def build_model(
    *,
    device: torch.device,
    vocab_size: int,
    spec: VariantSpec,
) -> ParallelDiagonalModel:
    return ParallelDiagonalModel(
        vocab_size=vocab_size,
        context_size=CONTEXT_SIZE,
        d_model=spec.d_model,
        feedforward_dim=spec.feedforward_dim,
        num_blocks=spec.num_blocks,
        rates=spec.rates,
        readout_mode=spec.readout_mode,
        token_injection=spec.token_injection,
        topology=spec.topology,
    ).to(device)


def append_log(log_path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def checkpoint_metrics(
    model: torch.nn.Module,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    batch_size: int,
    step: int,
) -> dict[str, float | int]:
    metrics = evaluate_model(model, val_inputs, val_targets, batch_size=batch_size)
    return {
        "step": step,
        "val_loss": round(metrics["loss"], 6),
        "val_accuracy": round(metrics["accuracy"], 6),
    }


def readout_weights_or_none(model: ParallelDiagonalModel) -> list[float] | None:
    mix_coefficients = model.mix_coefficients()
    readout_weights = mix_coefficients.get("readout_weights")
    if readout_weights is None:
        return None
    return [round(float(weight), 6) for weight in readout_weights]


def batch_to_device(
    batch: tuple[torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = batch
    return (
        inputs.to(device=device, dtype=torch.long, non_blocking=True),
        targets.to(device=device, dtype=torch.long, non_blocking=True),
    )


def build_dataloader(
    corpus: CorpusData,
    *,
    batch_size: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    return DataLoader(
        corpus.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )


def dataloader_batches(
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    while True:
        for batch in dataloader:
            yield batch_to_device(batch, device=device)


def mean_rounded(values: list[float]) -> float:
    return round(mean(values), 6)


def std_rounded(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(pstdev(values), 6)


def train_single_variant(
    *,
    seed: int,
    variant_key: str,
    spec: VariantSpec,
    args: argparse.Namespace,
    corpus: CorpusData,
    device: torch.device,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
) -> dict[str, object]:
    set_seed(seed)
    dataloader = build_dataloader(corpus, batch_size=args.batch_size)
    batch_iterator = dataloader_batches(dataloader, device=device)
    warmup_batches = [next(batch_iterator) for _ in range(WARMUP_STEPS)]

    set_seed(seed)
    model = build_model(device=device, vocab_size=corpus.vocab_size, spec=spec)
    optimizer = capturable_adamw(model, lr=args.learning_rate)
    trainer = GraphTrainer(
        model,
        optimizer,
        batch_size=args.batch_size,
        seq_len=CONTEXT_SIZE,
        device=device,
    )

    checkpoints = [
        checkpoint_metrics(
            model,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=0,
        )
    ]
    parameter_count = count_parameters(model)
    append_log(
        args.log_path,
        {
            "stage": "variant_started",
            "seed": seed,
            "variant": variant_key,
            "label": spec.label,
            "parameter_count": parameter_count,
            "initial_checkpoint": checkpoints[-1],
            "readout_weights": readout_weights_or_none(model),
        },
    )

    started_at = perf_counter()
    trainer.capture(warmup_batches)
    last_loss = trainer.static_loss.detach().clone()

    for step in range(WARMUP_STEPS + 1, args.training_steps + 1):
        batch_input, batch_target = next(batch_iterator)
        last_loss = trainer.step(batch_input, batch_target)
        if step % args.eval_interval != 0 and step != args.training_steps:
            continue

        trainer.synchronize()
        checkpoint = checkpoint_metrics(
            model,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=step,
        )
        checkpoints.append(checkpoint)
        append_log(
            args.log_path,
            {
                "stage": "checkpoint",
                "seed": seed,
                "variant": variant_key,
                **checkpoint,
                "readout_weights": readout_weights_or_none(model),
            },
        )

    trainer.synchronize()
    wall_seconds = perf_counter() - started_at
    result = {
        "seed": seed,
        "variant": variant_key,
        "label": spec.label,
        "class_name": "ParallelDiagonalModel",
        "d_model": spec.d_model,
        "feedforward_dim": spec.feedforward_dim,
        "num_blocks": spec.num_blocks,
        "rates": list(spec.rates),
        "readout_mode": spec.readout_mode,
        "token_injection": spec.token_injection,
        "topology": spec.topology,
        "parameter_count": parameter_count,
        "checkpoints": checkpoints,
        "best_checkpoint": min(checkpoints, key=lambda checkpoint: checkpoint["val_loss"]),
        "final_checkpoint": checkpoints[-1],
        "final_training_loss": round(last_loss.item(), 6),
        "wall_seconds": round(wall_seconds, 6),
        "mix_coefficients": model.mix_coefficients(),
    }
    append_log(
        args.log_path,
        {
            "stage": "variant_done",
            "seed": seed,
            "variant": variant_key,
            "final_checkpoint": result["final_checkpoint"],
            "final_training_loss": result["final_training_loss"],
            "wall_seconds": result["wall_seconds"],
            "readout_weights": readout_weights_or_none(model),
        },
    )

    del trainer
    del optimizer
    del model
    del dataloader
    del batch_iterator
    gc.collect()
    torch.cuda.empty_cache()
    return result


def summarize_results(
    *,
    per_seed_results: list[dict[str, object]],
    specs: dict[str, VariantSpec],
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {key: [] for key in specs}
    for result in per_seed_results:
        grouped[result["variant"]].append(result)

    summary: dict[str, object] = {}
    for key, runs in grouped.items():
        final_losses = [run["final_checkpoint"]["val_loss"] for run in runs]
        final_accuracies = [run["final_checkpoint"]["val_accuracy"] for run in runs]
        best_losses = [run["best_checkpoint"]["val_loss"] for run in runs]
        best_accuracies = [run["best_checkpoint"]["val_accuracy"] for run in runs]
        wall_seconds = [run["wall_seconds"] for run in runs]
        summary[key] = {
            "label": specs[key].label,
            "readout_mode": specs[key].readout_mode,
            "token_injection": specs[key].token_injection,
            "topology": specs[key].topology,
            "num_runs": len(runs),
            "mean_final_val_loss": mean_rounded(final_losses),
            "std_final_val_loss": std_rounded(final_losses),
            "mean_final_val_accuracy": mean_rounded(final_accuracies),
            "std_final_val_accuracy": std_rounded(final_accuracies),
            "mean_best_val_loss": mean_rounded(best_losses),
            "mean_best_val_accuracy": mean_rounded(best_accuracies),
            "mean_wall_seconds": mean_rounded(wall_seconds),
            "runs": runs,
        }
    return summary


def comparison(summary_by_variant: dict[str, object]) -> dict[str, float]:
    return {
        "mean_final_val_loss_delta_B_minus_A": round(
            summary_by_variant["B_4block_old"]["mean_final_val_loss"]
            - summary_by_variant["A_single_block"]["mean_final_val_loss"],
            6,
        ),
        "mean_final_val_accuracy_delta_B_minus_A": round(
            summary_by_variant["B_4block_old"]["mean_final_val_accuracy"]
            - summary_by_variant["A_single_block"]["mean_final_val_accuracy"],
            6,
        ),
        "mean_final_val_loss_delta_C_minus_A": round(
            summary_by_variant["C_4block_corrected"]["mean_final_val_loss"]
            - summary_by_variant["A_single_block"]["mean_final_val_loss"],
            6,
        ),
        "mean_final_val_accuracy_delta_C_minus_A": round(
            summary_by_variant["C_4block_corrected"]["mean_final_val_accuracy"]
            - summary_by_variant["A_single_block"]["mean_final_val_accuracy"],
            6,
        ),
        "mean_final_val_loss_delta_D_minus_A": round(
            summary_by_variant["D_4block_bidirectional"]["mean_final_val_loss"]
            - summary_by_variant["A_single_block"]["mean_final_val_loss"],
            6,
        ),
        "mean_final_val_accuracy_delta_D_minus_A": round(
            summary_by_variant["D_4block_bidirectional"]["mean_final_val_accuracy"]
            - summary_by_variant["A_single_block"]["mean_final_val_accuracy"],
            6,
        ),
        "mean_final_val_loss_delta_B_minus_C": round(
            summary_by_variant["B_4block_old"]["mean_final_val_loss"]
            - summary_by_variant["C_4block_corrected"]["mean_final_val_loss"],
            6,
        ),
        "mean_final_val_accuracy_delta_B_minus_C": round(
            summary_by_variant["B_4block_old"]["mean_final_val_accuracy"]
            - summary_by_variant["C_4block_corrected"]["mean_final_val_accuracy"],
            6,
        ),
        "mean_final_val_loss_delta_D_minus_C": round(
            summary_by_variant["D_4block_bidirectional"]["mean_final_val_loss"]
            - summary_by_variant["C_4block_corrected"]["mean_final_val_loss"],
            6,
        ),
        "mean_final_val_accuracy_delta_D_minus_C": round(
            summary_by_variant["D_4block_bidirectional"]["mean_final_val_accuracy"]
            - summary_by_variant["C_4block_corrected"]["mean_final_val_accuracy"],
            6,
        ),
    }


def main() -> int:
    args = parse_args()
    if len(args.seeds) == 0:
        raise ValueError("At least one seed is required.")
    if args.training_steps < WARMUP_STEPS:
        raise ValueError(
            f"training_steps must be at least {WARMUP_STEPS} so GraphTrainer.capture() can warm up."
        )
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}.")
    if args.eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {args.eval_batch_size}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/wikitext_baseline.py.")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    if args.log_path.exists():
        previous_log = args.log_path.read_text(encoding="utf-8")
        if previous_log:
            append_log(
                args.log_path,
                {
                    "stage": "run_restarted",
                    "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "previous_lines": len(previous_log.splitlines()),
                },
            )

    device = torch.device("cuda")
    corpus = load_corpus(
        train_path=args.train_path,
        val_path=args.val_path,
        context_size=CONTEXT_SIZE,
        eval_samples=1024,
    )
    val_inputs = corpus.val_inputs.to(device=device, dtype=torch.long)
    val_targets = corpus.val_targets.to(device=device, dtype=torch.long)

    specs = variant_specs()
    append_log(
        args.log_path,
        {
            "stage": "experiment_started",
            "seeds": args.seeds,
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "device": str(device),
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
            "vocab_size": corpus.vocab_size,
            "train_dataset_size": len(corpus.train_dataset),
            "val_examples": int(corpus.val_inputs.shape[0]),
            "trainer": "GraphTrainer",
        },
    )

    overall_started_at = perf_counter()
    per_seed_results: list[dict[str, object]] = []
    for seed in args.seeds:
        append_log(args.log_path, {"stage": "seed_started", "seed": seed})
        for variant_key, spec in specs.items():
            per_seed_results.append(
                train_single_variant(
                    seed=seed,
                    variant_key=variant_key,
                    spec=spec,
                    args=args,
                    corpus=corpus,
                    device=device,
                    val_inputs=val_inputs,
                    val_targets=val_targets,
                )
            )
        append_log(args.log_path, {"stage": "seed_done", "seed": seed})

    wall_seconds = perf_counter() - overall_started_at
    git_status_short = current_git_status_short()
    summary_by_variant = summarize_results(per_seed_results=per_seed_results, specs=specs)
    report = {
        "config": {
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
            "warmup_steps": WARMUP_STEPS,
            "context_size": CONTEXT_SIZE,
            "dataset_vocab_size": corpus.vocab_size,
            "trainer": "GraphTrainer",
            "dataset": "wikitext-103-raw",
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
        },
        "environment": {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": git_status_short == [],
            "git_status_short": git_status_short,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
        },
        "dataset": {
            "train_examples": len(corpus.train_dataset),
            "val_examples": int(corpus.val_inputs.shape[0]),
            "vocab_size": corpus.vocab_size,
        },
        "timing": {
            "overall_wall_seconds": round(wall_seconds, 6),
        },
        "variants": {key: asdict(spec) for key, spec in specs.items()},
        "per_seed_results": per_seed_results,
        "summary_by_variant": summary_by_variant,
        "comparison": comparison(summary_by_variant),
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "done",
            "report_path": str(args.report_path),
            "summary_by_variant": {
                key: {
                    "mean_final_val_loss": value["mean_final_val_loss"],
                    "mean_final_val_accuracy": value["mean_final_val_accuracy"],
                }
                for key, value in summary_by_variant.items()
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
