from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from base_experiments.transformer import build_fixed_length_split
from core.fixed_window_char import resolve_device, set_seed

from .char_model import AsyncGRUCharModel, TrainableConfig, VariantSpec, count_parameters, make_async_stale_variant, make_sync_variant, with_overrides
from .data import (
    artifact_stem,
    corpus_summary,
    current_git_sha,
    current_git_status_short,
    preprocess_corpus_text,
    resolve_text_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--corpus", choices=["tinyshakespeare", "enwik8"], default="tinyshakespeare")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-characters", type=int)
    parser.add_argument("--val-characters", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--early-stop-patience", type=int)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--allow-vocab-growth", action="store_true")
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def make_model(*, vocab_size: int, config: TrainableConfig, variant: VariantSpec, device: torch.device) -> AsyncGRUCharModel:
    return AsyncGRUCharModel(vocab_size=vocab_size, config=config, variant=variant).to(device)


def batched_pairs(inputs: Tensor, targets: Tensor, *, batch_size: int):
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        yield inputs[start:stop], targets[start:stop]


def evaluate_model(model: nn.Module, inputs: Tensor, targets: Tensor, *, batch_size: int) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in batched_pairs(inputs, targets, batch_size=batch_size):
            logits = model(batch_inputs)
            total_loss += F.cross_entropy(logits, batch_targets, reduction="sum").item()
            total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
            total_examples += batch_targets.shape[0]
    if was_training:
        model.train()
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def fixed_step_indices(size: int, *, steps: int, batch_size: int, seed: int, device: torch.device) -> list[Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return [torch.randint(0, size, (batch_size,), generator=generator).to(device) for _ in range(steps)]


def train_variant(
    *,
    model: AsyncGRUCharModel,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    config: TrainableConfig,
    step_indices: list[Tensor],
    eval_every: int,
    early_stop_patience: int | None,
    on_evaluation=None,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int | bool]] = []
    best_eval: dict[str, float | int] | None = None
    non_improving_evals = 0
    train_loss_sum = 0.0
    train_correct_sum = 0
    train_examples = 0
    interval_start = time.perf_counter()
    stop_reason = "max_steps_reached"

    model.train()
    for step, indices in enumerate(step_indices, start=1):
        batch_inputs = train_inputs[indices]
        batch_targets = train_targets[indices]
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

        batch_examples = batch_targets.shape[0]
        train_examples += batch_examples
        train_loss_sum += loss.item() * batch_examples
        train_correct_sum += (logits.argmax(dim=1) == batch_targets).sum().item()

        if step % eval_every != 0:
            continue

        if train_inputs.device.type == "cuda":
            torch.cuda.synchronize()
        eval_metrics = evaluate_model(model, val_inputs, val_targets, batch_size=config.eval_batch_size)
        if train_inputs.device.type == "cuda":
            torch.cuda.synchronize()
        interval_seconds = time.perf_counter() - interval_start
        train_metrics = {
            "loss": train_loss_sum / train_examples,
            "accuracy": train_correct_sum / train_examples,
        }
        is_best = best_eval is None or eval_metrics["loss"] < best_eval["val_loss"]
        if is_best:
            best_eval = {
                "step": step,
                "val_loss": round(eval_metrics["loss"], 6),
                "val_accuracy": round(eval_metrics["accuracy"], 6),
            }
            non_improving_evals = 0
        else:
            non_improving_evals += 1

        history.append(
            {
                "step": step,
                "train_loss": round(train_metrics["loss"], 6),
                "train_accuracy": round(train_metrics["accuracy"], 6),
                "val_loss": round(eval_metrics["loss"], 6),
                "val_accuracy": round(eval_metrics["accuracy"], 6),
                "interval_seconds": round(interval_seconds, 3),
                "is_best_so_far": is_best,
                "non_improving_evals": non_improving_evals,
            }
        )

        if on_evaluation is not None:
            on_evaluation(
                {
                    "history": history,
                    "best_eval": best_eval,
                    "final_eval": history[-1],
                    "completed_steps": history[-1]["step"],
                    "stop_reason": stop_reason,
                    "still_improving_at_end": best_eval["step"] == history[-1]["step"],
                    "final_minus_best_val_loss": round(history[-1]["val_loss"] - best_eval["val_loss"], 6),
                }
            )

        train_loss_sum = 0.0
        train_correct_sum = 0
        train_examples = 0
        interval_start = time.perf_counter()

        if early_stop_patience is not None and non_improving_evals >= early_stop_patience:
            stop_reason = "early_stop_non_improving_evals"
            break

    if best_eval is None:
        raise RuntimeError("No evaluation points were recorded.")

    final_eval = history[-1]
    return {
        "history": history,
        "best_eval": best_eval,
        "final_eval": final_eval,
        "completed_steps": final_eval["step"],
        "stop_reason": stop_reason,
        "still_improving_at_end": best_eval["step"] == final_eval["step"],
        "final_minus_best_val_loss": round(final_eval["val_loss"] - best_eval["val_loss"], 6),
    }


def analysis_summary(*, sync_result: dict[str, object], async_result: dict[str, object], max_steps: int) -> dict[str, object]:
    sync_best_step = int(sync_result["best_eval"]["step"])
    async_best_step = int(async_result["best_eval"]["step"])
    return {
        "best_val_gap_async_minus_sync": round(
            float(async_result["best_eval"]["val_loss"]) - float(sync_result["best_eval"]["val_loss"]),
            6,
        ),
        "best_step": {
            "synchronous_control": sync_best_step,
            "async_stale_reads": async_best_step,
        },
        "still_improving_at_end": {
            "synchronous_control": bool(sync_result["still_improving_at_end"]),
            "async_stale_reads": bool(async_result["still_improving_at_end"]),
        },
        "peaks_before_step_1500": {
            "synchronous_control": sync_best_step < 1500,
            "async_stale_reads": async_best_step < 1500,
        },
        "headroom_remaining_at_budget_end": bool(sync_result["still_improving_at_end"] or async_result["still_improving_at_end"]),
        "both_variants_peaked_before_1500": bool(sync_best_step < 1500 and async_best_step < 1500),
        "completed_max_steps": max_steps,
    }


def build_report(
    *,
    args: argparse.Namespace,
    config: TrainableConfig,
    device: torch.device,
    text_file: Path,
    raw_text: str,
    preprocessing: dict[str, object],
    split,
    sync_variant: VariantSpec,
    async_variant: VariantSpec,
    sync_model: AsyncGRUCharModel,
    sync_result: dict[str, object] | None,
    async_result: dict[str, object] | None,
) -> dict[str, object]:
    training = {}
    if sync_result is not None:
        training[sync_variant.label] = sync_result
    if async_result is not None:
        training[async_variant.label] = async_result

    report = {
        "config": asdict(config),
        "training_config": {
            "max_steps": args.max_steps,
            "eval_every": args.eval_every,
            "early_stop_patience": args.early_stop_patience,
        },
        "environment": {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": current_git_status_short() == [],
            "git_status_short": current_git_status_short(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "corpus_summary": corpus_summary(
            corpus=args.corpus,
            text_file=text_file,
            raw_text=raw_text,
            train_characters=len(split.train_text),
            val_characters=len(split.val_text),
            vocab_size=split.train_dataset.vocab_size,
            preprocessing=preprocessing,
        ),
        "model": {
            "parameter_count": count_parameters(sync_model),
            "d_model": config.d_model,
            "num_modules": config.num_modules,
            "num_ticks": config.num_ticks,
            "read_lags": {
                sync_variant.label: list(sync_variant.read_lags),
                async_variant.label: list(async_variant.read_lags),
            },
        },
        "training": training,
    }
    if sync_result is not None and async_result is not None:
        report["analysis"] = analysis_summary(sync_result=sync_result, async_result=async_result, max_steps=args.max_steps)
    return report


def main() -> None:
    args = parse_args()
    if args.max_steps % args.eval_every != 0:
        raise ValueError("max_steps must be divisible by eval_every so the last step is evaluated.")

    config = with_overrides(
        TrainableConfig(),
        train_characters=args.train_characters,
        val_characters=args.val_characters,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
    )
    text_file = resolve_text_file(
        repo_root=args.repo_root,
        corpus=args.corpus,
        text_file=args.text_file,
        download_missing=args.download_missing,
    )
    output_dir = args.output_dir or (args.repo_root / "experiments" / "async_gru_corpus" / "artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = resolve_device(args.device)
    source_text = text_file.read_text(encoding="utf-8")
    raw_text, preprocessing = preprocess_corpus_text(
        repo_root=args.repo_root,
        corpus=args.corpus,
        text_file=text_file,
        raw_text=source_text,
        allow_vocab_growth=args.allow_vocab_growth,
    )
    split = build_fixed_length_split(raw_text, config=config)
    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)
    vocab_size = split.train_dataset.vocab_size

    sync_variant = make_sync_variant(config)
    async_variant = make_async_stale_variant(config)
    sync_model = make_model(vocab_size=vocab_size, config=config, variant=sync_variant, device=device)
    async_model = make_model(vocab_size=vocab_size, config=config, variant=async_variant, device=device)
    async_model.load_state_dict(sync_model.state_dict())
    output_path = output_dir / f"training_{artifact_stem(corpus=args.corpus, train_characters=len(split.train_text), val_characters=len(split.val_text))}.json"
    partial_output_path = output_dir / f"training_{artifact_stem(corpus=args.corpus, train_characters=len(split.train_text), val_characters=len(split.val_text))}.partial.json"

    sync_result: dict[str, object] | None = None
    async_result: dict[str, object] | None = None

    def write_progress() -> None:
        report = build_report(
            args=args,
            config=config,
            device=device,
            text_file=text_file,
            raw_text=raw_text,
            preprocessing=preprocessing,
            split=split,
            sync_variant=sync_variant,
            async_variant=async_variant,
            sync_model=sync_model,
            sync_result=sync_result,
            async_result=async_result,
        )
        write_json(partial_output_path, report)

    def update_sync_progress(result: dict[str, object]) -> None:
        nonlocal sync_result
        sync_result = result
        write_progress()

    def update_async_progress(result: dict[str, object]) -> None:
        nonlocal async_result
        async_result = result
        write_progress()

    step_indices = fixed_step_indices(
        train_inputs.shape[0],
        steps=args.max_steps,
        batch_size=config.batch_size,
        seed=config.seed,
        device=device,
    )

    sync_result = train_variant(
        model=sync_model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        config=config,
        step_indices=step_indices,
        eval_every=args.eval_every,
        early_stop_patience=args.early_stop_patience,
        on_evaluation=update_sync_progress,
    )
    write_progress()
    async_result = train_variant(
        model=async_model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        config=config,
        step_indices=step_indices,
        eval_every=args.eval_every,
        early_stop_patience=args.early_stop_patience,
        on_evaluation=update_async_progress,
    )
    write_progress()
    report = build_report(
        args=args,
        config=config,
        device=device,
        text_file=text_file,
        raw_text=raw_text,
        preprocessing=preprocessing,
        split=split,
        sync_variant=sync_variant,
        async_variant=async_variant,
        sync_model=sync_model,
        sync_result=sync_result,
        async_result=async_result,
    )
    write_json(output_path, report)
    partial_output_path.unlink(missing_ok=True)
    summary = report["analysis"]
    print("PASS async_gru_corpus training")
    print(f"Wrote {output_path}")
    print(summary)


if __name__ == "__main__":
    main()
