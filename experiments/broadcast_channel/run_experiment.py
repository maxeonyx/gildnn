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
from experiments.async_gru_corpus.data import (
    artifact_stem,
    corpus_summary,
    current_git_sha,
    current_git_status_short,
    preprocess_corpus_text,
    resolve_text_file,
    write_json,
)

from .char_model import (
    BroadcastAsyncGRUCharModel,
    BroadcastVariantSpec,
    TrainableConfig,
    count_parameters,
    make_async_broadcast_variant,
    make_async_stale_variant,
    make_async_zero_variant,
    make_sync_variant,
    with_overrides,
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
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument(
        "--variant",
        choices=["all", "sync_no_broadcast", "async_no_broadcast", "async_with_broadcast"],
        default="all",
    )
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--allow-vocab-growth", action="store_true")
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def make_model(*, vocab_size: int, config: TrainableConfig, variant: BroadcastVariantSpec, device: torch.device) -> BroadcastAsyncGRUCharModel:
    return BroadcastAsyncGRUCharModel(vocab_size=vocab_size, config=config, variant=variant).to(device)


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


def clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def load_shared_weights(source_state: dict[str, Tensor], target_model: BroadcastAsyncGRUCharModel) -> None:
    load_result = target_model.load_state_dict(source_state, strict=False)
    missing_keys = set(load_result.missing_keys)
    unexpected_keys = set(load_result.unexpected_keys)
    expected_missing = set()
    if target_model.variant.use_broadcast:
        expected_missing = {"broadcast_proj.weight", "broadcast_proj.bias"}
    if missing_keys != expected_missing or unexpected_keys:
        raise RuntimeError(
            "Shared-weight load mismatch: "
            f"missing={sorted(missing_keys)} unexpected={sorted(unexpected_keys)}"
        )


def verify_zero_lag_equivalence(
    *,
    sync_model: BroadcastAsyncGRUCharModel,
    async_zero_model: BroadcastAsyncGRUCharModel,
    sample_inputs: Tensor,
) -> dict[str, object]:
    sync_model.eval()
    async_zero_model.eval()
    with torch.no_grad():
        sync_logits, sync_trace = sync_model.forward_with_trace(sample_inputs)
        async_logits, async_trace = async_zero_model.forward_with_trace(sample_inputs)
    logits_match = torch.equal(sync_logits, async_logits)
    trace_history_match = all(
        torch.equal(sync_hidden, async_hidden)
        for sync_hidden, async_hidden in zip(sync_trace.history, async_trace.history, strict=True)
    )
    tick_traces_match = sync_trace.tick_traces == async_trace.tick_traces
    max_abs_diff = float((sync_logits - async_logits).abs().max().item())
    if not (logits_match and trace_history_match and tick_traces_match):
        raise RuntimeError(
            "Zero-lag equivalence failed: "
            f"logits_match={logits_match} trace_history_match={trace_history_match} "
            f"tick_traces_match={tick_traces_match} max_abs_diff={max_abs_diff}"
        )
    return {
        "logits_match": logits_match,
        "trace_history_match": trace_history_match,
        "tick_traces_match": tick_traces_match,
        "max_abs_diff": max_abs_diff,
        "checked_batch_shape": list(sample_inputs.shape),
    }


def train_variant(
    *,
    model: BroadcastAsyncGRUCharModel,
    variant: BroadcastVariantSpec,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    config: TrainableConfig,
    step_indices: list[Tensor],
    eval_every: int,
    on_evaluation=None,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int | bool]] = []
    best_eval: dict[str, float | int] | None = None
    train_loss_sum = 0.0
    train_correct_sum = 0
    train_examples = 0
    interval_start = time.perf_counter()

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

        record = {
            "step": step,
            "train_loss": round(train_metrics["loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "val_loss": round(eval_metrics["loss"], 6),
            "val_accuracy": round(eval_metrics["accuracy"], 6),
            "interval_seconds": round(interval_seconds, 3),
            "is_best_so_far": is_best,
        }
        history.append(record)
        print(
            f"variant={variant.label} step={step} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={eval_metrics['loss']:.4f} val_acc={eval_metrics['accuracy']:.4f}"
        )

        if on_evaluation is not None:
            on_evaluation(
                {
                    "history": history,
                    "best_eval": best_eval,
                    "final_eval": history[-1],
                    "completed_steps": history[-1]["step"],
                    "stop_reason": "max_steps_reached",
                    "still_improving_at_end": False,
                    "final_minus_best_val_loss": round(history[-1]["val_loss"] - best_eval["val_loss"], 6),
                }
            )

        train_loss_sum = 0.0
        train_correct_sum = 0
        train_examples = 0
        interval_start = time.perf_counter()

    if best_eval is None:
        raise RuntimeError("No evaluation points were recorded.")

    final_eval = history[-1]
    return {
        "history": history,
        "best_eval": best_eval,
        "final_eval": final_eval,
        "completed_steps": final_eval["step"],
        "stop_reason": "max_steps_reached",
        "still_improving_at_end": best_eval["step"] == final_eval["step"],
        "final_minus_best_val_loss": round(final_eval["val_loss"] - best_eval["val_loss"], 6),
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
    zero_lag_equivalence: dict[str, object],
    variants: list[BroadcastVariantSpec],
    models: dict[str, BroadcastAsyncGRUCharModel],
    results: dict[str, dict[str, object] | None],
) -> dict[str, object]:
    parameter_counts = {
        variant.label: count_parameters(models[variant.label])
        for variant in variants
    }
    training = {
        label: result
        for label, result in results.items()
        if result is not None
    }
    sync_label = "sync_no_broadcast"
    async_label = "async_no_broadcast"
    broadcast_label = "async_with_broadcast"
    report = {
        "config": asdict(config),
        "training_config": {
            "max_steps": args.max_steps,
            "eval_every": args.eval_every,
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
        "zero_lag_equivalence": zero_lag_equivalence,
        "model": {
            "parameter_count": parameter_counts,
            "d_model": config.d_model,
            "num_modules": config.num_modules,
            "num_ticks": config.num_ticks,
            "read_lags": {variant.label: list(variant.read_lags) for variant in variants},
            "uses_broadcast": {variant.label: variant.use_broadcast for variant in variants},
        },
        "training": training,
    }
    if all(results.get(label) is not None for label in (sync_label, async_label, broadcast_label)):
        sync_best = float(results[sync_label]["best_eval"]["val_loss"])
        async_best = float(results[async_label]["best_eval"]["val_loss"])
        broadcast_best = float(results[broadcast_label]["best_eval"]["val_loss"])
        report["analysis"] = {
            "best_val_loss": {
                sync_label: sync_best,
                async_label: async_best,
                broadcast_label: broadcast_best,
            },
            "gap_async_minus_sync": round(async_best - sync_best, 6),
            "gap_async_broadcast_minus_sync": round(broadcast_best - sync_best, 6),
            "broadcast_improvement_vs_async": round(async_best - broadcast_best, 6),
            "broadcast_closed_fraction_of_gap": None if async_best == sync_best else round((async_best - broadcast_best) / (async_best - sync_best), 6),
            "best_step": {
                label: int(results[label]["best_eval"]["step"])
                for label in (sync_label, async_label, broadcast_label)
            },
        }
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
    output_dir = args.output_dir or (args.repo_root / "experiments" / "broadcast_channel" / "artifacts")
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

    variants = [
        make_sync_variant(config),
        make_async_zero_variant(config),
        make_async_stale_variant(config),
        make_async_broadcast_variant(config),
    ]
    models: dict[str, BroadcastAsyncGRUCharModel] = {}
    for variant in variants:
        set_seed(config.seed)
        models[variant.label] = make_model(vocab_size=vocab_size, config=config, variant=variant, device=device)

    initial_sync_state = clone_state_dict(models["sync_no_broadcast"])
    load_shared_weights(initial_sync_state, models["async_zero_no_broadcast"])
    load_shared_weights(initial_sync_state, models["async_no_broadcast"])
    load_shared_weights(initial_sync_state, models["async_with_broadcast"])

    zero_lag_equivalence = verify_zero_lag_equivalence(
        sync_model=models["sync_no_broadcast"],
        async_zero_model=models["async_zero_no_broadcast"],
        sample_inputs=train_inputs[: config.batch_size],
    )
    print(f"PASS zero-lag equivalence {zero_lag_equivalence}")

    parameter_counts = {label: count_parameters(model) for label, model in models.items() if label != "async_zero_no_broadcast"}
    print(f"parameter_counts={parameter_counts}")

    output_path = output_dir / f"training_{artifact_stem(corpus=args.corpus, train_characters=len(split.train_text), val_characters=len(split.val_text))}.json"
    partial_output_path = output_dir / f"training_{artifact_stem(corpus=args.corpus, train_characters=len(split.train_text), val_characters=len(split.val_text))}.partial.json"

    tracked_labels = ["sync_no_broadcast", "async_no_broadcast", "async_with_broadcast"]
    if args.variant != "all":
        tracked_labels = [args.variant]
    results: dict[str, dict[str, object] | None] = {label: None for label in tracked_labels}

    def write_progress() -> None:
        report = build_report(
            args=args,
            config=config,
            device=device,
            text_file=text_file,
            raw_text=raw_text,
            preprocessing=preprocessing,
            split=split,
            zero_lag_equivalence=zero_lag_equivalence,
            variants=[variant for variant in variants if variant.label != "async_zero_no_broadcast"],
            models={label: model for label, model in models.items() if label != "async_zero_no_broadcast"},
            results=results,
        )
        write_json(partial_output_path, report)

    step_indices = fixed_step_indices(
        train_inputs.shape[0],
        steps=args.max_steps,
        batch_size=config.batch_size,
        seed=config.seed,
        device=device,
    )

    for label in tracked_labels:
        variant = next(variant for variant in variants if variant.label == label)
        results[label] = train_variant(
            model=models[label],
            variant=variant,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            config=config,
            step_indices=step_indices,
            eval_every=args.eval_every,
            on_evaluation=lambda result, label=label: (results.__setitem__(label, result), write_progress()),
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
        zero_lag_equivalence=zero_lag_equivalence,
        variants=[variant for variant in variants if variant.label != "async_zero_no_broadcast"],
        models={label: model for label, model in models.items() if label != "async_zero_no_broadcast"},
        results=results,
    )
    if args.variant != "all":
        output_path = output_dir / f"training_{artifact_stem(corpus=args.corpus, train_characters=len(split.train_text), val_characters=len(split.val_text))}_{args.variant}.json"
    write_json(output_path, report)
    partial_output_path.unlink(missing_ok=True)
    print("PASS broadcast_channel training")
    print(f"Wrote {output_path}")
    if "analysis" in report:
        print(report["analysis"])


if __name__ == "__main__":
    main()
