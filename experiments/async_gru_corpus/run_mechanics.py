from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import torch

from base_experiments.transformer import build_fixed_length_split
from core.fixed_window_char import resolve_device, set_seed

from .char_model import (
    AsyncGRUCharModel,
    TrainableConfig,
    VariantSpec,
    count_parameters,
    make_async_stale_variant,
    make_async_zero_variant,
    make_sync_variant,
    with_overrides,
)
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
    parser.add_argument("--sample-batch-size", type=int, default=2)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--allow-vocab-growth", action="store_true")
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def make_model(*, vocab_size: int, config: TrainableConfig, variant: VariantSpec, device: torch.device) -> AsyncGRUCharModel:
    return AsyncGRUCharModel(vocab_size=vocab_size, config=config, variant=variant).to(device)


def check_shape(name: str, tensor: torch.Tensor, expected_shape: tuple[int, ...]) -> list[int]:
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise RuntimeError(f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}.")
    return list(actual_shape)


def max_history_abs_diff(left_history: list[torch.Tensor], right_history: list[torch.Tensor]) -> float:
    return max((left - right).abs().max().item() for left, right in zip(left_history, right_history, strict=True))


def zero_staleness_equivalence(sync_trace, zero_trace, sync_logits: torch.Tensor, zero_logits: torch.Tensor) -> dict[str, float]:
    history_diff = max_history_abs_diff(sync_trace.history, zero_trace.history)
    logits_diff = (sync_logits - zero_logits).abs().max().item()
    if history_diff > 1e-7 or logits_diff > 1e-7:
        raise RuntimeError(
            "Zero-staleness equivalence failed: "
            f"history_diff={history_diff}, logits_diff={logits_diff}."
        )
    return {
        "max_history_abs_diff": round(float(history_diff), 9),
        "max_logits_abs_diff": round(float(logits_diff), 9),
    }


def stale_read_witness(async_trace) -> dict[str, object]:
    for tick_trace in async_trace.tick_traces:
        if len(set(tick_trace.read_version_ids)) <= 1:
            continue
        stale_rows = []
        for module_index, (read_version_id, max_diff) in enumerate(
            zip(tick_trace.read_version_ids, tick_trace.max_read_vs_latest_abs_diff, strict=True)
        ):
            if read_version_id >= tick_trace.latest_version_id:
                continue
            stale_rows.append(
                {
                    "module_index": module_index,
                    "read_version_id": read_version_id,
                    "latest_version_id": tick_trace.latest_version_id,
                    "max_read_vs_latest_abs_diff": round(float(max_diff), 6),
                }
            )
        if stale_rows:
            max_gap = max(row["max_read_vs_latest_abs_diff"] for row in stale_rows)
            if max_gap <= 0.0:
                raise RuntimeError("Async stale-read witness found version skew but no numeric difference.")
            return {
                "witness_tick": tick_trace.tick,
                "latest_version_id": tick_trace.latest_version_id,
                "read_version_ids": list(tick_trace.read_version_ids),
                "stale_modules": stale_rows,
                "max_read_vs_latest_abs_diff": max_gap,
            }
    raise RuntimeError("No stale-read witness tick found.")


def trace_preview(trace) -> list[dict[str, object]]:
    return [
        {
            "tick": tick_trace.tick,
            "latest_version_id": tick_trace.latest_version_id,
            "read_version_ids": list(tick_trace.read_version_ids),
            "max_read_vs_latest_abs_diff": [round(float(value), 6) for value in tick_trace.max_read_vs_latest_abs_diff],
        }
        for tick_trace in trace.tick_traces
    ]


def main() -> None:
    args = parse_args()
    config = with_overrides(
        TrainableConfig(),
        train_characters=args.train_characters,
        val_characters=args.val_characters,
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
    sample_inputs = split.train_inputs[: args.sample_batch_size].to(device)
    vocab_size = split.train_dataset.vocab_size

    sync_variant = make_sync_variant(config)
    zero_variant = make_async_zero_variant(config)
    async_variant = make_async_stale_variant(config)

    sync_model = make_model(vocab_size=vocab_size, config=config, variant=sync_variant, device=device)
    zero_model = make_model(vocab_size=vocab_size, config=config, variant=zero_variant, device=device)
    async_model = make_model(vocab_size=vocab_size, config=config, variant=async_variant, device=device)
    zero_model.load_state_dict(sync_model.state_dict())
    async_model.load_state_dict(sync_model.state_dict())

    sync_logits, sync_trace = sync_model.forward_with_trace(sample_inputs)
    zero_logits, zero_trace = zero_model.forward_with_trace(sample_inputs)
    async_logits, async_trace = async_model.forward_with_trace(sample_inputs)

    expected_memory_shape = (args.sample_batch_size, config.context_size, config.d_model)
    expected_logits_shape = (args.sample_batch_size, vocab_size)
    sync_parameter_count = count_parameters(sync_model)
    zero_parameter_count = count_parameters(zero_model)
    async_parameter_count = count_parameters(async_model)
    if len({sync_parameter_count, zero_parameter_count, async_parameter_count}) != 1:
        raise RuntimeError(
            "Parameter count mismatch across variants: "
            f"sync={sync_parameter_count}, zero={zero_parameter_count}, async={async_parameter_count}."
        )

    report = {
        "config": asdict(config),
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
            vocab_size=vocab_size,
            preprocessing=preprocessing,
        ),
        "model": {
            "parameter_count": sync_parameter_count,
            "d_model": config.d_model,
            "num_modules": config.num_modules,
            "num_ticks": config.num_ticks,
            "read_lags": {
                sync_variant.label: list(sync_variant.read_lags),
                zero_variant.label: list(zero_variant.read_lags),
                async_variant.label: list(async_variant.read_lags),
            },
        },
        "checks": {
            "shapes": {
                "input_tokens": check_shape("input_tokens", sample_inputs, (args.sample_batch_size, config.context_size)),
                "final_memory_sync": check_shape("final_memory_sync", sync_trace.history[-1], expected_memory_shape),
                "final_memory_async": check_shape("final_memory_async", async_trace.history[-1], expected_memory_shape),
                "logits_sync": check_shape("logits_sync", sync_logits, expected_logits_shape),
                "logits_async": check_shape("logits_async", async_logits, expected_logits_shape),
            },
            "parameter_count_match": {
                "synchronous_control": sync_parameter_count,
                "async_zero_staleness": zero_parameter_count,
                "async_stale_reads": async_parameter_count,
            },
            "zero_staleness_equivalence": zero_staleness_equivalence(sync_trace, zero_trace, sync_logits, zero_logits),
            "stale_read_witness": stale_read_witness(async_trace),
            "async_vs_sync_output_delta": {
                "max_history_abs_diff": round(float(max_history_abs_diff(sync_trace.history, async_trace.history)), 6),
                "max_logits_abs_diff": round(float((sync_logits - async_logits).abs().max().item()), 6),
            },
        },
        "trace_preview": {
            sync_variant.label: trace_preview(sync_trace),
            async_variant.label: trace_preview(async_trace),
        },
    }
    output_path = output_dir / f"mechanics_{artifact_stem(corpus=args.corpus, train_characters=len(split.train_text), val_characters=len(split.val_text))}.json"
    write_json(output_path, report)
    print("PASS async_gru_corpus mechanics")
    print(f"Wrote {output_path}")
    print(
        {
            "parameter_count": sync_parameter_count,
            "zero_staleness_equivalence": report["checks"]["zero_staleness_equivalence"],
            "stale_read_witness": report["checks"]["stale_read_witness"],
        }
    )


if __name__ == "__main__":
    main()
