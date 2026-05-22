from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import load_dataset, resolve_device, set_seed
from core.model import (
    MultiRateResidualModel,
    count_parameters,
)


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 32
    d_model: int = 128
    feedforward_dim: int = 256
    temporal_window: int = 4
    num_heads: int = 4
    batch_size: int = 64
    eval_batch_size: int = 512
    learning_rate: float = 3e-3
    training_steps: int = 5_000
    eval_interval: int = 500
    seed: int = 42
    num_blocks: int = 4
    control_rates: tuple[int, ...] = (1, 1, 1, 1)
    multi_rates: tuple[int, ...] = (1, 1, 2, 4)
    overfit_steps: int = 2_000
    overfit_loss_threshold: float = 1e-3
    checkpoint_timing_warmup_passes: int = 20
    checkpoint_timing_passes: int = 100
    final_timing_warmup_passes: int = 100
    final_timing_passes: int = 1_000
    control_diagonal: bool = False
    multi_diagonal: bool = False


def parse_rate_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(","))


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--training-steps", type=int)
    parser.add_argument("--overfit-steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--checkpoint-timing-warmup-passes", type=int)
    parser.add_argument("--checkpoint-timing-passes", type=int)
    parser.add_argument("--final-timing-warmup-passes", type=int)
    parser.add_argument("--final-timing-passes", type=int)
    parser.add_argument("--control-rates")
    parser.add_argument("--multi-rates", default="1,1,2,4")
    parser.add_argument("--control-diagonal", action="store_true")
    parser.add_argument("--multi-diagonal", action="store_true")
    parser.add_argument("--smoke-check-only", action="store_true")
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def replace_config(config: RunConfig, **changes: object) -> RunConfig:
    payload = asdict(config)
    payload.update(changes)
    return RunConfig(**payload)


def current_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def current_git_status_short() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_model(
    *,
    vocab_size: int,
    config: RunConfig,
    rates: tuple[int, ...],
    device: torch.device,
    diagonal_enabled: bool,
 ) -> MultiRateResidualModel:
    return MultiRateResidualModel(
        vocab_size=vocab_size,
        context_size=config.context_size,
        d_model=config.d_model,
        feedforward_dim=config.feedforward_dim,
        temporal_window=config.temporal_window,
        num_heads=config.num_heads,
        rates=rates,
        diagonal_enabled=diagonal_enabled,
    ).to(device)


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
    with torch.inference_mode():
        for batch_inputs, batch_targets in batched_pairs(inputs, targets, batch_size=batch_size):
            logits = model(batch_inputs)
            batch_examples = batch_targets.shape[0]
            total_loss += F.cross_entropy(logits, batch_targets, reduction="sum").item()
            total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
            total_examples += batch_examples
    if was_training:
        model.train()
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def measure_forward_pass_ms(
    model: nn.Module,
    batch_inputs: Tensor,
    *,
    warmup_passes: int,
    timed_passes: int,
) -> float:
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup_passes):
            model(batch_inputs)
        if batch_inputs.device.type == "cuda":
            torch.cuda.synchronize()
        started_at = time.perf_counter()
        for _ in range(timed_passes):
            model(batch_inputs)
        if batch_inputs.device.type == "cuda":
            torch.cuda.synchronize()
    if was_training:
        model.train()
    total_ms = (time.perf_counter() - started_at) * 1000.0
    return total_ms / timed_passes


def fixed_step_indices(size: int, *, steps: int, batch_size: int, seed: int, device: torch.device) -> list[Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return [
        torch.randint(0, size, (batch_size,), generator=generator).to(device)
        for _ in range(steps)
    ]


def run_diagonal_smoke_check(model: MultiRateResidualModel, inputs: Tensor) -> dict[str, object]:
    sample_inputs = inputs[:2]
    logits, trace = model.forward_with_trace(sample_inputs)
    diagonal_block_traces = [
        block_trace
        for step_trace in trace.step_traces
        for block_trace in step_trace.blocks
        if block_trace.block_index > 0
    ]
    if len(diagonal_block_traces) == 0:
        raise ValueError("Smoke check requires at least two blocks.")
    return {
        "diagonal_enabled": model.diagonal_enabled,
        "batch_shape": list(sample_inputs.shape),
        "logits_shape": list(logits.shape),
        "steps": [
            {
                "time_index": step_trace.time_index,
                "blocks": [asdict(block_trace) for block_trace in step_trace.blocks],
            }
            for step_trace in trace.step_traces
        ],
        "checks": {
            "time0_second_block_diagonal_zero": abs(trace.step_traces[0].blocks[1].diagonal_input_rms) < 1e-8,
            "later_second_block_has_nonzero_diagonal": any(
                block_trace.diagonal_input_rms > 1e-8 for block_trace in diagonal_block_traces[1:]
            ),
            "all_diagonal_reads_match_previous_delta": all(
                block_trace.diagonal_matches_previous_delta for block_trace in diagonal_block_traces
            ),
            "all_diagonal_reads_only_apply_on_execute": all(
                (not block_trace.diagonal_applied) or block_trace.executed for block_trace in diagonal_block_traces
            ),
        },
    }


def compare_diagonal_paths(model: MultiRateResidualModel, inputs: Tensor) -> dict[str, object]:
    if not model.diagonal_enabled:
        raise ValueError("Diagonal path comparison requires diagonal_enabled=True.")
    sample_inputs = inputs[:2]
    diagonal_logits = model(sample_inputs)
    no_diagonal_model = MultiRateResidualModel(
        vocab_size=model.output.out_features,
        context_size=model.context_size,
        d_model=model.d_model,
        feedforward_dim=model.feedforward_dim,
        temporal_window=model.temporal_window,
        num_heads=model.num_heads,
        rates=model.rates,
        diagonal_enabled=False,
    ).to(sample_inputs.device)
    no_diagonal_model.load_state_dict(model.state_dict(), strict=False)
    without_diagonal_logits = no_diagonal_model(sample_inputs)
    max_logit_diff = (diagonal_logits - without_diagonal_logits).abs().max().item()
    return {
        "max_logit_abs_diff": max_logit_diff,
        "diagonal_changes_output": max_logit_diff > 1e-8,
    }


def overfit_one_batch(
    model: nn.Module,
    batch_inputs: Tensor,
    batch_targets: Tensor,
    *,
    steps: int,
    learning_rate: float,
    loss_threshold: float,
) -> dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trace: list[dict[str, float | int]] = []
    hit_step: int | None = None

    for step in range(steps + 1):
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)
        accuracy = (logits.argmax(dim=1) == batch_targets).float().mean().item()

        if step % 25 == 0 or step == steps:
            trace.append(
                {
                    "step": step,
                    "loss": round(loss.item(), 6),
                    "accuracy": round(accuracy, 6),
                }
            )

        if hit_step is None and accuracy == 1.0 and loss.item() < loss_threshold:
            hit_step = step
            break

        if step == steps:
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_logits = model(batch_inputs)
    final_loss = F.cross_entropy(final_logits, batch_targets).item()
    final_accuracy = (final_logits.argmax(dim=1) == batch_targets).float().mean().item()
    final_step = hit_step if hit_step is not None else trace[-1]["step"]
    return {
        "trace": trace,
        "final_loss": round(final_loss, 6),
        "final_accuracy": round(final_accuracy, 6),
        "memorized": final_accuracy == 1.0 and final_loss < loss_threshold,
        "hit_step": hit_step,
        "steps_run": final_step,
    }


def evaluate_pair(
    *,
    step: int,
    control_model: nn.Module,
    multi_model: nn.Module,
    val_inputs: Tensor,
    val_targets: Tensor,
    timing_batch: Tensor,
    config: RunConfig,
) -> dict[str, object]:
    control_metrics = evaluate_model(
        control_model,
        val_inputs,
        val_targets,
        batch_size=config.eval_batch_size,
    )
    multi_metrics = evaluate_model(
        multi_model,
        val_inputs,
        val_targets,
        batch_size=config.eval_batch_size,
    )
    control_ms = measure_forward_pass_ms(
        control_model,
        timing_batch,
        warmup_passes=config.checkpoint_timing_warmup_passes,
        timed_passes=config.checkpoint_timing_passes,
    )
    multi_ms = measure_forward_pass_ms(
        multi_model,
        timing_batch,
        warmup_passes=config.checkpoint_timing_warmup_passes,
        timed_passes=config.checkpoint_timing_passes,
    )
    speedup_pct = ((control_ms - multi_ms) / control_ms) * 100.0
    return {
        "step": step,
        "control": {
            "val_loss": round(control_metrics["loss"], 6),
            "val_accuracy": round(control_metrics["accuracy"], 6),
            "forward_ms_per_batch": round(control_ms, 6),
            "timing_passes": config.checkpoint_timing_passes,
        },
        "experiment": {
            "val_loss": round(multi_metrics["loss"], 6),
            "val_accuracy": round(multi_metrics["accuracy"], 6),
            "forward_ms_per_batch": round(multi_ms, 6),
            "timing_passes": config.checkpoint_timing_passes,
        },
        "comparison": {
            "val_loss_delta": round(multi_metrics["loss"] - control_metrics["loss"], 6),
            "forward_ms_delta": round(multi_ms - control_ms, 6),
            "speedup_percent": round(speedup_pct, 6),
        },
    }


def train_comparison(
    *,
    control_model: nn.Module,
    multi_model: nn.Module,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    config: RunConfig,
) -> list[dict[str, object]]:
    control_optimizer = torch.optim.Adam(control_model.parameters(), lr=config.learning_rate)
    multi_optimizer = torch.optim.Adam(multi_model.parameters(), lr=config.learning_rate)
    batch_schedule = fixed_step_indices(
        train_inputs.shape[0],
        steps=config.training_steps,
        batch_size=config.batch_size,
        seed=config.seed,
        device=train_inputs.device,
    )
    timing_batch = train_inputs[: config.batch_size]
    history = [
        evaluate_pair(
            step=0,
            control_model=control_model,
            multi_model=multi_model,
            val_inputs=val_inputs,
            val_targets=val_targets,
            timing_batch=timing_batch,
            config=config,
        )
    ]
    print(json.dumps({"checkpoint": history[-1]}, indent=2), flush=True)

    for step, batch_indices in enumerate(batch_schedule, start=1):
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]

        control_optimizer.zero_grad(set_to_none=True)
        control_loss = F.cross_entropy(control_model(batch_inputs), batch_targets)
        control_loss.backward()
        control_optimizer.step()

        multi_optimizer.zero_grad(set_to_none=True)
        multi_loss = F.cross_entropy(multi_model(batch_inputs), batch_targets)
        multi_loss.backward()
        multi_optimizer.step()

        if step % config.eval_interval == 0 or step == config.training_steps:
            history.append(
                evaluate_pair(
                    step=step,
                    control_model=control_model,
                    multi_model=multi_model,
                    val_inputs=val_inputs,
                    val_targets=val_targets,
                    timing_batch=timing_batch,
                    config=config,
                )
            )
            print(json.dumps({"checkpoint": history[-1]}, indent=2), flush=True)

    return history


def summarize_history(history: list[dict[str, object]]) -> dict[str, object]:
    final_record = history[-1]
    best_control = min(history, key=lambda record: record["control"]["val_loss"])
    best_multi = min(history, key=lambda record: record["experiment"]["val_loss"])
    return {
        "final": final_record,
        "best_control": {
            "step": best_control["step"],
            "val_loss": best_control["control"]["val_loss"],
            "val_accuracy": best_control["control"]["val_accuracy"],
        },
        "best_experiment": {
            "step": best_multi["step"],
            "val_loss": best_multi["experiment"]["val_loss"],
            "val_accuracy": best_multi["experiment"]["val_accuracy"],
        },
    }


def final_timing_summary(
    *,
    control_model: nn.Module,
    multi_model: nn.Module,
    timing_batch: Tensor,
    config: RunConfig,
) -> dict[str, object]:
    control_ms = measure_forward_pass_ms(
        control_model,
        timing_batch,
        warmup_passes=config.final_timing_warmup_passes,
        timed_passes=config.final_timing_passes,
    )
    multi_ms = measure_forward_pass_ms(
        multi_model,
        timing_batch,
        warmup_passes=config.final_timing_warmup_passes,
        timed_passes=config.final_timing_passes,
    )
    return {
        "control": {
            "forward_ms_per_batch": round(control_ms, 6),
            "timing_passes": config.final_timing_passes,
        },
        "experiment": {
            "forward_ms_per_batch": round(multi_ms, 6),
            "timing_passes": config.final_timing_passes,
        },
        "comparison": {
            "forward_ms_delta": round(multi_ms - control_ms, 6),
            "speedup_percent": round(((control_ms - multi_ms) / control_ms) * 100.0, 6),
        },
    }


def main() -> None:
    args = parse_args()
    multi_rates = parse_rate_tuple(args.multi_rates)
    control_rates = parse_rate_tuple(args.control_rates) if args.control_rates is not None else (1,) * len(multi_rates)
    assert len(control_rates) == len(multi_rates)
    config = replace_config(
        RunConfig(),
        **{
            key: value
            for key, value in {
                "training_steps": args.training_steps,
                "overfit_steps": args.overfit_steps,
                "eval_interval": args.eval_interval,
                "checkpoint_timing_warmup_passes": args.checkpoint_timing_warmup_passes,
                "checkpoint_timing_passes": args.checkpoint_timing_passes,
                "final_timing_warmup_passes": args.final_timing_warmup_passes,
                "final_timing_passes": args.final_timing_passes,
                "num_blocks": len(multi_rates),
                "control_rates": control_rates,
                "multi_rates": multi_rates,
                "control_diagonal": args.control_diagonal,
                "multi_diagonal": args.multi_diagonal,
            }.items()
            if value is not None
        },
    )
    output_dir = args.output_dir or (
        args.repo_root / "experiments" / "fixed_multi_rate" / "artifacts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    device = resolve_device(args.device)
    train_data, val_data, vocab_size = load_dataset(context_size=config.context_size)
    train_inputs, train_targets = train_data
    val_inputs, val_targets = val_data
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    val_inputs = val_inputs.to(device)
    val_targets = val_targets.to(device)

    overfit_model = build_model(
        vocab_size=vocab_size,
        config=config,
        rates=config.multi_rates,
        device=device,
        diagonal_enabled=config.multi_diagonal,
    )
    smoke_check = run_diagonal_smoke_check(overfit_model, train_inputs)
    diagonal_effect = compare_diagonal_paths(overfit_model, train_inputs) if config.multi_diagonal else None
    if args.smoke_check_only:
        smoke_report = {
            "config": asdict(config),
            "smoke_check": smoke_check,
            "diagonal_effect": diagonal_effect,
        }
        write_json(output_dir / "smoke_check.json", smoke_report)
        print("PASS fixed_multi_rate smoke_check")
        print(f"Wrote {output_dir / 'smoke_check.json'}")
        print(json.dumps(smoke_report, indent=2))
        return
    overfit_inputs = train_inputs[: config.batch_size]
    overfit_targets = train_targets[: config.batch_size]
    overfit_result = overfit_one_batch(
        overfit_model,
        overfit_inputs,
        overfit_targets,
        steps=config.overfit_steps,
        learning_rate=config.learning_rate,
        loss_threshold=config.overfit_loss_threshold,
    )

    control_model = build_model(
        vocab_size=vocab_size,
        config=config,
        rates=config.control_rates,
        device=device,
        diagonal_enabled=config.control_diagonal,
    )
    multi_model = build_model(
        vocab_size=vocab_size,
        config=config,
        rates=config.multi_rates,
        device=device,
        diagonal_enabled=config.multi_diagonal,
    )
    multi_model.load_state_dict(control_model.state_dict())
    history = train_comparison(
        control_model=control_model,
        multi_model=multi_model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        config=config,
    )
    final_timing = final_timing_summary(
        control_model=control_model,
        multi_model=multi_model,
        timing_batch=train_inputs[: config.batch_size],
        config=config,
    )

    git_status_short = current_git_status_short()
    report = {
        "config": asdict(config),
        "environment": {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": git_status_short == [],
            "git_status_short": git_status_short,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "model": {
            "parameter_count": count_parameters(control_model),
            "num_blocks": config.num_blocks,
            "control": {
                "rates": list(config.control_rates),
                "diagonal_enabled": config.control_diagonal,
                "mix_coefficients": control_model.mix_coefficients(),
            },
            "experiment": {
                "rates": list(config.multi_rates),
                "diagonal_enabled": config.multi_diagonal,
                "mix_coefficients": multi_model.mix_coefficients(),
            },
        },
        "overfit": overfit_result,
        "smoke_check": smoke_check,
        "diagonal_effect": diagonal_effect,
        "comparison_history": history,
        "summary": summarize_history(history),
        "final_timing": final_timing,
    }
    write_json(output_dir / "report.json", report)

    final_record = history[-1]
    print("PASS fixed_multi_rate")
    print(f"Wrote {output_dir / 'report.json'}")
    print(
        json.dumps(
            {
                "overfit": {
                    "memorized": overfit_result["memorized"],
                    "hit_step": overfit_result["hit_step"],
                    "final_loss": overfit_result["final_loss"],
                    "final_accuracy": overfit_result["final_accuracy"],
                },
                "final": final_record,
                "final_timing": final_timing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
