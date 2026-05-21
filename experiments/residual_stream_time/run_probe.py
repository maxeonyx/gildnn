from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from base_experiments.transformer import build_fixed_length_split
from core.fixed_window_char import (
    FixedWindowCharDataset,
    generate_text,
    render_predictions,
    resolve_device,
    set_seed,
)

from .model import ResidualRunTrace, ResidualStreamTimeCharModel, ResidualStreamTimeConfig, count_parameters


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 32
    d_model: int = 128
    feedforward_dim: int = 512
    num_heads: int = 4
    temporal_window: int = 4
    train_characters: int = 4_096
    val_characters: int = 1_024
    tiny_epochs: int = 3
    batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 0.003
    gradient_clip_norm: float = 1.0
    overfit_batch_size: int = 32
    overfit_steps: int = 2_000
    overfit_learning_rate: float = 0.01
    overfit_log_interval: int = 25
    sample_length: int = 160
    progression_sample_length: int = 120
    sample_checkpoints: tuple[int, ...] = (0, 1, 2, 3)
    seed: int = 42


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


def trace_to_payload(trace: ResidualRunTrace) -> dict[str, object]:
    return {
        "final_stream_rms": trace.final_stream_rms,
        "step_traces": [asdict(step_trace) for step_trace in trace.step_traces],
    }


def batched_pairs(inputs: Tensor, targets: Tensor, *, batch_size: int):
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        yield inputs[start:stop], targets[start:stop]


def evaluate_model(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
) -> dict[str, float]:
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


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_inputs: Tensor,
    train_targets: Tensor,
    *,
    batch_size: int,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    permutation = torch.randperm(train_inputs.shape[0], device=train_inputs.device)
    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    for start in range(0, permutation.shape[0], batch_size):
        batch_indices = permutation[start : start + batch_size]
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        batch_examples = batch_targets.shape[0]
        total_examples += batch_examples
        total_loss += loss.item() * batch_examples
        total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def capture_sample(
    model: ResidualStreamTimeCharModel,
    dataset: FixedWindowCharDataset,
    prompt: str,
    *,
    length: int,
    device: torch.device,
) -> str:
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        sample = generate_text(model, dataset, prompt, length=length, device=device)
    if was_training:
        model.train()
    return sample


def write_loss_curve_svg(
    path: Path,
    *,
    title: str,
    series: dict[str, list[tuple[float, float]]],
) -> None:
    width = 800
    height = 420
    margin_left = 60
    margin_right = 20
    margin_top = 30
    margin_bottom = 40
    inner_width = width - margin_left - margin_right
    inner_height = height - margin_top - margin_bottom
    xs = [x for points in series.values() for x, _ in points]
    ys = [y for points in series.values() for _, y in points]
    if not xs or not ys:
        raise ValueError("Cannot plot empty loss curve.")
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    if max_x == min_x:
        max_x = min_x + 1.0
    if max_y == min_y:
        max_y = min_y + 1.0

    def scale_x(value: float) -> float:
        return margin_left + ((value - min_x) / (max_x - min_x)) * inner_width

    def scale_y(value: float) -> float:
        return margin_top + inner_height - ((value - min_y) / (max_y - min_y)) * inner_height

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    lines: list[str] = []
    for index, (label, points) in enumerate(series.items()):
        point_string = " ".join(f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in points)
        color = colors[index % len(colors)]
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{point_string}" />')
        legend_y = margin_top + 18 * index
        lines.append(f'<rect x="{width - 180}" y="{legend_y - 10}" width="12" height="12" fill="{color}" />')
        lines.append(f'<text x="{width - 162}" y="{legend_y}" font-size="12" font-family="monospace">{label}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{margin_left}" y="20" font-size="16" font-family="monospace">{title}</text>
<line x1="{margin_left}" y1="{margin_top + inner_height}" x2="{width - margin_right}" y2="{margin_top + inner_height}" stroke="black" />
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + inner_height}" stroke="black" />
<text x="{margin_left}" y="{height - 10}" font-size="12" font-family="monospace">x:[{min_x:.0f}, {max_x:.0f}]</text>
<text x="{margin_left + 160}" y="{height - 10}" font-size="12" font-family="monospace">loss:[{min_y:.3f}, {max_y:.3f}]</text>
{''.join(lines)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_metadata(
    output_dir: Path,
    *,
    config: RunConfig,
    split,
    model: ResidualStreamTimeCharModel,
    text_file: Path,
    raw_text: str,
    device: torch.device,
    stage: str,
) -> None:
    git_status_short = current_git_status_short()
    write_json(output_dir / "config.json", {**asdict(config), "stage": stage})
    write_json(
        output_dir / "environment.json",
        {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": git_status_short == [],
            "git_status_short": git_status_short,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_is_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
    )
    used_text = split.train_text + split.val_text
    write_json(
        output_dir / "corpus_summary.json",
        {
            "source_file": str(text_file),
            "source_total_characters": len(raw_text),
            "source_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "used_total_characters": len(used_text),
            "used_sha256": hashlib.sha256(used_text.encode("utf-8")).hexdigest(),
            "train_characters": len(split.train_text),
            "val_characters": len(split.val_text),
            "val_start": split.val_start,
            "val_stop": split.val_start + len(split.val_text),
            "train_windows": int(split.train_inputs.shape[0]),
            "val_windows": int(split.val_inputs.shape[0]),
            "vocab_size": split.train_dataset.vocab_size,
        },
    )
    write_json(
        output_dir / "model_summary.json",
        {
            "model_family": "residual_stream_time",
            "parameter_count": count_parameters(model),
            "context_size": config.context_size,
            "d_model": config.d_model,
            "feedforward_dim": config.feedforward_dim,
            "num_heads": config.num_heads,
            "temporal_window": config.temporal_window,
        },
    )


def run_forward_sanity_check(
    model: ResidualStreamTimeCharModel,
    inputs: Tensor,
) -> dict[str, object]:
    logits, trace = model.forward_with_trace(inputs[:2])
    return {
        "batch_shape": list(inputs[:2].shape),
        "logits_shape": list(logits.shape),
        "trace": trace_to_payload(trace),
    }


def overfit_one_batch(
    *,
    model: ResidualStreamTimeCharModel,
    dataset: FixedWindowCharDataset,
    inputs: Tensor,
    targets: Tensor,
    config: RunConfig,
    device: torch.device,
) -> tuple[list[dict[str, float | int]], dict[str, object], str]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.overfit_learning_rate)
    trace: list[dict[str, float | int]] = []
    final_step = 0
    final_loss = float("nan")
    final_accuracy = 0.0
    for step in range(config.overfit_steps + 1):
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
        if step == 0 or step == config.overfit_steps or step % config.overfit_log_interval == 0:
            trace.append(
                {
                    "step": step,
                    "loss": round(loss.item(), 6),
                    "accuracy": round(accuracy, 6),
                }
            )
        final_step = step
        final_loss = loss.item()
        final_accuracy = accuracy
        if accuracy == 1.0 and loss.item() < 1e-3:
            break
        if step == config.overfit_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_logits, final_trace = model.forward_with_trace(inputs[:1])
    predictions = model(inputs).argmax(dim=1)
    sample_prompt = dataset.decode(inputs[0].detach().cpu().tolist())
    sample = capture_sample(model, dataset, sample_prompt, length=80, device=device)
    return trace, {
        "steps_run": final_step,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "memorized_batch": final_accuracy == 1.0 and final_loss < 1e-3,
        "first_example_trace": trace_to_payload(final_trace),
        "first_example_logit_argmax": int(final_logits.argmax(dim=1)[0].item()),
    }, render_predictions(dataset, inputs.cpu(), targets.cpu(), predictions.cpu()) + "\nSAMPLE\n" + sample + "\n"


def run_tiny_training(
    *,
    model: ResidualStreamTimeCharModel,
    dataset: FixedWindowCharDataset,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    prompt: str,
    config: RunConfig,
    device: torch.device,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int | str]], dict[str, object], str]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []
    progression_samples: list[dict[str, float | int | str]] = []
    sample_epochs = {epoch for epoch in config.sample_checkpoints if epoch <= config.tiny_epochs}
    started_at = time.perf_counter()

    initial_train_metrics = evaluate_model(model, train_inputs, train_targets, batch_size=config.eval_batch_size)
    initial_val_metrics = evaluate_model(model, val_inputs, val_targets, batch_size=config.eval_batch_size)
    history.append(
        {
            "epoch": 0,
            "train_loss": round(initial_train_metrics["loss"], 6),
            "train_accuracy": round(initial_train_metrics["accuracy"], 6),
            "val_loss": round(initial_val_metrics["loss"], 6),
            "val_accuracy": round(initial_val_metrics["accuracy"], 6),
        }
    )
    if 0 in sample_epochs:
        progression_samples.append(
            {
                "epoch": 0,
                "val_loss": round(initial_val_metrics["loss"], 6),
                "sample": capture_sample(model, dataset, prompt, length=config.progression_sample_length, device=device),
            }
        )

    for epoch in range(1, config.tiny_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_inputs,
            train_targets,
            batch_size=config.batch_size,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        val_metrics = evaluate_model(model, val_inputs, val_targets, batch_size=config.eval_batch_size)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "train_accuracy": round(train_metrics["accuracy"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
            }
        )
        if epoch in sample_epochs:
            progression_samples.append(
                {
                    "epoch": epoch,
                    "val_loss": round(val_metrics["loss"], 6),
                    "sample": capture_sample(model, dataset, prompt, length=config.progression_sample_length, device=device),
                }
            )

    runtime_seconds = time.perf_counter() - started_at
    final_metrics = evaluate_model(model, val_inputs, val_targets, batch_size=config.eval_batch_size)
    final_sample = capture_sample(model, dataset, prompt, length=config.sample_length, device=device)
    prompt_tokens = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)
    _, prompt_trace = model.forward_with_trace(prompt_tokens)
    best_epoch_record = min(history[1:] or history, key=lambda record: float(record["val_loss"]))
    summary = {
        "parameter_count": count_parameters(model),
        "runtime_seconds": runtime_seconds,
        "final_val_loss": final_metrics["loss"],
        "final_val_accuracy": final_metrics["accuracy"],
        "best_val_loss": float(best_epoch_record["val_loss"]),
        "best_val_accuracy": float(best_epoch_record["val_accuracy"]),
        "best_epoch": int(best_epoch_record["epoch"]),
        "prompt": prompt.replace("\n", "\\n"),
        "prompt_trace": trace_to_payload(prompt_trace),
    }
    return history, progression_samples, summary, final_sample


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["overfit", "tiny"], required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overfit-steps", type=int)
    parser.add_argument("--tiny-epochs", type=int)
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--feedforward-dim", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--temporal-window", type=int)
    parser.add_argument("--train-characters", type=int)
    parser.add_argument("--val-characters", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gradient-clip-norm", type=float)
    parser.add_argument("--sample-length", type=int)
    parser.add_argument("--progression-sample-length", type=int)
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig()
    override_map = {
        "overfit_steps": args.overfit_steps,
        "tiny_epochs": args.tiny_epochs,
        "context_size": args.context_size,
        "d_model": args.d_model,
        "feedforward_dim": args.feedforward_dim,
        "num_heads": args.num_heads,
        "temporal_window": args.temporal_window,
        "train_characters": args.train_characters,
        "val_characters": args.val_characters,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "gradient_clip_norm": args.gradient_clip_norm,
        "sample_length": args.sample_length,
        "progression_sample_length": args.progression_sample_length,
    }
    config = replace_config(
        config,
        **{key: value for key, value in override_map.items() if value is not None},
    )
    set_seed(config.seed)
    device = resolve_device(args.device)
    text_file = args.text_file or (
        args.repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    )
    output_dir = args.output_dir or (
        args.repo_root / "experiments" / "residual_stream_time" / "artifacts" / args.stage
    )
    raw_text = text_file.read_text(encoding="utf-8")
    split = build_fixed_length_split(raw_text, config=config)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = ResidualStreamTimeCharModel(
        vocab_size=split.train_dataset.vocab_size,
        config=ResidualStreamTimeConfig(
            context_size=config.context_size,
            d_model=config.d_model,
            feedforward_dim=config.feedforward_dim,
            temporal_window=config.temporal_window,
            num_heads=config.num_heads,
        ),
    ).to(device)
    write_metadata(
        output_dir,
        config=config,
        split=split,
        model=model,
        text_file=text_file,
        raw_text=raw_text,
        device=device,
        stage=args.stage,
    )
    write_json(
        output_dir / "forward_sanity_check.json",
        run_forward_sanity_check(model, split.train_inputs.to(device)),
    )

    if args.stage == "overfit":
        overfit_inputs = split.train_inputs[: config.overfit_batch_size].to(device)
        overfit_targets = split.train_targets[: config.overfit_batch_size].to(device)
        trace, metrics, predictions_text = overfit_one_batch(
            model=model,
            dataset=split.train_dataset,
            inputs=overfit_inputs,
            targets=overfit_targets,
            config=config,
            device=device,
        )
        write_json(output_dir / "overfit_metrics.json", {"trace": trace, **metrics})
        (output_dir / "overfit_predictions.txt").write_text(predictions_text, encoding="utf-8")
        write_loss_curve_svg(
            output_dir / "overfit_loss.svg",
            title="Residual stream across time: overfit one batch",
            series={"overfit_loss": [(float(record["step"]), float(record["loss"])) for record in trace]},
        )
        print(
            f"overfit final_loss={metrics['final_loss']:.6f} final_accuracy={metrics['final_accuracy']:.6f} steps={metrics['steps_run']}"
        )
        return

    prompt = split.train_text[: config.context_size]
    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)
    history, progression_samples, summary, final_sample = run_tiny_training(
        model=model,
        dataset=split.train_dataset,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        prompt=prompt,
        config=config,
        device=device,
    )
    write_json(output_dir / "tiny_metrics.json", summary)
    write_json(output_dir / "training_history.json", history)
    write_json(
        output_dir / "progression_samples.json",
        {
            "prompt": prompt.replace("\n", "\\n"),
            "sample_length": config.progression_sample_length,
            "checkpoints": progression_samples,
        },
    )
    (output_dir / "sample.txt").write_text(final_sample, encoding="utf-8")
    write_loss_curve_svg(
        output_dir / "tiny_loss.svg",
        title="Residual stream across time: tiny run",
        series={
            "train_loss": [(float(record["epoch"]), float(record["train_loss"])) for record in history],
            "val_loss": [(float(record["epoch"]), float(record["val_loss"])) for record in history],
        },
    )
    print(
        f"tiny best_val_loss={summary['best_val_loss']:.6f} final_val_loss={summary['final_val_loss']:.6f} runtime_seconds={summary['runtime_seconds']:.2f}"
    )


if __name__ == "__main__":
    main()
