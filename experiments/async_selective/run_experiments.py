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
from torch import Tensor
from torch.nn import functional as F

from base_experiments.transformer import build_fixed_length_split
from core.fixed_window_char import FixedWindowCharDataset, generate_text, resolve_device, set_seed

from .model import AsyncSelectiveCharModel, ModelRun, VariantSpec, count_parameters


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 32
    train_characters: int = 100_000
    val_characters: int = 20_000
    d_model: int = 72
    batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 0.003
    gradient_clip_norm: float = 1.0
    seed: int = 42
    overfit_batch_size: int = 32
    overfit_steps: int = 1_500
    overfit_learning_rate: float = 0.01
    overfit_log_interval: int = 25
    sample_length: int = 320
    progression_sample_length: int = 200
    sample_checkpoints: tuple[int, ...] = (0, 1, 2, 4)
    tiny_train_characters: int = 4_096
    tiny_val_characters: int = 1_024
    tiny_epochs: int = 2
    full_epochs: int = 13
    budget_weight: float = 1.0


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


def replace_config(config: RunConfig, **changes: object) -> RunConfig:
    payload = asdict(config)
    payload.update(changes)
    return RunConfig(**payload)


def make_variant(
    name: str,
    *,
    config: RunConfig,
    target_open_rate: float | None,
) -> VariantSpec:
    return VariantSpec(
        name=name,  # type: ignore[arg-type]
        target_open_rate=target_open_rate,
        budget_weight=config.budget_weight,
    )


def make_model(
    *,
    vocab_size: int,
    config: RunConfig,
    variant: VariantSpec,
    device: torch.device,
) -> AsyncSelectiveCharModel:
    return AsyncSelectiveCharModel(
        vocab_size=vocab_size,
        context_size=config.context_size,
        d_model=config.d_model,
        variant=variant,
    ).to(device)


def batched_pairs(inputs: Tensor, targets: Tensor, *, batch_size: int):
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        yield inputs[start:stop], targets[start:stop]


def capture_sample(
    model: AsyncSelectiveCharModel,
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


def format_target_suffix(target_open_rate: float | None) -> str:
    if target_open_rate is None:
        return "none"
    return f"r{int(round(target_open_rate * 100)):02d}"


class GateAccumulator:
    def __init__(self, *, context_size: int) -> None:
        self.context_size = context_size
        self.block_open_sum: dict[int, float] = {}
        self.block_probability_sum: dict[int, float] = {}
        self.block_token_count: dict[int, int] = {}
        self.position_open_sum: dict[int, Tensor] = {}
        self.position_probability_sum: dict[int, Tensor] = {}
        self.position_example_count: dict[int, int] = {}

    def update(self, run: ModelRun) -> None:
        for trace in run.block_traces[1:]:
            block_index = trace.block_index
            gate_values = trace.gate.values.detach().float().squeeze(-1)
            gate_probabilities = (
                trace.gate.probabilities.detach().float().squeeze(-1)
                if trace.gate.probabilities is not None
                else gate_values
            )
            self.block_open_sum[block_index] = self.block_open_sum.get(block_index, 0.0) + gate_values.sum().item()
            self.block_probability_sum[block_index] = self.block_probability_sum.get(block_index, 0.0) + gate_probabilities.sum().item()
            self.block_token_count[block_index] = self.block_token_count.get(block_index, 0) + gate_values.numel()
            if block_index not in self.position_open_sum:
                self.position_open_sum[block_index] = torch.zeros(self.context_size)
                self.position_probability_sum[block_index] = torch.zeros(self.context_size)
                self.position_example_count[block_index] = 0
            self.position_open_sum[block_index] += gate_values.sum(dim=0).cpu()
            self.position_probability_sum[block_index] += gate_probabilities.sum(dim=0).cpu()
            self.position_example_count[block_index] += gate_values.shape[0]

    def summary(self) -> dict[str, object]:
        block_rows: dict[str, object] = {}
        gated_means: list[float] = []
        for block_index in sorted(self.block_open_sum):
            open_rate = self.block_open_sum[block_index] / self.block_token_count[block_index]
            probability_mean = self.block_probability_sum[block_index] / self.block_token_count[block_index]
            gated_means.append(open_rate)
            example_count = self.position_example_count[block_index]
            block_rows[f"block_{block_index + 1}"] = {
                "mean_open_rate": open_rate,
                "mean_probability": probability_mean,
                "per_position_mean_open_rate": (
                    self.position_open_sum[block_index] / example_count
                ).tolist(),
                "per_position_mean_probability": (
                    self.position_probability_sum[block_index] / example_count
                ).tolist(),
            }
        effective_ratio = (1.0 + sum(gated_means)) / 3.0 if gated_means else 1.0
        return {
            "blocks": block_rows,
            "effective_executed_block_ratio": effective_ratio,
        }


def gate_collapse_status(gate_summary: dict[str, object], *, threshold: float = 0.98) -> str | None:
    block_rows = gate_summary["blocks"]
    if not isinstance(block_rows, dict) or not block_rows:
        return None
    open_rates = [row["mean_open_rate"] for row in block_rows.values() if isinstance(row, dict)]
    if open_rates and all(rate >= threshold for rate in open_rates):
        return "all_open"
    if open_rates and all(rate <= 1.0 - threshold for rate in open_rates):
        return "all_closed"
    return None


def forward_metrics(run: ModelRun, targets: Tensor) -> dict[str, float]:
    lm_loss = F.cross_entropy(run.last_logits, targets)
    total_loss = lm_loss + run.budget_loss
    accuracy = (run.last_logits.argmax(dim=1) == targets).float().mean()
    return {
        "lm_loss": lm_loss.item(),
        "budget_loss": run.budget_loss.item(),
        "total_loss": total_loss.item(),
        "accuracy": accuracy.item(),
    }


def write_metadata(
    output_dir: Path,
    *,
    config: RunConfig,
    split,
    variant: VariantSpec,
    model: AsyncSelectiveCharModel,
    text_file: Path,
    raw_text: str,
    device: torch.device,
    stage: str,
) -> None:
    git_status_short = current_git_status_short()
    write_json(
        output_dir / "config.json",
        {
            **asdict(config),
            "stage": stage,
            "variant": variant.name,
            "target_open_rate": variant.target_open_rate,
            "budget_weight": variant.budget_weight,
        },
    )
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
            "variant": variant.name,
            "target_open_rate": variant.target_open_rate,
            "parameter_count": count_parameters(model),
            "context_size": config.context_size,
            "d_model": config.d_model,
            "num_layers": variant.num_layers,
            "num_heads": variant.num_heads,
            "feedforward_dims": list(variant.feedforward_dims),
        },
    )


def evaluate_model(
    model: AsyncSelectiveCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
) -> dict[str, object]:
    was_training = model.training
    model.eval()
    total_examples = 0
    lm_loss_sum = 0.0
    budget_loss_sum = 0.0
    total_loss_sum = 0.0
    total_correct = 0
    gate_accumulator = GateAccumulator(context_size=model.context_size)
    with torch.no_grad():
        for batch_inputs, batch_targets in batched_pairs(inputs, targets, batch_size=batch_size):
            run = model.run(batch_inputs)
            metrics = forward_metrics(run, batch_targets)
            batch_examples = batch_targets.shape[0]
            lm_loss_sum += metrics["lm_loss"] * batch_examples
            budget_loss_sum += metrics["budget_loss"] * batch_examples
            total_loss_sum += metrics["total_loss"] * batch_examples
            total_correct += (run.last_logits.argmax(dim=1) == batch_targets).sum().item()
            total_examples += batch_examples
            gate_accumulator.update(run)
    if was_training:
        model.train()
    return {
        "lm_loss": lm_loss_sum / total_examples,
        "budget_loss": budget_loss_sum / total_examples,
        "total_loss": total_loss_sum / total_examples,
        "accuracy": total_correct / total_examples,
        "gate_summary": gate_accumulator.summary(),
    }


def train_one_epoch(
    model: AsyncSelectiveCharModel,
    optimizer: torch.optim.Optimizer,
    train_inputs: Tensor,
    train_targets: Tensor,
    *,
    batch_size: int,
    gradient_clip_norm: float,
) -> dict[str, object]:
    model.train()
    permutation = torch.randperm(train_inputs.shape[0], device=train_inputs.device)
    total_examples = 0
    lm_loss_sum = 0.0
    budget_loss_sum = 0.0
    total_loss_sum = 0.0
    total_correct = 0
    gate_accumulator = GateAccumulator(context_size=model.context_size)
    for start in range(0, permutation.shape[0], batch_size):
        batch_indices = permutation[start : start + batch_size]
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]
        run = model.run(batch_inputs)
        lm_loss = F.cross_entropy(run.last_logits, batch_targets)
        total_loss = lm_loss + run.budget_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        batch_examples = batch_targets.shape[0]
        lm_loss_sum += lm_loss.item() * batch_examples
        budget_loss_sum += run.budget_loss.item() * batch_examples
        total_loss_sum += total_loss.item() * batch_examples
        total_correct += (run.last_logits.argmax(dim=1) == batch_targets).sum().item()
        total_examples += batch_examples
        gate_accumulator.update(run)
    return {
        "lm_loss": lm_loss_sum / total_examples,
        "budget_loss": budget_loss_sum / total_examples,
        "total_loss": total_loss_sum / total_examples,
        "accuracy": total_correct / total_examples,
        "gate_summary": gate_accumulator.summary(),
    }


def history_record(epoch: int, train_metrics: dict[str, object], val_metrics: dict[str, object]) -> dict[str, object]:
    train_gate_summary = train_metrics["gate_summary"]
    val_gate_summary = val_metrics["gate_summary"]
    return {
        "epoch": epoch,
        "train_lm_loss": round(float(train_metrics["lm_loss"]), 6),
        "train_budget_loss": round(float(train_metrics["budget_loss"]), 6),
        "train_total_loss": round(float(train_metrics["total_loss"]), 6),
        "train_accuracy": round(float(train_metrics["accuracy"]), 6),
        "train_effective_executed_block_ratio": round(
            float(train_gate_summary["effective_executed_block_ratio"]), 6
        ),
        "val_lm_loss": round(float(val_metrics["lm_loss"]), 6),
        "val_budget_loss": round(float(val_metrics["budget_loss"]), 6),
        "val_total_loss": round(float(val_metrics["total_loss"]), 6),
        "val_accuracy": round(float(val_metrics["accuracy"]), 6),
        "val_effective_executed_block_ratio": round(
            float(val_gate_summary["effective_executed_block_ratio"]), 6
        ),
    }


def overfit_one_batch(
    *,
    model: AsyncSelectiveCharModel,
    inputs: Tensor,
    targets: Tensor,
    config: RunConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.overfit_learning_rate)
    trace: list[dict[str, object]] = []
    final_metrics: dict[str, object] = {}
    final_step = 0
    started_at = time.perf_counter()
    for step in range(config.overfit_steps + 1):
        run = model.run(inputs)
        lm_loss = F.cross_entropy(run.last_logits, targets)
        total_loss = lm_loss + run.budget_loss
        accuracy = (run.last_logits.argmax(dim=1) == targets).float().mean().item()
        gate_accumulator = GateAccumulator(context_size=model.context_size)
        gate_accumulator.update(run)
        gate_summary = gate_accumulator.summary()
        if step == 0 or step == config.overfit_steps or step % config.overfit_log_interval == 0:
            trace.append(
                {
                    "step": step,
                    "lm_loss": round(lm_loss.item(), 6),
                    "budget_loss": round(run.budget_loss.item(), 6),
                    "total_loss": round(total_loss.item(), 6),
                    "accuracy": round(accuracy, 6),
                    "gate_summary": gate_summary,
                }
            )
        final_step = step
        final_metrics = {
            "steps_run": step,
            "lm_loss": lm_loss.item(),
            "budget_loss": run.budget_loss.item(),
            "total_loss": total_loss.item(),
            "accuracy": accuracy,
            "gate_summary": gate_summary,
            "gate_collapse": gate_collapse_status(gate_summary),
        }
        if accuracy == 1.0 and lm_loss.item() < 0.02:
            break
        if step == config.overfit_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
    final_metrics["runtime_seconds"] = time.perf_counter() - started_at
    final_metrics["steps_run"] = final_step
    return trace, final_metrics


def train_stage(
    *,
    model: AsyncSelectiveCharModel,
    dataset: FixedWindowCharDataset,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    prompt: str,
    config: RunConfig,
    device: torch.device,
    epochs: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], float, str]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, object]] = []
    progression_samples: list[dict[str, object]] = []
    sample_epochs = {epoch for epoch in config.sample_checkpoints if epoch <= epochs}
    started_at = time.perf_counter()

    initial_train_metrics = evaluate_model(model, train_inputs, train_targets, batch_size=config.eval_batch_size)
    initial_val_metrics = evaluate_model(model, val_inputs, val_targets, batch_size=config.eval_batch_size)
    history.append(history_record(0, initial_train_metrics, initial_val_metrics))
    if 0 in sample_epochs:
        progression_samples.append(
            {
                "epoch": 0,
                "val_lm_loss": round(float(initial_val_metrics["lm_loss"]), 6),
                "sample": capture_sample(
                    model,
                    dataset,
                    prompt,
                    length=config.progression_sample_length,
                    device=device,
                ),
            }
        )

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_inputs,
            train_targets,
            batch_size=config.batch_size,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        val_metrics = evaluate_model(model, val_inputs, val_targets, batch_size=config.eval_batch_size)
        history.append(history_record(epoch, train_metrics, val_metrics))
        if epoch in sample_epochs:
            progression_samples.append(
                {
                    "epoch": epoch,
                    "val_lm_loss": round(float(val_metrics["lm_loss"]), 6),
                    "sample": capture_sample(
                        model,
                        dataset,
                        prompt,
                        length=config.progression_sample_length,
                        device=device,
                    ),
                }
            )

    runtime_seconds = time.perf_counter() - started_at
    final_train_metrics = evaluate_model(model, train_inputs, train_targets, batch_size=config.eval_batch_size)
    final_val_metrics = evaluate_model(model, val_inputs, val_targets, batch_size=config.eval_batch_size)
    sample = capture_sample(model, dataset, prompt, length=config.sample_length, device=device)
    final_metrics = {
        "train": final_train_metrics,
        "val": final_val_metrics,
        "gate_collapse": gate_collapse_status(final_val_metrics["gate_summary"]),
    }
    return history, progression_samples, final_metrics, runtime_seconds, sample


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["overfit", "tiny", "full"], required=True)
    parser.add_argument(
        "--variant",
        choices=["synchronous_control", "learned_gate", "random_skip", "forced_open"],
        required=True,
    )
    parser.add_argument("--target-open-rate", type=float, choices=[0.5, 0.75])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig()
    if args.stage == "tiny":
        config = replace_config(
            config,
            train_characters=config.tiny_train_characters,
            val_characters=config.tiny_val_characters,
            sample_length=160,
            progression_sample_length=120,
        )
    if args.variant in {"learned_gate", "random_skip"} and args.target_open_rate is None:
        raise ValueError(f"Variant {args.variant} requires --target-open-rate.")
    if args.variant in {"synchronous_control", "forced_open"} and args.target_open_rate is not None:
        raise ValueError(f"Variant {args.variant} does not accept --target-open-rate.")

    variant = make_variant(
        args.variant,
        config=config,
        target_open_rate=args.target_open_rate,
    )
    text_file = args.text_file or (
        args.repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    )
    default_output_name = (
        f"{variant.name}_{format_target_suffix(variant.target_open_rate)}"
        if variant.target_open_rate is not None
        else variant.name
    )
    output_dir = args.output_dir or (
        args.repo_root / "experiments" / "async_selective" / "artifacts" / args.stage / default_output_name
    )

    set_seed(config.seed)
    device = resolve_device(args.device)
    raw_text = text_file.read_text(encoding="utf-8")
    split = build_fixed_length_split(raw_text, config=config)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = make_model(
        vocab_size=split.train_dataset.vocab_size,
        config=config,
        variant=variant,
        device=device,
    )
    write_metadata(
        output_dir,
        config=config,
        split=split,
        variant=variant,
        model=model,
        text_file=text_file,
        raw_text=raw_text,
        device=device,
        stage=args.stage,
    )

    if args.stage == "overfit":
        batch_inputs = split.train_inputs[: config.overfit_batch_size].to(device)
        batch_targets = split.train_targets[: config.overfit_batch_size].to(device)
        trace, final_metrics = overfit_one_batch(
            model=model,
            inputs=batch_inputs,
            targets=batch_targets,
            config=config,
        )
        write_json(output_dir / "overfit_trace.json", trace)
        write_json(
            output_dir / "final_metrics.json",
            {
                "parameter_count": count_parameters(model),
                **final_metrics,
            },
        )
        print(
            f"{variant.label} overfit lm_loss={final_metrics['lm_loss']:.6f} "
            f"gate_collapse={final_metrics['gate_collapse']}"
        )
        return

    prompt = split.train_text[: config.context_size]
    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)
    epochs = config.tiny_epochs if args.stage == "tiny" else config.full_epochs
    history, progression_samples, final_metrics, runtime_seconds, sample = train_stage(
        model=model,
        dataset=split.train_dataset,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        prompt=prompt,
        config=config,
        device=device,
        epochs=epochs,
    )
    write_json(output_dir / "training_history.json", history)
    write_json(
        output_dir / "progression_samples.json",
        {
            "prompt": prompt.replace("\n", "\\n"),
            "sample_length": config.progression_sample_length,
            "checkpoints": progression_samples,
        },
    )
    best_epoch_record = min(history[1:] or history, key=lambda record: float(record["val_lm_loss"]))
    write_json(
        output_dir / "final_metrics.json",
        {
            "parameter_count": count_parameters(model),
            "runtime_seconds": runtime_seconds,
            "best_epoch": int(best_epoch_record["epoch"]),
            "best_val_lm_loss": float(best_epoch_record["val_lm_loss"]),
            "best_val_total_loss": float(best_epoch_record["val_total_loss"]),
            "final_val_lm_loss": float(final_metrics["val"]["lm_loss"]),
            "final_val_total_loss": float(final_metrics["val"]["total_loss"]),
            "final_val_accuracy": float(final_metrics["val"]["accuracy"]),
            "final_val_gate_summary": final_metrics["val"]["gate_summary"],
            "final_train_gate_summary": final_metrics["train"]["gate_summary"],
            "gate_collapse": final_metrics["gate_collapse"],
            "prompt": prompt.replace("\n", "\\n"),
        },
    )
    (output_dir / "sample.txt").write_text(sample, encoding="utf-8")
    print(
        f"{variant.label} {args.stage} best_val_lm_loss={float(best_epoch_record['val_lm_loss']):.6f} "
        f"runtime_seconds={runtime_seconds:.2f} gate_collapse={final_metrics['gate_collapse']}"
    )


if __name__ == "__main__":
    main()
