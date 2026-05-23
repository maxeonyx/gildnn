from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from jaxtyping import Float, Int
from torch import Tensor

from core.fixed_window_char import load_dataset, set_seed
from core.model import ParallelDiagonalForwardState, ParallelDiagonalModel, count_parameters
from core.training import (
    GraphTrainer,
    capturable_adamw,
    current_git_sha,
    current_git_status_short,
    evaluate_model,
    fixed_step_indices,
    write_json,
)

CONTEXT_SIZE = 32
VOCAB_SIZE = 67
TRAINING_STEPS = 20_000
EVAL_INTERVAL = 1_000
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 3e-4
DEFAULT_SEEDS = (42, 43, 44)
WARMUP_STEPS = 3
NUM_BLOCKS = 4
RATES = (1, 2, 4, 8)
D_MODEL = 256
FEEDFORWARD_DIM = 512


class GradientCondition(StrEnum):
    FULL_BACKPROP = "a_full_backprop"
    LATERAL_DETACHED = "b_lateral_detached"
    ONE_HOP_TRUNCATED = "c_one_hop_truncated"


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    label: str
    gradient_condition: str
    d_model: int
    feedforward_dim: int
    num_blocks: int
    rates: tuple[int, ...]
    readout_mode: str


class LocalLearningParallelDiagonalModel(ParallelDiagonalModel):
    def __init__(self, *, gradient_condition: GradientCondition, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.gradient_condition = gradient_condition

    def _apply_block_update(
        self,
        *,
        block_index: int,
        state_input: Float[Tensor, "batch d_model"],
        neighbor_state: Float[Tensor, "batch d_model"] | None,
    ) -> Float[Tensor, "batch d_model"]:
        block = self.blocks[block_index]
        block_mix = self.block_mixes[block_index]
        if neighbor_state is None:
            block_input = state_input
            block_delta = block(block_input)
            return block_mix(block_input, block_delta)

        match self.gradient_condition:
            case GradientCondition.FULL_BACKPROP:
                block_input = 0.5 * (state_input + neighbor_state)
                block_delta = block(block_input)
                return block_mix(block_input, block_delta)
            case GradientCondition.LATERAL_DETACHED:
                block_input = 0.5 * (state_input + neighbor_state.detach())
                block_delta = block(block_input)
                return block_mix(block_input, block_delta)
            case GradientCondition.ONE_HOP_TRUNCATED:
                full_block_input = 0.5 * (state_input + neighbor_state)
                proxy_block_input = 0.5 * (state_input + neighbor_state.detach())
                full_block_delta = block(full_block_input)
                proxy_block_delta = block(proxy_block_input)
                full_state = block_mix(full_block_input, full_block_delta)
                proxy_state = block_mix(proxy_block_input, proxy_block_delta)
                return full_state.detach() + (proxy_state - proxy_state.detach())
            case _:
                raise ValueError(f"Unsupported gradient condition: {self.gradient_condition!r}")

    def _forward_impl(
        self,
        tokens: Int[Tensor, "batch context"],
        *,
        return_state: bool,
    ) -> tuple[Float[Tensor, "batch vocab"], ParallelDiagonalForwardState | None]:
        embeddings = self.embedded_tokens(tokens)
        batch_size = tokens.shape[0]
        previous_states = [
            torch.zeros(batch_size, self.d_model, device=tokens.device, dtype=embeddings.dtype)
            for _ in range(self.num_blocks)
        ]
        block_output_history = [[] for _ in range(self.num_blocks)] if return_state else None

        for time_index in range(self.context_size):
            token_state = embeddings[:, time_index, :]
            seeded_states = [
                token_mix(previous_state, token_state)
                for token_mix, previous_state in zip(self.token_mixes, previous_states, strict=True)
            ]
            current_states = list(previous_states)
            for internal_step in range(self.internal_steps):
                next_states = list(current_states)
                for block_index, rate in enumerate(self.rates):
                    if time_index % rate != 0:
                        continue
                    state_input = seeded_states[block_index] if internal_step == 0 else current_states[block_index]
                    neighbor_state = None
                    if block_index > 0:
                        neighbor_state = (
                            previous_states[block_index - 1]
                            if internal_step == 0
                            else current_states[block_index - 1]
                        )
                    next_states[block_index] = self._apply_block_update(
                        block_index=block_index,
                        state_input=state_input,
                        neighbor_state=neighbor_state,
                    )
                current_states = next_states
            previous_states = current_states
            if block_output_history is not None:
                for block_index, state in enumerate(previous_states):
                    block_output_history[block_index].append(state)

        logits = self.output(self._readout_state(previous_states))
        if block_output_history is None:
            return logits, None
        return logits, ParallelDiagonalForwardState(
            block_outputs=[torch.stack(history, dim=1) for history in block_output_history]
        )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "fixed_multi_rate" / "artifacts" / "local_learning_gradient"
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=[condition.value for condition in GradientCondition],
    )
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def condition_specs() -> dict[str, ConditionSpec]:
    return {
        GradientCondition.FULL_BACKPROP.value: ConditionSpec(
            key=GradientCondition.FULL_BACKPROP.value,
            label="local_learning_gradient_a_full_backprop",
            gradient_condition=GradientCondition.FULL_BACKPROP.value,
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=NUM_BLOCKS,
            rates=RATES,
            readout_mode="all",
        ),
        GradientCondition.LATERAL_DETACHED.value: ConditionSpec(
            key=GradientCondition.LATERAL_DETACHED.value,
            label="local_learning_gradient_b_lateral_detached",
            gradient_condition=GradientCondition.LATERAL_DETACHED.value,
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=NUM_BLOCKS,
            rates=RATES,
            readout_mode="all",
        ),
        GradientCondition.ONE_HOP_TRUNCATED.value: ConditionSpec(
            key=GradientCondition.ONE_HOP_TRUNCATED.value,
            label="local_learning_gradient_c_one_hop_truncated",
            gradient_condition=GradientCondition.ONE_HOP_TRUNCATED.value,
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=NUM_BLOCKS,
            rates=RATES,
            readout_mode="all",
        ),
    }


def build_model(*, device: torch.device, gradient_condition: GradientCondition) -> LocalLearningParallelDiagonalModel:
    return LocalLearningParallelDiagonalModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        num_blocks=NUM_BLOCKS,
        rates=RATES,
        readout_mode="all",
        gradient_condition=gradient_condition,
    ).to(device)


def materialize_batch(
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return train_inputs[indices], train_targets[indices]


def checkpoint_metrics(
    model: torch.nn.Module,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    batch_size: int,
    step: int,
    wall_seconds: float,
) -> dict[str, float | int]:
    metrics = evaluate_model(model, val_inputs, val_targets, batch_size=batch_size)
    return {
        "step": step,
        "val_loss": round(metrics["loss"], 6),
        "val_accuracy": round(metrics["accuracy"], 6),
        "wall_seconds": round(wall_seconds, 6),
    }


def append_log(log_path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def mean_rounded(values: list[float]) -> float:
    return round(mean(values), 6)


def pairwise_condition_deltas(seed_results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    by_seed = {
        result["seed"]: {condition_result["condition"]: condition_result for condition_result in result["conditions"]}
        for result in seed_results
    }
    pairs = [
        (GradientCondition.FULL_BACKPROP.value, GradientCondition.LATERAL_DETACHED.value),
        (GradientCondition.FULL_BACKPROP.value, GradientCondition.ONE_HOP_TRUNCATED.value),
        (GradientCondition.ONE_HOP_TRUNCATED.value, GradientCondition.LATERAL_DETACHED.value),
    ]
    deltas: dict[str, dict[str, float]] = {}
    for left_condition, right_condition in pairs:
        final_val_losses = []
        final_val_accuracies = []
        wall_seconds = []
        for condition_results in by_seed.values():
            left = condition_results[left_condition]
            right = condition_results[right_condition]
            final_val_losses.append(
                float(left["final_checkpoint"]["val_loss"]) - float(right["final_checkpoint"]["val_loss"])
            )
            final_val_accuracies.append(
                float(left["final_checkpoint"]["val_accuracy"]) - float(right["final_checkpoint"]["val_accuracy"])
            )
            wall_seconds.append(float(left["wall_seconds"]) - float(right["wall_seconds"]))
        deltas[f"{left_condition}_minus_{right_condition}"] = {
            "final_val_loss": mean_rounded(final_val_losses),
            "final_val_accuracy": mean_rounded(final_val_accuracies),
            "wall_seconds": mean_rounded(wall_seconds),
        }
    return deltas


def build_dry_run_report(
    *,
    args: argparse.Namespace,
    device: torch.device,
    vocab_size: int,
    specs: dict[str, ConditionSpec],
    model_checks: dict[str, dict[str, object]],
) -> dict[str, object]:
    git_status_short = current_git_status_short()
    return {
        "dry_run": True,
        "config": {
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
            "conditions": args.conditions,
            "warmup_steps": WARMUP_STEPS,
            "context_size": CONTEXT_SIZE,
            "dataset_vocab_size": vocab_size,
            "model_vocab_size": VOCAB_SIZE,
            "trainer": "GraphTrainer",
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
        "conditions": {key: asdict(spec) for key, spec in specs.items() if key in args.conditions},
        "model_checks": model_checks,
    }


def run_dry_run(*, args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/local_learning_gradient.py.")

    specs = condition_specs()
    for condition in args.conditions:
        if condition not in specs:
            raise ValueError(f"Unknown condition {condition!r}. Expected one of {sorted(specs)}.")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text("", encoding="utf-8")

    device = torch.device("cuda")
    set_seed(args.seeds[0])
    (train_inputs, _), (_, _), vocab_size = load_dataset(context_size=CONTEXT_SIZE)
    if vocab_size > VOCAB_SIZE:
        raise ValueError(
            f"Loaded dataset vocab_size {vocab_size} exceeds configured model vocab_size {VOCAB_SIZE}."
        )

    sample_tokens = train_inputs[:2].to(device)
    model_checks: dict[str, dict[str, object]] = {}
    for condition_key in args.conditions:
        model = build_model(device=device, gradient_condition=GradientCondition(condition_key))
        with torch.no_grad():
            output = model(sample_tokens)
        model_checks[condition_key] = {
            "parameter_count": count_parameters(model),
            "output_shape": list(output.shape),
            "mix_coefficients": model.mix_coefficients(),
        }
        del model
    torch.cuda.empty_cache()

    report = build_dry_run_report(
        args=args,
        device=device,
        vocab_size=vocab_size,
        specs=specs,
        model_checks=model_checks,
    )
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "dry_run",
            "report_path": str(args.report_path),
            "conditions": args.conditions,
            "model_checks": model_checks,
        },
    )
    return 0


def train_single_condition(
    *,
    seed: int,
    condition_key: str,
    spec: ConditionSpec,
    args: argparse.Namespace,
    device: torch.device,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
) -> dict[str, object]:
    batch_schedule = fixed_step_indices(
        train_inputs.shape[0],
        steps=args.training_steps,
        batch_size=args.batch_size,
        seed=seed,
        device=device,
    )
    warmup_batches = [
        materialize_batch(train_inputs, train_targets, indices)
        for indices in batch_schedule[:WARMUP_STEPS]
    ]

    set_seed(seed)
    model = build_model(device=device, gradient_condition=GradientCondition(condition_key))
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
            wall_seconds=0.0,
        )
    ]
    parameter_count = count_parameters(model)
    append_log(
        args.log_path,
        {
            "stage": "condition_started",
            "seed": seed,
            "condition": condition_key,
            "label": spec.label,
            "parameter_count": parameter_count,
            "initial_checkpoint": checkpoints[-1],
        },
    )

    started_at = perf_counter()
    trainer.capture(warmup_batches)
    last_loss = trainer.static_loss.detach().clone()

    for zero_based_index, indices in enumerate(batch_schedule[WARMUP_STEPS:], start=WARMUP_STEPS):
        batch_input, batch_target = materialize_batch(train_inputs, train_targets, indices)
        last_loss = trainer.step(batch_input, batch_target)
        step = zero_based_index + 1
        if step % args.eval_interval != 0 and step != args.training_steps:
            continue

        trainer.synchronize()
        wall_seconds = perf_counter() - started_at
        checkpoint = checkpoint_metrics(
            model,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=step,
            wall_seconds=wall_seconds,
        )
        checkpoints.append(checkpoint)
        append_log(
            args.log_path,
            {
                "stage": "checkpoint",
                "seed": seed,
                "condition": condition_key,
                **checkpoint,
            },
        )

    trainer.synchronize()
    wall_seconds = perf_counter() - started_at
    result = {
        "condition": condition_key,
        "label": spec.label,
        "gradient_condition": spec.gradient_condition,
        "d_model": spec.d_model,
        "feedforward_dim": spec.feedforward_dim,
        "num_blocks": spec.num_blocks,
        "rates": list(spec.rates),
        "readout_mode": spec.readout_mode,
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
            "stage": "condition_finished",
            "seed": seed,
            "condition": condition_key,
            "final_checkpoint": result["final_checkpoint"],
            "wall_seconds": result["wall_seconds"],
        },
    )

    del trainer
    del optimizer
    del model
    del warmup_batches
    del batch_schedule
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    specs = condition_specs()
    for condition in args.conditions:
        if condition not in specs:
            raise ValueError(f"Unknown condition {condition!r}. Expected one of {sorted(specs)}.")
    if args.dry_run:
        return run_dry_run(args=args)
    if args.training_steps < WARMUP_STEPS:
        raise ValueError(
            f"training_steps must be at least {WARMUP_STEPS} so GraphTrainer.capture() can warm up."
        )
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    if len(args.seeds) == 0:
        raise ValueError("At least one seed is required.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/local_learning_gradient.py.")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text("", encoding="utf-8")

    device = torch.device("cuda")
    set_seed(args.seeds[0])
    (train_inputs, train_targets), (val_inputs, val_targets), vocab_size = load_dataset(
        context_size=CONTEXT_SIZE
    )
    if vocab_size > VOCAB_SIZE:
        raise ValueError(
            f"Loaded dataset vocab_size {vocab_size} exceeds configured model vocab_size {VOCAB_SIZE}."
        )

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    val_inputs = val_inputs.to(device)
    val_targets = val_targets.to(device)

    append_log(
        args.log_path,
        {
            "stage": "run_started",
            "seeds": args.seeds,
            "conditions": args.conditions,
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "trainer": "GraphTrainer",
        },
    )

    seed_results: list[dict[str, object]] = []
    overall_started_at = perf_counter()
    for seed in args.seeds:
        append_log(args.log_path, {"stage": "seed_started", "seed": seed})
        condition_results = [
            train_single_condition(
                seed=seed,
                condition_key=condition_key,
                spec=specs[condition_key],
                args=args,
                device=device,
                train_inputs=train_inputs,
                train_targets=train_targets,
                val_inputs=val_inputs,
                val_targets=val_targets,
            )
            for condition_key in args.conditions
        ]
        seed_results.append(
            {
                "seed": seed,
                "conditions": condition_results,
            }
        )
        append_log(
            args.log_path,
            {
                "stage": "seed_finished",
                "seed": seed,
                "conditions": {
                    result["condition"]: {
                        "final_val_loss": result["final_checkpoint"]["val_loss"],
                        "wall_seconds": result["wall_seconds"],
                    }
                    for result in condition_results
                },
            },
        )

    overall_wall_seconds = perf_counter() - overall_started_at
    git_status_short = current_git_status_short()
    averages = {
        condition_key: {
            "final_val_loss": mean_rounded(
                [
                    next(
                        result
                        for result in seed_result["conditions"]
                        if result["condition"] == condition_key
                    )["final_checkpoint"]["val_loss"]
                    for seed_result in seed_results
                ]
            ),
            "final_val_accuracy": mean_rounded(
                [
                    next(
                        result
                        for result in seed_result["conditions"]
                        if result["condition"] == condition_key
                    )["final_checkpoint"]["val_accuracy"]
                    for seed_result in seed_results
                ]
            ),
            "best_val_loss": mean_rounded(
                [
                    next(
                        result
                        for result in seed_result["conditions"]
                        if result["condition"] == condition_key
                    )["best_checkpoint"]["val_loss"]
                    for seed_result in seed_results
                ]
            ),
            "final_training_loss": mean_rounded(
                [
                    next(
                        result
                        for result in seed_result["conditions"]
                        if result["condition"] == condition_key
                    )["final_training_loss"]
                    for seed_result in seed_results
                ]
            ),
            "wall_seconds": mean_rounded(
                [
                    next(
                        result
                        for result in seed_result["conditions"]
                        if result["condition"] == condition_key
                    )["wall_seconds"]
                    for seed_result in seed_results
                ]
            ),
        }
        for condition_key in args.conditions
    }
    report = {
        "config": {
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
            "conditions": args.conditions,
            "warmup_steps": WARMUP_STEPS,
            "context_size": CONTEXT_SIZE,
            "dataset_vocab_size": vocab_size,
            "model_vocab_size": VOCAB_SIZE,
            "trainer": "GraphTrainer",
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
        "timing": {
            "overall_wall_seconds": round(overall_wall_seconds, 6),
        },
        "conditions": {key: asdict(specs[key]) for key in args.conditions},
        "seed_results": seed_results,
        "averages": averages,
        "pairwise_condition_deltas": pairwise_condition_deltas(seed_results),
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "done",
            "report_path": str(args.report_path),
            "averages": averages,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
