from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from core.fixed_window_char import load_dataset, set_seed
from core.model import ParallelDiagonalModel, count_parameters
from core.self_prediction import SelfPredictionLoss
from core.training import current_git_sha, current_git_status_short, fixed_step_indices, write_json

CONTEXT_SIZE = 32
VOCAB_SIZE = 67
TRAINING_STEPS = 20_000
EVAL_INTERVAL = 1_000
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
SEED = 42
NUM_BLOCKS = 4
RATES = (1, 2, 4, 8)
D_MODEL = 128
FEEDFORWARD_DIM = 256
D_AUX = 32
LAMBDA_WARMUP_STEPS = 1_000


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    lambda_aux_max: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = (
        repo_root / "experiments" / "fixed_multi_rate" / "artifacts" / "self_prediction_sweep"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.jsonl")
    return parser.parse_args()


def variant_specs() -> list[VariantSpec]:
    lambdas = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    specs: list[VariantSpec] = []
    for lambda_aux in lambdas:
        suffix = format(lambda_aux, ".0e") if lambda_aux != 0.0 else "0"
        specs.append(
            VariantSpec(
                key=f"parallel_4block_rate1248_self_prediction_lambda_{suffix}",
                label=f"parallel_4block_rate1248_self_prediction_lambda_{suffix}",
                lambda_aux_max=lambda_aux,
            )
        )
    return specs


def build_model(*, device: torch.device) -> ParallelDiagonalModel:
    return ParallelDiagonalModel(
        vocab_size=VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        num_blocks=NUM_BLOCKS,
        rates=RATES,
        readout_mode="all",
    ).to(device)


def build_aux_loss(*, device: torch.device) -> SelfPredictionLoss:
    return SelfPredictionLoss(
        d_model=D_MODEL,
        d_aux=D_AUX,
        num_blocks=NUM_BLOCKS,
        rates=RATES,
    ).to(device)


def materialize_batch(
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return train_inputs[indices], train_targets[indices]


def lambda_at_step(*, step: int, lambda_aux_max: float) -> float:
    if lambda_aux_max == 0.0:
        return 0.0
    return lambda_aux_max * min(step / LAMBDA_WARMUP_STEPS, 1.0)


def append_log(log_path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def evaluate_variant(
    model: ParallelDiagonalModel,
    aux_loss_module: SelfPredictionLoss,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    step: int,
    lambda_aux: float,
) -> dict[str, float | int]:
    was_training_model = model.training
    was_training_aux = aux_loss_module.training
    model.eval()
    aux_loss_module.eval()
    total_examples = 0
    total_task_loss = 0.0
    total_aux_loss = 0.0
    total_correct = 0
    with torch.inference_mode():
        for start in range(0, inputs.shape[0], batch_size):
            stop = min(start + batch_size, inputs.shape[0])
            batch_inputs = inputs[start:stop]
            batch_targets = targets[start:stop]
            logits, state = model.forward_with_state(batch_inputs)
            task_loss = F.cross_entropy(logits, batch_targets, reduction="sum")
            aux_loss = aux_loss_module(state.block_outputs)
            batch_examples = batch_targets.shape[0]
            total_examples += batch_examples
            total_task_loss += task_loss.item()
            total_aux_loss += aux_loss.item() * batch_examples
            total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
    if was_training_model:
        model.train()
    if was_training_aux:
        aux_loss_module.train()
    mean_task_loss = total_task_loss / total_examples
    mean_aux_loss = total_aux_loss / total_examples
    return {
        "step": step,
        "val_loss": round(mean_task_loss, 6),
        "val_task_loss": round(mean_task_loss, 6),
        "val_aux_loss": round(mean_aux_loss, 6),
        "val_total_loss": round(mean_task_loss + (lambda_aux * mean_aux_loss), 6),
        "val_accuracy": round(total_correct / total_examples, 6),
        "lambda_aux": round(lambda_aux, 8),
    }


def run_variant(
    spec: VariantSpec,
    *,
    args: argparse.Namespace,
    device: torch.device,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    batch_schedule: list[torch.Tensor],
    log_path: Path,
) -> dict[str, object]:
    set_seed(args.seed)
    model = build_model(device=device)
    aux_loss_module = build_aux_loss(device=device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(aux_loss_module.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    checkpoints = [
        evaluate_variant(
            model,
            aux_loss_module,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=0,
            lambda_aux=0.0,
        )
    ]
    variant_started_at = perf_counter()
    last_task_loss = torch.zeros((), device=device)
    last_aux_loss = torch.zeros((), device=device)
    last_total_loss = torch.zeros((), device=device)
    last_lambda_aux = 0.0

    append_log(
        log_path,
        {
            "stage": "variant_start",
            "variant": spec.key,
            "num_blocks": NUM_BLOCKS,
            "rates": list(RATES),
            "readout_mode": "all",
            "d_aux": D_AUX,
            "lambda_aux_max": spec.lambda_aux_max,
            "lambda_warmup_steps": LAMBDA_WARMUP_STEPS,
            "trainer": "manual_adamw",
        },
    )
    append_log(
        log_path,
        {
            "stage": "checkpoint",
            "variant": spec.key,
            "train_task_loss": None,
            "train_aux_loss": None,
            "train_total_loss": None,
            **checkpoints[-1],
        },
    )

    model.train()
    aux_loss_module.train()
    for zero_based_index, indices in enumerate(batch_schedule):
        batch_input, batch_target = materialize_batch(train_inputs, train_targets, indices)
        step = zero_based_index + 1
        current_lambda_aux = lambda_at_step(step=step, lambda_aux_max=spec.lambda_aux_max)

        optimizer.zero_grad(set_to_none=True)
        logits, state = model.forward_with_state(batch_input)
        task_loss = F.cross_entropy(logits, batch_target)
        aux_loss = aux_loss_module(state.block_outputs)
        total_loss = task_loss + (current_lambda_aux * aux_loss)
        total_loss.backward()
        optimizer.step()

        last_task_loss = task_loss.detach()
        last_aux_loss = aux_loss.detach()
        last_total_loss = total_loss.detach()
        last_lambda_aux = current_lambda_aux

        if step % args.eval_interval != 0 and step != args.training_steps:
            continue

        checkpoint = evaluate_variant(
            model,
            aux_loss_module,
            val_inputs,
            val_targets,
            batch_size=args.eval_batch_size,
            step=step,
            lambda_aux=current_lambda_aux,
        )
        checkpoints.append(checkpoint)
        append_log(
            log_path,
            {
                "stage": "checkpoint",
                "variant": spec.key,
                "train_task_loss": round(last_task_loss.item(), 6),
                "train_aux_loss": round(last_aux_loss.item(), 6),
                "train_total_loss": round(last_total_loss.item(), 6),
                **checkpoint,
            },
        )

    wall_seconds = perf_counter() - variant_started_at
    result = {
        **asdict(spec),
        "d_model": D_MODEL,
        "feedforward_dim": FEEDFORWARD_DIM,
        "num_blocks": NUM_BLOCKS,
        "rates": list(RATES),
        "readout_mode": "all",
        "d_aux": D_AUX,
        "lambda_warmup_steps": LAMBDA_WARMUP_STEPS,
        "parameter_count": count_parameters(model) + count_parameters(aux_loss_module),
        "checkpoints": checkpoints,
        "best_checkpoint": min(checkpoints, key=lambda checkpoint: checkpoint["val_loss"]),
        "final_checkpoint": checkpoints[-1],
        "final_training_task_loss": round(last_task_loss.item(), 6),
        "final_training_aux_loss": round(last_aux_loss.item(), 6),
        "final_training_total_loss": round(last_total_loss.item(), 6),
        "final_lambda_aux": round(last_lambda_aux, 8),
        "wall_seconds": round(wall_seconds, 6),
    }
    append_log(
        log_path,
        {
            "stage": "variant_done",
            "variant": spec.key,
            "final_checkpoint": result["final_checkpoint"],
            "wall_seconds": result["wall_seconds"],
        },
    )

    del optimizer
    del aux_loss_module
    del model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if args.training_steps <= 0:
        raise ValueError(f"training_steps must be positive, got {args.training_steps}.")
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    if args.training_steps < LAMBDA_WARMUP_STEPS:
        raise ValueError(
            f"training_steps must be at least {LAMBDA_WARMUP_STEPS} for lambda warmup, got {args.training_steps}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/self_prediction_sweep.py.")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.write_text("", encoding="utf-8")

    device = torch.device("cuda")
    set_seed(args.seed)
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
    batch_schedule = fixed_step_indices(
        train_inputs.shape[0],
        steps=args.training_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    specs = variant_specs()

    append_log(
        args.log_path,
        {
            "stage": "start",
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "variant_count": len(specs),
            "trainer": "manual_adamw",
        },
    )

    overall_started_at = perf_counter()
    results = [
        run_variant(
            spec,
            args=args,
            device=device,
            train_inputs=train_inputs,
            train_targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            batch_schedule=batch_schedule,
            log_path=args.log_path,
        )
        for spec in specs
    ]
    overall_wall_seconds = perf_counter() - overall_started_at

    git_status_short = current_git_status_short()
    report = {
        "config": {
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "context_size": CONTEXT_SIZE,
            "dataset_vocab_size": vocab_size,
            "model_vocab_size": VOCAB_SIZE,
            "schedule_seed": args.seed,
            "trainer": "manual_adamw",
            "d_model": D_MODEL,
            "feedforward_dim": FEEDFORWARD_DIM,
            "num_blocks": NUM_BLOCKS,
            "rates": list(RATES),
            "readout_mode": "all",
            "d_aux": D_AUX,
            "lambda_warmup_steps": LAMBDA_WARMUP_STEPS,
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
        "variants": [asdict(spec) for spec in specs],
        "results": results,
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "done",
            "report_path": str(args.report_path),
            "overall_wall_seconds": round(overall_wall_seconds, 6),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
