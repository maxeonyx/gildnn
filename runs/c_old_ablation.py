from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

# Ensure repo root is importable regardless of how this script is launched
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaxtyping import Int
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.dataset import CorpusData, load_corpus
from core.fixed_window_char import set_seed
from core.model import ParallelDiagonalModel, count_parameters
from core.training import GraphTrainer, capturable_adamw, current_git_sha, current_git_status_short, write_json

CONTEXT_SIZE = 128
TRAINING_STEPS = 20_000
EVAL_INTERVAL = 1_000
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 3e-4
DEFAULT_SEEDS = (42, 43)
DEFAULT_VARIANTS = ("C_lateral", "C_isolated")
EVAL_SAMPLES = 4096
WARMUP_STEPS = 3
EXPECTED_PARAMETER_COUNT = 3_639_168


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    num_blocks: int
    internal_steps: int
    d_model: int
    feedforward_dim: int
    token_injection: str
    readout_mode: str
    topology: str
    rates: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "wikitext_103" / "artifacts" / "c_old_ablation"
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--eval-samples", type=int, default=EVAL_SAMPLES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--compile", dest="compile_model", action="store_true")
    parser.add_argument("--no-compile", dest="compile_model", action="store_false")
    parser.set_defaults(compile_model=False)
    parser.add_argument("--sanity-check-only", action="store_true")
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
        "C_lateral": VariantSpec(
            key="C_lateral",
            label="c_old_lateral",
            num_blocks=4,
            internal_steps=1,
            d_model=256,
            feedforward_dim=512,
            token_injection="all",
            readout_mode="all",
            topology="upward",
            rates=(1, 1, 1, 1),
        ),
        "C_isolated": VariantSpec(
            key="C_isolated",
            label="c_old_isolated",
            num_blocks=4,
            internal_steps=1,
            d_model=256,
            feedforward_dim=512,
            token_injection="all",
            readout_mode="all",
            topology="isolated",
            rates=(1, 1, 1, 1),
        ),
    }


def append_log(log_path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(requested_device)


def maybe_compile_model(model: ParallelDiagonalModel, *, enabled: bool) -> nn.Module:
    if not enabled:
        return model
    return torch.compile(model, backend="aot_eager")


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
        internal_steps=spec.internal_steps,
        readout_mode=spec.readout_mode,
        token_injection=spec.token_injection,
        topology=spec.topology,
    ).to(device)


def validate_parameter_counts(*, vocab_size: int, selected_specs: dict[str, VariantSpec]) -> dict[str, object]:
    cpu_device = torch.device("cpu")
    counts_by_variant: dict[str, int] = {}
    for key, spec in selected_specs.items():
        model = build_model(device=cpu_device, vocab_size=vocab_size, spec=spec)
        counts_by_variant[key] = count_parameters(model)
        del model

    if set(counts_by_variant) != set(DEFAULT_VARIANTS):
        raise ValueError(
            "Parameter validation requires both C_lateral and C_isolated variants. "
            f"Got: {sorted(counts_by_variant)}."
        )

    shared_count = counts_by_variant["C_lateral"]
    if counts_by_variant["C_isolated"] != shared_count:
        raise ValueError(
            "C_lateral and C_isolated must have identical parameter counts. "
            f"Got {counts_by_variant['C_lateral']} vs {counts_by_variant['C_isolated']}."
        )
    if shared_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            "C_old ablation expected parameter count mismatch. "
            f"Expected {EXPECTED_PARAMETER_COUNT}, got {shared_count}."
        )

    return {
        "counts_by_variant": counts_by_variant,
        "shared_parameter_count": shared_count,
        "matches_expected_parameter_count": shared_count == EXPECTED_PARAMETER_COUNT,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
    }


def build_optimizer(
    model: ParallelDiagonalModel,
    *,
    device: torch.device,
    compile_model: bool,
    learning_rate: float,
) -> torch.optim.AdamW:
    if device.type == "cuda" and not compile_model:
        return capturable_adamw(model, lr=learning_rate)
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )


def random_batches(
    encoded_corpus: Int[Tensor, "tokens"],
    *,
    context_size: int,
    batch_size: int,
    device: torch.device,
    rng: torch.Generator,
) -> Iterator[tuple[Int[Tensor, "batch context"], Int[Tensor, "batch"]]]:
    if encoded_corpus.ndim != 1:
        raise ValueError(f"random_batches expects a 1D corpus tensor, got shape {tuple(encoded_corpus.shape)}.")
    if encoded_corpus.dtype != torch.long:
        raise ValueError(f"random_batches expects torch.long tokens, got {encoded_corpus.dtype}.")

    max_start = encoded_corpus.numel() - context_size
    if max_start <= 0:
        raise ValueError(
            "random_batches needs more encoded tokens than context_size. "
            f"Got corpus length {encoded_corpus.numel()} and context_size {context_size}."
        )

    offsets = torch.arange(context_size, dtype=torch.long)
    pin_memory = device.type == "cuda"
    while True:
        starts = torch.randint(0, max_start, (batch_size,), generator=rng)
        inputs = encoded_corpus[starts[:, None] + offsets]
        targets = encoded_corpus[starts + context_size]
        if pin_memory:
            inputs = inputs.pin_memory()
            targets = targets.pin_memory()
        yield (
            inputs.to(device=device, dtype=torch.long, non_blocking=pin_memory),
            targets.to(device=device, dtype=torch.long, non_blocking=pin_memory),
        )


@torch.inference_mode()
def evaluate_model_call(
    *,
    model: ParallelDiagonalModel,
    model_call: nn.Module,
    inputs: Int[Tensor, "examples context"],
    targets: Int[Tensor, "examples"],
    batch_size: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    if model_call is not model:
        model_call.eval()

    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        batch_inputs = inputs[start:stop]
        batch_targets = targets[start:stop]
        logits = model_call(batch_inputs)
        batch_examples = batch_targets.shape[0]
        total_loss += F.cross_entropy(logits, batch_targets, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
        total_examples += batch_examples

    if was_training:
        model.train()
        if model_call is not model:
            model_call.train()

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def checkpoint_metrics(
    *,
    model: ParallelDiagonalModel,
    model_call: nn.Module,
    val_inputs: Int[Tensor, "examples context"],
    val_targets: Int[Tensor, "examples"],
    batch_size: int,
    step: int,
    tokens_per_second: float | None,
) -> dict[str, float | int | None]:
    metrics = evaluate_model_call(
        model=model,
        model_call=model_call,
        inputs=val_inputs,
        targets=val_targets,
        batch_size=batch_size,
    )
    return {
        "step": step,
        "val_loss": round(metrics["loss"], 6),
        "val_accuracy": round(metrics["accuracy"], 6),
        "tokens_per_second": None if tokens_per_second is None else round(tokens_per_second, 6),
    }


def verify_forward_and_gradients(
    *,
    model: ParallelDiagonalModel,
    model_call: nn.Module,
    device: torch.device,
    vocab_size: int,
) -> dict[str, object]:
    saved_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    model.train()
    if model_call is not model:
        model_call.train()

    dummy_inputs = torch.randint(0, vocab_size, (2, CONTEXT_SIZE), device=device)
    dummy_targets = torch.randint(0, vocab_size, (2,), device=device)
    logits = model_call(dummy_inputs)
    if tuple(logits.shape) != (2, vocab_size):
        raise RuntimeError(
            f"Verification failed: expected output shape (2, {vocab_size}), got {tuple(logits.shape)}."
        )

    loss = F.cross_entropy(logits, dummy_targets)
    if not torch.isfinite(loss):
        raise RuntimeError("Verification failed: dummy loss is not finite.")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    missing_gradient_parameters: list[str] = []
    zero_gradient_parameters: list[str] = []
    gradient_norm_sum = 0.0
    total_trainable_parameters = 0
    nonzero_gradient_parameters = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        total_trainable_parameters += 1
        if parameter.grad is None:
            missing_gradient_parameters.append(name)
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Verification failed: parameter {name} has NaN or Inf gradients.")
        grad_norm = parameter.grad.detach().norm().item()
        gradient_norm_sum += grad_norm
        if grad_norm > 0.0:
            nonzero_gradient_parameters += 1
        else:
            zero_gradient_parameters.append(name)

    if len(missing_gradient_parameters) > 0 or len(zero_gradient_parameters) > 0:
        raise RuntimeError(
            "Verification failed: some parameters did not receive gradients. "
            f"missing={missing_gradient_parameters}, zero={zero_gradient_parameters}"
        )

    optimizer.step()
    model.zero_grad(set_to_none=True)
    if model_call is not model:
        model_call.zero_grad(set_to_none=True)
    model.load_state_dict(saved_state)

    return {
        "output_shape": list(logits.shape),
        "dummy_loss": round(loss.item(), 6),
        "total_trainable_parameters": total_trainable_parameters,
        "nonzero_gradient_parameters": nonzero_gradient_parameters,
        "gradient_norm_sum": round(gradient_norm_sum, 6),
        "missing_gradient_parameters": missing_gradient_parameters,
        "zero_gradient_parameters": zero_gradient_parameters,
        "forward_pass_ok": True,
        "backward_pass_ok": True,
    }


def training_throughput_tokens_per_second(
    *,
    current_step: int,
    previous_step: int,
    batch_size: int,
    elapsed_seconds: float,
) -> float:
    if elapsed_seconds <= 0.0:
        raise ValueError(f"elapsed_seconds must be positive, got {elapsed_seconds}.")
    steps_completed = current_step - previous_step
    if steps_completed <= 0:
        raise ValueError(
            f"current_step must be greater than previous_step, got current_step={current_step}, previous_step={previous_step}."
        )
    tokens_processed = steps_completed * batch_size * CONTEXT_SIZE
    return tokens_processed / elapsed_seconds


def save_temporary_state_dict(*, model: ParallelDiagonalModel, checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(state_dict, checkpoint_path)


def train_single_variant(
    *,
    seed: int,
    variant_key: str,
    spec: VariantSpec,
    args: argparse.Namespace,
    corpus: CorpusData,
    device: torch.device,
    val_inputs: Int[Tensor, "examples context"],
    val_targets: Int[Tensor, "examples"],
    parameter_audit: dict[str, object],
) -> tuple[dict[str, object], Path | None]:
    set_seed(seed)
    encoded_corpus = getattr(corpus.train_dataset, "encoded_corpus", None)
    if not isinstance(encoded_corpus, torch.Tensor):
        raise TypeError(
            "train_single_variant expects corpus.train_dataset to expose encoded_corpus as a torch.Tensor."
        )

    model = build_model(device=device, vocab_size=corpus.vocab_size, spec=spec)
    parameter_count = count_parameters(model)
    model_call: nn.Module = maybe_compile_model(model, enabled=args.compile_model)
    verification = verify_forward_and_gradients(
        model=model,
        model_call=model_call,
        device=device,
        vocab_size=corpus.vocab_size,
    )

    initial_checkpoint = checkpoint_metrics(
        model=model,
        model_call=model_call,
        val_inputs=val_inputs,
        val_targets=val_targets,
        batch_size=args.eval_batch_size,
        step=0,
        tokens_per_second=None,
    )

    append_log(
        args.log_path,
        {
            "stage": "variant_started",
            "seed": seed,
            "variant": variant_key,
            "label": spec.label,
            "parameter_count": parameter_count,
            "parameter_audit": parameter_audit,
            "compiled": args.compile_model,
            "verification": verification,
            "initial_checkpoint": initial_checkpoint,
        },
    )

    expected_parameter_count = int(parameter_audit["counts_by_variant"][variant_key])
    if parameter_count != expected_parameter_count:
        raise RuntimeError(
            f"Parameter count mismatch for {variant_key}: built model has {parameter_count}, audit expected {expected_parameter_count}."
        )

    temporary_checkpoint_path: Path | None = None
    if args.sanity_check_only:
        result = {
            "seed": seed,
            "variant": variant_key,
            "label": spec.label,
            "class_name": "ParallelDiagonalModel",
            "config": asdict(spec),
            "parameter_count": parameter_count,
            "parameter_audit": parameter_audit,
            "compiled": args.compile_model,
            "verification": verification,
            "checkpoints": [initial_checkpoint],
            "best_checkpoint": initial_checkpoint,
            "final_checkpoint": initial_checkpoint,
            "final_training_loss": None,
            "wall_seconds": 0.0,
        }
        append_log(
            args.log_path,
            {
                "stage": "variant_finished",
                "seed": seed,
                "variant": variant_key,
                "final_val_loss": initial_checkpoint["val_loss"],
                "final_val_accuracy": initial_checkpoint["val_accuracy"],
                "tokens_per_second": initial_checkpoint["tokens_per_second"],
                "wall_seconds": 0.0,
            },
        )
        del model_call
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result, temporary_checkpoint_path

    optimizer = build_optimizer(
        model,
        device=device,
        compile_model=args.compile_model,
        learning_rate=args.learning_rate,
    )

    batch_rng = torch.Generator(device="cpu")
    batch_rng.manual_seed(seed)
    batch_iterator = random_batches(
        encoded_corpus,
        context_size=CONTEXT_SIZE,
        batch_size=args.batch_size,
        device=device,
        rng=batch_rng,
    )

    checkpoints = [initial_checkpoint]
    trainer: GraphTrainer | None = None
    last_loss: Tensor | None = None
    started_at = perf_counter()
    last_eval_started_at = started_at
    last_eval_step = 0

    if device.type == "cuda" and not args.compile_model:
        warmup_batches = [next(batch_iterator) for _ in range(WARMUP_STEPS)]
        trainer = GraphTrainer(
            model,
            optimizer,
            batch_size=args.batch_size,
            seq_len=CONTEXT_SIZE,
            device=device,
        )
        trainer.capture(warmup_batches)
        last_loss = trainer.static_loss.detach().clone()
        training_step_start = WARMUP_STEPS + 1
    else:
        training_step_start = 1

    for step in range(training_step_start, args.training_steps + 1):
        batch_input, batch_target = next(batch_iterator)
        if trainer is not None:
            last_loss = trainer.step(batch_input, batch_target)
        else:
            optimizer.zero_grad(set_to_none=True)
            logits = model_call(batch_input)
            loss = F.cross_entropy(logits, batch_target)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Training diverged for variant {variant_key} at step {step}: loss is NaN or Inf."
                )
            loss.backward()
            optimizer.step()
            last_loss = loss.detach().clone()

        if step % args.eval_interval != 0 and step != args.training_steps:
            continue

        if trainer is not None:
            trainer.synchronize()
        now = perf_counter()
        tokens_per_second = training_throughput_tokens_per_second(
            current_step=step,
            previous_step=last_eval_step,
            batch_size=args.batch_size,
            elapsed_seconds=now - last_eval_started_at,
        )
        checkpoint = checkpoint_metrics(
            model=model,
            model_call=model_call,
            val_inputs=val_inputs,
            val_targets=val_targets,
            batch_size=args.eval_batch_size,
            step=step,
            tokens_per_second=tokens_per_second,
        )
        checkpoints.append(checkpoint)
        append_log(
            args.log_path,
            {
                "stage": "checkpoint",
                "seed": seed,
                "variant": variant_key,
                **checkpoint,
            },
        )
        last_eval_started_at = perf_counter()
        last_eval_step = step

    if trainer is not None:
        trainer.synchronize()
    if last_loss is None:
        raise RuntimeError(f"Variant {variant_key} completed without recording a training loss.")

    if variant_key == "C_lateral":
        temporary_checkpoint_path = args.report_path.parent / f"c_lateral_seed_{seed}.pt"
        save_temporary_state_dict(model=model, checkpoint_path=temporary_checkpoint_path)

    wall_seconds = perf_counter() - started_at
    result = {
        "seed": seed,
        "variant": variant_key,
        "label": spec.label,
        "class_name": "ParallelDiagonalModel",
        "config": asdict(spec),
        "parameter_count": parameter_count,
        "parameter_audit": parameter_audit,
        "compiled": args.compile_model,
        "verification": verification,
        "checkpoints": checkpoints,
        "best_checkpoint": min(checkpoints, key=lambda checkpoint: checkpoint["val_loss"]),
        "final_checkpoint": checkpoints[-1],
        "final_training_loss": round(last_loss.item(), 6),
        "wall_seconds": round(wall_seconds, 6),
    }
    append_log(
        args.log_path,
        {
            "stage": "variant_finished",
            "seed": seed,
            "variant": variant_key,
            "final_val_loss": result["final_checkpoint"]["val_loss"],
            "final_val_accuracy": result["final_checkpoint"]["val_accuracy"],
            "tokens_per_second": result["final_checkpoint"]["tokens_per_second"],
            "wall_seconds": result["wall_seconds"],
            "temporary_checkpoint_path": None if temporary_checkpoint_path is None else str(temporary_checkpoint_path),
        },
    )

    del optimizer
    del trainer
    del model_call
    del model
    del batch_iterator
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, temporary_checkpoint_path


def ablation_logits(*, ablation_name: str, num_blocks: int) -> Tensor | None:
    if num_blocks != 4:
        raise ValueError(f"C_old ablation expects num_blocks=4, got {num_blocks}.")
    match ablation_name:
        case "full":
            return None
        case "block0_only":
            return torch.tensor([100.0, -100.0, -100.0, -100.0], dtype=torch.float32)
        case "blocks_0_1":
            return torch.tensor([100.0, 100.0, -100.0, -100.0], dtype=torch.float32)
        case "blocks_0_1_2":
            return torch.tensor([100.0, 100.0, 100.0, -100.0], dtype=torch.float32)
        case _:
            raise ValueError(f"Unknown ablation {ablation_name!r}.")


def run_ablation_suite(
    *,
    seed: int,
    checkpoint_path: Path,
    spec: VariantSpec,
    vocab_size: int,
    device: torch.device,
    val_inputs: Int[Tensor, "examples context"],
    val_targets: Int[Tensor, "examples"],
    eval_batch_size: int,
    log_path: Path,
) -> dict[str, dict[str, float]]:
    model = build_model(device=device, vocab_size=vocab_size, spec=spec)
    loaded_state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(loaded_state, dict):
        raise TypeError(f"Expected checkpoint at {checkpoint_path} to be a state_dict dict.")
    model.load_state_dict(loaded_state)
    if model.readout_logits is None:
        raise RuntimeError("C_lateral ablations require readout_mode='all' with readout_logits present.")

    append_log(log_path, {"stage": "ablation_started", "seed": seed, "source_variant": "C_lateral"})

    ablation_names = ("full", "block0_only", "blocks_0_1", "blocks_0_1_2")
    original_logits = model.readout_logits.detach().clone()
    results: dict[str, dict[str, float]] = {}
    for name in ablation_names:
        replacement_logits = ablation_logits(ablation_name=name, num_blocks=spec.num_blocks)
        with torch.no_grad():
            if replacement_logits is None:
                model.readout_logits.copy_(original_logits)
            else:
                model.readout_logits.copy_(replacement_logits.to(device=model.readout_logits.device))
        metrics = evaluate_model_call(
            model=model,
            model_call=model,
            inputs=val_inputs,
            targets=val_targets,
            batch_size=eval_batch_size,
        )
        rounded_metrics = {
            "val_loss": round(metrics["loss"], 6),
            "val_accuracy": round(metrics["accuracy"], 6),
        }
        results[name] = rounded_metrics
        append_log(
            log_path,
            {
                "stage": "ablation_result",
                "seed": seed,
                "source_variant": "C_lateral",
                "ablation": name,
                **rounded_metrics,
            },
        )

    append_log(log_path, {"stage": "ablation_finished", "seed": seed, "source_variant": "C_lateral"})

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def mean_rounded(values: list[float]) -> float:
    return round(mean(values), 6)


def std_rounded(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(pstdev(values), 6)


def summarize_results(*, per_seed_results: list[dict[str, object]], sanity_check_only: bool) -> dict[str, object]:
    results_by_variant: dict[str, list[dict[str, object]]] = {}
    for result in per_seed_results:
        results_by_variant.setdefault(str(result["variant"]), []).append(result)

    summary: dict[str, object] = {}
    for variant, runs in results_by_variant.items():
        if sanity_check_only:
            summary[variant] = {
                "num_runs": len(runs),
                "parameter_count": runs[0]["parameter_count"],
                "parameter_audit": runs[0]["parameter_audit"],
                "verification": runs[0]["verification"],
            }
            continue

        final_losses = [float(run["final_checkpoint"]["val_loss"]) for run in runs]
        final_accuracies = [float(run["final_checkpoint"]["val_accuracy"]) for run in runs]
        final_tokens_per_second = [float(run["final_checkpoint"]["tokens_per_second"]) for run in runs]
        wall_seconds = [float(run["wall_seconds"]) for run in runs]
        summary[variant] = {
            "num_runs": len(runs),
            "mean_final_val_loss": mean_rounded(final_losses),
            "std_final_val_loss": std_rounded(final_losses),
            "mean_final_val_accuracy": mean_rounded(final_accuracies),
            "std_final_val_accuracy": std_rounded(final_accuracies),
            "mean_final_tokens_per_second": mean_rounded(final_tokens_per_second),
            "std_final_tokens_per_second": std_rounded(final_tokens_per_second),
            "mean_wall_seconds": mean_rounded(wall_seconds),
            "runs": runs,
        }
    return summary


def main() -> int:
    args = parse_args()
    if len(args.seeds) == 0:
        raise ValueError("At least one seed is required.")
    if args.training_steps <= 0:
        raise ValueError(f"training_steps must be positive, got {args.training_steps}.")
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}.")
    if args.eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {args.eval_batch_size}.")
    if args.eval_samples <= 0:
        raise ValueError(f"eval_samples must be positive, got {args.eval_samples}.")
    if not args.sanity_check_only and args.training_steps < WARMUP_STEPS and not args.compile_model:
        raise ValueError(
            f"training_steps must be at least {WARMUP_STEPS} for CUDA graph warmup when compile is disabled."
        )

    specs = variant_specs()
    unknown_variants = [variant for variant in args.variants if variant not in specs]
    if len(unknown_variants) > 0:
        raise ValueError(f"Unknown variants requested: {unknown_variants}. Available variants: {list(specs)}.")
    if set(args.variants) != set(DEFAULT_VARIANTS):
        raise ValueError("This experiment requires both C_lateral and C_isolated so the comparison stays matched.")

    selected_specs = {key: specs[key] for key in args.variants}
    device = resolve_device(args.device)

    lock_path = Path("runs/active.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_content = (
        f"PID: {os.getpid()}\n"
        f"Experiment: c_old_ablation\n"
        f"Variants: {args.variants}\n"
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    )
    lock_path.write_text(lock_content, encoding="utf-8")

    def _remove_lock() -> None:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    atexit.register(_remove_lock)

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

    corpus = load_corpus(
        train_path=args.train_path,
        val_path=args.val_path,
        context_size=CONTEXT_SIZE,
        eval_samples=args.eval_samples,
    )
    parameter_audit = validate_parameter_counts(vocab_size=corpus.vocab_size, selected_specs=selected_specs)
    val_inputs = corpus.val_inputs.to(device=device, dtype=torch.long)
    val_targets = corpus.val_targets.to(device=device, dtype=torch.long)

    append_log(
        args.log_path,
        {
            "stage": "experiment_started",
            "sanity_check_only": args.sanity_check_only,
            "seeds": args.seeds,
            "variants": args.variants,
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "context_size": CONTEXT_SIZE,
            "compile_model": args.compile_model,
            "device": str(device),
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
            "vocab_size": corpus.vocab_size,
            "train_dataset_size": len(corpus.train_dataset),
            "val_examples": int(corpus.val_inputs.shape[0]),
            "parameter_audit": parameter_audit,
        },
    )

    overall_started_at = perf_counter()
    per_seed_results: list[dict[str, object]] = []
    lateral_checkpoints: dict[int, Path] = {}
    for seed in args.seeds:
        append_log(args.log_path, {"stage": "seed_started", "seed": seed})
        for variant_key in args.variants:
            result, temporary_checkpoint_path = train_single_variant(
                seed=seed,
                variant_key=variant_key,
                spec=selected_specs[variant_key],
                args=args,
                corpus=corpus,
                device=device,
                val_inputs=val_inputs,
                val_targets=val_targets,
                parameter_audit=parameter_audit,
            )
            per_seed_results.append(result)
            if temporary_checkpoint_path is not None:
                lateral_checkpoints[seed] = temporary_checkpoint_path
        append_log(args.log_path, {"stage": "seed_finished", "seed": seed})

    ablation_results: dict[str, dict[str, dict[str, float]]] = {}
    if not args.sanity_check_only:
        missing_checkpoints = [seed for seed in args.seeds if seed not in lateral_checkpoints]
        if len(missing_checkpoints) > 0:
            raise RuntimeError(f"Missing C_lateral checkpoints for ablations: {missing_checkpoints}.")
        for seed in args.seeds:
            ablation_results[str(seed)] = run_ablation_suite(
                seed=seed,
                checkpoint_path=lateral_checkpoints[seed],
                spec=selected_specs["C_lateral"],
                vocab_size=corpus.vocab_size,
                device=device,
                val_inputs=val_inputs,
                val_targets=val_targets,
                eval_batch_size=args.eval_batch_size,
                log_path=args.log_path,
            )
        for checkpoint_path in lateral_checkpoints.values():
            checkpoint_path.unlink(missing_ok=True)

    wall_seconds = perf_counter() - overall_started_at
    git_status_short = current_git_status_short()
    summary_by_variant = summarize_results(
        per_seed_results=per_seed_results,
        sanity_check_only=args.sanity_check_only,
    )
    report = {
        "config": {
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "eval_samples": args.eval_samples,
            "seeds": args.seeds,
            "context_size": CONTEXT_SIZE,
            "compile_model": args.compile_model,
            "sanity_check_only": args.sanity_check_only,
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
            "cuda_device_name": None if device.type != "cuda" else torch.cuda.get_device_name(device),
        },
        "dataset": {
            "train_examples": len(corpus.train_dataset),
            "val_examples": int(corpus.val_inputs.shape[0]),
            "vocab_size": corpus.vocab_size,
        },
        "timing": {
            "overall_wall_seconds": round(wall_seconds, 6),
        },
        "parameter_audit": parameter_audit,
        "variants": {key: asdict(value) for key, value in selected_specs.items()},
        "per_seed_results": per_seed_results,
        "summary_by_variant": summary_by_variant,
        "ablation_results": ablation_results,
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "experiment_finished",
            "report_path": str(args.report_path),
            "summary_by_variant": summary_by_variant,
            "ablation_results": ablation_results,
        },
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        import traceback

        args = parse_args()
        append_log(args.log_path, {"stage": "crash", "traceback": traceback.format_exc()})
        raise
