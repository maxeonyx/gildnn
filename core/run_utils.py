from __future__ import annotations

import atexit
import gc
import json
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

import torch
from jaxtyping import Int
from torch import Tensor, nn
from torch.nn import functional as F

from core.training import GraphTrainer, capturable_adamw


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


def maybe_compile_model(model: nn.Module, *, enabled: bool) -> nn.Module:
    if not enabled:
        return model
    return torch.compile(model, backend="aot_eager")


def build_optimizer(
    model: nn.Module,
    *,
    device: torch.device,
    compile_model: bool,
    learning_rate: float,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    if device.type == "cuda" and not compile_model:
        return capturable_adamw(model, lr=learning_rate, weight_decay=weight_decay)
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
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
    model: nn.Module,
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
    model: nn.Module,
    model_call: nn.Module,
    val_inputs: Int[Tensor, "examples context"],
    val_targets: Int[Tensor, "examples"],
    batch_size: int,
    step: int,
    tokens_per_second: float | None = None,
    include_tokens_per_second: bool = True,
) -> dict[str, float | int | None]:
    metrics = evaluate_model_call(
        model=model,
        model_call=model_call,
        inputs=val_inputs,
        targets=val_targets,
        batch_size=batch_size,
    )
    checkpoint: dict[str, float | int | None] = {
        "step": step,
        "val_loss": round(metrics["loss"], 6),
        "val_accuracy": round(metrics["accuracy"], 6),
    }
    if include_tokens_per_second:
        checkpoint["tokens_per_second"] = None if tokens_per_second is None else round(tokens_per_second, 6)
    return checkpoint


@dataclass(frozen=True)
class GradientVerificationSummary:
    output_shape: list[int]
    dummy_loss: float
    total_trainable_parameters: int
    nonzero_gradient_parameters: int
    gradient_norm_sum: float
    missing_gradient_parameters: list[str]
    zero_gradient_parameters: list[str]
    expected_missing_gradient_parameters: list[str]
    unexpected_missing_gradient_parameters: list[str]


def verify_forward_and_gradients(
    *,
    model: nn.Module,
    model_call: nn.Module,
    device: torch.device,
    vocab_size: int,
    context_size: int,
    error_prefix: str,
    missing_gradient_is_expected: Callable[[str], bool] | None = None,
) -> GradientVerificationSummary:
    saved_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    model.train()
    if model_call is not model:
        model_call.train()

    dummy_inputs = torch.randint(0, vocab_size, (2, context_size), device=device)
    dummy_targets = torch.randint(0, vocab_size, (2,), device=device)
    logits = model_call(dummy_inputs)
    if tuple(logits.shape) != (2, vocab_size):
        raise RuntimeError(
            f"{error_prefix}: expected output shape (2, {vocab_size}), got {tuple(logits.shape)}."
        )

    loss = F.cross_entropy(logits, dummy_targets)
    if not torch.isfinite(loss):
        raise RuntimeError(f"{error_prefix}: dummy loss is not finite.")

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
            raise RuntimeError(f"{error_prefix}: parameter {name} has NaN or Inf gradients.")
        grad_norm = parameter.grad.detach().norm().item()
        gradient_norm_sum += grad_norm
        if grad_norm > 0.0:
            nonzero_gradient_parameters += 1
        else:
            zero_gradient_parameters.append(name)

    expected_missing_gradient_parameters = sorted(
        name
        for name in missing_gradient_parameters
        if missing_gradient_is_expected is not None and missing_gradient_is_expected(name)
    )
    unexpected_missing_gradient_parameters = sorted(
        name for name in missing_gradient_parameters if name not in expected_missing_gradient_parameters
    )
    if len(unexpected_missing_gradient_parameters) > 0 or len(zero_gradient_parameters) > 0:
        raise RuntimeError(
            f"{error_prefix}: some parameters did not receive gradients. "
            f"missing={unexpected_missing_gradient_parameters}, zero={zero_gradient_parameters}"
        )

    optimizer.step()
    model.zero_grad(set_to_none=True)
    if model_call is not model:
        model_call.zero_grad(set_to_none=True)
    model.load_state_dict(saved_state)

    return GradientVerificationSummary(
        output_shape=list(logits.shape),
        dummy_loss=round(loss.item(), 6),
        total_trainable_parameters=total_trainable_parameters,
        nonzero_gradient_parameters=nonzero_gradient_parameters,
        gradient_norm_sum=round(gradient_norm_sum, 6),
        missing_gradient_parameters=missing_gradient_parameters,
        zero_gradient_parameters=zero_gradient_parameters,
        expected_missing_gradient_parameters=expected_missing_gradient_parameters,
        unexpected_missing_gradient_parameters=unexpected_missing_gradient_parameters,
    )


def verification_payload(
    summary: GradientVerificationSummary,
    *,
    include_expected_missing: bool = False,
    include_unexpected_missing: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "output_shape": summary.output_shape,
        "dummy_loss": summary.dummy_loss,
        "total_trainable_parameters": summary.total_trainable_parameters,
        "nonzero_gradient_parameters": summary.nonzero_gradient_parameters,
        "gradient_norm_sum": summary.gradient_norm_sum,
        "missing_gradient_parameters": summary.missing_gradient_parameters,
        "zero_gradient_parameters": summary.zero_gradient_parameters,
        "forward_pass_ok": True,
        "backward_pass_ok": True,
    }
    if include_expected_missing:
        payload["expected_missing_gradient_parameters"] = summary.expected_missing_gradient_parameters
    if include_unexpected_missing:
        payload["unexpected_missing_gradient_parameters"] = summary.unexpected_missing_gradient_parameters
    return payload


def training_throughput_tokens_per_second(
    *,
    current_step: int,
    previous_step: int,
    batch_size: int,
    context_size: int,
    elapsed_seconds: float,
) -> float:
    if elapsed_seconds <= 0.0:
        raise ValueError(f"elapsed_seconds must be positive, got {elapsed_seconds}.")
    steps_completed = current_step - previous_step
    if steps_completed <= 0:
        raise ValueError(
            f"current_step must be greater than previous_step, got current_step={current_step}, previous_step={previous_step}."
        )
    tokens_processed = steps_completed * batch_size * context_size
    return tokens_processed / elapsed_seconds


@dataclass(frozen=True)
class TrainingLoopArtifacts:
    checkpoints: list[dict[str, object]]
    final_training_loss: float
    wall_seconds: float


def run_training_loop(
    *,
    model: nn.Module,
    model_call: nn.Module,
    optimizer: torch.optim.Optimizer,
    encoded_corpus: Int[Tensor, "tokens"],
    seed: int,
    device: torch.device,
    context_size: int,
    batch_size: int,
    training_steps: int,
    eval_interval: int,
    compile_model: bool,
    warmup_steps: int,
    initial_checkpoint: dict[str, object],
    checkpoint_builder: Callable[[int, float | None], dict[str, object]],
    checkpoint_logger: Callable[[dict[str, object]], None],
    divergence_error_message: str,
    missing_loss_error_message: str,
    measure_tokens_per_second: bool,
) -> TrainingLoopArtifacts:
    batch_rng = torch.Generator(device="cpu")
    batch_rng.manual_seed(seed)
    batch_iterator = random_batches(
        encoded_corpus,
        context_size=context_size,
        batch_size=batch_size,
        device=device,
        rng=batch_rng,
    )

    checkpoints = [initial_checkpoint]
    trainer: GraphTrainer | None = None
    last_loss: Tensor | None = None
    started_at = perf_counter()
    last_eval_started_at = started_at
    last_eval_step = 0

    if device.type == "cuda" and not compile_model:
        warmup_batches = [next(batch_iterator) for _ in range(warmup_steps)]
        trainer = GraphTrainer(
            model,
            optimizer,
            batch_size=batch_size,
            seq_len=context_size,
            device=device,
        )
        trainer.capture(warmup_batches)
        last_loss = trainer.static_loss.detach().clone()
        training_step_start = warmup_steps + 1
    else:
        training_step_start = 1

    for step in range(training_step_start, training_steps + 1):
        batch_input, batch_target = next(batch_iterator)
        if trainer is not None:
            last_loss = trainer.step(batch_input, batch_target)
        else:
            optimizer.zero_grad(set_to_none=True)
            logits = model_call(batch_input)
            loss = F.cross_entropy(logits, batch_target)
            if not torch.isfinite(loss):
                raise RuntimeError(divergence_error_message.format(step=step))
            loss.backward()
            optimizer.step()
            last_loss = loss.detach().clone()

        if step % eval_interval != 0 and step != training_steps:
            continue

        if trainer is not None:
            trainer.synchronize()

        tokens_per_second: float | None = None
        if measure_tokens_per_second:
            now = perf_counter()
            tokens_per_second = training_throughput_tokens_per_second(
                current_step=step,
                previous_step=last_eval_step,
                batch_size=batch_size,
                context_size=context_size,
                elapsed_seconds=now - last_eval_started_at,
            )

        checkpoint = checkpoint_builder(step, tokens_per_second)
        checkpoints.append(checkpoint)
        checkpoint_logger(checkpoint)

        if measure_tokens_per_second:
            last_eval_started_at = perf_counter()
            last_eval_step = step

    if trainer is not None:
        trainer.synchronize()
    if last_loss is None:
        raise RuntimeError(missing_loss_error_message)

    return TrainingLoopArtifacts(
        checkpoints=checkpoints,
        final_training_loss=round(last_loss.item(), 6),
        wall_seconds=round(perf_counter() - started_at, 6),
    )


def mean_rounded(values: Sequence[float]) -> float:
    return round(mean(values), 6)


def std_rounded(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(pstdev(values), 6)


def summarize_variant_runs(
    *,
    per_seed_results: list[dict[str, object]],
    sanity_check_only: bool,
    carry_forward_keys: Sequence[str],
    include_tokens_per_second: bool,
) -> dict[str, object]:
    results_by_variant: dict[str, list[dict[str, object]]] = {}
    for result in per_seed_results:
        results_by_variant.setdefault(str(result["variant"]), []).append(result)

    summary: dict[str, object] = {}
    for variant, runs in results_by_variant.items():
        if sanity_check_only:
            variant_summary: dict[str, object] = {"num_runs": len(runs)}
            for key in carry_forward_keys:
                variant_summary[key] = runs[0][key]
            summary[variant] = variant_summary
            continue

        final_losses = [float(run["final_checkpoint"]["val_loss"]) for run in runs]
        final_accuracies = [float(run["final_checkpoint"]["val_accuracy"]) for run in runs]
        wall_seconds = [float(run["wall_seconds"]) for run in runs]
        variant_summary = {
            "num_runs": len(runs),
            "mean_final_val_loss": mean_rounded(final_losses),
            "std_final_val_loss": std_rounded(final_losses),
            "mean_final_val_accuracy": mean_rounded(final_accuracies),
            "std_final_val_accuracy": std_rounded(final_accuracies),
            "mean_wall_seconds": mean_rounded(wall_seconds),
            "runs": runs,
        }
        if include_tokens_per_second:
            final_tokens_per_second = [float(run["final_checkpoint"]["tokens_per_second"]) for run in runs]
            variant_summary["mean_final_tokens_per_second"] = mean_rounded(final_tokens_per_second)
            variant_summary["std_final_tokens_per_second"] = std_rounded(final_tokens_per_second)
        summary[variant] = variant_summary
    return summary


def validate_common_training_args(args: object, *, warmup_steps: int) -> None:
    seeds = getattr(args, "seeds")
    if len(seeds) == 0:
        raise ValueError("At least one seed is required.")
    training_steps = getattr(args, "training_steps")
    if training_steps <= 0:
        raise ValueError(f"training_steps must be positive, got {training_steps}.")
    eval_interval = getattr(args, "eval_interval")
    if eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {eval_interval}.")
    batch_size = getattr(args, "batch_size")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    eval_batch_size = getattr(args, "eval_batch_size")
    if eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {eval_batch_size}.")
    eval_samples = getattr(args, "eval_samples")
    if eval_samples <= 0:
        raise ValueError(f"eval_samples must be positive, got {eval_samples}.")
    if not getattr(args, "sanity_check_only") and training_steps < warmup_steps and not getattr(args, "compile_model"):
        raise ValueError(
            f"training_steps must be at least {warmup_steps} for CUDA graph warmup when compile is disabled."
        )


def register_active_lock(
    *,
    experiment_name: str,
    variants: object,
    enabled: bool = True,
    started_at: str | None = None,
    lock_path: Path = Path("runs/active.lock"),
) -> None:
    if not enabled:
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_content = (
        f"PID: {os.getpid()}\n"
        f"Experiment: {experiment_name}\n"
        f"Variants: {variants}\n"
        f"Started: {started_at or datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    )
    lock_path.write_text(lock_content, encoding="utf-8")

    def _remove_lock() -> None:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    atexit.register(_remove_lock)


def prepare_output_paths(*, report_path: Path, log_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)


def log_run_restarted(log_path: Path) -> None:
    if not log_path.exists():
        return
    previous_log = log_path.read_text(encoding="utf-8")
    if previous_log:
        append_log(
            log_path,
            {
                "stage": "run_restarted",
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "previous_lines": len(previous_log.splitlines()),
            },
        )


def release_memory(*, device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
