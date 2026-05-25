from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from jaxtyping import Int
import torch
from torch import Tensor
from torch.nn import functional as F

from core.dataset import CorpusData, load_corpus
from core.fixed_window_char import set_seed
from core.model import ParallelDiagonalModel, count_parameters
from core.run_utils import (
    append_log,
    build_optimizer,
    checkpoint_metrics,
    log_run_restarted,
    prepare_output_paths,
    random_batches,
    redirect_sanity_check_paths,
    register_active_lock,
    release_memory,
    resolve_device,
    summarize_variant_runs,
    training_throughput_tokens_per_second,
    validate_common_training_args,
    verification_payload,
    verify_forward_and_gradients as shared_verify_forward_and_gradients,
)
from core.training import GraphTrainer, capturable_adamw, current_git_sha, current_git_status_short, write_json

CONTEXT_SIZE = 128
TRAINING_STEPS = 20_000
EVAL_INTERVAL = 1_000
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 3e-4
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_VARIANTS = ("warm12_detach", "warm15_detach")
EVAL_SAMPLES = 4096
WARMUP_STEPS = 3
EXPECTED_PARAMETER_COUNT = 3_639_168
ABLATION_MASK_LOGIT = -100.0


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
    switch_step: int


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "wikitext_103" / "artifacts" / "warmup_detach"
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
    parser.add_argument("--sanity-check-only", action="store_true")
    parser.add_argument("--no-lock", action="store_true")
    parser.set_defaults(compile_model=False)
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
    shared_kwargs = {
        "num_blocks": 4,
        "internal_steps": 1,
        "d_model": 256,
        "feedforward_dim": 512,
        "token_injection": "all",
        "readout_mode": "all",
        "topology": "upward",
        "rates": (1, 1, 1, 1),
    }
    return {
        "warm12_detach": VariantSpec(
            key="warm12_detach",
            label="bridge_warm12_detach",
            switch_step=12_000,
            **shared_kwargs,
        ),
        "warm15_detach": VariantSpec(
            key="warm15_detach",
            label="bridge_warm15_detach",
            switch_step=15_000,
            **shared_kwargs,
        ),
    }


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
        detach_lateral=False,
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
            "Parameter validation requires both warm12_detach and warm15_detach variants. "
            f"Got: {sorted(counts_by_variant)}."
        )

    shared_count = counts_by_variant["warm12_detach"]
    if counts_by_variant["warm15_detach"] != shared_count:
        raise ValueError(
            "warm12_detach and warm15_detach must have identical parameter counts. "
            f"Got {counts_by_variant['warm12_detach']} vs {counts_by_variant['warm15_detach']}."
        )
    if shared_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            "warmup_detach expected parameter count mismatch. "
            f"Expected {EXPECTED_PARAMETER_COUNT}, got {shared_count}."
        )

    return {
        "counts_by_variant": counts_by_variant,
        "shared_parameter_count": shared_count,
        "matches_expected_parameter_count": shared_count == EXPECTED_PARAMETER_COUNT,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
    }


def verify_forward_and_gradients(
    *,
    model: ParallelDiagonalModel,
    device: torch.device,
    vocab_size: int,
) -> dict[str, object]:
    summary = shared_verify_forward_and_gradients(
        model=model,
        model_call=model,
        device=device,
        vocab_size=vocab_size,
        context_size=CONTEXT_SIZE,
        error_prefix="Verification failed",
    )
    return verification_payload(summary)


def cumulative_ablation_names(*, num_blocks: int) -> tuple[str, ...]:
    if num_blocks != 4:
        raise ValueError(f"Warmup-detach readout ablation expects num_blocks=4, got {num_blocks}.")
    return ("block0_only", "blocks_0_1", "blocks_0_1_2", "full")


def apply_per_block_ablation(*, trained_logits: Tensor, block_index: int) -> Tensor:
    replacement = trained_logits.clone()
    replacement[block_index] = ABLATION_MASK_LOGIT
    return replacement


def apply_cumulative_ablation(*, trained_logits: Tensor, ablation_name: str, num_blocks: int) -> Tensor:
    replacement = torch.full_like(trained_logits, ABLATION_MASK_LOGIT)
    match ablation_name:
        case "block0_only":
            keep_count = 1
        case "blocks_0_1":
            keep_count = 2
        case "blocks_0_1_2":
            keep_count = 3
        case "full":
            keep_count = num_blocks
        case _:
            raise ValueError(f"Unknown cumulative ablation {ablation_name!r}.")
    replacement[:keep_count] = trained_logits[:keep_count]
    return replacement


def run_readout_ablation(
    *,
    seed: int,
    variant_key: str,
    model: ParallelDiagonalModel,
    spec: VariantSpec,
    val_inputs: Int[Tensor, "examples context"],
    val_targets: Int[Tensor, "examples"],
    eval_batch_size: int,
    log_path: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    if model.readout_logits is None:
        raise RuntimeError(f"{variant_key} readout ablation requires readout_mode='all' with readout_logits present.")

    trained_logits = model.readout_logits.detach().clone()
    results: dict[str, dict[str, dict[str, float]]] = {
        "per_block": {},
        "cumulative": {},
    }

    try:
        for block_index in range(spec.num_blocks):
            ablation_name = f"zero_block_{block_index}"
            with torch.no_grad():
                model.readout_logits.copy_(
                    apply_per_block_ablation(trained_logits=trained_logits, block_index=block_index)
                )
            metrics = checkpoint_metrics(
                model=model,
                model_call=model,
                val_inputs=val_inputs,
                val_targets=val_targets,
                batch_size=eval_batch_size,
                step=0,
                include_tokens_per_second=False,
            )
            result = {
                "val_loss": float(metrics["val_loss"]),
                "val_accuracy": float(metrics["val_accuracy"]),
            }
            results["per_block"][ablation_name] = result
            append_log(
                log_path,
                {
                    "stage": "readout_ablation",
                    "seed": seed,
                    "variant": variant_key,
                    "ablation_type": "per_block",
                    "ablation_name": ablation_name,
                    **result,
                },
            )

        for ablation_name in cumulative_ablation_names(num_blocks=spec.num_blocks):
            with torch.no_grad():
                model.readout_logits.copy_(
                    apply_cumulative_ablation(
                        trained_logits=trained_logits,
                        ablation_name=ablation_name,
                        num_blocks=spec.num_blocks,
                    )
                )
            metrics = checkpoint_metrics(
                model=model,
                model_call=model,
                val_inputs=val_inputs,
                val_targets=val_targets,
                batch_size=eval_batch_size,
                step=0,
                include_tokens_per_second=False,
            )
            result = {
                "val_loss": float(metrics["val_loss"]),
                "val_accuracy": float(metrics["val_accuracy"]),
            }
            results["cumulative"][ablation_name] = result
            append_log(
                log_path,
                {
                    "stage": "readout_ablation",
                    "seed": seed,
                    "variant": variant_key,
                    "ablation_type": "cumulative",
                    "ablation_name": ablation_name,
                    **result,
                },
            )
    finally:
        with torch.no_grad():
            model.readout_logits.copy_(trained_logits)

    return results


def validate_switch_schedule(*, selected_specs: dict[str, VariantSpec], training_steps: int) -> None:
    for spec in selected_specs.values():
        if spec.switch_step <= 0:
            raise ValueError(f"{spec.key} switch_step must be positive, got {spec.switch_step}.")
        if spec.switch_step >= training_steps:
            raise ValueError(
                f"{spec.key} switch_step ({spec.switch_step}) must be smaller than training_steps ({training_steps})."
            )


def load_bridge_baselines(*, baseline_report_path: Path, allow_missing: bool) -> dict[str, object]:
    if not baseline_report_path.exists():
        if allow_missing:
            return {
                "source_report_path": str(baseline_report_path),
                "available": False,
                "reason": "bridge_detach report not found",
            }
        raise FileNotFoundError(
            "warmup_detach expects bridge_detach baselines from an existing report. "
            f"Missing report: {baseline_report_path}"
        )

    payload = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    summary_by_variant = payload.get("summary_by_variant")
    if not isinstance(summary_by_variant, dict):
        raise ValueError(
            f"Baseline report {baseline_report_path} is malformed: missing summary_by_variant object."
        )
    required_variants = ("full_backprop", "detached")
    missing_variants = [variant for variant in required_variants if variant not in summary_by_variant]
    if len(missing_variants) > 0:
        raise ValueError(
            f"Baseline report {baseline_report_path} is missing variants: {missing_variants}."
        )
    return {
        "source_report_path": str(baseline_report_path),
        "available": True,
        "summary_by_variant": {variant: summary_by_variant[variant] for variant in required_variants},
    }


def build_training_optimizer(*, model: ParallelDiagonalModel, device: torch.device, learning_rate: float) -> torch.optim.AdamW:
    if device.type == "cuda":
        return capturable_adamw(model, lr=learning_rate)
    return build_optimizer(
        model,
        device=device,
        compile_model=False,
        learning_rate=learning_rate,
    )


def check_finite_loss(*, loss: Tensor, step: int, variant_key: str) -> None:
    if not torch.isfinite(loss):
        raise RuntimeError(f"Training diverged for variant {variant_key} at step {step}: loss is NaN or Inf.")


def capture_graph_segment(
    *,
    trainer: GraphTrainer,
    batch_iterator: Iterator[tuple[Int[Tensor, "batch context"], Int[Tensor, "batch"]]],
    variant_key: str,
    start_step: int,
) -> tuple[Tensor, int]:
    warmup_batches = [next(batch_iterator) for _ in range(WARMUP_STEPS)]
    loss = trainer.capture(warmup_batches)
    completed_step = start_step + WARMUP_STEPS
    check_finite_loss(loss=loss, step=completed_step, variant_key=variant_key)
    return loss, completed_step


def maybe_log_checkpoint(
    *,
    checkpoints: list[dict[str, object]],
    current_step: int,
    training_steps: int,
    eval_interval: int,
    measure_tokens_per_second: bool,
    batch_size: int,
    context_size: int,
    last_eval_started_at: float,
    last_eval_step: int,
    model: ParallelDiagonalModel,
    val_inputs: Int[Tensor, "examples context"],
    val_targets: Int[Tensor, "examples"],
    eval_batch_size: int,
    log_path: Path,
    seed: int,
    variant_key: str,
) -> tuple[float, int]:
    if current_step % eval_interval != 0 and current_step != training_steps:
        return last_eval_started_at, last_eval_step

    tokens_per_second: float | None = None
    if measure_tokens_per_second:
        now = perf_counter()
        tokens_per_second = training_throughput_tokens_per_second(
            current_step=current_step,
            previous_step=last_eval_step,
            batch_size=batch_size,
            context_size=context_size,
            elapsed_seconds=now - last_eval_started_at,
        )
    else:
        now = perf_counter()

    checkpoint = checkpoint_metrics(
        model=model,
        model_call=model,
        val_inputs=val_inputs,
        val_targets=val_targets,
        batch_size=eval_batch_size,
        step=current_step,
        tokens_per_second=tokens_per_second,
    )
    checkpoints.append(checkpoint)
    append_log(
        log_path,
        {
            "stage": "checkpoint",
            "seed": seed,
            "variant": variant_key,
            **checkpoint,
        },
    )
    return now, current_step


@dataclass(frozen=True)
class WarmupDetachTrainingArtifacts:
    checkpoints: list[dict[str, object]]
    final_training_loss: float
    wall_seconds: float
    switch_event: dict[str, int | str | bool]


def run_warmup_detach_training_loop(
    *,
    model: ParallelDiagonalModel,
    optimizer: torch.optim.AdamW,
    encoded_corpus: Int[Tensor, "tokens"],
    seed: int,
    device: torch.device,
    context_size: int,
    batch_size: int,
    training_steps: int,
    eval_interval: int,
    switch_step: int,
    initial_checkpoint: dict[str, object],
    val_inputs: Int[Tensor, "examples context"],
    val_targets: Int[Tensor, "examples"],
    eval_batch_size: int,
    log_path: Path,
    variant_key: str,
    measure_tokens_per_second: bool,
) -> WarmupDetachTrainingArtifacts:
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
    started_at = perf_counter()
    last_eval_started_at = started_at
    last_eval_step = 0
    current_step = 0
    last_loss: Tensor | None = None
    switched = False
    switch_event: dict[str, int | str | bool] = {
        "switch_step": switch_step,
        "switched": False,
        "mode_before": "full_backprop",
        "mode_after": "detached",
    }
    trainer: GraphTrainer | None = None

    if device.type == "cuda":
        trainer = GraphTrainer(
            model,
            optimizer,
            batch_size=batch_size,
            seq_len=context_size,
            device=device,
        )
        last_loss, current_step = capture_graph_segment(
            trainer=trainer,
            batch_iterator=batch_iterator,
            variant_key=variant_key,
            start_step=0,
        )

    while current_step < training_steps:
        if not switched and current_step == switch_step:
            if trainer is not None:
                trainer.synchronize()
                del trainer
                trainer = None

            model.detach_lateral = True
            switch_event = {
                "switch_step": switch_step,
                "switched": True,
                "mode_before": "full_backprop",
                "mode_after": "detached",
                "switch_triggered_at_step": current_step,
                "recapture_warmup_steps": WARMUP_STEPS if device.type == "cuda" else 0,
            }

            append_log(
                log_path,
                {
                    "stage": "detach_switch",
                    "seed": seed,
                    "variant": variant_key,
                    **switch_event,
                },
            )

            if device.type == "cuda":
                trainer = GraphTrainer(
                    model,
                    optimizer,
                    batch_size=batch_size,
                    seq_len=context_size,
                    device=device,
                )
                last_loss, current_step = capture_graph_segment(
                    trainer=trainer,
                    batch_iterator=batch_iterator,
                    variant_key=variant_key,
                    start_step=current_step,
                )
                switch_event["recapture_completed_at_step"] = current_step
                append_log(
                    log_path,
                    {
                        "stage": "detach_recapture_finished",
                        "seed": seed,
                        "variant": variant_key,
                        "recapture_completed_at_step": current_step,
                    },
                )
                last_eval_started_at, last_eval_step = maybe_log_checkpoint(
                    checkpoints=checkpoints,
                    current_step=current_step,
                    training_steps=training_steps,
                    eval_interval=eval_interval,
                    measure_tokens_per_second=measure_tokens_per_second,
                    batch_size=batch_size,
                    context_size=context_size,
                    last_eval_started_at=last_eval_started_at,
                    last_eval_step=last_eval_step,
                    model=model,
                    val_inputs=val_inputs,
                    val_targets=val_targets,
                    eval_batch_size=eval_batch_size,
                    log_path=log_path,
                    seed=seed,
                    variant_key=variant_key,
                )
                switched = True
                continue

            switched = True

        if current_step >= training_steps:
            break

        batch_input, batch_target = next(batch_iterator)
        current_step += 1
        if trainer is not None:
            last_loss = trainer.step(batch_input, batch_target)
        else:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_input)
            loss = F.cross_entropy(logits, batch_target)
            check_finite_loss(loss=loss, step=current_step, variant_key=variant_key)
            loss.backward()
            optimizer.step()
            last_loss = loss.detach().clone()

        check_finite_loss(loss=last_loss, step=current_step, variant_key=variant_key)

        if trainer is not None and (current_step % eval_interval == 0 or current_step == training_steps):
            trainer.synchronize()

        last_eval_started_at, last_eval_step = maybe_log_checkpoint(
            checkpoints=checkpoints,
            current_step=current_step,
            training_steps=training_steps,
            eval_interval=eval_interval,
            measure_tokens_per_second=measure_tokens_per_second,
            batch_size=batch_size,
            context_size=context_size,
            last_eval_started_at=last_eval_started_at,
            last_eval_step=last_eval_step,
            model=model,
            val_inputs=val_inputs,
            val_targets=val_targets,
            eval_batch_size=eval_batch_size,
            log_path=log_path,
            seed=seed,
            variant_key=variant_key,
        )

    if trainer is not None:
        trainer.synchronize()
    if last_loss is None:
        raise RuntimeError(f"Variant {variant_key} completed without recording a training loss.")
    if not switched:
        raise RuntimeError(
            f"Variant {variant_key} finished without performing the detach switch at step {switch_step}."
        )

    return WarmupDetachTrainingArtifacts(
        checkpoints=checkpoints,
        final_training_loss=round(last_loss.item(), 6),
        wall_seconds=round(perf_counter() - started_at, 6),
        switch_event=switch_event,
    )


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
) -> tuple[dict[str, object], dict[str, object]]:
    set_seed(seed)
    encoded_corpus = getattr(corpus.train_dataset, "encoded_corpus", None)
    if not isinstance(encoded_corpus, torch.Tensor):
        raise TypeError(
            "train_single_variant expects corpus.train_dataset to expose encoded_corpus as a torch.Tensor."
        )

    model = build_model(device=device, vocab_size=corpus.vocab_size, spec=spec)
    parameter_count = count_parameters(model)
    verification = verify_forward_and_gradients(
        model=model,
        device=device,
        vocab_size=corpus.vocab_size,
    )
    initial_checkpoint = checkpoint_metrics(
        model=model,
        model_call=model,
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
            "verification": verification,
            "initial_checkpoint": initial_checkpoint,
            "switch_step": spec.switch_step,
            "detach_lateral_before_switch": False,
            "detach_lateral_after_switch": True,
        },
    )

    expected_parameter_count = int(parameter_audit["counts_by_variant"][variant_key])
    if parameter_count != expected_parameter_count:
        raise RuntimeError(
            f"Parameter count mismatch for {variant_key}: built model has {parameter_count}, audit expected {expected_parameter_count}."
        )

    if args.sanity_check_only:
        readout_ablation = run_readout_ablation(
            seed=seed,
            variant_key=variant_key,
            model=model,
            spec=spec,
            val_inputs=val_inputs,
            val_targets=val_targets,
            eval_batch_size=args.eval_batch_size,
            log_path=args.log_path,
        )
        result = {
            "seed": seed,
            "variant": variant_key,
            "label": spec.label,
            "class_name": "ParallelDiagonalModel",
            "config": asdict(spec),
            "parameter_count": parameter_count,
            "parameter_audit": parameter_audit,
            "verification": verification,
            "checkpoints": [initial_checkpoint],
            "best_checkpoint": initial_checkpoint,
            "final_checkpoint": initial_checkpoint,
            "final_training_loss": None,
            "wall_seconds": 0.0,
            "switch_event": {
                "switch_step": spec.switch_step,
                "switched": False,
                "reason": "sanity_check_only skipped training",
            },
            "readout_ablation": readout_ablation,
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
        del model
        release_memory(device=device)
        return result, {
            "readout_ablation": readout_ablation,
        }

    optimizer = build_training_optimizer(model=model, device=device, learning_rate=args.learning_rate)
    training = run_warmup_detach_training_loop(
        model=model,
        optimizer=optimizer,
        encoded_corpus=encoded_corpus,
        seed=seed,
        device=device,
        context_size=CONTEXT_SIZE,
        batch_size=args.batch_size,
        training_steps=args.training_steps,
        eval_interval=args.eval_interval,
        switch_step=spec.switch_step,
        initial_checkpoint=initial_checkpoint,
        val_inputs=val_inputs,
        val_targets=val_targets,
        eval_batch_size=args.eval_batch_size,
        log_path=args.log_path,
        variant_key=variant_key,
        measure_tokens_per_second=True,
    )

    readout_ablation = run_readout_ablation(
        seed=seed,
        variant_key=variant_key,
        model=model,
        spec=spec,
        val_inputs=val_inputs,
        val_targets=val_targets,
        eval_batch_size=args.eval_batch_size,
        log_path=args.log_path,
    )

    result = {
        "seed": seed,
        "variant": variant_key,
        "label": spec.label,
        "class_name": "ParallelDiagonalModel",
        "config": asdict(spec),
        "parameter_count": parameter_count,
        "parameter_audit": parameter_audit,
        "verification": verification,
        "checkpoints": training.checkpoints,
        "best_checkpoint": min(training.checkpoints, key=lambda checkpoint: checkpoint["val_loss"]),
        "final_checkpoint": training.checkpoints[-1],
        "final_training_loss": training.final_training_loss,
        "wall_seconds": training.wall_seconds,
        "switch_event": training.switch_event,
        "readout_ablation": readout_ablation,
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
        },
    )

    del optimizer
    del model
    release_memory(device=device)
    return result, {
        "readout_ablation": readout_ablation,
    }


def summarize_results(*, per_seed_results: list[dict[str, object]], sanity_check_only: bool) -> dict[str, object]:
    return summarize_variant_runs(
        per_seed_results=per_seed_results,
        sanity_check_only=sanity_check_only,
        carry_forward_keys=("parameter_count", "parameter_audit", "verification"),
        include_tokens_per_second=True,
    )


def main() -> int:
    args = parse_args()
    validate_common_training_args(args, warmup_steps=WARMUP_STEPS)
    if args.sanity_check_only:
        redirect_sanity_check_paths(args)

    specs = variant_specs()
    unknown_variants = [variant for variant in args.variants if variant not in specs]
    if len(unknown_variants) > 0:
        raise ValueError(f"Unknown variants requested: {unknown_variants}. Available variants: {list(specs)}.")
    if set(args.variants) != set(DEFAULT_VARIANTS):
        raise ValueError(
            "This experiment requires both warm12_detach and warm15_detach variants so the comparison stays matched."
        )

    selected_specs = {key: specs[key] for key in args.variants}
    validate_switch_schedule(selected_specs=selected_specs, training_steps=args.training_steps)
    device = resolve_device(args.device)

    register_active_lock(
        experiment_name="warmup_detach",
        variants=args.variants,
        enabled=not args.no_lock and not args.sanity_check_only,
    )

    prepare_output_paths(report_path=args.report_path, log_path=args.log_path)
    log_run_restarted(args.log_path)

    repo_root = Path(__file__).resolve().parents[1]
    baseline_report_path = repo_root / "experiments" / "wikitext_103" / "artifacts" / "bridge_detach" / "report.json"
    baseline_summary = load_bridge_baselines(
        baseline_report_path=baseline_report_path,
        allow_missing=args.sanity_check_only,
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
            "device": str(device),
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
            "vocab_size": corpus.vocab_size,
            "train_dataset_size": len(corpus.train_dataset),
            "val_examples": int(corpus.val_inputs.shape[0]),
            "parameter_audit": parameter_audit,
            "bridge_baselines": baseline_summary,
        },
    )

    overall_started_at = perf_counter()
    per_seed_results: list[dict[str, object]] = []
    ablation_results: dict[str, dict[str, dict[str, object]]] = {}
    for seed in args.seeds:
        append_log(args.log_path, {"stage": "seed_started", "seed": seed})
        ablation_results[str(seed)] = {}
        for variant_key in args.variants:
            result, variant_ablation_results = train_single_variant(
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
            ablation_results[str(seed)][variant_key] = variant_ablation_results
        append_log(args.log_path, {"stage": "seed_finished", "seed": seed})

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
        "bridge_baselines": baseline_summary,
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
        if args.sanity_check_only:
            redirect_sanity_check_paths(args)
        append_log(args.log_path, {"stage": "crash", "traceback": traceback.format_exc()})
        raise
