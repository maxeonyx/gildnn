from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

# Ensure repo root is importable regardless of how this script is launched
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jaxtyping import Int
import torch
from torch import Tensor, nn

from core.dataset import CorpusData, load_corpus
from core.fixed_window_char import set_seed
from core.model import ParallelDiagonalModel, count_parameters
from core.run_utils import (
    append_log,
    build_optimizer,
    checkpoint_metrics,
    log_run_restarted,
    maybe_compile_model,
    prepare_output_paths,
    redirect_sanity_check_paths,
    register_active_lock,
    release_memory,
    resolve_device,
    run_training_loop,
    summarize_variant_runs,
    validate_common_training_args,
    verification_payload,
    verify_forward_and_gradients as shared_verify_forward_and_gradients,
)
from core.training import current_git_sha, current_git_status_short, write_json

CONTEXT_SIZE = 128
TRAINING_STEPS = 20_000
EVAL_INTERVAL = 1_000
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_VARIANTS = ("A_all", "C8_all", "W8_all")
EVAL_SAMPLES = 4096
WARMUP_STEPS = 3
REFERENCE_PARAMETER_COUNT = 3_639_168
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
    temporal_window: int
    temporal_window_mode: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "temporal-window" / "artifacts"
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
    parser.add_argument("--no-lock", action="store_true")
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
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "4block_report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "4block_run.jsonl")
    return parser.parse_args()


def variant_specs() -> dict[str, VariantSpec]:
    shared_kwargs = {
        "num_blocks": 4,
        "internal_steps": 1,
        "d_model": 256,
        "feedforward_dim": 512,
        "token_injection": "block0",
        "readout_mode": "all",
        "topology": "upward",
        "rates": (1, 1, 1, 1),
    }
    return {
        "A_all": VariantSpec(
            key="A_all",
            label="temporal_window_4block_a_all",
            temporal_window=0,
            temporal_window_mode="history",
            **shared_kwargs,
        ),
        "C8_all": VariantSpec(
            key="C8_all",
            label="temporal_window_4block_c8_all",
            temporal_window=8,
            temporal_window_mode="current",
            **shared_kwargs,
        ),
        "W8_all": VariantSpec(
            key="W8_all",
            label="temporal_window_4block_w8_all",
            temporal_window=8,
            temporal_window_mode="history",
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
        temporal_window=spec.temporal_window,
        temporal_window_mode=spec.temporal_window_mode,
    ).to(device)


def parameter_reference_summary(parameter_count: int) -> dict[str, float | int]:
    difference = parameter_count - REFERENCE_PARAMETER_COUNT
    difference_ratio = abs(difference) / REFERENCE_PARAMETER_COUNT
    return {
        "reference_parameter_count": REFERENCE_PARAMETER_COUNT,
        "parameter_count_difference": difference,
        "parameter_count_difference_ratio": round(difference_ratio, 6),
    }


def validate_parameter_counts(*, vocab_size: int, selected_specs: dict[str, VariantSpec]) -> dict[str, object]:
    cpu_device = torch.device("cpu")
    counts_by_variant: dict[str, int] = {}
    for key, spec in selected_specs.items():
        model = build_model(device=cpu_device, vocab_size=vocab_size, spec=spec)
        counts_by_variant[key] = count_parameters(model)
        del model

    required_variants = set(DEFAULT_VARIANTS)
    if set(counts_by_variant) != required_variants:
        raise ValueError(
            f"Parameter validation requires {sorted(required_variants)}, got {sorted(counts_by_variant)}."
        )

    a_all_reference_summary = parameter_reference_summary(counts_by_variant["A_all"])
    if counts_by_variant["C8_all"] != counts_by_variant["W8_all"]:
        raise ValueError(
            "C8_all and W8_all must have identical parameter counts. "
            f"Got {counts_by_variant['C8_all']} vs {counts_by_variant['W8_all']}."
        )
    if counts_by_variant["W8_all"] <= counts_by_variant["A_all"]:
        raise ValueError(
            "W8_all must have more parameters than A_all because window_proj is active. "
            f"Got W8_all={counts_by_variant['W8_all']} and A_all={counts_by_variant['A_all']}."
        )

    return {
        "counts_by_variant": counts_by_variant,
        "a_all_reference_summary": a_all_reference_summary,
        "window_parameter_delta": counts_by_variant["W8_all"] - counts_by_variant["A_all"],
        "c8_all_equals_w8_all": True,
    }


def verify_forward_and_gradients(
    *,
    model: ParallelDiagonalModel,
    model_call: nn.Module,
    device: torch.device,
    vocab_size: int,
) -> dict[str, object]:
    summary = shared_verify_forward_and_gradients(
        model=model,
        model_call=model_call,
        device=device,
        vocab_size=vocab_size,
        context_size=CONTEXT_SIZE,
        error_prefix="Verification failed",
        missing_gradient_is_expected=lambda name: model.token_injection == "block0"
        and any(name == f"token_mixes.{block_index}.alpha_logit" for block_index in range(1, model.num_blocks)),
    )
    return verification_payload(
        summary,
        include_expected_missing=True,
        include_unexpected_missing=True,
    )


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
        release_memory(device=device)
        return result, temporary_checkpoint_path

    optimizer = build_optimizer(
        model,
        device=device,
        compile_model=args.compile_model,
        learning_rate=args.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    training = run_training_loop(
        model=model,
        model_call=model_call,
        optimizer=optimizer,
        encoded_corpus=encoded_corpus,
        seed=seed,
        device=device,
        context_size=CONTEXT_SIZE,
        batch_size=args.batch_size,
        training_steps=args.training_steps,
        eval_interval=args.eval_interval,
        compile_model=args.compile_model,
        warmup_steps=WARMUP_STEPS,
        initial_checkpoint=initial_checkpoint,
        checkpoint_builder=lambda step, tokens_per_second: checkpoint_metrics(
            model=model,
            model_call=model_call,
            val_inputs=val_inputs,
            val_targets=val_targets,
            batch_size=args.eval_batch_size,
            step=step,
            tokens_per_second=tokens_per_second,
        ),
        checkpoint_logger=lambda checkpoint: append_log(
            args.log_path,
            {
                "stage": "checkpoint",
                "seed": seed,
                "variant": variant_key,
                **checkpoint,
            },
        ),
        divergence_error_message=f"Training diverged for variant {variant_key} at step {{step}}: loss is NaN or Inf.",
        missing_loss_error_message=f"Variant {variant_key} completed without recording a training loss.",
        measure_tokens_per_second=True,
    )

    temporary_checkpoint_path = args.report_path.parent / f"{variant_key}_seed_{seed}.pt"
    save_temporary_state_dict(model=model, checkpoint_path=temporary_checkpoint_path)

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
        "checkpoints": training.checkpoints,
        "best_checkpoint": min(training.checkpoints, key=lambda checkpoint: checkpoint["val_loss"]),
        "final_checkpoint": training.checkpoints[-1],
        "final_training_loss": training.final_training_loss,
        "wall_seconds": training.wall_seconds,
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
            "temporary_checkpoint_path": str(temporary_checkpoint_path),
        },
    )

    del optimizer
    del model_call
    del model
    release_memory(device=device)
    return result, temporary_checkpoint_path


def cumulative_ablation_names(*, num_blocks: int) -> tuple[str, ...]:
    if num_blocks != 4:
        raise ValueError(f"Temporal-window 4-block ablation expects num_blocks=4, got {num_blocks}.")
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


def run_ablation_suite(
    *,
    seed: int,
    variant_key: str,
    checkpoint_path: Path,
    spec: VariantSpec,
    vocab_size: int,
    device: torch.device,
    val_inputs: Int[Tensor, "examples context"],
    val_targets: Int[Tensor, "examples"],
    eval_batch_size: int,
    log_path: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    model = build_model(device=device, vocab_size=vocab_size, spec=spec)
    loaded_state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(loaded_state, dict):
        raise TypeError(f"Expected checkpoint at {checkpoint_path} to be a state_dict dict.")
    model.load_state_dict(loaded_state)
    if model.readout_logits is None:
        raise RuntimeError(f"{variant_key} ablations require readout_mode='all' with readout_logits present.")

    trained_logits = model.readout_logits.detach().clone()
    results: dict[str, dict[str, dict[str, float]]] = {
        "per_block": {},
        "cumulative": {},
    }
    append_log(log_path, {"stage": "ablation_started", "seed": seed, "variant": variant_key})

    for block_index in range(spec.num_blocks):
        ablation_name = f"zero_block_{block_index}"
        with torch.no_grad():
            model.readout_logits.copy_(apply_per_block_ablation(trained_logits=trained_logits, block_index=block_index))
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
            "val_loss": metrics["val_loss"],
            "val_accuracy": metrics["val_accuracy"],
        }
        results["per_block"][ablation_name] = result
        append_log(
            log_path,
            {
                "stage": "ablation_result",
                "variant": variant_key,
                "seed": seed,
                "ablation_type": "per_block",
                "ablation_name": ablation_name,
                "val_loss": result["val_loss"],
                "val_accuracy": result["val_accuracy"],
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
            "val_loss": metrics["val_loss"],
            "val_accuracy": metrics["val_accuracy"],
        }
        results["cumulative"][ablation_name] = result
        append_log(
            log_path,
            {
                "stage": "ablation_result",
                "variant": variant_key,
                "seed": seed,
                "ablation_type": "cumulative",
                "ablation_name": ablation_name,
                "val_loss": result["val_loss"],
                "val_accuracy": result["val_accuracy"],
            },
        )

    append_log(log_path, {"stage": "ablation_finished", "seed": seed, "variant": variant_key})

    del model
    release_memory(device=device)
    return results


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
            "This experiment requires A_all, C8_all, and W8_all so the pre-registered comparison stays matched."
        )

    selected_specs = {key: specs[key] for key in args.variants}
    device = resolve_device(args.device)

    register_active_lock(
        experiment_name="temporal_window_4block",
        variants=args.variants,
        enabled=not args.no_lock and not args.sanity_check_only,
        started_at=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z"),
    )

    prepare_output_paths(report_path=args.report_path, log_path=args.log_path)
    log_run_restarted(args.log_path)

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
            "no_lock": args.no_lock,
            "seeds": args.seeds,
            "variants": args.variants,
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": WEIGHT_DECAY,
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
    checkpoints_by_variant_and_seed: dict[str, dict[int, Path]] = {variant_key: {} for variant_key in args.variants}
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
                checkpoints_by_variant_and_seed[variant_key][seed] = temporary_checkpoint_path
        append_log(args.log_path, {"stage": "seed_finished", "seed": seed})

    ablation_results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    if not args.sanity_check_only:
        for variant_key in args.variants:
            missing_checkpoints = [seed for seed in args.seeds if seed not in checkpoints_by_variant_and_seed[variant_key]]
            if len(missing_checkpoints) > 0:
                raise RuntimeError(f"Missing checkpoints for {variant_key} ablations: {missing_checkpoints}.")
        for variant_key in args.variants:
            ablation_results[variant_key] = {}
            for seed in args.seeds:
                ablation_results[variant_key][str(seed)] = run_ablation_suite(
                    seed=seed,
                    variant_key=variant_key,
                    checkpoint_path=checkpoints_by_variant_and_seed[variant_key][seed],
                    spec=selected_specs[variant_key],
                    vocab_size=corpus.vocab_size,
                    device=device,
                    val_inputs=val_inputs,
                    val_targets=val_targets,
                    eval_batch_size=args.eval_batch_size,
                    log_path=args.log_path,
                )
        for checkpoints_by_seed in checkpoints_by_variant_and_seed.values():
            for checkpoint_path in checkpoints_by_seed.values():
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
            "weight_decay": WEIGHT_DECAY,
            "eval_samples": args.eval_samples,
            "seeds": args.seeds,
            "context_size": CONTEXT_SIZE,
            "compile_model": args.compile_model,
            "sanity_check_only": args.sanity_check_only,
            "no_lock": args.no_lock,
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
        if args.sanity_check_only:
            redirect_sanity_check_paths(args)
        append_log(args.log_path, {"stage": "crash", "traceback": traceback.format_exc()})
        raise
