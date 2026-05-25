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
    )
    return verification_payload(summary)


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

    if variant_key == "C_lateral":
        temporary_checkpoint_path = args.report_path.parent / f"c_lateral_seed_{seed}.pt"
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
            "temporary_checkpoint_path": None if temporary_checkpoint_path is None else str(temporary_checkpoint_path),
        },
    )

    del optimizer
    del model_call
    del model
    release_memory(device=device)
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
        metrics = checkpoint_metrics(
            model=model,
            model_call=model,
            val_inputs=val_inputs,
            val_targets=val_targets,
            batch_size=eval_batch_size,
            step=0,
            include_tokens_per_second=False,
        )
        rounded_metrics = {
            "val_loss": metrics["val_loss"],
            "val_accuracy": metrics["val_accuracy"],
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
        raise ValueError("This experiment requires both C_lateral and C_isolated so the comparison stays matched.")

    selected_specs = {key: specs[key] for key in args.variants}
    device = resolve_device(args.device)

    register_active_lock(experiment_name="c_old_ablation", variants=args.variants)

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
        if args.sanity_check_only:
            redirect_sanity_check_paths(args)
        append_log(args.log_path, {"stage": "crash", "traceback": traceback.format_exc()})
        raise
