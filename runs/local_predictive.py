from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

# Ensure repo root is importable regardless of how this script is launched
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from core.dataset import CorpusData, load_corpus
from core.fixed_window_char import set_seed
from core.model import ParallelDiagonalForwardState, ParallelDiagonalModel, count_parameters
from core.run_utils import (
    append_log,
    checkpoint_metrics,
    prepare_output_paths,
    random_batches,
    redirect_sanity_check_paths,
    resolve_device,
)

CONTEXT_SIZE = 128
TRAINING_STEPS = 2_000
EVAL_INTERVAL = 500
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 3e-4
SEED = 42
EVAL_SAMPLES = 4_096
LAMBDA_LOCAL = 1.0
D_MODEL = 256
FEEDFORWARD_DIM = 512


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    label: str
    num_blocks: int
    topology: str
    readout_mode: str
    token_injection: str
    detach_lateral: bool
    internal_steps: int
    rates: tuple[int, ...]
    use_local_predictive_loss: bool


@dataclass(frozen=True)
class ConditionModules:
    model: ParallelDiagonalModel
    prediction_head: nn.Linear | None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "wikitext_103" / "artifacts" / "local_predictive"
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--eval-samples", type=int, default=EVAL_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lambda-local", type=float, default=LAMBDA_LOCAL)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
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


def condition_specs() -> tuple[ConditionSpec, ConditionSpec]:
    return (
        ConditionSpec(
            key="baseline",
            label="single_block_ce",
            num_blocks=1,
            topology="upward",
            readout_mode="first",
            token_injection="block0",
            detach_lateral=True,
            internal_steps=1,
            rates=(1,),
            use_local_predictive_loss=False,
        ),
        ConditionSpec(
            key="treatment",
            label="two_block_local_predictive",
            num_blocks=2,
            topology="top_down_to_first",
            readout_mode="first",
            token_injection="block0",
            detach_lateral=True,
            internal_steps=1,
            rates=(1, 1),
            use_local_predictive_loss=True,
        ),
    )


def build_modules(*, spec: ConditionSpec, vocab_size: int, device: torch.device) -> ConditionModules:
    model = ParallelDiagonalModel(
        vocab_size=vocab_size,
        context_size=CONTEXT_SIZE,
        d_model=D_MODEL,
        feedforward_dim=FEEDFORWARD_DIM,
        num_blocks=spec.num_blocks,
        rates=spec.rates,
        internal_steps=spec.internal_steps,
        readout_mode=spec.readout_mode,
        token_injection=spec.token_injection,
        topology=spec.topology,
        detach_lateral=spec.detach_lateral,
    ).to(device)
    prediction_head = None
    if spec.use_local_predictive_loss:
        prediction_head = nn.Linear(D_MODEL, D_MODEL).to(device)
    return ConditionModules(model=model, prediction_head=prediction_head)


def total_parameter_count(modules: ConditionModules) -> dict[str, int]:
    model_parameter_count = count_parameters(modules.model)
    prediction_head_parameter_count = 0
    if modules.prediction_head is not None:
        prediction_head_parameter_count = count_parameters(modules.prediction_head)
    return {
        "model_parameter_count": model_parameter_count,
        "prediction_head_parameter_count": prediction_head_parameter_count,
        "total_parameter_count": model_parameter_count + prediction_head_parameter_count,
    }


def verify_args(args: argparse.Namespace) -> None:
    errors: list[str] = []
    if args.training_steps <= 0:
        errors.append(f"training_steps must be positive, got {args.training_steps}")
    if args.eval_interval <= 0:
        errors.append(f"eval_interval must be positive, got {args.eval_interval}")
    if args.batch_size <= 0:
        errors.append(f"batch_size must be positive, got {args.batch_size}")
    if args.eval_batch_size <= 0:
        errors.append(f"eval_batch_size must be positive, got {args.eval_batch_size}")
    if args.eval_samples <= 0:
        errors.append(f"eval_samples must be positive, got {args.eval_samples}")
    if args.lambda_local < 0.0:
        errors.append(f"lambda_local must be non-negative, got {args.lambda_local}")
    if errors:
        raise ValueError("Argument validation failed:\n- " + "\n- ".join(errors))


def combined_named_parameters(modules: ConditionModules) -> list[tuple[str, nn.Parameter]]:
    named_parameters = [(f"model.{name}", parameter) for name, parameter in modules.model.named_parameters()]
    if modules.prediction_head is not None:
        named_parameters.extend(
            (f"prediction_head.{name}", parameter) for name, parameter in modules.prediction_head.named_parameters()
        )
    return named_parameters


def expected_zero_gradients(spec: ConditionSpec, *, lambda_local: float) -> set[str]:
    expected: set[str] = set()
    if spec.token_injection == "block0" and spec.num_blocks > 1:
        for block_index in range(1, spec.num_blocks):
            expected.add(f"model.token_mixes.{block_index}.alpha_logit")
    # When lambda_local=0, interior blocks and prediction head get no gradient (control condition)
    if spec.use_local_predictive_loss and lambda_local == 0.0 and spec.num_blocks > 1:
        for block_index in range(1, spec.num_blocks):
            expected.add(f"model.blocks.{block_index}.proj_in.weight")
            expected.add(f"model.blocks.{block_index}.proj_in.bias")
            expected.add(f"model.blocks.{block_index}.proj_out.weight")
            expected.add(f"model.blocks.{block_index}.proj_out.bias")
            expected.add(f"model.block_mixes.{block_index}.alpha_logit")
        expected.add("prediction_head.weight")
        expected.add("prediction_head.bias")
    return expected


def compute_losses(
    *,
    modules: ConditionModules,
    spec: ConditionSpec,
    batch_inputs: Int[Tensor, "batch context"],
    batch_targets: Int[Tensor, "batch"],
    lambda_local: float,
) -> tuple[Float[Tensor, ""], dict[str, float], ParallelDiagonalForwardState]:
    logits, state = modules.model.forward_with_state(batch_inputs)
    ce_loss = F.cross_entropy(logits, batch_targets)
    local_loss_tensor = torch.zeros((), device=logits.device, dtype=logits.dtype)
    if spec.use_local_predictive_loss:
        if modules.prediction_head is None:
            raise RuntimeError("Treatment condition requires a prediction head.")
        if len(state.block_outputs) < 2:
            raise RuntimeError("Treatment condition requires two block outputs for local predictive loss.")
        predictions = modules.prediction_head(state.block_outputs[1])
        local_targets = state.block_outputs[0].detach()
        local_loss_tensor = F.mse_loss(predictions, local_targets)
    total_loss = ce_loss + (lambda_local * local_loss_tensor)
    metrics = {
        "ce_loss": round(ce_loss.detach().item(), 6),
        "local_loss": round(local_loss_tensor.detach().item(), 6),
        "total_loss": round(total_loss.detach().item(), 6),
    }
    return total_loss, metrics, state


def verify_forward_and_gradients(
    *,
    modules: ConditionModules,
    spec: ConditionSpec,
    vocab_size: int,
    device: torch.device,
    lambda_local: float,
) -> dict[str, object]:
    model_state = {name: tensor.detach().clone() for name, tensor in modules.model.state_dict().items()}
    prediction_head_state = None
    if modules.prediction_head is not None:
        prediction_head_state = {
            name: tensor.detach().clone() for name, tensor in modules.prediction_head.state_dict().items()
        }
    optimizer = torch.optim.SGD(
        [parameter for _, parameter in combined_named_parameters(modules)],
        lr=1e-3,
    )
    modules.model.train()
    if modules.prediction_head is not None:
        modules.prediction_head.train()

    dummy_inputs = torch.randint(0, vocab_size, (2, CONTEXT_SIZE), device=device)
    dummy_targets = torch.randint(0, vocab_size, (2,), device=device)
    total_loss, _, state = compute_losses(
        modules=modules,
        spec=spec,
        batch_inputs=dummy_inputs,
        batch_targets=dummy_targets,
        lambda_local=lambda_local,
    )
    if tuple(state.block_outputs[0].shape) != (2, CONTEXT_SIZE, D_MODEL):
        raise RuntimeError(
            f"Verification failed: expected block output shape (2, {CONTEXT_SIZE}, {D_MODEL}), got {tuple(state.block_outputs[0].shape)}."
        )
    if not torch.isfinite(total_loss):
        raise RuntimeError("Verification failed: dummy total loss is not finite.")

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()

    missing_gradient_parameters: list[str] = []
    zero_gradient_parameters: list[str] = []
    expected_missing = expected_zero_gradients(spec, lambda_local=lambda_local)
    unexpected_missing_gradient_parameters: list[str] = []
    gradient_norm_sum = 0.0
    nonzero_gradient_parameters = 0
    total_trainable_parameters = 0
    for name, parameter in combined_named_parameters(modules):
        if not parameter.requires_grad:
            continue
        total_trainable_parameters += 1
        if parameter.grad is None:
            missing_gradient_parameters.append(name)
            if name not in expected_missing:
                unexpected_missing_gradient_parameters.append(name)
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Verification failed: parameter {name} has NaN or Inf gradients.")
        gradient_norm = parameter.grad.detach().norm().item()
        gradient_norm_sum += gradient_norm
        if gradient_norm > 0.0:
            nonzero_gradient_parameters += 1
        elif name not in expected_missing:
            zero_gradient_parameters.append(name)

    if len(unexpected_missing_gradient_parameters) > 0 or len(zero_gradient_parameters) > 0:
        raise RuntimeError(
            "Verification failed: some parameters did not receive gradients. "
            f"unexpected_missing={unexpected_missing_gradient_parameters}, zero={zero_gradient_parameters}"
        )

    modules.model.zero_grad(set_to_none=True)
    if modules.prediction_head is not None:
        modules.prediction_head.zero_grad(set_to_none=True)
    modules.model.load_state_dict(model_state)
    if modules.prediction_head is not None:
        if prediction_head_state is None:
            raise RuntimeError("Prediction head state missing during verification restore.")
        modules.prediction_head.load_state_dict(prediction_head_state)

    return {
        "forward_pass_ok": True,
        "backward_pass_ok": True,
        "block_output_shape": list(state.block_outputs[0].shape),
        "dummy_total_loss": round(total_loss.detach().item(), 6),
        "total_trainable_parameters": total_trainable_parameters,
        "nonzero_gradient_parameters": nonzero_gradient_parameters,
        "gradient_norm_sum": round(gradient_norm_sum, 6),
        "expected_missing_gradient_parameters": sorted(expected_missing),
        "missing_gradient_parameters": sorted(missing_gradient_parameters),
    }


def write_report(report_path: Path, payload: dict[str, object]) -> None:
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def train_condition(
    *,
    spec: ConditionSpec,
    corpus: CorpusData,
    val_inputs: Int[Tensor, "examples context"],
    val_targets: Int[Tensor, "examples"],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    set_seed(args.seed)
    modules = build_modules(spec=spec, vocab_size=corpus.vocab_size, device=device)
    parameter_counts = total_parameter_count(modules)
    verification = verify_forward_and_gradients(
        modules=modules,
        spec=spec,
        vocab_size=corpus.vocab_size,
        device=device,
        lambda_local=args.lambda_local,
    )

    optimizer_parameters = [parameter for _, parameter in combined_named_parameters(modules)]
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=0.01)
    encoded_corpus = getattr(corpus.train_dataset, "encoded_corpus", None)
    if not isinstance(encoded_corpus, torch.Tensor):
        raise TypeError("Expected corpus.train_dataset.encoded_corpus to be a torch.Tensor.")
    batch_rng = torch.Generator(device="cpu")
    batch_rng.manual_seed(args.seed)
    batch_iterator = random_batches(
        encoded_corpus,
        context_size=CONTEXT_SIZE,
        batch_size=args.batch_size,
        device=device,
        rng=batch_rng,
    )

    initial_checkpoint = checkpoint_metrics(
        model=modules.model,
        model_call=modules.model,
        val_inputs=val_inputs,
        val_targets=val_targets,
        batch_size=args.eval_batch_size,
        step=0,
        include_tokens_per_second=False,
    )
    checkpoints: list[dict[str, object]] = [initial_checkpoint]
    append_log(
        args.log_path,
        {
            "stage": "condition_started",
            "condition": spec.key,
            "label": spec.label,
            "config": asdict(spec),
            "parameter_counts": parameter_counts,
            "verification": verification,
            "initial_checkpoint": initial_checkpoint,
        },
    )

    print(
        f"[{spec.key}] params total={parameter_counts['total_parameter_count']} "
        f"(model={parameter_counts['model_parameter_count']}, prediction_head={parameter_counts['prediction_head_parameter_count']})",
        flush=True,
    )
    print(f"[{spec.key}] step 0 val_loss={initial_checkpoint['val_loss']:.6f}", flush=True)

    last_training_metrics = {"ce_loss": None, "local_loss": None, "total_loss": None}
    started_at = perf_counter()
    for step in range(1, args.training_steps + 1):
        batch_inputs, batch_targets = next(batch_iterator)
        optimizer.zero_grad(set_to_none=True)
        total_loss, training_metrics, _ = compute_losses(
            modules=modules,
            spec=spec,
            batch_inputs=batch_inputs,
            batch_targets=batch_targets,
            lambda_local=args.lambda_local,
        )
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Training diverged for {spec.key} at step {step}: loss is NaN or Inf.")
        total_loss.backward()
        optimizer.step()
        last_training_metrics = training_metrics

        if step % args.eval_interval != 0 and step != args.training_steps:
            continue

        checkpoint = checkpoint_metrics(
            model=modules.model,
            model_call=modules.model,
            val_inputs=val_inputs,
            val_targets=val_targets,
            batch_size=args.eval_batch_size,
            step=step,
            include_tokens_per_second=False,
        )
        checkpoints.append(
            {
                **checkpoint,
                "training_ce_loss": training_metrics["ce_loss"],
                "training_local_loss": training_metrics["local_loss"],
                "training_total_loss": training_metrics["total_loss"],
            }
        )
        append_log(
            args.log_path,
            {
                "stage": "checkpoint",
                "condition": spec.key,
                "step": step,
                "val_loss": checkpoint["val_loss"],
                "val_accuracy": checkpoint["val_accuracy"],
                "training_ce_loss": training_metrics["ce_loss"],
                "training_local_loss": training_metrics["local_loss"],
                "training_total_loss": training_metrics["total_loss"],
            },
        )
        print(
            f"[{spec.key}] step {step} val_loss={checkpoint['val_loss']:.6f} "
            f"ce={training_metrics['ce_loss']:.6f} local={training_metrics['local_loss']:.6f}",
            flush=True,
        )

    final_checkpoint = checkpoints[-1]
    result = {
        "condition": spec.key,
        "label": spec.label,
        "config": asdict(spec),
        "parameter_counts": parameter_counts,
        "verification": verification,
        "checkpoints": checkpoints,
        "final_checkpoint": final_checkpoint,
        "best_checkpoint": min(checkpoints, key=lambda checkpoint: float(checkpoint["val_loss"])),
        "final_training_metrics": last_training_metrics,
        "wall_seconds": round(perf_counter() - started_at, 6),
    }
    append_log(
        args.log_path,
        {
            "stage": "condition_finished",
            "condition": spec.key,
            "final_checkpoint": final_checkpoint,
            "wall_seconds": result["wall_seconds"],
        },
    )
    return result


def comparison_summary(*, baseline: dict[str, object], treatment: dict[str, object]) -> dict[str, object]:
    baseline_loss = float(baseline["final_checkpoint"]["val_loss"])
    treatment_loss = float(treatment["final_checkpoint"]["val_loss"])
    delta = round(treatment_loss - baseline_loss, 6)
    return {
        "baseline_final_val_loss": baseline_loss,
        "treatment_final_val_loss": treatment_loss,
        "treatment_minus_baseline": delta,
        "treatment_beats_baseline": treatment_loss < baseline_loss,
    }


def main() -> int:
    args = parse_args()
    verify_args(args)
    if args.sanity_check_only:
        redirect_sanity_check_paths(args)

    device = resolve_device(args.device)
    prepare_output_paths(report_path=args.report_path, log_path=args.log_path)
    corpus = load_corpus(
        train_path=args.train_path,
        val_path=args.val_path,
        context_size=CONTEXT_SIZE,
        eval_samples=args.eval_samples,
    )
    val_inputs = corpus.val_inputs.to(device=device, dtype=torch.long)
    val_targets = corpus.val_targets.to(device=device, dtype=torch.long)

    baseline_spec, treatment_spec = condition_specs()
    append_log(
        args.log_path,
        {
            "stage": "experiment_started",
            "sanity_check_only": args.sanity_check_only,
            "seed": args.seed,
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "eval_samples": args.eval_samples,
            "lambda_local": args.lambda_local,
            "context_size": CONTEXT_SIZE,
            "device": str(device),
            "vocab_size": corpus.vocab_size,
            "train_dataset_size": len(corpus.train_dataset),
            "val_examples": int(corpus.val_inputs.shape[0]),
            "conditions": [asdict(baseline_spec), asdict(treatment_spec)],
        },
    )

    baseline_result = train_condition(
        spec=baseline_spec,
        corpus=corpus,
        val_inputs=val_inputs,
        val_targets=val_targets,
        args=args,
        device=device,
    )
    treatment_result = train_condition(
        spec=treatment_spec,
        corpus=corpus,
        val_inputs=val_inputs,
        val_targets=val_targets,
        args=args,
        device=device,
    )
    summary = comparison_summary(baseline=baseline_result, treatment=treatment_result)
    report = {
        "config": {
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "eval_samples": args.eval_samples,
            "seed": args.seed,
            "lambda_local": args.lambda_local,
            "context_size": CONTEXT_SIZE,
            "d_model": D_MODEL,
            "feedforward_dim": FEEDFORWARD_DIM,
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
            "sanity_check_only": args.sanity_check_only,
        },
        "results": {
            "baseline": baseline_result,
            "treatment": treatment_result,
        },
        "comparison": summary,
    }
    write_report(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "experiment_finished",
            "report_path": str(args.report_path),
            "comparison": summary,
        },
    )
    print(
        "Comparison summary: "
        f"baseline={summary['baseline_final_val_loss']:.6f}, "
        f"treatment={summary['treatment_final_val_loss']:.6f}, "
        f"delta={summary['treatment_minus_baseline']:.6f}, "
        f"treatment_beats_baseline={summary['treatment_beats_baseline']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
