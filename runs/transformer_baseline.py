from __future__ import annotations

import argparse
import atexit
import os
from collections.abc import Iterator
import gc
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

# Ensure repo root is importable regardless of how this script is launched
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from einops import rearrange, repeat
from jaxtyping import Float, Int
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.dataset import CorpusData, load_corpus
from core.fixed_window_char import set_seed
from core.model import count_parameters
from core.training import GraphTrainer, capturable_adamw, current_git_sha, current_git_status_short, write_json

CONTEXT_SIZE = 128
TRAINING_STEPS = 20_000
EVAL_INTERVAL = 1_000
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 3e-4
DEFAULT_SEEDS = (42, 43)
EVAL_SAMPLES = 4096
WARMUP_STEPS = 3
TARGET_PARAMETER_COUNT = 2_851_446
MAX_PARAMETER_COUNT_DIFFERENCE_RATIO = 0.05


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 160
    n_layers: int = 4
    n_heads: int = 8
    ffn_dim: int = 640
    context_size: int = CONTEXT_SIZE


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    config: TransformerConfig


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "wikitext_103" / "artifacts" / "transformer_baseline"
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--eval-samples", type=int, default=EVAL_SAMPLES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
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
        "transformer": VariantSpec(
            key="transformer",
            label="transformer_baseline",
            config=TransformerConfig(),
        )
    }


def append_log(log_path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


class CausalSelfAttention(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: Float[Tensor, "batch context d_model"],
    ) -> Float[Tensor, "batch context d_model"]:
        q = rearrange(
            self.q_proj(x),
            "batch context (heads head_dim) -> batch heads context head_dim",
            heads=self.n_heads,
            head_dim=self.head_dim,
        )
        k = rearrange(
            self.k_proj(x),
            "batch context (heads head_dim) -> batch heads context head_dim",
            heads=self.n_heads,
            head_dim=self.head_dim,
        )
        v = rearrange(
            self.v_proj(x),
            "batch context (heads head_dim) -> batch heads context head_dim",
            heads=self.n_heads,
            head_dim=self.head_dim,
        )
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        return self.out_proj(
            rearrange(attended, "batch heads context head_dim -> batch context (heads head_dim)")
        )


class FeedForward(nn.Module):
    def __init__(self, *, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, ffn_dim)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: Float[Tensor, "batch context d_model"],
    ) -> Float[Tensor, "batch context d_model"]:
        return self.out_proj(self.activation(self.in_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, ffn_dim=ffn_dim)

    def forward(
        self,
        x: Float[Tensor, "batch context d_model"],
    ) -> Float[Tensor, "batch context d_model"]:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, *, vocab_size: int, config: TransformerConfig) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    ffn_dim=config.ffn_dim,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=True)

    def embedded_tokens(
        self,
        tokens: Int[Tensor, "batch context"],
    ) -> Float[Tensor, "batch context d_model"]:
        context_size = tokens.shape[1]
        if context_size != self.config.context_size:
            raise ValueError(f"Expected context length {self.config.context_size}, got {context_size}.")
        positions = torch.arange(context_size, device=tokens.device)
        return self.token_embedding(tokens) + repeat(
            self.position_embedding(positions),
            "context d_model -> batch context d_model",
            batch=tokens.shape[0],
        )

    def forward(
        self,
        tokens: Int[Tensor, "batch context"],
    ) -> Float[Tensor, "batch vocab"]:
        x = self.embedded_tokens(tokens)
        for block in self.blocks:
            x = block(x)
        final_state = self.final_norm(x[:, -1, :])
        return self.lm_head(final_state)


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(requested_device)


def maybe_compile_model(model: DecoderOnlyTransformer, *, enabled: bool) -> nn.Module:
    if not enabled:
        return model
    return torch.compile(model, backend="aot_eager")


def build_model(
    *,
    device: torch.device,
    vocab_size: int,
    spec: VariantSpec,
) -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(vocab_size=vocab_size, config=spec.config).to(device)


def parameter_match_summary(parameter_count: int) -> dict[str, float | int | bool]:
    difference = parameter_count - TARGET_PARAMETER_COUNT
    difference_ratio = abs(difference) / TARGET_PARAMETER_COUNT
    return {
        "target_parameter_count": TARGET_PARAMETER_COUNT,
        "parameter_count_difference": difference,
        "parameter_count_difference_ratio": round(difference_ratio, 6),
        "within_5_percent": difference_ratio <= MAX_PARAMETER_COUNT_DIFFERENCE_RATIO,
    }


def validate_parameter_count(parameter_count: int) -> dict[str, float | int | bool]:
    summary = parameter_match_summary(parameter_count)
    if summary["within_5_percent"] is not True:
        raise ValueError(
            "Transformer parameter count is too far from the A_single target. "
            f"Got {parameter_count}, target {TARGET_PARAMETER_COUNT}, ratio {summary['parameter_count_difference_ratio']}."
        )
    return summary


def build_optimizer(
    model: DecoderOnlyTransformer,
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
    model: DecoderOnlyTransformer,
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
    model: DecoderOnlyTransformer,
    model_call: nn.Module,
    val_inputs: Int[Tensor, "examples context"],
    val_targets: Int[Tensor, "examples"],
    batch_size: int,
    step: int,
) -> dict[str, float | int]:
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
    }


def verify_forward_and_gradients(
    *,
    model: DecoderOnlyTransformer,
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
        raise RuntimeError(f"Transformer verification failed: expected output shape (2, {vocab_size}), got {tuple(logits.shape)}.")

    loss = F.cross_entropy(logits, dummy_targets)
    if not torch.isfinite(loss):
        raise RuntimeError("Transformer verification failed: dummy loss is not finite.")

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
            raise RuntimeError(f"Transformer verification failed: parameter {name} has NaN or Inf gradients.")
        grad_norm = parameter.grad.detach().norm().item()
        gradient_norm_sum += grad_norm
        if grad_norm > 0.0:
            nonzero_gradient_parameters += 1
        else:
            zero_gradient_parameters.append(name)

    if len(missing_gradient_parameters) > 0 or len(zero_gradient_parameters) > 0:
        raise RuntimeError(
            "Transformer verification failed: some parameters did not receive gradients. "
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
) -> dict[str, object]:
    set_seed(seed)
    encoded_corpus = getattr(corpus.train_dataset, "encoded_corpus", None)
    if not isinstance(encoded_corpus, torch.Tensor):
        raise TypeError(
            "train_single_variant expects corpus.train_dataset to expose encoded_corpus as a torch.Tensor."
        )

    model = build_model(device=device, vocab_size=corpus.vocab_size, spec=spec)
    parameter_count = count_parameters(model)
    parameter_match = validate_parameter_count(parameter_count)
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
    )

    append_log(
        args.log_path,
        {
            "stage": "variant_started",
            "seed": seed,
            "variant": variant_key,
            "label": spec.label,
            "parameter_count": parameter_count,
            "parameter_match": parameter_match,
            "compiled": args.compile_model,
            "verification": verification,
            "initial_checkpoint": initial_checkpoint,
        },
    )

    if args.sanity_check_only:
        result = {
            "seed": seed,
            "variant": variant_key,
            "label": spec.label,
            "class_name": "DecoderOnlyTransformer",
            "config": asdict(spec.config),
            "parameter_count": parameter_count,
            "parameter_match": parameter_match,
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
                "wall_seconds": 0.0,
            },
        )
        del model_call
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result

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
        checkpoint = checkpoint_metrics(
            model=model,
            model_call=model_call,
            val_inputs=val_inputs,
            val_targets=val_targets,
            batch_size=args.eval_batch_size,
            step=step,
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

    if trainer is not None:
        trainer.synchronize()
    if last_loss is None:
        raise RuntimeError(f"Variant {variant_key} completed without recording a training loss.")

    wall_seconds = perf_counter() - started_at
    result = {
        "seed": seed,
        "variant": variant_key,
        "label": spec.label,
        "class_name": "DecoderOnlyTransformer",
        "config": asdict(spec.config),
        "parameter_count": parameter_count,
        "parameter_match": parameter_match,
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
            "wall_seconds": result["wall_seconds"],
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
    return result


def mean_rounded(values: list[float]) -> float:
    return round(mean(values), 6)


def std_rounded(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(pstdev(values), 6)


def summarize_results(*, per_seed_results: list[dict[str, object]], sanity_check_only: bool) -> dict[str, object]:
    if sanity_check_only:
        return {
            "transformer": {
                "num_runs": len(per_seed_results),
                "parameter_count": per_seed_results[0]["parameter_count"],
                "parameter_match": per_seed_results[0]["parameter_match"],
                "verification": per_seed_results[0]["verification"],
            }
        }

    final_losses = [result["final_checkpoint"]["val_loss"] for result in per_seed_results]
    final_accuracies = [result["final_checkpoint"]["val_accuracy"] for result in per_seed_results]
    wall_seconds = [result["wall_seconds"] for result in per_seed_results]
    return {
        "transformer": {
            "num_runs": len(per_seed_results),
            "mean_final_val_loss": mean_rounded(final_losses),
            "std_final_val_loss": std_rounded(final_losses),
            "mean_final_val_accuracy": mean_rounded(final_accuracies),
            "std_final_val_accuracy": std_rounded(final_accuracies),
            "mean_wall_seconds": mean_rounded(wall_seconds),
            "runs": per_seed_results,
        }
    }


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

    device = resolve_device(args.device)
    specs = variant_specs()

    # Manage runs/active.lock — tied to process lifetime
    lock_path = Path("runs/active.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_content = f"PID: {os.getpid()}\nExperiment: transformer_baseline\nVariants: {list(specs)}\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
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
    val_inputs = corpus.val_inputs.to(device=device, dtype=torch.long)
    val_targets = corpus.val_targets.to(device=device, dtype=torch.long)

    append_log(
        args.log_path,
        {
            "stage": "experiment_started",
            "sanity_check_only": args.sanity_check_only,
            "seeds": args.seeds,
            "variants": list(specs),
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
        },
    )

    overall_started_at = perf_counter()
    per_seed_results: list[dict[str, object]] = []
    for seed in args.seeds:
        append_log(args.log_path, {"stage": "seed_started", "seed": seed})
        per_seed_results.append(
            train_single_variant(
                seed=seed,
                variant_key="transformer",
                spec=specs["transformer"],
                args=args,
                corpus=corpus,
                device=device,
                val_inputs=val_inputs,
                val_targets=val_targets,
            )
        )
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
        "variants": {key: {"key": value.key, "label": value.label, "config": asdict(value.config)} for key, value in specs.items()},
        "per_seed_results": per_seed_results,
        "summary_by_variant": summary_by_variant,
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "experiment_finished",
            "report_path": str(args.report_path),
            "summary_by_variant": summary_by_variant,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
