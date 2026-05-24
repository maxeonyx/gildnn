from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from core.dataset import CorpusData, RandomWindowCharDataset, download_wikitext_103_raw, load_corpus
from core.fixed_window_char import set_seed
from core.model import ParallelDiagonalForwardState, ParallelDiagonalModel, count_parameters
from core.training import GraphTrainer, capturable_adamw, current_git_sha, current_git_status_short, write_json

CONTEXT_SIZE = 128
TRAINING_STEPS = 20_000
EVAL_INTERVAL = 1_000
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 3e-4
EVAL_SAMPLES = 4_096
WARMUP_STEPS = 3
DEFAULT_SEEDS = (42,)


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    model_class: str
    num_blocks: int
    rates: tuple[int, ...]
    topology: str
    token_injection: str
    readout_mode: str
    d_model: int = 256
    feedforward_dim: int = 512
    internal_steps: int = 1
    detach_lateral: bool = False


def variant_specs() -> dict[str, VariantSpec]:
    return {
        "A_single": VariantSpec(
            key="A_single",
            label="gated_wikitext_A_single",
            model_class="ParallelDiagonalModel",
            num_blocks=1,
            rates=(1,),
            topology="upward",
            token_injection="block0",
            readout_mode="last",
        ),
        "B_gated": VariantSpec(
            key="B_gated",
            label="gated_wikitext_B_gated",
            model_class="GatedParallelModel",
            num_blocks=4,
            rates=(1, 2, 4, 8),
            topology="upward",
            token_injection="block0",
            readout_mode="all",
        ),
    }


class GatedParallelModel(ParallelDiagonalModel):
    def __init__(
        self,
        *,
        vocab_size: int,
        context_size: int,
        d_model: int,
        feedforward_dim: int,
        num_blocks: int,
        rates: tuple[int, ...] | list[int] | None = None,
        internal_steps: int = 1,
        readout_mode: str = "last",
        token_injection: str = "block0",
        topology: str = "upward",
        token_mix_init: float = 0.5,
        block_mix_init: float = 0.9,
        detach_lateral: bool = False,
        temporal_window: int = 0,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            context_size=context_size,
            d_model=d_model,
            feedforward_dim=feedforward_dim,
            num_blocks=num_blocks,
            rates=rates,
            internal_steps=internal_steps,
            readout_mode=readout_mode,
            token_injection=token_injection,
            topology=topology,
            token_mix_init=token_mix_init,
            block_mix_init=block_mix_init,
            detach_lateral=detach_lateral,
            temporal_window=temporal_window,
        )
        if self.topology != "upward":
            raise ValueError("GatedParallelModel currently supports topology='upward' only.")
        self.lateral_gates = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, dtype=torch.float32)) for _ in range(max(0, self.num_blocks - 1))]
        )

    def gate_for_block(self, block_index: int) -> Tensor:
        if block_index <= 0:
            raise ValueError(f"Block {block_index} does not receive lateral input in topology='upward'.")
        return self.lateral_gates[block_index - 1]

    def mix_lateral_input(
        self,
        *,
        block_index: int,
        state_input: Float[Tensor, "batch d_model"],
        neighbor_state: Float[Tensor, "batch d_model"],
    ) -> Float[Tensor, "batch d_model"]:
        gate = self.gate_for_block(block_index).to(device=state_input.device, dtype=state_input.dtype)
        return state_input + (gate * neighbor_state)

    def _forward_impl(
        self,
        tokens: Int[Tensor, "batch context"],
        *,
        return_state: bool,
    ) -> tuple[Float[Tensor, "batch vocab"], ParallelDiagonalForwardState | None]:
        return forward_parallel_model(
            model=self,
            tokens=tokens,
            return_state=return_state,
            ablated_block_index=None,
        )

    @torch.no_grad()
    def gate_values(self) -> list[float]:
        return [round(parameter.detach().item(), 6) for parameter in self.lateral_gates]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    artifact_dir = repo_root / "experiments" / "gated_wikitext" / "artifacts.ignore"
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--eval-samples", type=int, default=EVAL_SAMPLES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--variants", nargs="+", default=["A_single", "B_gated"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--sanity-steps", type=int, default=16)
    parser.add_argument("--sanity-eval-interval", type=int, default=8)
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.jsonl")
    return parser.parse_args()


def append_log(log_path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(requested_device)


def maybe_mixed_input(
    *,
    model: ParallelDiagonalModel,
    block_index: int,
    state_input: Float[Tensor, "batch d_model"],
    neighbor_state: Float[Tensor, "batch d_model"],
) -> Float[Tensor, "batch d_model"]:
    if isinstance(model, GatedParallelModel):
        return model.mix_lateral_input(
            block_index=block_index,
            state_input=state_input,
            neighbor_state=neighbor_state,
        )
    return 0.5 * (state_input + neighbor_state)


def forward_parallel_model(
    *,
    model: ParallelDiagonalModel,
    tokens: Int[Tensor, "batch context"],
    return_state: bool,
    ablated_block_index: int | None,
) -> tuple[Float[Tensor, "batch vocab"], ParallelDiagonalForwardState | None]:
    embeddings = model.embedded_tokens(tokens)
    batch_size = tokens.shape[0]
    previous_states = [
        torch.zeros(batch_size, model.d_model, device=tokens.device, dtype=embeddings.dtype)
        for _ in range(model.num_blocks)
    ]
    temporal_history = (
        [
            torch.zeros(
                batch_size,
                model.temporal_window,
                model.d_model,
                device=tokens.device,
                dtype=embeddings.dtype,
            )
            for _ in range(model.num_blocks)
        ]
        if model.temporal_window > 0
        else None
    )
    block_output_history = [[] for _ in range(model.num_blocks)] if return_state else None

    for time_index in range(model.context_size):
        token_state = embeddings[:, time_index, :]
        if model.token_injection == "all":
            seeded_states = [
                token_mix(previous_state, token_state)
                for token_mix, previous_state in zip(model.token_mixes, previous_states, strict=True)
            ]
        else:
            seeded_states = list(previous_states)
            seeded_states[0] = model.token_mixes[0](previous_states[0], token_state)

        current_states = list(previous_states)
        for internal_step in range(model.internal_steps):
            next_states = list(current_states)
            for block_index, (block, block_mix, rate) in enumerate(
                zip(model.blocks, model.block_mixes, model.rates, strict=True)
            ):
                if time_index % rate != 0:
                    continue
                if ablated_block_index == block_index:
                    next_states[block_index] = torch.zeros_like(current_states[block_index])
                    continue

                state_input = seeded_states[block_index] if internal_step == 0 else current_states[block_index]
                if block_index == 0 and model.topology == "upward":
                    block_input = state_input
                else:
                    if block_index == 0:
                        lateral_source = previous_states[1] if internal_step == 0 else current_states[1]
                    else:
                        lateral_source = (
                            previous_states[block_index - 1]
                            if internal_step == 0
                            else current_states[block_index - 1]
                        )
                    neighbor_state = model._maybe_detach_lateral(lateral_source)
                    if block_index > 0 and model.window_proj is not None and temporal_history is not None:
                        lower_history = model._maybe_detach_lateral(temporal_history[block_index - 1])
                        temporal_neighbor = model.window_proj(
                            rearrange(lower_history, "batch window d_model -> batch (window d_model)")
                        )
                        neighbor_state = 0.5 * (neighbor_state + temporal_neighbor)
                    block_input = maybe_mixed_input(
                        model=model,
                        block_index=block_index,
                        state_input=state_input,
                        neighbor_state=neighbor_state,
                    )
                block_delta = block(block_input)
                next_states[block_index] = block_mix(block_input, block_delta)
            current_states = next_states

        previous_states = current_states
        if temporal_history is not None:
            for block_index, (state, rate) in enumerate(zip(previous_states, model.rates, strict=True)):
                if time_index % rate != 0:
                    continue
                updated_history = torch.roll(temporal_history[block_index], shifts=-1, dims=1)
                updated_history[:, -1, :] = state
                temporal_history[block_index] = updated_history
        if block_output_history is not None:
            for block_index, state in enumerate(previous_states):
                block_output_history[block_index].append(state)

    logits = model.output(model._readout_state(previous_states))
    if block_output_history is None:
        return logits, None
    return logits, ParallelDiagonalForwardState(
        block_outputs=[torch.stack(history, dim=1) for history in block_output_history]
    )


def build_model(
    *,
    device: torch.device,
    vocab_size: int,
    spec: VariantSpec,
) -> ParallelDiagonalModel:
    model_cls: type[ParallelDiagonalModel]
    if spec.model_class == "ParallelDiagonalModel":
        model_cls = ParallelDiagonalModel
    elif spec.model_class == "GatedParallelModel":
        model_cls = GatedParallelModel
    else:
        raise ValueError(f"Unknown model_class {spec.model_class!r}.")
    return model_cls(
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
        detach_lateral=spec.detach_lateral,
    ).to(device)


def build_optimizer(
    model: ParallelDiagonalModel,
    *,
    device: torch.device,
    learning_rate: float,
) -> torch.optim.AdamW:
    if device.type == "cuda":
        return capturable_adamw(model, lr=learning_rate)
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01)


def random_batches(
    encoded_corpus: Int[Tensor, "tokens"],
    *,
    context_size: int,
    batch_size: int,
    device: torch.device,
    rng: torch.Generator,
) -> tuple[Int[Tensor, "batch context"], Int[Tensor, "batch"]]:
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
    inputs: Int[Tensor, "examples context"],
    targets: Int[Tensor, "examples"],
    batch_size: int,
    ablated_block_index: int | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        batch_inputs = inputs[start:stop]
        batch_targets = targets[start:stop]
        if ablated_block_index is None:
            logits = model(batch_inputs)
        else:
            logits, _ = forward_parallel_model(
                model=model,
                tokens=batch_inputs,
                return_state=False,
                ablated_block_index=ablated_block_index,
            )
        batch_examples = batch_targets.shape[0]
        total_loss += F.cross_entropy(logits, batch_targets, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
        total_examples += batch_examples

    if was_training:
        model.train()

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def checkpoint_metrics(
    *,
    model: ParallelDiagonalModel,
    val_inputs: Int[Tensor, "examples context"],
    val_targets: Int[Tensor, "examples"],
    batch_size: int,
    step: int,
) -> dict[str, float | int]:
    metrics = evaluate_model_call(
        model=model,
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
    model: ParallelDiagonalModel,
    device: torch.device,
    vocab_size: int,
) -> dict[str, object]:
    saved_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    model.train()
    dummy_inputs = torch.randint(0, vocab_size, (2, CONTEXT_SIZE), device=device)
    dummy_targets = torch.randint(0, vocab_size, (2,), device=device)
    logits = model(dummy_inputs)
    if tuple(logits.shape) != (2, vocab_size):
        raise RuntimeError(f"Verification failed: expected output shape (2, {vocab_size}), got {tuple(logits.shape)}.")

    loss = F.cross_entropy(logits, dummy_targets)
    if not torch.isfinite(loss):
        raise RuntimeError("Verification failed: dummy loss is not finite.")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    allowed_missing_gradients = {
        f"token_mixes.{block_index}.alpha_logit"
        for block_index in range(1, model.num_blocks)
        if model.token_injection == "block0"
    }
    missing_gradient_parameters: list[str] = []
    allowed_missing_gradient_parameters: list[str] = []
    zero_gradient_parameters: list[str] = []
    gradient_norm_sum = 0.0
    total_trainable_parameters = 0
    nonzero_gradient_parameters = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        total_trainable_parameters += 1
        if parameter.grad is None:
            if name in allowed_missing_gradients:
                allowed_missing_gradient_parameters.append(name)
            else:
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
    model.load_state_dict(saved_state)
    return {
        "output_shape": list(logits.shape),
        "dummy_loss": round(loss.item(), 6),
        "total_trainable_parameters": total_trainable_parameters,
        "nonzero_gradient_parameters": nonzero_gradient_parameters,
        "gradient_norm_sum": round(gradient_norm_sum, 6),
        "allowed_missing_gradient_parameters": allowed_missing_gradient_parameters,
    }


def block_ablation_report(
    *,
    model: ParallelDiagonalModel,
    val_inputs: Int[Tensor, "examples context"],
    val_targets: Int[Tensor, "examples"],
    batch_size: int,
) -> list[dict[str, float | int]]:
    baseline = evaluate_model_call(
        model=model,
        inputs=val_inputs,
        targets=val_targets,
        batch_size=batch_size,
    )
    reports: list[dict[str, float | int]] = []
    for block_index in range(model.num_blocks):
        ablated = evaluate_model_call(
            model=model,
            inputs=val_inputs,
            targets=val_targets,
            batch_size=batch_size,
            ablated_block_index=block_index,
        )
        reports.append(
            {
                "block_index": block_index,
                "baseline_val_loss": round(baseline["loss"], 6),
                "ablated_val_loss": round(ablated["loss"], 6),
                "ablation_effect_nats": round(ablated["loss"] - baseline["loss"], 6),
                "ablated_val_accuracy": round(ablated["accuracy"], 6),
            }
        )
    return reports


def gate_summary(model: ParallelDiagonalModel) -> dict[str, object] | None:
    if not isinstance(model, GatedParallelModel):
        return None
    return {
        "raw_values": model.gate_values(),
        "abs_mean": round(mean(abs(parameter.detach().item()) for parameter in model.lateral_gates), 6)
        if len(model.lateral_gates) > 0
        else 0.0,
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
    training_steps: int,
    eval_interval: int,
) -> dict[str, object]:
    set_seed(seed)
    train_dataset = corpus.train_dataset
    if not isinstance(train_dataset, RandomWindowCharDataset):
        raise TypeError(
            "gated_wikitext expects corpus.train_dataset to be RandomWindowCharDataset so encoded_corpus is available."
        )

    model = build_model(device=device, vocab_size=corpus.vocab_size, spec=spec)
    parameter_count = count_parameters(model)
    verification = verify_forward_and_gradients(model=model, device=device, vocab_size=corpus.vocab_size)
    initial_checkpoint = checkpoint_metrics(
        model=model,
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
            "verification": verification,
            "initial_checkpoint": initial_checkpoint,
            "gate_summary": gate_summary(model),
        },
    )

    optimizer = build_optimizer(model, device=device, learning_rate=args.learning_rate)
    batch_rng = torch.Generator(device="cpu")
    batch_rng.manual_seed(seed)
    batch_iterator = random_batches(
        train_dataset.encoded_corpus,
        context_size=CONTEXT_SIZE,
        batch_size=args.batch_size,
        device=device,
        rng=batch_rng,
    )

    checkpoints = [initial_checkpoint]
    trainer: GraphTrainer | None = None
    last_loss: Tensor | None = None
    started_at = perf_counter()

    if device.type == "cuda":
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

    for step in range(training_step_start, training_steps + 1):
        batch_input, batch_target = next(batch_iterator)
        if trainer is not None:
            last_loss = trainer.step(batch_input, batch_target)
        else:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_input)
            loss = F.cross_entropy(logits, batch_target)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Training diverged for variant {variant_key} at step {step}: loss is NaN or Inf."
                )
            loss.backward()
            optimizer.step()
            last_loss = loss.detach().clone()

        if step % eval_interval != 0 and step != training_steps:
            continue

        if trainer is not None:
            trainer.synchronize()
        checkpoint = checkpoint_metrics(
            model=model,
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
                "gate_summary": gate_summary(model),
            },
        )

    if trainer is not None:
        trainer.synchronize()
    if last_loss is None:
        raise RuntimeError(f"Variant {variant_key} completed without recording a training loss.")

    ablation = block_ablation_report(
        model=model,
        val_inputs=val_inputs,
        val_targets=val_targets,
        batch_size=args.eval_batch_size,
    )
    wall_seconds = perf_counter() - started_at
    result = {
        "seed": seed,
        "variant": variant_key,
        "label": spec.label,
        "class_name": type(model).__name__,
        "config": asdict(spec),
        "parameter_count": parameter_count,
        "verification": verification,
        "checkpoints": checkpoints,
        "best_checkpoint": min(checkpoints, key=lambda checkpoint: checkpoint["val_loss"]),
        "final_checkpoint": checkpoints[-1],
        "final_training_loss": round(last_loss.item(), 6),
        "wall_seconds": round(wall_seconds, 6),
        "gate_summary": gate_summary(model),
        "block_ablation": ablation,
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
            "gate_summary": result["gate_summary"],
            "block_ablation": ablation,
        },
    )

    del optimizer
    del trainer
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


def summarize_results(*, per_seed_results: list[dict[str, object]]) -> dict[str, object]:
    results_by_variant: dict[str, list[dict[str, object]]] = {}
    for result in per_seed_results:
        results_by_variant.setdefault(str(result["variant"]), []).append(result)

    summary: dict[str, object] = {}
    for variant, runs in results_by_variant.items():
        final_losses = [float(run["final_checkpoint"]["val_loss"]) for run in runs]
        final_accuracies = [float(run["final_checkpoint"]["val_accuracy"]) for run in runs]
        wall_seconds = [float(run["wall_seconds"]) for run in runs]
        summary[variant] = {
            "num_runs": len(runs),
            "mean_final_val_loss": mean_rounded(final_losses),
            "std_final_val_loss": std_rounded(final_losses),
            "mean_final_val_accuracy": mean_rounded(final_accuracies),
            "std_final_val_accuracy": std_rounded(final_accuracies),
            "mean_wall_seconds": mean_rounded(wall_seconds),
            "runs": runs,
        }
    return summary


def ensure_wikitext_corpus(*, context_size: int, eval_samples: int) -> tuple[CorpusData, dict[str, str]]:
    paths = download_wikitext_103_raw()
    corpus = load_corpus(
        train_path=paths["wiki.train.raw"],
        val_path=paths["wiki.valid.raw"],
        context_size=context_size,
        eval_samples=eval_samples,
    )
    return corpus, {name: str(path) for name, path in paths.items()}


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
    if args.sanity_steps <= 0:
        raise ValueError(f"sanity_steps must be positive, got {args.sanity_steps}.")
    if args.sanity_eval_interval <= 0:
        raise ValueError(f"sanity_eval_interval must be positive, got {args.sanity_eval_interval}.")
    if args.device == "cuda" and args.training_steps < WARMUP_STEPS and not args.sanity_check:
        raise ValueError(
            f"training_steps must be at least {WARMUP_STEPS} for CUDA graph warmup, got {args.training_steps}."
        )

    specs = variant_specs()
    unknown_variants = [variant for variant in args.variants if variant not in specs]
    if len(unknown_variants) > 0:
        raise ValueError(f"Unknown variants requested: {unknown_variants}. Available variants: {list(specs)}.")

    device = resolve_device(args.device)
    training_steps = args.sanity_steps if args.sanity_check else args.training_steps
    eval_interval = args.sanity_eval_interval if args.sanity_check else args.eval_interval

    lock_path = Path("runs/active.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_content = (
        f"PID: {os.getpid()}\n"
        f"Experiment: gated_wikitext\n"
        f"Variants: {args.variants}\n"
        f"Started: {datetime.now().astimezone().isoformat()}\n"
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

    corpus, corpus_paths = ensure_wikitext_corpus(context_size=CONTEXT_SIZE, eval_samples=args.eval_samples)
    val_inputs = corpus.val_inputs.to(device=device, dtype=torch.long)
    val_targets = corpus.val_targets.to(device=device, dtype=torch.long)

    append_log(
        args.log_path,
        {
            "stage": "experiment_started",
            "sanity_check": args.sanity_check,
            "seeds": args.seeds,
            "variants": args.variants,
            "training_steps": training_steps,
            "eval_interval": eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "context_size": CONTEXT_SIZE,
            "device": str(device),
            "vocab_size": corpus.vocab_size,
            "train_dataset_size": len(corpus.train_dataset),
            "val_examples": int(corpus.val_inputs.shape[0]),
            "corpus_paths": corpus_paths,
        },
    )

    overall_started_at = perf_counter()
    per_seed_results: list[dict[str, object]] = []
    for seed in args.seeds:
        append_log(args.log_path, {"stage": "seed_started", "seed": seed})
        for variant_key in args.variants:
            per_seed_results.append(
                train_single_variant(
                    seed=seed,
                    variant_key=variant_key,
                    spec=specs[variant_key],
                    args=args,
                    corpus=corpus,
                    device=device,
                    val_inputs=val_inputs,
                    val_targets=val_targets,
                    training_steps=training_steps,
                    eval_interval=eval_interval,
                )
            )
        append_log(args.log_path, {"stage": "seed_finished", "seed": seed})

    wall_seconds = perf_counter() - overall_started_at
    git_status_short = current_git_status_short()
    summary_by_variant = summarize_results(per_seed_results=per_seed_results)
    report = {
        "config": {
            "training_steps": training_steps,
            "eval_interval": eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "eval_samples": args.eval_samples,
            "seeds": args.seeds,
            "context_size": CONTEXT_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "dataset": "wikitext-103-raw",
            "sanity_check": args.sanity_check,
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
            "paths": corpus_paths,
        },
        "timing": {
            "overall_wall_seconds": round(wall_seconds, 6),
        },
        "variants": {key: asdict(value) for key, value in specs.items() if key in args.variants},
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
