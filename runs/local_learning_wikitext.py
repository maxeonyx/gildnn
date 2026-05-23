from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
import gc
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from core.dataset import CorpusData, load_corpus
from core.fixed_window_char import set_seed
from core.model import ParallelDiagonalForwardState, ParallelDiagonalModel, count_parameters
from core.training import (
    GraphTrainer,
    capturable_adamw,
    current_git_sha,
    current_git_status_short,
    write_json,
)

CONTEXT_SIZE = 32
TRAINING_STEPS = 20_000
EVAL_INTERVAL = 1_000
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
LEARNING_RATE = 3e-4
DEFAULT_SEEDS = (42, 43)
WARMUP_STEPS = 3
D_MODEL = 256
FEEDFORWARD_DIM = 512
RATES = (1, 2, 4, 8)
ABLATION_LOGIT = -100.0
AUX_LOSS_WEIGHT = 0.3
AUX_BLOCK_INDICES = (1, 2, 3)
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
ADAMW_WEIGHT_DECAY = 1e-2


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    d_model: int
    feedforward_dim: int
    num_blocks: int
    rates: tuple[int, ...]
    readout_mode: str
    token_injection: str
    topology: str
    aux_loss_weight: float


@dataclass
class LocalAuxLossTensors:
    task_loss: Float[Tensor, ""]
    aux_loss: Float[Tensor, ""]
    total_loss: Float[Tensor, ""]
    per_block_aux_losses: Float[Tensor, "num_aux_blocks"]


class UpperBlockAuxHeads(nn.Module):
    def __init__(self, *, d_model: int, vocab_size: int, block_indices: tuple[int, ...]) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.block_indices = tuple(block_indices)
        self.heads = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in self.block_indices])

    def compute_losses(
        self,
        *,
        state: ParallelDiagonalForwardState,
        final_targets: Int[Tensor, "batch"],
        final_logits: Float[Tensor, "batch vocab"],
        aux_loss_weight: float,
    ) -> LocalAuxLossTensors:
        if len(state.block_outputs) <= max(self.block_indices):
            raise ValueError(
                f"Need block outputs through index {max(self.block_indices)}, got {len(state.block_outputs)} blocks."
            )

        task_loss = F.cross_entropy(final_logits, final_targets)
        per_block_aux_losses = torch.stack(
            [
                F.cross_entropy(head(state.block_outputs[block_index][:, -1, :]), final_targets)
                for head, block_index in zip(self.heads, self.block_indices, strict=True)
            ]
        )
        aux_loss = per_block_aux_losses.mean()
        total_loss = task_loss + (aux_loss_weight * aux_loss)
        return LocalAuxLossTensors(
            task_loss=task_loss,
            aux_loss=aux_loss,
            total_loss=total_loss,
            per_block_aux_losses=per_block_aux_losses,
        )


class LocalAuxGraphTrainer:
    def __init__(
        self,
        *,
        model: ParallelDiagonalModel,
        aux_heads: UpperBlockAuxHeads,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        aux_loss_weight: float,
    ) -> None:
        if device.type != "cuda":
            raise ValueError(f"LocalAuxGraphTrainer requires CUDA, got {device}.")
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError(f"Expected AdamW, got {type(optimizer).__name__}.")

        self.model = model
        self.aux_heads = aux_heads
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.aux_loss_weight = aux_loss_weight
        self.capture_stream = torch.cuda.Stream(device=device)
        self.graph = torch.cuda.CUDAGraph()
        self.is_captured = False

        self.static_input = torch.empty((batch_size, seq_len), device=device, dtype=torch.long)
        self.static_target = torch.empty((batch_size,), device=device, dtype=torch.long)
        self.static_task_loss = torch.zeros((), device=device)
        self.static_aux_loss = torch.zeros((), device=device)
        self.static_total_loss = torch.zeros((), device=device)
        self.static_per_block_aux_losses = torch.zeros((len(aux_heads.block_indices),), device=device)

    def _copy_batch(
        self,
        *,
        batch_input: Int[Tensor, "batch context"],
        batch_target: Int[Tensor, "batch"],
    ) -> None:
        if tuple(batch_input.shape) != (self.batch_size, self.seq_len):
            raise ValueError(f"Expected input shape {(self.batch_size, self.seq_len)}, got {tuple(batch_input.shape)}.")
        if tuple(batch_target.shape) != (self.batch_size,):
            raise ValueError(f"Expected target shape {(self.batch_size,)}, got {tuple(batch_target.shape)}.")
        if batch_input.dtype != torch.long:
            raise ValueError(f"Expected input dtype torch.long, got {batch_input.dtype}.")
        if batch_target.dtype != torch.long:
            raise ValueError(f"Expected target dtype torch.long, got {batch_target.dtype}.")

        self.static_input.copy_(batch_input, non_blocking=True)
        self.static_target.copy_(batch_target, non_blocking=True)

    def _training_step(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        logits, state = self.model.forward_with_state(self.static_input)
        losses = self.aux_heads.compute_losses(
            state=state,
            final_targets=self.static_target,
            final_logits=logits,
            aux_loss_weight=self.aux_loss_weight,
        )
        losses.total_loss.backward()
        self.optimizer.step()
        self.static_task_loss.copy_(losses.task_loss.detach())
        self.static_aux_loss.copy_(losses.aux_loss.detach())
        self.static_total_loss.copy_(losses.total_loss.detach())
        self.static_per_block_aux_losses.copy_(losses.per_block_aux_losses.detach())

    def capture(
        self,
        warmup_batches: Sequence[tuple[Int[Tensor, "batch context"], Int[Tensor, "batch"]]],
    ) -> None:
        if self.is_captured:
            raise RuntimeError("capture() may only be called once.")
        if len(warmup_batches) < 3:
            raise ValueError(f"Need at least 3 warmup batches, got {len(warmup_batches)}.")

        self.model.train()
        self.aux_heads.train()
        current_stream = torch.cuda.current_stream(device=self.device)
        self.capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(self.capture_stream):
            for batch_input, batch_target in warmup_batches[:3]:
                self._copy_batch(batch_input=batch_input, batch_target=batch_target)
                self._training_step()
        current_stream.wait_stream(self.capture_stream)
        torch.cuda.synchronize(self.device)

        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self._training_step()

        current_stream.wait_stream(self.capture_stream)
        self.is_captured = True

    def step(
        self,
        *,
        batch_input: Int[Tensor, "batch context"],
        batch_target: Int[Tensor, "batch"],
    ) -> LocalAuxLossTensors:
        if not self.is_captured:
            raise RuntimeError("step() requires capture() first.")
        self._copy_batch(batch_input=batch_input, batch_target=batch_target)
        self.graph.replay()
        return self.snapshot()

    def snapshot(self) -> LocalAuxLossTensors:
        return LocalAuxLossTensors(
            task_loss=self.static_task_loss.detach().clone(),
            aux_loss=self.static_aux_loss.detach().clone(),
            total_loss=self.static_total_loss.detach().clone(),
            per_block_aux_losses=self.static_per_block_aux_losses.detach().clone(),
        )

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "wikitext_103" / "artifacts" / "local_learning"
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-steps", type=int, default=TRAINING_STEPS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
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
        "A_control": VariantSpec(
            key="A_control",
            label="local_learning_wikitext_a_control",
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=4,
            rates=RATES,
            readout_mode="all",
            token_injection="block0",
            topology="upward",
            aux_loss_weight=0.0,
        ),
        "B_aux_loss": VariantSpec(
            key="B_aux_loss",
            label="local_learning_wikitext_b_aux_loss",
            d_model=D_MODEL,
            feedforward_dim=FEEDFORWARD_DIM,
            num_blocks=4,
            rates=RATES,
            readout_mode="all",
            token_injection="block0",
            topology="upward",
            aux_loss_weight=AUX_LOSS_WEIGHT,
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
        readout_mode=spec.readout_mode,
        token_injection=spec.token_injection,
        topology=spec.topology,
    ).to(device)


def build_aux_heads(*, device: torch.device, vocab_size: int) -> UpperBlockAuxHeads:
    return UpperBlockAuxHeads(
        d_model=D_MODEL,
        vocab_size=vocab_size,
        block_indices=AUX_BLOCK_INDICES,
    ).to(device)


def append_log(log_path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def readout_weights(model: ParallelDiagonalModel) -> list[float]:
    weights = model.mix_coefficients()["readout_weights"]
    if weights is None:
        raise RuntimeError("Expected readout weights for readout_mode='all'.")
    return [round(float(weight), 6) for weight in weights]


def evaluate_with_block_ablated(
    model: ParallelDiagonalModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
    block_index: int,
) -> dict[str, float]:
    if model.readout_logits is None:
        raise RuntimeError("Expected readout_logits for block ablation.")
    with torch.no_grad():
        original_logits = model.readout_logits.detach().clone()
        model.readout_logits.copy_(original_logits)
        model.readout_logits[block_index] = ABLATION_LOGIT
    try:
        return evaluate_variant(
            model=model,
            aux_heads=None,
            inputs=inputs,
            targets=targets,
            batch_size=batch_size,
        )
    finally:
        with torch.no_grad():
            model.readout_logits.copy_(original_logits)


def ablation_losses(
    model: ParallelDiagonalModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
) -> list[float]:
    return [
        round(
            evaluate_with_block_ablated(
                model,
                inputs,
                targets,
                batch_size=batch_size,
                block_index=block_index,
            )["loss"],
            6,
        )
        for block_index in range(model.num_blocks)
    ]


@torch.inference_mode()
def evaluate_variant(
    *,
    model: ParallelDiagonalModel,
    aux_heads: UpperBlockAuxHeads | None,
    inputs: Int[Tensor, "examples context"],
    targets: Int[Tensor, "examples"],
    batch_size: int,
) -> dict[str, float | list[float] | None]:
    was_training_model = model.training
    was_training_aux = aux_heads.training if aux_heads is not None else False
    model.eval()
    if aux_heads is not None:
        aux_heads.eval()

    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    total_aux_losses = torch.zeros((len(AUX_BLOCK_INDICES),), device=inputs.device, dtype=torch.float64)

    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        batch_inputs = inputs[start:stop]
        batch_targets = targets[start:stop]
        batch_examples = batch_targets.shape[0]

        if aux_heads is None:
            logits = model(batch_inputs)
        else:
            logits, state = model.forward_with_state(batch_inputs)
            losses = aux_heads.compute_losses(
                state=state,
                final_targets=batch_targets,
                final_logits=logits,
                aux_loss_weight=AUX_LOSS_WEIGHT,
            )
            total_aux_losses += losses.per_block_aux_losses.detach().to(dtype=torch.float64) * batch_examples

        total_loss += F.cross_entropy(logits, batch_targets, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
        total_examples += batch_examples

    if was_training_model:
        model.train()
    if aux_heads is not None and was_training_aux:
        aux_heads.train()

    per_block_aux_val_losses: list[float] | None
    if aux_heads is None:
        per_block_aux_val_losses = None
    else:
        per_block_aux_val_losses = [
            round(float(loss), 6) for loss in (total_aux_losses / total_examples).cpu().tolist()
        ]

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "per_block_aux_val_losses": per_block_aux_val_losses,
    }


def checkpoint_metrics(
    *,
    model: ParallelDiagonalModel,
    aux_heads: UpperBlockAuxHeads | None,
    val_inputs: Tensor,
    val_targets: Tensor,
    batch_size: int,
    step: int,
) -> dict[str, float | int | list[float] | None]:
    metrics = evaluate_variant(
        model=model,
        aux_heads=aux_heads,
        inputs=val_inputs,
        targets=val_targets,
        batch_size=batch_size,
    )
    return {
        "step": step,
        "val_loss": round(float(metrics["loss"]), 6),
        "val_accuracy": round(float(metrics["accuracy"]), 6),
        "readout_weights": readout_weights(model),
        "per_block_aux_val_losses": metrics["per_block_aux_val_losses"],
        "ablation_losses": ablation_losses(
            model,
            val_inputs,
            val_targets,
            batch_size=batch_size,
        ),
    }


def random_batches(
    encoded_corpus: torch.Tensor,
    *,
    context_size: int,
    batch_size: int,
    device: torch.device,
    rng: torch.Generator,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
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


def capturable_adamw_with_aux(
    *,
    model: ParallelDiagonalModel,
    aux_heads: UpperBlockAuxHeads,
    lr: float,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        list(model.parameters()) + list(aux_heads.parameters()),
        lr=lr,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=ADAMW_WEIGHT_DECAY,
        capturable=True,
    )


def mean_rounded(values: list[float]) -> float:
    return round(mean(values), 6)


def std_rounded(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(pstdev(values), 6)


def mean_vector_rounded(vectors: list[list[float]]) -> list[float]:
    if len(vectors) == 0:
        raise ValueError("Expected at least one vector.")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ValueError("All vectors must have the same length.")
    return [round(mean(vector[index] for vector in vectors), 6) for index in range(width)]


def optional_mean_vector_rounded(vectors: list[list[float] | None]) -> list[float] | None:
    realized_vectors = [vector for vector in vectors if vector is not None]
    if len(realized_vectors) == 0:
        return None
    return mean_vector_rounded(realized_vectors)


def train_single_variant(
    *,
    seed: int,
    variant_key: str,
    spec: VariantSpec,
    args: argparse.Namespace,
    corpus: CorpusData,
    device: torch.device,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
) -> dict[str, object]:
    set_seed(seed)
    encoded_corpus = getattr(corpus.train_dataset, "encoded_corpus", None)
    if not isinstance(encoded_corpus, torch.Tensor):
        raise TypeError(
            "train_single_variant expects corpus.train_dataset to expose encoded_corpus as a torch.Tensor."
        )

    batch_rng = torch.Generator()
    batch_rng.manual_seed(seed)
    batch_iterator = random_batches(
        encoded_corpus,
        context_size=CONTEXT_SIZE,
        batch_size=args.batch_size,
        device=device,
        rng=batch_rng,
    )
    warmup_batches = [next(batch_iterator) for _ in range(WARMUP_STEPS)]

    set_seed(seed)
    model = build_model(device=device, vocab_size=corpus.vocab_size, spec=spec)
    aux_heads = build_aux_heads(device=device, vocab_size=corpus.vocab_size) if spec.aux_loss_weight > 0.0 else None
    if aux_heads is None:
        optimizer = capturable_adamw(model, lr=args.learning_rate)
        trainer: GraphTrainer | LocalAuxGraphTrainer = GraphTrainer(
            model,
            optimizer,
            batch_size=args.batch_size,
            seq_len=CONTEXT_SIZE,
            device=device,
        )
        trainer_name = "GraphTrainer"
    else:
        optimizer = capturable_adamw_with_aux(model=model, aux_heads=aux_heads, lr=args.learning_rate)
        trainer = LocalAuxGraphTrainer(
            model=model,
            aux_heads=aux_heads,
            optimizer=optimizer,
            batch_size=args.batch_size,
            seq_len=CONTEXT_SIZE,
            device=device,
            aux_loss_weight=spec.aux_loss_weight,
        )
        trainer_name = "LocalAuxGraphTrainer"

    checkpoints = [
        checkpoint_metrics(
            model=model,
            aux_heads=aux_heads,
            val_inputs=val_inputs,
            val_targets=val_targets,
            batch_size=args.eval_batch_size,
            step=0,
        )
    ]
    parameter_count = count_parameters(model) + (count_parameters(aux_heads) if aux_heads is not None else 0)
    append_log(
        args.log_path,
        {
            "stage": "variant_started",
            "seed": seed,
            "variant": variant_key,
            "label": spec.label,
            "parameter_count": parameter_count,
            "trainer": trainer_name,
            "initial_checkpoint": checkpoints[-1],
        },
    )

    started_at = perf_counter()
    trainer.capture(warmup_batches)
    if isinstance(trainer, GraphTrainer):
        last_task_loss = trainer.static_loss.detach().clone()
        last_aux_loss = torch.zeros((), device=device)
        last_total_loss = trainer.static_loss.detach().clone()
        last_per_block_aux_losses = torch.zeros((len(AUX_BLOCK_INDICES),), device=device)
    else:
        snapshot = trainer.snapshot()
        last_task_loss = snapshot.task_loss.detach().clone()
        last_aux_loss = snapshot.aux_loss.detach().clone()
        last_total_loss = snapshot.total_loss.detach().clone()
        last_per_block_aux_losses = snapshot.per_block_aux_losses.detach().clone()

    for step in range(WARMUP_STEPS + 1, args.training_steps + 1):
        batch_input, batch_target = next(batch_iterator)
        if isinstance(trainer, GraphTrainer):
            loss = trainer.step(batch_input, batch_target)
            last_task_loss = loss.detach().clone()
            last_aux_loss = torch.zeros((), device=device)
            last_total_loss = loss.detach().clone()
            last_per_block_aux_losses = torch.zeros((len(AUX_BLOCK_INDICES),), device=device)
        else:
            snapshot = trainer.step(batch_input=batch_input, batch_target=batch_target)
            last_task_loss = snapshot.task_loss.detach().clone()
            last_aux_loss = snapshot.aux_loss.detach().clone()
            last_total_loss = snapshot.total_loss.detach().clone()
            last_per_block_aux_losses = snapshot.per_block_aux_losses.detach().clone()

        if step % args.eval_interval != 0 and step != args.training_steps:
            continue

        trainer.synchronize()
        checkpoint = checkpoint_metrics(
            model=model,
            aux_heads=aux_heads,
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
                "train_task_loss": round(last_task_loss.item(), 6),
                "train_aux_loss": round(last_aux_loss.item(), 6),
                "train_total_loss": round(last_total_loss.item(), 6),
                "train_per_block_aux_losses": [
                    round(float(loss), 6) for loss in last_per_block_aux_losses.detach().cpu().tolist()
                ],
                **checkpoint,
            },
        )

    trainer.synchronize()
    wall_seconds = perf_counter() - started_at
    result = {
        "seed": seed,
        "variant": variant_key,
        "label": spec.label,
        "class_name": "ParallelDiagonalModel",
        "d_model": spec.d_model,
        "feedforward_dim": spec.feedforward_dim,
        "num_blocks": spec.num_blocks,
        "rates": list(spec.rates),
        "readout_mode": spec.readout_mode,
        "token_injection": spec.token_injection,
        "topology": spec.topology,
        "aux_loss_weight": spec.aux_loss_weight,
        "parameter_count": parameter_count,
        "trainer": trainer_name,
        "checkpoints": checkpoints,
        "best_checkpoint": min(checkpoints, key=lambda checkpoint: float(checkpoint["val_loss"])),
        "final_checkpoint": checkpoints[-1],
        "final_training_task_loss": round(last_task_loss.item(), 6),
        "final_training_aux_loss": round(last_aux_loss.item(), 6),
        "final_training_total_loss": round(last_total_loss.item(), 6),
        "final_training_per_block_aux_losses": [
            round(float(loss), 6) for loss in last_per_block_aux_losses.detach().cpu().tolist()
        ],
        "wall_seconds": round(wall_seconds, 6),
        "mix_coefficients": model.mix_coefficients(),
    }
    append_log(
        args.log_path,
        {
            "stage": "variant_done",
            "seed": seed,
            "variant": variant_key,
            "final_checkpoint": result["final_checkpoint"],
            "final_training_task_loss": result["final_training_task_loss"],
            "final_training_aux_loss": result["final_training_aux_loss"],
            "final_training_total_loss": result["final_training_total_loss"],
            "wall_seconds": result["wall_seconds"],
        },
    )

    del trainer
    del optimizer
    del aux_heads
    del model
    del batch_iterator
    gc.collect()
    torch.cuda.empty_cache()
    return result


def summarize_results(
    *,
    per_seed_results: list[dict[str, object]],
    specs: dict[str, VariantSpec],
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {key: [] for key in specs}
    for result in per_seed_results:
        grouped[result["variant"]].append(result)

    summary: dict[str, object] = {}
    for key, runs in grouped.items():
        final_losses = [run["final_checkpoint"]["val_loss"] for run in runs]
        final_accuracies = [run["final_checkpoint"]["val_accuracy"] for run in runs]
        best_losses = [run["best_checkpoint"]["val_loss"] for run in runs]
        best_accuracies = [run["best_checkpoint"]["val_accuracy"] for run in runs]
        wall_seconds = [run["wall_seconds"] for run in runs]
        final_readout_weights = [run["final_checkpoint"]["readout_weights"] for run in runs]
        final_ablation_losses = [run["final_checkpoint"]["ablation_losses"] for run in runs]
        final_aux_losses = [run["final_checkpoint"]["per_block_aux_val_losses"] for run in runs]
        summary[key] = {
            "label": specs[key].label,
            "rates": list(specs[key].rates),
            "aux_loss_weight": specs[key].aux_loss_weight,
            "num_runs": len(runs),
            "mean_final_val_loss": mean_rounded(final_losses),
            "std_final_val_loss": std_rounded(final_losses),
            "mean_final_val_accuracy": mean_rounded(final_accuracies),
            "std_final_val_accuracy": std_rounded(final_accuracies),
            "mean_best_val_loss": mean_rounded(best_losses),
            "mean_best_val_accuracy": mean_rounded(best_accuracies),
            "mean_wall_seconds": mean_rounded(wall_seconds),
            "mean_final_readout_weights": mean_vector_rounded(final_readout_weights),
            "mean_final_ablation_losses": mean_vector_rounded(final_ablation_losses),
            "mean_final_per_block_aux_val_losses": optional_mean_vector_rounded(final_aux_losses),
            "runs": runs,
        }
    return summary


def comparison(summary_by_variant: dict[str, object]) -> dict[str, float]:
    return {
        "mean_final_val_loss_delta_B_minus_A": round(
            summary_by_variant["B_aux_loss"]["mean_final_val_loss"]
            - summary_by_variant["A_control"]["mean_final_val_loss"],
            6,
        ),
        "mean_final_val_accuracy_delta_B_minus_A": round(
            summary_by_variant["B_aux_loss"]["mean_final_val_accuracy"]
            - summary_by_variant["A_control"]["mean_final_val_accuracy"],
            6,
        ),
    }


def main() -> int:
    args = parse_args()
    if len(args.seeds) == 0:
        raise ValueError("At least one seed is required.")
    if args.training_steps < WARMUP_STEPS:
        raise ValueError(
            f"training_steps must be at least {WARMUP_STEPS} so trainer.capture() can warm up."
        )
    if args.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {args.eval_interval}.")
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}.")
    if args.eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {args.eval_batch_size}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for runs/local_learning_wikitext.py.")

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

    device = torch.device("cuda")
    corpus = load_corpus(
        train_path=args.train_path,
        val_path=args.val_path,
        context_size=CONTEXT_SIZE,
        eval_samples=4096,
    )
    val_inputs = corpus.val_inputs.to(device=device, dtype=torch.long)
    val_targets = corpus.val_targets.to(device=device, dtype=torch.long)

    specs = variant_specs()
    append_log(
        args.log_path,
        {
            "stage": "experiment_started",
            "seeds": args.seeds,
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "device": str(device),
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
            "vocab_size": corpus.vocab_size,
            "train_dataset_size": len(corpus.train_dataset),
            "val_examples": int(corpus.val_inputs.shape[0]),
            "trainer_by_variant": {
                "A_control": "GraphTrainer",
                "B_aux_loss": "LocalAuxGraphTrainer",
            },
        },
    )

    overall_started_at = perf_counter()
    per_seed_results: list[dict[str, object]] = []
    for seed in args.seeds:
        append_log(args.log_path, {"stage": "seed_started", "seed": seed})
        for variant_key, spec in specs.items():
            per_seed_results.append(
                train_single_variant(
                    seed=seed,
                    variant_key=variant_key,
                    spec=spec,
                    args=args,
                    corpus=corpus,
                    device=device,
                    val_inputs=val_inputs,
                    val_targets=val_targets,
                )
            )
        append_log(args.log_path, {"stage": "seed_done", "seed": seed})

    wall_seconds = perf_counter() - overall_started_at
    git_status_short = current_git_status_short()
    summary_by_variant = summarize_results(per_seed_results=per_seed_results, specs=specs)
    report = {
        "config": {
            "training_steps": args.training_steps,
            "eval_interval": args.eval_interval,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
            "warmup_steps": WARMUP_STEPS,
            "context_size": CONTEXT_SIZE,
            "dataset_vocab_size": corpus.vocab_size,
            "dataset": "wikitext-103-raw",
            "train_path": str(args.train_path),
            "val_path": str(args.val_path),
            "aux_loss_weight": AUX_LOSS_WEIGHT,
            "aux_block_indices": list(AUX_BLOCK_INDICES),
            "ablation_logit": ABLATION_LOGIT,
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
        "dataset": {
            "train_examples": len(corpus.train_dataset),
            "val_examples": int(corpus.val_inputs.shape[0]),
            "vocab_size": corpus.vocab_size,
        },
        "timing": {
            "overall_wall_seconds": round(wall_seconds, 6),
        },
        "variants": {key: asdict(spec) for key, spec in specs.items()},
        "per_seed_results": per_seed_results,
        "summary_by_variant": summary_by_variant,
        "comparison": comparison(summary_by_variant),
    }
    write_json(args.report_path, report)
    append_log(
        args.log_path,
        {
            "stage": "done",
            "report_path": str(args.report_path),
            "summary_by_variant": {
                key: {
                    "mean_final_val_loss": value["mean_final_val_loss"],
                    "mean_final_val_accuracy": value["mean_final_val_accuracy"],
                    "mean_final_readout_weights": value["mean_final_readout_weights"],
                    "mean_final_ablation_losses": value["mean_final_ablation_losses"],
                    "mean_final_per_block_aux_val_losses": value["mean_final_per_block_aux_val_losses"],
                }
                for key, value in summary_by_variant.items()
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
