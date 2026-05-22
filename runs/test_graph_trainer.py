from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor
from torch.nn import functional as F

from core.fixed_window_char import load_dataset, set_seed
from core.model import ResidualStreamTimeMixAddCharModel, ResidualStreamTimeMixAddConfig
from core.training import GraphTrainer, capturable_adamw, evaluate_model, fixed_step_indices


@dataclass(frozen=True)
class TrainingResult:
    train_loss: float
    val_loss: float
    elapsed_seconds: float


def build_model(*, vocab_size: int, device: torch.device) -> ResidualStreamTimeMixAddCharModel:
    config = ResidualStreamTimeMixAddConfig(
        context_size=32,
        d_model=128,
        feedforward_dim=256,
        temporal_window=4,
        num_heads=4,
    )
    return ResidualStreamTimeMixAddCharModel(vocab_size=vocab_size, config=config).to(device)


def materialize_batches(
    inputs: Tensor,
    targets: Tensor,
    schedule: list[Tensor],
) -> list[tuple[Tensor, Tensor]]:
    return [(inputs[indices], targets[indices]) for indices in schedule]


def compute_loss(model: torch.nn.Module, batch_inputs: Tensor, batch_targets: Tensor) -> float:
    logits = model(batch_inputs)
    return F.cross_entropy(logits, batch_targets).item()


def run_eager_training(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    warmup_batches: list[tuple[Tensor, Tensor]],
    train_batches: list[tuple[Tensor, Tensor]],
    val_inputs: Tensor,
    val_targets: Tensor,
) -> TrainingResult:
    device = next(model.parameters()).device
    for batch_inputs, batch_targets in warmup_batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = perf_counter()
    for batch_inputs, batch_targets in train_batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = perf_counter() - started_at

    train_loss = compute_loss(model, *train_batches[-1])
    val_loss = evaluate_model(model, val_inputs, val_targets, batch_size=512)["loss"]
    return TrainingResult(train_loss=train_loss, val_loss=val_loss, elapsed_seconds=elapsed_seconds)


def run_graph_training(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    warmup_batches: list[tuple[Tensor, Tensor]],
    train_batches: list[tuple[Tensor, Tensor]],
    val_inputs: Tensor,
    val_targets: Tensor,
) -> TrainingResult:
    first_inputs, _ = warmup_batches[0]
    trainer = GraphTrainer(
        model,
        optimizer,
        batch_size=first_inputs.shape[0],
        seq_len=first_inputs.shape[1],
    )
    trainer.capture(warmup_batches)
    trainer.synchronize()

    started_at = perf_counter()
    last_loss: Tensor | None = None
    for batch_inputs, batch_targets in train_batches:
        last_loss = trainer.step(batch_inputs, batch_targets)
    trainer.synchronize()
    elapsed_seconds = perf_counter() - started_at

    if last_loss is None:
        raise RuntimeError("Expected at least one graphed training batch.")

    val_loss = evaluate_model(model, val_inputs, val_targets, batch_size=512)["loss"]
    return TrainingResult(train_loss=last_loss.item(), val_loss=val_loss, elapsed_seconds=elapsed_seconds)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GraphTrainer verification.")

    device = torch.device("cuda")
    seed = 1234
    warmup_steps = 3
    train_steps = 100
    total_steps = warmup_steps + train_steps

    set_seed(seed)
    (train_inputs, train_targets), (val_inputs, val_targets), vocab_size = load_dataset(context_size=32)
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    val_inputs = val_inputs.to(device)
    val_targets = val_targets.to(device)

    schedule = fixed_step_indices(
        train_inputs.shape[0],
        steps=total_steps,
        batch_size=64,
        seed=seed,
        device=device,
    )
    all_batches = materialize_batches(train_inputs, train_targets, schedule)
    warmup_batches = all_batches[:warmup_steps]
    train_batches = all_batches[warmup_steps:]

    set_seed(seed)
    reference_model = build_model(vocab_size=vocab_size, device=device)
    initial_state = reference_model.state_dict()

    eager_model = build_model(vocab_size=vocab_size, device=device)
    eager_model.load_state_dict(initial_state)
    eager_optimizer = capturable_adamw(eager_model, lr=3e-3, weight_decay=0.01)

    graph_model = build_model(vocab_size=vocab_size, device=device)
    graph_model.load_state_dict(initial_state)
    graph_optimizer = capturable_adamw(graph_model, lr=3e-3, weight_decay=0.01)

    eager_result = run_eager_training(
        model=eager_model,
        optimizer=eager_optimizer,
        warmup_batches=warmup_batches,
        train_batches=train_batches,
        val_inputs=val_inputs,
        val_targets=val_targets,
    )
    graph_result = run_graph_training(
        model=graph_model,
        optimizer=graph_optimizer,
        warmup_batches=warmup_batches,
        train_batches=train_batches,
        val_inputs=val_inputs,
        val_targets=val_targets,
    )

    train_loss_delta = abs(graph_result.train_loss - eager_result.train_loss)
    if train_loss_delta > 0.05:
        raise AssertionError(
            f"Final train loss delta {train_loss_delta:.6f} exceeded tolerance 0.05 "
            f"(graph={graph_result.train_loss:.6f}, eager={eager_result.train_loss:.6f})."
        )

    speedup = eager_result.elapsed_seconds / graph_result.elapsed_seconds
    print(f"eager_train_loss={eager_result.train_loss:.6f}")
    print(f"graph_train_loss={graph_result.train_loss:.6f}")
    print(f"eager_val_loss={eager_result.val_loss:.6f}")
    print(f"graph_val_loss={graph_result.val_loss:.6f}")
    print(f"train_loss_delta={train_loss_delta:.6f}")
    print(f"eager_seconds={eager_result.elapsed_seconds:.3f}")
    print(f"graph_seconds={graph_result.elapsed_seconds:.3f}")
    print(f"speedup={speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
