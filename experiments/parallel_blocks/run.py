from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import (
    FixedWindowCharDataset,
    _choose_validation_text,
    _encode_windows,
    resolve_device,
    set_seed,
)
from core.model import MixAdd, ResidualFeedForwardBlock, TemporalWindowAttention, count_parameters
from core.training import current_git_sha, current_git_status_short, evaluate_model, fixed_step_indices, write_json


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 32
    d_model: int = 128
    feedforward_dim: int = 512
    temporal_window: int = 4
    num_heads: int = 4
    batch_size: int = 64
    eval_batch_size: int = 512
    learning_rate: float = 3e-4
    training_steps: int = 2_000
    eval_interval: int = 500
    seed: int = 42
    rates: tuple[int, ...] = (1, 2, 4, 8)
    train_characters: int = 100_000
    val_characters: int = 20_000
    token_mix_init: float = 0.5
    block_mix_init: float = 0.9
    time_mix_init: float = 0.9


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "experiments" / "parallel_blocks" / "results.json",
    )
    return parser.parse_args()


def load_tiny_shakespeare(
    *,
    context_size: int,
    train_characters: int,
    val_characters: int,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], int]:
    repo_root = Path(__file__).resolve().parents[2]
    text_path = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    raw_text = text_path.read_text(encoding="utf-8")
    required_characters = train_characters + val_characters
    if len(raw_text) < required_characters:
        raise ValueError(f"Need at least {required_characters} characters, got {len(raw_text)}.")

    train_text = raw_text[:train_characters]
    val_text = _choose_validation_text(
        raw_text,
        train_text=train_text,
        val_characters=val_characters,
    )
    train_dataset = FixedWindowCharDataset(train_text, context_size=context_size)
    val_inputs, val_targets = _encode_windows(
        val_text,
        context_size=context_size,
        stoi=train_dataset.stoi,
    )
    return (
        (train_dataset.inputs, train_dataset.targets),
        (val_inputs, val_targets),
        train_dataset.vocab_size,
    )


class MultiRateComparisonModel(nn.Module):
    def __init__(self, *, vocab_size: int, config: RunConfig) -> None:
        super().__init__()
        self.context_size = config.context_size
        self.d_model = config.d_model
        self.temporal_window = config.temporal_window
        self.rates = config.rates
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_size, config.d_model)
        self.mix_token = MixAdd(init=config.token_mix_init)
        self.mix_time = MixAdd(init=config.time_mix_init)
        self.temporal_attention = TemporalWindowAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
        )
        self.blocks = nn.ModuleList(
            [
                ResidualFeedForwardBlock(
                    d_model=config.d_model,
                    feedforward_dim=config.feedforward_dim,
                )
                for _ in config.rates
            ]
        )
        self.output = nn.Linear(config.d_model, vocab_size)

    def embedded_tokens(self, tokens: Tensor) -> Tensor:
        sequence_length = tokens.shape[1]
        if sequence_length != self.context_size:
            raise ValueError(f"Expected context length {self.context_size}, got {sequence_length}.")
        positions = torch.arange(sequence_length, device=tokens.device)
        return self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)

    def apply_blocks(
        self,
        stream: Tensor,
        *,
        time_index: int,
        cached_outputs: list[Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        raise NotImplementedError

    def forward(self, tokens: Tensor) -> Tensor:
        embeddings = self.embedded_tokens(tokens)
        batch_size = tokens.shape[0]
        stream = torch.zeros(batch_size, self.d_model, device=tokens.device, dtype=embeddings.dtype)
        history: list[Tensor] = []
        cached_outputs = [torch.zeros_like(stream) for _ in self.blocks]

        for time_index in range(self.context_size):
            stream = self.mix_token(stream, embeddings[:, time_index, :])
            past_states = history[-self.temporal_window :]
            if len(past_states) == 0:
                stacked_past = torch.empty(
                    batch_size,
                    0,
                    self.d_model,
                    device=tokens.device,
                    dtype=embeddings.dtype,
                )
            else:
                stacked_past = torch.stack(past_states, dim=1)
            temporal_context, _ = self.temporal_attention(stream, stacked_past, capture_weights=False)
            stream, cached_outputs = self.apply_blocks(
                stream,
                time_index=time_index,
                cached_outputs=cached_outputs,
            )
            stream = self.mix_time(stream, temporal_context)
            history.append(stream)

        return self.output(stream)


class SequentialBlocksModel(MultiRateComparisonModel):
    def __init__(self, *, vocab_size: int, config: RunConfig) -> None:
        super().__init__(vocab_size=vocab_size, config=config)
        self.block_mixes = nn.ModuleList([MixAdd(init=config.block_mix_init) for _ in config.rates])

    def apply_blocks(
        self,
        stream: Tensor,
        *,
        time_index: int,
        cached_outputs: list[Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        for block_index, (block, block_mix, rate) in enumerate(
            zip(self.blocks, self.block_mixes, self.rates, strict=True)
        ):
            if time_index % rate == 0:
                cached_outputs[block_index] = block(stream)
            stream = block_mix(stream, cached_outputs[block_index])
        return stream, cached_outputs


class ParallelBlocksModel(MultiRateComparisonModel):
    def __init__(self, *, vocab_size: int, config: RunConfig) -> None:
        super().__init__(vocab_size=vocab_size, config=config)
        self.parallel_mix = MixAdd(init=config.block_mix_init)

    def apply_blocks(
        self,
        stream: Tensor,
        *,
        time_index: int,
        cached_outputs: list[Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        shared_stream = stream
        deltas: list[Tensor] = []
        for block_index, (block, rate) in enumerate(zip(self.blocks, self.rates, strict=True)):
            if time_index % rate == 0:
                cached_outputs[block_index] = block(shared_stream)
            deltas.append(cached_outputs[block_index])
        combined_delta = torch.stack(deltas, dim=0).mean(dim=0)
        stream = self.parallel_mix(stream, combined_delta)
        return stream, cached_outputs


def copy_shared_initialization(source: SequentialBlocksModel, target: ParallelBlocksModel) -> None:
    target.token_embedding.load_state_dict(source.token_embedding.state_dict())
    target.position_embedding.load_state_dict(source.position_embedding.state_dict())
    target.mix_token.load_state_dict(source.mix_token.state_dict())
    target.mix_time.load_state_dict(source.mix_time.state_dict())
    target.temporal_attention.load_state_dict(source.temporal_attention.state_dict())
    target.blocks.load_state_dict(source.blocks.state_dict())
    target.output.load_state_dict(source.output.state_dict())
    target.parallel_mix.load_state_dict(source.block_mixes[0].state_dict())


def train_variant(
    *,
    name: str,
    model: nn.Module,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    batch_schedule: list[Tensor],
    config: RunConfig,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    checkpoints: list[dict[str, float | int]] = []

    for step, batch_indices in enumerate(batch_schedule, start=1):
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % config.eval_interval == 0 or step == config.training_steps:
            metrics = evaluate_model(
                model,
                val_inputs,
                val_targets,
                batch_size=config.eval_batch_size,
            )
            checkpoint = {
                "step": step,
                "val_loss": round(metrics["loss"], 6),
                "val_accuracy": round(metrics["accuracy"], 6),
            }
            checkpoints.append(checkpoint)
            print({"variant": name, "checkpoint": checkpoint}, flush=True)

    final_checkpoint = checkpoints[-1]
    return {
        "parameter_count": count_parameters(model),
        "final_val_loss": final_checkpoint["val_loss"],
        "final_val_accuracy": final_checkpoint["val_accuracy"],
        "checkpoints": checkpoints,
    }


def main() -> None:
    args = parse_args()
    config = RunConfig()
    set_seed(config.seed)
    device = resolve_device(args.device)
    train_data, val_data, vocab_size = load_tiny_shakespeare(
        context_size=config.context_size,
        train_characters=config.train_characters,
        val_characters=config.val_characters,
    )
    train_inputs, train_targets = train_data
    val_inputs, val_targets = val_data
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    val_inputs = val_inputs.to(device)
    val_targets = val_targets.to(device)
    batch_schedule = fixed_step_indices(
        train_inputs.shape[0],
        steps=config.training_steps,
        batch_size=config.batch_size,
        seed=config.seed,
        device=train_inputs.device,
    )

    sequential_model = SequentialBlocksModel(vocab_size=vocab_size, config=config).to(device)
    parallel_model = ParallelBlocksModel(vocab_size=vocab_size, config=config).to(device)
    copy_shared_initialization(sequential_model, parallel_model)

    sequential_results = train_variant(
        name="sequential",
        model=sequential_model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        batch_schedule=batch_schedule,
        config=config,
    )
    parallel_results = train_variant(
        name="parallel",
        model=parallel_model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        batch_schedule=batch_schedule,
        config=config,
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "config": asdict(config),
        "environment": {
            "git_sha": current_git_sha(),
            "git_status_short": current_git_status_short(),
            "device": str(device),
            "torch_version": torch.__version__,
        },
        "sequential": sequential_results,
        "parallel": parallel_results,
        "comparison": {
            "final_val_loss_delta": round(
                parallel_results["final_val_loss"] - sequential_results["final_val_loss"],
                6,
            )
        },
    }
    write_json(output_path, results)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
