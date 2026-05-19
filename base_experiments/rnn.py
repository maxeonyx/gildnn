from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import (
    FixedWindowCharDataset,
    generate_text,
    resolve_device,
    set_seed,
)


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 32
    train_characters: int = 100_000
    val_characters: int = 20_000
    batch_size: int = 256
    epochs: int = 13
    learning_rate: float = 0.003
    gradient_clip_norm: float = 1.0
    eval_batch_size: int = 512
    embedding_dim: int = 64
    hidden_dim: int = 368
    num_layers: int = 1
    sample_length: int = 320
    seed: int = 42
    check_batch_size: int = 8
    memorization_batch_size: int = 32
    memorization_steps: int = 200
    memorization_learning_rate: float = 0.02


@dataclass(frozen=True)
class DatasetSplit:
    train_dataset: FixedWindowCharDataset
    train_text: str
    val_text: str
    val_start: int
    train_inputs: Tensor
    train_targets: Tensor
    val_inputs: Tensor
    val_targets: Tensor


class TinyRnnCharModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        embedded = self.embedding(tokens)
        outputs, _ = self.rnn(embedded)
        return self.output(outputs[:, -1, :])


def replace_config(config: RunConfig, **changes: object) -> RunConfig:
    payload = asdict(config)
    payload.update(changes)
    return RunConfig(**payload)


def current_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def current_git_status_short() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def encode_windows(
    text: str,
    *,
    context_size: int,
    stoi: dict[str, int],
) -> tuple[Tensor, Tensor]:
    if len(text) <= context_size:
        raise ValueError("Text split must be longer than the context size.")
    encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    inputs = []
    targets = []
    for start in range(len(encoded) - context_size):
        stop = start + context_size
        inputs.append(encoded[start:stop])
        targets.append(encoded[stop])
    return torch.stack(inputs), torch.stack(targets)


def choose_validation_text(text: str, *, train_text: str, config: RunConfig) -> tuple[str, int]:
    start = config.train_characters
    stop = start + config.val_characters
    candidate = text[start:stop]
    if len(candidate) < config.val_characters:
        raise ValueError(
            f"Need validation slice of {config.val_characters} characters, got {len(candidate)}."
        )
    if set(candidate).issubset(set(train_text)):
        return candidate, start

    max_start = len(text) - config.val_characters
    for candidate_start in range(start + 1, max_start + 1):
        candidate_stop = candidate_start + config.val_characters
        candidate = text[candidate_start:candidate_stop]
        if set(candidate).issubset(set(train_text)):
            return candidate, candidate_start

    missing = sorted(set(text[start : max_start + config.val_characters]) - set(train_text))
    raise ValueError(
        "Could not find a validation slice whose characters are all present in the "
        f"training slice. Missing training vocabulary coverage for: {missing}"
    )


def build_fixed_length_split(text: str, *, config: RunConfig) -> DatasetSplit:
    required_characters = config.train_characters + config.val_characters
    if len(text) < required_characters:
        raise ValueError(
            f"Need at least {required_characters} characters, got {len(text)}."
        )

    train_text = text[: config.train_characters]
    val_text, val_start = choose_validation_text(
        text,
        train_text=train_text,
        config=config,
    )
    train_dataset = FixedWindowCharDataset(train_text, context_size=config.context_size)
    missing_val_chars = sorted(set(val_text) - set(train_text))
    if missing_val_chars:
        raise ValueError(
            f"Validation text contains characters absent from training text: {missing_val_chars}"
        )

    val_inputs, val_targets = encode_windows(
        val_text,
        context_size=config.context_size,
        stoi=train_dataset.stoi,
    )
    return DatasetSplit(
        train_dataset=train_dataset,
        train_text=train_text,
        val_text=val_text,
        val_start=val_start,
        train_inputs=train_dataset.inputs,
        train_targets=train_dataset.targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
    )


def make_model(
    config: RunConfig,
    *,
    vocab_size: int,
    device: torch.device,
) -> TinyRnnCharModel:
    return TinyRnnCharModel(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    ).to(device)


def batched_pairs(inputs: Tensor, targets: Tensor, *, batch_size: int):
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        yield inputs[start:stop], targets[start:stop]


def evaluate_model(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_examples = 0
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in batched_pairs(
            inputs,
            targets,
            batch_size=batch_size,
        ):
            logits = model(batch_inputs)
            total_loss += F.cross_entropy(
                logits,
                batch_targets,
                reduction="sum",
            ).item()
            total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
            total_examples += batch_targets.shape[0]
    if was_training:
        model.train()
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_inputs: Tensor,
    train_targets: Tensor,
    *,
    batch_size: int,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    permutation = torch.randperm(train_inputs.shape[0], device=train_inputs.device)
    total_examples = 0
    total_loss = 0.0
    total_correct = 0

    for start in range(0, permutation.shape[0], batch_size):
        batch_indices = permutation[start : start + batch_size]
        batch_inputs = train_inputs[batch_indices]
        batch_targets = train_targets[batch_indices]
        logits = model(batch_inputs)
        loss = F.cross_entropy(logits, batch_targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        batch_examples = batch_targets.shape[0]
        total_examples += batch_examples
        total_loss += loss.item() * batch_examples
        total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_model(
    model: nn.Module,
    *,
    train_inputs: Tensor,
    train_targets: Tensor,
    val_inputs: Tensor,
    val_targets: Tensor,
    config: RunConfig,
) -> tuple[list[dict[str, float | int]], dict[str, float], float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []
    started_at = time.perf_counter()

    initial_train_metrics = evaluate_model(
        model,
        train_inputs,
        train_targets,
        batch_size=config.eval_batch_size,
    )
    initial_val_metrics = evaluate_model(
        model,
        val_inputs,
        val_targets,
        batch_size=config.eval_batch_size,
    )
    history.append(
        {
            "epoch": 0,
            "train_loss": round(initial_train_metrics["loss"], 6),
            "train_accuracy": round(initial_train_metrics["accuracy"], 6),
            "val_loss": round(initial_val_metrics["loss"], 6),
            "val_accuracy": round(initial_val_metrics["accuracy"], 6),
        }
    )
    print(
        f"epoch=0 train_loss={initial_train_metrics['loss']:.4f} "
        f"val_loss={initial_val_metrics['loss']:.4f}"
    )

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_inputs,
            train_targets,
            batch_size=config.batch_size,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        val_metrics = evaluate_model(
            model,
            val_inputs,
            val_targets,
            batch_size=config.eval_batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "train_accuracy": round(train_metrics["accuracy"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
            }
        )
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )

    runtime_seconds = time.perf_counter() - started_at
    final_metrics = evaluate_model(
        model,
        val_inputs,
        val_targets,
        batch_size=config.eval_batch_size,
    )
    return history, final_metrics, runtime_seconds


def run_correctness_checks(
    split: DatasetSplit,
    *,
    config: RunConfig,
    device: torch.device,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    sample_text = split.train_text[: config.context_size]
    round_trip = split.train_dataset.decode(split.train_dataset.encode(sample_text))
    if round_trip != sample_text:
        raise RuntimeError("Dataset encode/decode round-trip failed.")
    checks.append(
        {
            "name": "dataset_round_trip",
            "ok": True,
            "sample": sample_text.replace("\n", "\\n"),
        }
    )

    set_seed(config.seed)
    shape_model = make_model(config, vocab_size=split.train_dataset.vocab_size, device=device)
    batch_inputs = split.train_inputs[: config.check_batch_size].to(device)
    batch_targets = split.train_targets[: config.check_batch_size].to(device)
    logits = shape_model(batch_inputs)
    expected_shape = [config.check_batch_size, split.train_dataset.vocab_size]
    actual_shape = list(logits.shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"Forward output shape mismatch: expected {expected_shape}, got {actual_shape}."
        )
    checks.append(
        {
            "name": "forward_shape",
            "ok": True,
            "expected_shape": expected_shape,
            "actual_shape": actual_shape,
        }
    )

    tiny_loss = F.cross_entropy(logits, batch_targets).item()
    expected_low = math.log(split.train_dataset.vocab_size) - 1.0
    expected_high = math.log(split.train_dataset.vocab_size) + 1.0
    if not expected_low <= tiny_loss <= expected_high:
        raise RuntimeError(
            f"Known tiny-batch loss {tiny_loss:.6f} fell outside expected range "
            f"[{expected_low:.6f}, {expected_high:.6f}]."
        )
    checks.append(
        {
            "name": "known_tiny_batch_loss",
            "ok": True,
            "loss": tiny_loss,
            "expected_range": [expected_low, expected_high],
        }
    )

    set_seed(config.seed)
    memorization_model = make_model(
        config,
        vocab_size=split.train_dataset.vocab_size,
        device=device,
    )
    memorize_inputs = split.train_inputs[: config.memorization_batch_size].to(device)
    memorize_targets = split.train_targets[: config.memorization_batch_size].to(device)
    optimizer = torch.optim.AdamW(
        memorization_model.parameters(),
        lr=config.memorization_learning_rate,
    )
    trace: list[dict[str, float | int]] = []
    final_loss = float("nan")
    final_accuracy = 0.0
    for step in range(config.memorization_steps + 1):
        logits = memorization_model(memorize_inputs)
        loss = F.cross_entropy(logits, memorize_targets)
        accuracy = (
            (logits.argmax(dim=1) == memorize_targets).float().mean().item()
        )
        if step in {0, config.memorization_steps} or step % 20 == 0:
            trace.append(
                {
                    "step": step,
                    "loss": round(loss.item(), 6),
                    "accuracy": round(accuracy, 6),
                }
            )
        final_loss = loss.item()
        final_accuracy = accuracy
        if step == config.memorization_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if not (final_loss < 0.05 and final_accuracy == 1.0):
        raise RuntimeError(
            "One-batch memorization check failed: expected loss < 0.05 and accuracy 1.0, "
            f"got loss={final_loss:.6f}, accuracy={final_accuracy:.6f}."
        )
    checks.append(
        {
            "name": "one_batch_memorization",
            "ok": True,
            "final_loss": final_loss,
            "final_accuracy": final_accuracy,
            "trace": trace,
        }
    )
    return checks


def make_default_paths(repo_root: Path, mode: str) -> tuple[Path, Path]:
    text_file = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    base_output_dir = repo_root / "base-experiments" / "rnn" / "artifacts"
    if mode == "full":
        return text_file, base_output_dir
    return text_file, base_output_dir / mode


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["checks-only", "tiny", "full"], default="full")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--train-characters", type=int)
    parser.add_argument("--val-characters", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--sample-length", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--memorization-steps", type=int)
    parser.add_argument("--memorization-learning-rate", type=float)
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def resolved_config(args: argparse.Namespace) -> RunConfig:
    config = RunConfig()
    if args.mode == "tiny":
        config = replace_config(
            config,
            train_characters=4_096,
            val_characters=1_024,
            epochs=2,
            sample_length=160,
        )
    if args.mode == "checks-only":
        config = replace_config(
            config,
            train_characters=2_048,
            val_characters=512,
            epochs=0,
            sample_length=0,
        )

    overrides = {
        "train_characters": args.train_characters,
        "val_characters": args.val_characters,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "sample_length": args.sample_length,
        "seed": args.seed,
        "memorization_steps": args.memorization_steps,
        "memorization_learning_rate": args.memorization_learning_rate,
    }
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    if clean_overrides:
        config = replace_config(config, **clean_overrides)
    return config


def main() -> None:
    args = parse_args()
    config = resolved_config(args)
    default_text_file, default_output_dir = make_default_paths(args.repo_root, args.mode)
    text_file = args.text_file or default_text_file
    output_dir = args.output_dir or default_output_dir

    set_seed(config.seed)
    device = resolve_device(args.device)
    raw_text = text_file.read_text(encoding="utf-8")
    split = build_fixed_length_split(raw_text, config=config)
    output_dir.mkdir(parents=True, exist_ok=True)

    git_status_short = current_git_status_short()
    parameter_count = count_parameters(
        make_model(config, vocab_size=split.train_dataset.vocab_size, device=device)
    )

    write_json(output_dir / "config.json", asdict(config))
    write_json(
        output_dir / "environment.json",
        {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": git_status_short == [],
            "git_status_short": git_status_short,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_is_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None,
        },
    )
    used_text = split.train_text + split.val_text
    write_json(
        output_dir / "corpus_summary.json",
        {
            "source_file": str(text_file),
            "source_total_characters": len(raw_text),
            "source_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "used_total_characters": len(used_text),
            "used_sha256": hashlib.sha256(used_text.encode("utf-8")).hexdigest(),
            "train_characters": len(split.train_text),
            "val_characters": len(split.val_text),
            "val_start": split.val_start,
            "val_stop": split.val_start + len(split.val_text),
            "train_windows": int(split.train_inputs.shape[0]),
            "val_windows": int(split.val_inputs.shape[0]),
            "vocab_size": split.train_dataset.vocab_size,
        },
    )
    write_json(
        output_dir / "model_summary.json",
        {
            "model_family": "ordinary_rnn",
            "rnn_type": "nn.RNN",
            "parameter_count": parameter_count,
            "context_size": config.context_size,
            "embedding_dim": config.embedding_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "nonlinearity": "tanh",
            "vocab_size": split.train_dataset.vocab_size,
            "hidden_state_handling": "reset_to_zero_per_forward_call_no_carry_between_windows_or_batches",
        },
    )

    checks = run_correctness_checks(split, config=config, device=device)
    write_json(output_dir / "correctness_checks.json", checks)
    print("correctness checks passed")

    if args.mode == "checks-only":
        return

    train_inputs = split.train_inputs.to(device)
    train_targets = split.train_targets.to(device)
    val_inputs = split.val_inputs.to(device)
    val_targets = split.val_targets.to(device)

    set_seed(config.seed)
    model = make_model(config, vocab_size=split.train_dataset.vocab_size, device=device)
    history, final_val_metrics, runtime_seconds = train_model(
        model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        val_inputs=val_inputs,
        val_targets=val_targets,
        config=config,
    )

    prompt = split.train_text[: config.context_size]
    sample = generate_text(
        model,
        split.train_dataset,
        prompt,
        length=config.sample_length,
        device=device,
    )
    torch.save(model.state_dict(), output_dir / "model_state.pt")
    write_json(output_dir / "training_history.json", history)
    best_epoch_record = min(history[1:] or history, key=lambda record: float(record["val_loss"]))
    write_json(
        output_dir / "final_metrics.json",
        {
            "parameter_count": parameter_count,
            "runtime_seconds": runtime_seconds,
            "final_val_loss": final_val_metrics["loss"],
            "final_val_accuracy": final_val_metrics["accuracy"],
            "best_val_loss": float(best_epoch_record["val_loss"]),
            "best_val_accuracy": float(best_epoch_record["val_accuracy"]),
            "best_epoch": int(best_epoch_record["epoch"]),
            "prompt": prompt.replace("\n", "\\n"),
        },
    )
    (output_dir / "sample.txt").write_text(sample, encoding="utf-8")


if __name__ == "__main__":
    main()
