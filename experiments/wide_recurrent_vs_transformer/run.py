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
from einops import rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F

from core.dataset import CorpusData, RandomWindowCharDataset, load_corpus
from core.fixed_window_char import generate_text, set_seed
from core.model import ResidualFeedForwardBlock, count_parameters
from core.training import batched_pairs, write_json

TRAIN_CHARACTERS = 100_000
VAL_CHARACTERS = 20_000
TEXT_PROMPT = "First Citizen:\nBefore we proceed"
UNKNOWN_CHAR_TOKEN = "<unk>"


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 32
    train_characters: int = TRAIN_CHARACTERS
    val_characters: int = VAL_CHARACTERS
    batch_size: int = 256
    epochs: int = 13
    learning_rates: tuple[float, ...] = (0.003, 0.001)
    gradient_clip_norm: float = 1.0
    eval_batch_size: int = 512
    d_model: int = 72
    feedforward_dim: int = 256
    num_heads: int = 4
    depth: int = 3
    sample_length: int = 320
    progression_sample_length: int = 200
    sample_checkpoints: tuple[int, ...] = (0, 1, 3, 7, 13)
    seed: int = 42
    check_batch_size: int = 8
    memorization_batch_size: int = 32
    memorization_steps: int = 120
    memorization_learning_rate: float = 0.02


class CausalSelfAttention(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: Float[Tensor, "batch context d_model"],
    ) -> Float[Tensor, "batch context d_model"]:
        query = rearrange(
            self.q_proj(x),
            "batch context (heads head_dim) -> batch heads context head_dim",
            heads=self.num_heads,
            head_dim=self.head_dim,
        )
        key = rearrange(
            self.k_proj(x),
            "batch context (heads head_dim) -> batch heads context head_dim",
            heads=self.num_heads,
            head_dim=self.head_dim,
        )
        value = rearrange(
            self.v_proj(x),
            "batch context (heads head_dim) -> batch heads context head_dim",
            heads=self.num_heads,
            head_dim=self.head_dim,
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        return self.out_proj(
            rearrange(attended, "batch heads context head_dim -> batch context (heads head_dim)")
        )


class TiedDepthTransformer(nn.Module):
    def __init__(self, *, vocab_size: int, config: RunConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_size, config.d_model)
        self.attention = CausalSelfAttention(d_model=config.d_model, num_heads=config.num_heads)
        self.feedforward = ResidualFeedForwardBlock(
            d_model=config.d_model,
            feedforward_dim=config.feedforward_dim,
        )
        self.attention_norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.depth)]
        )
        self.feedforward_norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.depth)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=True)

    def embedded_tokens(
        self,
        tokens: Int[Tensor, "batch context"],
    ) -> Float[Tensor, "batch context d_model"]:
        sequence_length = tokens.shape[1]
        if sequence_length != self.config.context_size:
            raise ValueError(
                f"Expected context length {self.config.context_size}, got {sequence_length}."
            )
        positions = torch.arange(sequence_length, device=tokens.device)
        return self.token_embedding(tokens) + repeat(
            self.position_embedding(positions),
            "context d_model -> batch context d_model",
            batch=tokens.shape[0],
        )

    def depth_logits(
        self,
        tokens: Int[Tensor, "batch context"],
    ) -> list[Float[Tensor, "batch vocab"]]:
        stream = self.embedded_tokens(tokens)
        logits_by_depth: list[Tensor] = []
        for depth_index in range(self.config.depth):
            stream = stream + self.attention(self.attention_norms[depth_index](stream))
            stream = stream + self.feedforward(self.feedforward_norms[depth_index](stream))
            final_state = self.final_norm(stream[:, -1, :])
            logits_by_depth.append(self.lm_head(final_state))
        return logits_by_depth

    def forward(
        self,
        tokens: Int[Tensor, "batch context"],
    ) -> Float[Tensor, "batch vocab"]:
        return self.depth_logits(tokens)[-1]


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


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(requested_device)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sanity", "full"), default="sanity")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--sample-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--feedforward-dim", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def replace_config(config: RunConfig, **changes: object) -> RunConfig:
    payload = asdict(config)
    payload.update(changes)
    return RunConfig(**payload)


def resolved_config(args: argparse.Namespace) -> RunConfig:
    config = RunConfig()
    overrides = {
        "learning_rates": tuple(args.learning_rates) if args.learning_rates is not None else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "sample_length": args.sample_length,
        "seed": args.seed,
        "d_model": args.d_model,
        "feedforward_dim": args.feedforward_dim,
        "num_heads": args.num_heads,
    }
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    if clean_overrides:
        config = replace_config(config, **clean_overrides)
    return config


def prepare_tinyshakespeare_split_files(
    *,
    repo_root: Path,
    config: RunConfig,
    destination_dir: Path,
) -> tuple[Path, Path, str]:
    source_path = repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    raw_text = source_path.read_text(encoding="utf-8")
    required_characters = config.train_characters + config.val_characters
    if len(raw_text) < required_characters:
        raise ValueError(f"Need at least {required_characters} characters, got {len(raw_text)}.")

    train_text = raw_text[: config.train_characters]
    val_text = raw_text[config.train_characters : config.train_characters + config.val_characters]
    missing_val_characters = sorted(set(val_text) - set(train_text))
    if len(missing_val_characters) > 0:
        raise ValueError(
            "Validation slice contains characters absent from the training slice: "
            f"{missing_val_characters}"
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    train_path = destination_dir / "tinyshakespeare_train.txt"
    val_path = destination_dir / "tinyshakespeare_val.txt"
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")
    return train_path, val_path, raw_text


def build_corpus(
    *,
    repo_root: Path,
    config: RunConfig,
    prepared_dir: Path,
) -> tuple[CorpusData, str, str]:
    train_path, val_path, raw_text = prepare_tinyshakespeare_split_files(
        repo_root=repo_root,
        config=config,
        destination_dir=prepared_dir,
    )
    eval_samples = config.val_characters - config.context_size
    corpus = load_corpus(
        train_path=train_path,
        val_path=val_path,
        context_size=config.context_size,
        eval_samples=eval_samples,
        seed=config.seed,
    )
    return corpus, raw_text, train_path.read_text(encoding="utf-8")


def effective_vocab_size(corpus: CorpusData) -> int:
    unknown_index = corpus.char_to_idx.get(UNKNOWN_CHAR_TOKEN)
    if unknown_index is None:
        return corpus.vocab_size
    train_dataset = corpus.train_dataset
    if not isinstance(train_dataset, RandomWindowCharDataset):
        raise TypeError("Expected RandomWindowCharDataset for effective_vocab_size.")
    if (train_dataset.encoded_corpus == unknown_index).any():
        raise ValueError("Training corpus unexpectedly uses the unknown token index.")
    if (corpus.val_inputs == unknown_index).any() or (corpus.val_targets == unknown_index).any():
        raise ValueError("Validation corpus unexpectedly uses the unknown token index.")
    if unknown_index != corpus.vocab_size - 1:
        raise ValueError(
            f"Expected unknown token index {corpus.vocab_size - 1}, got {unknown_index}."
        )
    return corpus.vocab_size - 1


def dataset_tensors(
    dataset: RandomWindowCharDataset,
) -> tuple[Int[Tensor, "examples context"], Int[Tensor, "examples"]]:
    encoded = dataset.encoded_corpus
    context_size = dataset.context_size
    inputs = encoded.unfold(0, context_size, 1)[:-1].clone()
    targets = encoded[context_size:].clone()
    return inputs, targets


def corpus_summary(
    *,
    repo_root: Path,
    raw_text: str,
    train_text: str,
    corpus: CorpusData,
    config: RunConfig,
) -> dict[str, object]:
    used_text = raw_text[: config.train_characters + config.val_characters]
    return {
        "source_file": str(repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"),
        "source_total_characters": len(raw_text),
        "source_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "used_total_characters": len(used_text),
        "used_sha256": hashlib.sha256(used_text.encode("utf-8")).hexdigest(),
        "train_characters": config.train_characters,
        "val_characters": config.val_characters,
        "val_start": config.train_characters,
        "val_stop": config.train_characters + config.val_characters,
        "train_windows": len(corpus.train_dataset),
        "val_windows": int(corpus.val_inputs.shape[0]),
        "vocab_size": corpus.vocab_size,
        "prompt": train_text[: config.context_size].replace("\n", "\\n"),
    }


def capture_sample(
    model: nn.Module,
    *,
    corpus: CorpusData,
    prompt: str,
    length: int,
    device: torch.device,
) -> str:
    class DatasetAdapter:
        def __init__(self, corpus_data: CorpusData, context_size: int) -> None:
            self.context_size = context_size
            self._corpus = corpus_data

        def encode(self, text: str) -> list[int]:
            return self._corpus.encode_text(text)

        def decode(self, tokens: list[int]) -> str:
            return self._corpus.decode_tokens(tokens)

    was_training = model.training
    model.eval()
    with torch.inference_mode():
        sample = generate_text(
            model,
            DatasetAdapter(corpus, corpus.context_size),
            prompt,
            length=length,
            device=device,
        )
    if was_training:
        model.train()
    return sample


def evaluate_depth_losses(
    model: TiedDepthTransformer,
    inputs: Int[Tensor, "examples context"],
    targets: Int[Tensor, "examples"],
    *,
    batch_size: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_examples = 0
    total_losses = [0.0 for _ in range(model.config.depth)]
    with torch.inference_mode():
        for batch_inputs, batch_targets in batched_pairs(inputs, targets, batch_size=batch_size):
            logits_by_depth = model.depth_logits(batch_inputs)
            batch_examples = batch_targets.shape[0]
            total_examples += batch_examples
            for depth_index, logits in enumerate(logits_by_depth, start=1):
                total_losses[depth_index - 1] += F.cross_entropy(
                    logits,
                    batch_targets,
                    reduction="sum",
                ).item()
    if was_training:
        model.train()
    return {
        f"depth_{depth_index}": round(total_loss / total_examples, 6)
        for depth_index, total_loss in enumerate(total_losses, start=1)
    }


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
    with torch.inference_mode():
        for batch_inputs, batch_targets in batched_pairs(inputs, targets, batch_size=batch_size):
            logits = model(batch_inputs)
            total_loss += F.cross_entropy(logits, batch_targets, reduction="sum").item()
            total_correct += (logits.argmax(dim=1) == batch_targets).sum().item()
            total_examples += batch_targets.shape[0]
    if was_training:
        model.train()
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def run_shape_check(
    *,
    model: TiedDepthTransformer,
    corpus: CorpusData,
    device: torch.device,
    config: RunConfig,
    vocab_size: int,
) -> dict[str, object]:
    batch_inputs = corpus.val_inputs[: config.check_batch_size].to(device)
    logits_by_depth = model.depth_logits(batch_inputs)
    expected_shape = [config.check_batch_size, vocab_size]
    actual_shapes = [list(logits.shape) for logits in logits_by_depth]
    if any(shape != expected_shape for shape in actual_shapes):
        raise RuntimeError(
            f"Forward output shape mismatch: expected every depth to produce {expected_shape}, got {actual_shapes}."
        )
    return {
        "name": "forward_shape",
        "ok": True,
        "expected_shape": expected_shape,
        "actual_shapes": actual_shapes,
    }


def run_one_batch_memorization_check(
    *,
    model: TiedDepthTransformer,
    train_inputs: Int[Tensor, "examples context"],
    train_targets: Int[Tensor, "examples"],
    device: torch.device,
    config: RunConfig,
) -> dict[str, object]:
    memorize_inputs = train_inputs[: config.memorization_batch_size].to(device)
    memorize_targets = train_targets[: config.memorization_batch_size].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.memorization_learning_rate)
    trace: list[dict[str, float | int]] = []
    final_loss = float("nan")
    final_accuracy = 0.0

    for step in range(config.memorization_steps + 1):
        logits = model(memorize_inputs)
        loss = F.cross_entropy(logits, memorize_targets)
        accuracy = (logits.argmax(dim=1) == memorize_targets).float().mean().item()
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

    return {
        "name": "one_batch_memorization",
        "ok": True,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "trace": trace,
    }


def run_sanity_checks(
    *,
    corpus: CorpusData,
    train_inputs: Int[Tensor, "examples context"],
    train_targets: Int[Tensor, "examples"],
    device: torch.device,
    config: RunConfig,
    vocab_size: int,
) -> dict[str, object]:
    set_seed(config.seed)
    shape_model = TiedDepthTransformer(vocab_size=vocab_size, config=config).to(device)
    shape_check = run_shape_check(
        model=shape_model,
        corpus=corpus,
        device=device,
        config=config,
        vocab_size=vocab_size,
    )

    set_seed(config.seed)
    memorization_model = TiedDepthTransformer(vocab_size=vocab_size, config=config).to(device)
    memorization_check = run_one_batch_memorization_check(
        model=memorization_model,
        train_inputs=train_inputs,
        train_targets=train_targets,
        device=device,
        config=config,
    )

    return {
        "shape_check": shape_check,
        "memorization_check": memorization_check,
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


def train_full_run(
    *,
    corpus: CorpusData,
    train_inputs: Int[Tensor, "examples context"],
    train_targets: Int[Tensor, "examples"],
    val_inputs: Int[Tensor, "examples context"],
    val_targets: Int[Tensor, "examples"],
    prompt: str,
    learning_rate: float,
    device: torch.device,
    config: RunConfig,
    vocab_size: int,
) -> dict[str, object]:
    set_seed(config.seed)
    model = TiedDepthTransformer(vocab_size=vocab_size, config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    started_at = time.perf_counter()
    history: list[dict[str, object]] = []
    progression_samples: list[dict[str, object]] = []
    sample_epochs = {epoch for epoch in config.sample_checkpoints if epoch <= config.epochs}
    parameter_count = count_parameters(model)

    for epoch in range(0, config.epochs + 1):
        if epoch == 0:
            train_metrics = evaluate_model(
                model,
                train_inputs,
                train_targets,
                batch_size=config.eval_batch_size,
            )
        else:
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
        depth_losses = evaluate_depth_losses(
            model,
            val_inputs,
            val_targets,
            batch_size=config.eval_batch_size,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "depth_val_losses": depth_losses,
        }
        history.append(epoch_record)
        print(
            f"lr={learning_rate:.4f} epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} depth_losses={depth_losses}",
            flush=True,
        )
        if epoch in sample_epochs:
            progression_samples.append(
                {
                    "epoch": epoch,
                    "val_loss": round(val_metrics["loss"], 6),
                    "depth_val_losses": depth_losses,
                    "sample": capture_sample(
                        model,
                        corpus=corpus,
                        prompt=prompt,
                        length=config.progression_sample_length,
                        device=device,
                    ),
                }
            )

    runtime_seconds = time.perf_counter() - started_at
    best_epoch_record = min(history[1:] or history, key=lambda record: float(record["val_loss"]))
    sample = capture_sample(
        model,
        corpus=corpus,
        prompt=prompt,
        length=config.sample_length,
        device=device,
    )
    return {
        "learning_rate": learning_rate,
        "parameter_count": parameter_count,
        "runtime_seconds": runtime_seconds,
        "history": history,
        "best_epoch": int(best_epoch_record["epoch"]),
        "best_val_loss": float(best_epoch_record["val_loss"]),
        "best_val_accuracy": float(best_epoch_record["val_accuracy"]),
        "best_depth_val_losses": best_epoch_record["depth_val_losses"],
        "final_val_loss": float(history[-1]["val_loss"]),
        "final_val_accuracy": float(history[-1]["val_accuracy"]),
        "final_depth_val_losses": history[-1]["depth_val_losses"],
        "sample": sample,
        "progression_samples": progression_samples,
    }


def write_run_outputs(
    *,
    output_dir: Path,
    config: RunConfig,
    corpus: CorpusData,
    raw_text: str,
    train_text: str,
    device: torch.device,
    sanity_checks: dict[str, object],
    full_results: list[dict[str, object]] | None,
    vocab_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    git_status_short = current_git_status_short()
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
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
    )
    write_json(
        output_dir / "corpus_summary.json",
        corpus_summary(
            repo_root=Path(__file__).resolve().parents[2],
            raw_text=raw_text,
            train_text=train_text,
            corpus=corpus,
            config=config,
        ),
    )
    write_json(
        output_dir / "model_summary.json",
        {
            "model_family": "tied_depth_transformer",
            "parameter_count": count_parameters(TiedDepthTransformer(vocab_size=vocab_size, config=config)),
            "context_size": config.context_size,
            "d_model": config.d_model,
            "feedforward_dim": config.feedforward_dim,
            "num_heads": config.num_heads,
            "depth": config.depth,
            "shared_modules": ["attention", "feedforward"],
            "separate_modules_per_depth": ["attention_norm", "feedforward_norm"],
            "vocab_size": vocab_size,
            "raw_corpus_vocab_size": corpus.vocab_size,
        },
    )
    write_json(output_dir / "sanity_checks.json", sanity_checks)
    if full_results is None:
        return

    for run_result in full_results:
        learning_rate_key = str(run_result["learning_rate"]).replace(".", "p")
        run_dir = output_dir / f"lr_{learning_rate_key}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "training_history.json", run_result["history"])
        write_json(
            run_dir / "final_metrics.json",
            {
                "learning_rate": run_result["learning_rate"],
                "parameter_count": run_result["parameter_count"],
                "runtime_seconds": run_result["runtime_seconds"],
                "final_val_loss": run_result["final_val_loss"],
                "final_val_accuracy": run_result["final_val_accuracy"],
                "final_depth_val_losses": run_result["final_depth_val_losses"],
                "best_val_loss": run_result["best_val_loss"],
                "best_val_accuracy": run_result["best_val_accuracy"],
                "best_depth_val_losses": run_result["best_depth_val_losses"],
                "best_epoch": run_result["best_epoch"],
                "prompt": TEXT_PROMPT.replace("\n", "\\n"),
            },
        )
        write_json(
            run_dir / "progression_samples.json",
            {
                "prompt": TEXT_PROMPT.replace("\n", "\\n"),
                "sample_length": config.progression_sample_length,
                "checkpoints": run_result["progression_samples"],
            },
        )
        (run_dir / "sample.txt").write_text(str(run_result["sample"]), encoding="utf-8")

    best_result = min(full_results, key=lambda result: float(result["best_val_loss"]))
    write_json(
        output_dir / "summary.json",
        {
            "best_learning_rate": best_result["learning_rate"],
            "best_val_loss": best_result["best_val_loss"],
            "runs": [
                {
                    "learning_rate": result["learning_rate"],
                    "best_val_loss": result["best_val_loss"],
                    "best_epoch": result["best_epoch"],
                    "runtime_seconds": result["runtime_seconds"],
                    "parameter_count": result["parameter_count"],
                    "best_depth_val_losses": result["best_depth_val_losses"],
                }
                for result in full_results
            ],
        },
    )


def main() -> int:
    args = parse_args()
    config = resolved_config(args)
    device = resolve_device(args.device)
    repo_root = args.repo_root
    default_output_dir = repo_root / "experiments" / "wide_recurrent_vs_transformer" / "artifacts"
    output_dir = args.output_dir or default_output_dir
    prepared_dir = repo_root / "experiments" / "wide_recurrent_vs_transformer" / "prepared.ignore"

    set_seed(config.seed)
    corpus, raw_text, train_text = build_corpus(
        repo_root=repo_root,
        config=config,
        prepared_dir=prepared_dir,
    )
    vocab_size = effective_vocab_size(corpus)
    train_inputs_cpu, train_targets_cpu = dataset_tensors(corpus.train_dataset)
    sanity_checks = run_sanity_checks(
        corpus=corpus,
        train_inputs=train_inputs_cpu,
        train_targets=train_targets_cpu,
        device=device,
        config=config,
        vocab_size=vocab_size,
    )

    full_results: list[dict[str, object]] | None = None
    if args.mode == "full":
        train_inputs = train_inputs_cpu.to(device)
        train_targets = train_targets_cpu.to(device)
        val_inputs = corpus.val_inputs.to(device)
        val_targets = corpus.val_targets.to(device)
        prompt = train_text[: config.context_size]
        full_results = [
            train_full_run(
                corpus=corpus,
                train_inputs=train_inputs,
                train_targets=train_targets,
                val_inputs=val_inputs,
                val_targets=val_targets,
                prompt=prompt,
                learning_rate=learning_rate,
                device=device,
                config=config,
                vocab_size=vocab_size,
            )
            for learning_rate in config.learning_rates
        ]

    write_run_outputs(
        output_dir=output_dir,
        config=config,
        corpus=corpus,
        raw_text=raw_text,
        train_text=train_text,
        device=device,
        sanity_checks=sanity_checks,
        full_results=full_results,
        vocab_size=vocab_size,
    )

    summary = {
        "mode": args.mode,
        "device": str(device),
        "parameter_count": count_parameters(TiedDepthTransformer(vocab_size=vocab_size, config=config)),
        "effective_vocab_size": vocab_size,
        "raw_corpus_vocab_size": corpus.vocab_size,
        "sanity_checks": sanity_checks,
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
