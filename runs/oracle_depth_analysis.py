from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    raise RuntimeError(
        "Run from the repo root with `.\\.venv\\Scripts\\python.exe -m runs.oracle_depth_analysis ...` so `core` imports resolve cleanly."
    )

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import load_dataset, set_seed
from core.run_utils import append_log, prepare_output_paths, redirect_sanity_check_paths, register_active_lock, resolve_device
from core.tied_readout import CausalSelfAttention, FeedForward, normalize_hidden, tied_logits

CONTEXT_SIZE = 128
RECURRENT_ITERATIONS = 4
TRAINING_STEPS = 20_000
SANITY_CHECK_STEPS = 10
TRAIN_BATCH_SIZE = 128
SANITY_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 2_048
LEARNING_RATE = 3e-4
DEFAULT_SEED = 42
TRAIN_CHARACTERS = 900_000
VAL_CHARACTERS = 20_000
DELTA = 0.01
PRINT_INTERVAL = 1_000


@dataclass(frozen=True)
class ExperimentConfig:
    context_size: int = CONTEXT_SIZE
    d_model: int = 256
    n_heads: int = 4
    ff_dim: int = 1_024
    recurrent_iterations: int = RECURRENT_ITERATIONS
    temperature: float = 0.07
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainingSummary:
    steps: int
    batch_size: int
    final_train_loss: float
    wall_seconds: float


@dataclass(frozen=True)
class OracleMetrics:
    mean_loss_per_depth: tuple[float, ...]
    oracle_best_loss: float
    oracle_depth_histogram: tuple[int, ...]
    no_regret_depth_histogram: tuple[int, ...]
    mean_no_regret_depth: float
    oracle_speedup_delta_0_01: float
    fraction_harmed_by_full_depth: float
    total_examples: int


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "tinyshakespeare" / "artifacts" / "oracle_depth_analysis"
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--steps", type=positive_int, default=TRAINING_STEPS)
    parser.add_argument("--batch-size", type=positive_int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=positive_int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--train-characters", type=positive_int, default=TRAIN_CHARACTERS)
    parser.add_argument("--val-characters", type=positive_int, default=VAL_CHARACTERS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--recurrent-iterations", type=positive_int, default=RECURRENT_ITERATIONS)
    parser.add_argument("--sanity-check-only", action="store_true")
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--log-path", type=Path, default=artifact_dir / "run.jsonl")
    return parser.parse_args()


def resolve_training_steps(args: argparse.Namespace) -> int:
    return SANITY_CHECK_STEPS if args.sanity_check_only else args.steps


def resolve_batch_size(args: argparse.Namespace) -> int:
    return SANITY_BATCH_SIZE if args.sanity_check_only else args.batch_size


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def dataset_to_device(dataset: tuple[Tensor, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    inputs, targets = dataset
    return inputs.to(device), targets.to(device)


def sample_batch(dataset: tuple[Tensor, Tensor], *, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
    inputs, targets = dataset
    indices = torch.randint(0, targets.shape[0], (batch_size,), device=device)
    return inputs[indices], targets[indices]


class SharedRecurrentCore(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, ff_dim: int, dropout: float, iterations: int) -> None:
        super().__init__()
        self.iterations = iterations
        self.attn_norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(iterations))
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(iterations))
        self.ffn = FeedForward(d_model=d_model, ff_dim=ff_dim)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, *, collect_iteration_states: bool) -> tuple[Tensor, list[Tensor]]:
        iteration_states: list[Tensor] = []
        for iteration_index in range(self.iterations):
            hidden = hidden + self.attn_dropout(self.attn(self.attn_norms[iteration_index](hidden)))
            hidden = hidden + self.ffn_dropout(self.ffn(self.ffn_norms[iteration_index](hidden)))
            if collect_iteration_states:
                iteration_states.append(hidden)
        return hidden, iteration_states


class RecurrentDepthLM(nn.Module):
    def __init__(self, *, vocab_size: int, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.core = SharedRecurrentCore(
            d_model=config.d_model,
            n_heads=config.n_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            iterations=config.recurrent_iterations,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.config.d_model**-0.5)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=self.config.d_model**-0.5)

    def embed(self, tokens: Tensor) -> Tensor:
        positions = torch.arange(self.config.context_size, device=tokens.device)
        token_hidden = normalize_hidden(self.token_embedding(tokens))
        hidden = token_hidden + self.position_embedding(positions)
        return self.embedding_dropout(hidden)

    def iteration_states(self, tokens: Tensor, *, collect_iteration_states: bool) -> tuple[Tensor, list[Tensor]]:
        hidden = self.embed(tokens)
        return self.core(hidden, collect_iteration_states=collect_iteration_states)

    def forward(self, tokens: Tensor) -> Tensor:
        hidden, _ = self.iteration_states(tokens, collect_iteration_states=False)
        last_hidden = hidden[:, -1, :]
        return tied_logits(last_hidden, self.token_embedding, temperature=self.config.temperature, normalize=True)


def build_model(*, vocab_size: int, config: ExperimentConfig) -> RecurrentDepthLM:
    return RecurrentDepthLM(vocab_size=vocab_size, config=config)


def build_optimizer(model: nn.Module, *, learning_rate: float, device: torch.device) -> torch.optim.Optimizer:
    optimizer_kwargs: dict[str, object] = {"lr": learning_rate}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    return torch.optim.Adam(model.parameters(), **optimizer_kwargs)


def train_model(
    model: RecurrentDepthLM,
    train_dataset: tuple[Tensor, Tensor],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    log_path: Path,
) -> TrainingSummary:
    optimizer = build_optimizer(model, learning_rate=learning_rate, device=device)
    last_train_loss = float("nan")
    started_at = perf_counter()

    for step in range(1, steps + 1):
        model.train()
        batch_inputs, batch_targets = sample_batch(train_dataset, batch_size=batch_size, device=device)
        with autocast_context(device):
            logits = model(batch_inputs)
            loss = F.cross_entropy(logits.float(), batch_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_train_loss = float(loss.item())

        if step % PRINT_INTERVAL == 0 or step == steps:
            print(f"step={step:05d}/{steps} train_loss={last_train_loss:.4f}", flush=True)
            append_log(
                log_path,
                {
                    "stage": "train_progress",
                    "step": step,
                    "steps": steps,
                    "train_loss": round(last_train_loss, 6),
                },
            )

    return TrainingSummary(
        steps=steps,
        batch_size=batch_size,
        final_train_loss=last_train_loss,
        wall_seconds=perf_counter() - started_at,
    )


@torch.inference_mode()
def evaluate_oracle_metrics(
    model: RecurrentDepthLM,
    dataset: tuple[Tensor, Tensor],
    *,
    eval_batch_size: int,
    delta: float,
    device: torch.device,
) -> OracleMetrics:
    model.eval()
    inputs, targets = dataset
    depth_count = model.config.recurrent_iterations
    loss_sums = torch.zeros(depth_count, dtype=torch.float64)
    oracle_depth_histogram = torch.zeros(depth_count, dtype=torch.int64)
    no_regret_depth_histogram = torch.zeros(depth_count, dtype=torch.int64)
    oracle_best_loss_sum = 0.0
    harmed_by_full_depth = 0
    no_regret_depth_sum = 0.0
    total_examples = 0

    for start in range(0, targets.shape[0], eval_batch_size):
        stop = min(start + eval_batch_size, targets.shape[0])
        batch_inputs = inputs[start:stop]
        batch_targets = targets[start:stop]

        with autocast_context(device):
            _, iteration_states = model.iteration_states(batch_inputs, collect_iteration_states=True)
            batch_losses = []
            for iteration_hidden in iteration_states:
                logits = tied_logits(
                    iteration_hidden[:, -1, :],
                    model.token_embedding,
                    temperature=model.config.temperature,
                    normalize=True,
                )
                batch_losses.append(F.cross_entropy(logits.float(), batch_targets, reduction="none"))

        per_depth_losses = torch.stack(batch_losses, dim=1)
        loss_sums += per_depth_losses.sum(dim=0).double().cpu()

        oracle_best_losses, oracle_depth_indices = per_depth_losses.min(dim=1)
        oracle_best_loss_sum += oracle_best_losses.sum().item()
        oracle_depth_histogram += torch.bincount(oracle_depth_indices.cpu(), minlength=depth_count)

        full_depth_losses = per_depth_losses[:, -1]
        harmed_by_full_depth += int((full_depth_losses > oracle_best_losses + 1e-12).sum().item())

        no_regret_mask = per_depth_losses <= (full_depth_losses.unsqueeze(1) + delta)
        if not bool(no_regret_mask[:, -1].all().item()):
            raise RuntimeError("full depth must always satisfy the no-regret condition")
        no_regret_depth_indices = no_regret_mask.float().argmax(dim=1)
        no_regret_depth_histogram += torch.bincount(no_regret_depth_indices.cpu(), minlength=depth_count)
        no_regret_depth_sum += float((no_regret_depth_indices + 1).sum().item())

        total_examples += batch_targets.shape[0]

    if total_examples == 0:
        raise RuntimeError("validation dataset was empty")

    mean_no_regret_depth = no_regret_depth_sum / total_examples
    return OracleMetrics(
        mean_loss_per_depth=tuple((loss_sums / total_examples).tolist()),
        oracle_best_loss=oracle_best_loss_sum / total_examples,
        oracle_depth_histogram=tuple(int(value) for value in oracle_depth_histogram.tolist()),
        no_regret_depth_histogram=tuple(int(value) for value in no_regret_depth_histogram.tolist()),
        mean_no_regret_depth=mean_no_regret_depth,
        oracle_speedup_delta_0_01=model.config.recurrent_iterations / mean_no_regret_depth,
        fraction_harmed_by_full_depth=harmed_by_full_depth / total_examples,
        total_examples=total_examples,
    )


def print_metrics(metrics: OracleMetrics, *, delta: float) -> None:
    print("=== Oracle depth analysis ===", flush=True)
    print("Mean loss per depth:", flush=True)
    for depth, loss in enumerate(metrics.mean_loss_per_depth, start=1):
        print(f"  depth {depth}: {loss:.6f}", flush=True)

    print(f"Oracle-best loss: {metrics.oracle_best_loss:.6f}", flush=True)
    print("Oracle depth histogram:", flush=True)
    for depth, count in enumerate(metrics.oracle_depth_histogram, start=1):
        fraction = count / metrics.total_examples
        print(f"  depth {depth}: {count} ({fraction:.4%})", flush=True)

    print(f"No-regret depth histogram (delta={delta:.2f}):", flush=True)
    for depth, count in enumerate(metrics.no_regret_depth_histogram, start=1):
        fraction = count / metrics.total_examples
        print(f"  depth {depth}: {count} ({fraction:.4%})", flush=True)

    print(f"Oracle speedup at delta={delta:.2f}: {metrics.oracle_speedup_delta_0_01:.6f}x", flush=True)
    print(f"Fraction harmed by full depth: {metrics.fraction_harmed_by_full_depth:.6f}", flush=True)


def write_report(
    report_path: Path,
    *,
    args: argparse.Namespace,
    config: ExperimentConfig,
    training: TrainingSummary,
    metrics: OracleMetrics,
    device: torch.device,
) -> None:
    payload = {
        "seed": args.seed,
        "device": device.type,
        "sanity_check_only": args.sanity_check_only,
        "config": asdict(config),
        "train_characters": args.train_characters,
        "val_characters": args.val_characters,
        "steps": training.steps,
        "batch_size": training.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "delta": DELTA,
        "training": {
            "final_train_loss": round(training.final_train_loss, 6),
            "wall_seconds": round(training.wall_seconds, 6),
        },
        "oracle_metrics": {
            "mean_loss_per_depth": [round(value, 6) for value in metrics.mean_loss_per_depth],
            "oracle_best_loss": round(metrics.oracle_best_loss, 6),
            "oracle_depth_histogram": list(metrics.oracle_depth_histogram),
            "no_regret_depth_histogram": list(metrics.no_regret_depth_histogram),
            "mean_no_regret_depth": round(metrics.mean_no_regret_depth, 6),
            "oracle_speedup_delta_0_01": round(metrics.oracle_speedup_delta_0_01, 6),
            "fraction_harmed_by_full_depth": round(metrics.fraction_harmed_by_full_depth, 6),
            "total_examples": metrics.total_examples,
        },
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {args.learning_rate}")

    config = ExperimentConfig(recurrent_iterations=args.recurrent_iterations)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.sanity_check_only:
        redirect_sanity_check_paths(args)

    prepare_output_paths(report_path=args.report_path, log_path=args.log_path)
    if not args.sanity_check_only:
        register_active_lock(
            experiment_name="oracle_depth_analysis",
            variants=[f"recurrent_{config.recurrent_iterations}"],
            enabled=not args.no_lock,
        )

    append_log(
        args.log_path,
        {
            "stage": "run_started",
            "seed": args.seed,
            "device": device.type,
            "sanity_check_only": args.sanity_check_only,
            "steps": resolve_training_steps(args),
            "batch_size": resolve_batch_size(args),
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "train_characters": args.train_characters,
            "val_characters": args.val_characters,
            "config": asdict(config),
        },
    )

    set_seed(args.seed)
    train_dataset, val_dataset, vocab_size = load_dataset(
        context_size=config.context_size,
        train_characters=args.train_characters,
        val_characters=args.val_characters,
    )
    train_dataset = dataset_to_device(train_dataset, device)
    val_dataset = dataset_to_device(val_dataset, device)

    append_log(
        args.log_path,
        {
            "stage": "dataset_loaded",
            "train_examples": train_dataset[1].shape[0],
            "val_examples": val_dataset[1].shape[0],
            "vocab_size": vocab_size,
        },
    )

    model = build_model(vocab_size=vocab_size, config=config).to(device)
    training = train_model(
        model,
        train_dataset,
        steps=resolve_training_steps(args),
        batch_size=resolve_batch_size(args),
        learning_rate=args.learning_rate,
        device=device,
        log_path=args.log_path,
    )
    metrics = evaluate_oracle_metrics(
        model,
        val_dataset,
        eval_batch_size=args.eval_batch_size,
        delta=DELTA,
        device=device,
    )

    print_metrics(metrics, delta=DELTA)
    write_report(
        args.report_path,
        args=args,
        config=config,
        training=training,
        metrics=metrics,
        device=device,
    )
    append_log(
        args.log_path,
        {
            "stage": "run_completed",
            "report_path": str(args.report_path),
            "oracle_metrics": {
                "mean_loss_per_depth": [round(value, 6) for value in metrics.mean_loss_per_depth],
                "oracle_best_loss": round(metrics.oracle_best_loss, 6),
                "oracle_depth_histogram": list(metrics.oracle_depth_histogram),
                "no_regret_depth_histogram": list(metrics.no_regret_depth_histogram),
                "oracle_speedup_delta_0_01": round(metrics.oracle_speedup_delta_0_01, 6),
                "fraction_harmed_by_full_depth": round(metrics.fraction_harmed_by_full_depth, 6),
            },
        },
    )


if __name__ == "__main__":
    main()
