from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    raise RuntimeError(
        "Run from the repo root with `.\\.venv\\Scripts\\python.exe -m runs.recurrent_depth_lm ...` so `core` imports resolve cleanly."
    )

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from core.fixed_window_char import load_dataset, set_seed
from core.model import count_parameters
from core.run_utils import (
    append_log,
    prepare_output_paths,
    redirect_sanity_check_paths,
    register_active_lock,
    resolve_device,
)
from core.tied_readout import CausalSelfAttention, FeedForward, normalize_hidden, tied_logits

CONTEXT_SIZE = 128
DISTINCT_LAYERS = 4
RECURRENT_ITERATIONS = 4
TRAINING_STEPS = 20_000
SANITY_CHECK_STEPS = 10
TRAIN_BATCH_SIZE = 128
SANITY_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 2_048
LEARNING_RATE = 3e-4
DEFAULT_SEED = 42
D_MODEL = 128
N_HEADS = 4
FF_DIM = 512
DROPOUT = 0.1
TEMPERATURE = 0.07
GRAD_LOG_INTERVAL = 100
ACTIVATION_RMS_SAMPLE_BATCHES = 1


@dataclass(frozen=True)
class ExperimentConfig:
    context_size: int = CONTEXT_SIZE
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    ff_dim: int = FF_DIM
    dropout: float = DROPOUT
    temperature: float = TEMPERATURE
    distinct_layers: int = DISTINCT_LAYERS
    recurrent_iterations: int = RECURRENT_ITERATIONS


@dataclass(frozen=True)
class ConditionResult:
    condition: str
    params: int
    val_loss: float
    train_time_seconds: float


@dataclass(frozen=True)
class GradientNormPoint:
    step: int
    grad_norm: float
    train_loss: float


@dataclass(frozen=True)
class RecurrentDiagnostics:
    per_iteration_val_loss: tuple[float, ...]
    activation_rms_by_iteration: tuple[float, ...]


@dataclass(frozen=True)
class TrainedCondition:
    result: ConditionResult
    gradient_norm_trace: tuple[GradientNormPoint, ...]
    recurrent_diagnostics: RecurrentDiagnostics | None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "tinyshakespeare" / "artifacts" / "recurrent_depth_lm"
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--steps", type=positive_int, default=TRAINING_STEPS)
    parser.add_argument("--batch-size", type=positive_int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=positive_int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
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


def gradient_norm(model: nn.Module) -> float:
    squared_norm = 0.0
    found_gradient = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        found_gradient = True
        grad = parameter.grad.detach().float()
        squared_norm += grad.square().sum().item()
    if not found_gradient:
        raise RuntimeError("expected gradients after backward pass, but none were present")
    return squared_norm**0.5


class DistinctTransformerBlock(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, ff_dim=ff_dim)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attn_dropout(self.attn(self.attn_norm(hidden)))
        hidden = hidden + self.ffn_dropout(self.ffn(self.ffn_norm(hidden)))
        return hidden


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


class DistinctDepthLM(nn.Module):
    def __init__(self, *, vocab_size: int, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            DistinctTransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
            )
            for _ in range(config.distinct_layers)
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

    def encode(self, tokens: Tensor) -> Tensor:
        hidden = self.embed(tokens)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def forward(self, tokens: Tensor) -> Tensor:
        hidden = self.encode(tokens)
        last_hidden = hidden[:, -1, :]
        return tied_logits(last_hidden, self.token_embedding, temperature=self.config.temperature, normalize=True)


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


def build_model(condition: str, *, vocab_size: int, config: ExperimentConfig) -> nn.Module:
    if condition == "distinct_4":
        return DistinctDepthLM(vocab_size=vocab_size, config=config)
    if condition == "recurrent_4":
        return RecurrentDepthLM(vocab_size=vocab_size, config=config)
    raise ValueError(f"unsupported condition: {condition}")


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    dataset: tuple[Tensor, Tensor],
    *,
    eval_batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    inputs, targets = dataset
    total_loss = 0.0
    total_examples = 0
    for start in range(0, targets.shape[0], eval_batch_size):
        stop = min(start + eval_batch_size, targets.shape[0])
        batch_inputs = inputs[start:stop]
        batch_targets = targets[start:stop]
        with autocast_context(device):
            logits = model(batch_inputs)
        loss = F.cross_entropy(logits.float(), batch_targets, reduction="sum")
        total_loss += loss.item()
        total_examples += batch_targets.shape[0]
    return total_loss / total_examples


@torch.inference_mode()
def evaluate_recurrent_diagnostics(
    model: RecurrentDepthLM,
    dataset: tuple[Tensor, Tensor],
    *,
    eval_batch_size: int,
    device: torch.device,
) -> RecurrentDiagnostics:
    model.eval()
    inputs, targets = dataset
    loss_sums = [0.0 for _ in range(model.config.recurrent_iterations)]
    total_examples = 0
    activation_rms_sums = [0.0 for _ in range(model.config.recurrent_iterations)]
    activation_rms_batches = 0

    for batch_index, start in enumerate(range(0, targets.shape[0], eval_batch_size)):
        stop = min(start + eval_batch_size, targets.shape[0])
        batch_inputs = inputs[start:stop]
        batch_targets = targets[start:stop]
        with autocast_context(device):
            _, iteration_states = model.iteration_states(batch_inputs, collect_iteration_states=True)
            for iteration_index, iteration_hidden in enumerate(iteration_states):
                logits = tied_logits(
                    iteration_hidden[:, -1, :],
                    model.token_embedding,
                    temperature=model.config.temperature,
                    normalize=True,
                )
                loss = F.cross_entropy(logits.float(), batch_targets, reduction="sum")
                loss_sums[iteration_index] += loss.item()
                if batch_index < ACTIVATION_RMS_SAMPLE_BATCHES:
                    activation_rms_sums[iteration_index] += torch.sqrt(
                        torch.mean(iteration_hidden.detach().float().square())
                    ).item()
        total_examples += batch_targets.shape[0]
        if batch_index < ACTIVATION_RMS_SAMPLE_BATCHES:
            activation_rms_batches += 1

    if activation_rms_batches == 0:
        raise RuntimeError("expected at least one activation RMS sample batch")

    return RecurrentDiagnostics(
        per_iteration_val_loss=tuple(loss_sum / total_examples for loss_sum in loss_sums),
        activation_rms_by_iteration=tuple(value / activation_rms_batches for value in activation_rms_sums),
    )


def train_condition(
    condition: str,
    train_dataset: tuple[Tensor, Tensor],
    val_dataset: tuple[Tensor, Tensor],
    *,
    vocab_size: int,
    config: ExperimentConfig,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    log_path: Path,
) -> TrainedCondition:
    set_seed(seed)
    model = build_model(condition, vocab_size=vocab_size, config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=device.type == "cuda")
    gradient_trace: list[GradientNormPoint] = []

    append_log(
        log_path,
        {
            "stage": "condition_started",
            "condition": condition,
            "params": count_parameters(model),
            "steps": steps,
            "batch_size": batch_size,
        },
    )

    started_at = perf_counter()
    for step in range(1, steps + 1):
        model.train()
        batch_inputs, batch_targets = sample_batch(train_dataset, batch_size=batch_size, device=device)
        with autocast_context(device):
            logits = model(batch_inputs)
            loss = F.cross_entropy(logits.float(), batch_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_total_norm = gradient_norm(model)
        optimizer.step()

        if step % GRAD_LOG_INTERVAL == 0 or step == steps:
            point = GradientNormPoint(step=step, grad_norm=grad_total_norm, train_loss=float(loss.item()))
            gradient_trace.append(point)
            print(
                f"[{condition}] step={step:05d}/{steps} train_loss={point.train_loss:.4f} grad_norm={point.grad_norm:.4f}",
                flush=True,
            )
            append_log(log_path, {"stage": "grad_norm", "condition": condition, **asdict(point)})

    train_time_seconds = perf_counter() - started_at
    val_loss = evaluate_model(model, val_dataset, eval_batch_size=eval_batch_size, device=device)
    recurrent_diagnostics = (
        evaluate_recurrent_diagnostics(model, val_dataset, eval_batch_size=eval_batch_size, device=device)
        if isinstance(model, RecurrentDepthLM)
        else None
    )

    append_log(
        log_path,
        {
            "stage": "condition_completed",
            "condition": condition,
            "val_loss": round(val_loss, 6),
            "train_time_seconds": round(train_time_seconds, 6),
            "params": count_parameters(model),
            "recurrent_diagnostics": None
            if recurrent_diagnostics is None
            else {
                "per_iteration_val_loss": [round(value, 6) for value in recurrent_diagnostics.per_iteration_val_loss],
                "activation_rms_by_iteration": [round(value, 6) for value in recurrent_diagnostics.activation_rms_by_iteration],
            },
        },
    )

    return TrainedCondition(
        result=ConditionResult(
            condition=condition,
            params=count_parameters(model),
            val_loss=val_loss,
            train_time_seconds=train_time_seconds,
        ),
        gradient_norm_trace=tuple(gradient_trace),
        recurrent_diagnostics=recurrent_diagnostics,
    )


def print_results_table(results: list[ConditionResult]) -> None:
    print("=== Results ===", flush=True)
    print("Condition       | Params  | Val Loss | Train Time", flush=True)
    for result in results:
        print(
            f"{result.condition:<15} | {result.params:<7} | {result.val_loss:.4f}   | {result.train_time_seconds:.2f}s",
            flush=True,
        )


def print_recurrent_tables(diagnostics: RecurrentDiagnostics) -> None:
    print("", flush=True)
    print("=== Per-iteration eval (recurrent_4) ===", flush=True)
    print("Iteration | Val Loss", flush=True)
    for index, value in enumerate(diagnostics.per_iteration_val_loss, start=1):
        print(f"{index:<9} | {value:.4f}", flush=True)

    print("", flush=True)
    print("=== Activation RMS (recurrent_4, sampled eval) ===", flush=True)
    print("Iteration | RMS", flush=True)
    for index, value in enumerate(diagnostics.activation_rms_by_iteration, start=1):
        print(f"{index:<9} | {value:.4f}", flush=True)


def gradient_trace_payload(trace: tuple[GradientNormPoint, ...]) -> list[dict[str, float | int]]:
    return [asdict(point) for point in trace]


def write_report(
    report_path: Path,
    *,
    args: argparse.Namespace,
    config: ExperimentConfig,
    results: list[TrainedCondition],
) -> None:
    payload = {
        "seed": args.seed,
        "device": args.device,
        "sanity_check_only": args.sanity_check_only,
        "config": asdict(config),
        "steps": resolve_training_steps(args),
        "batch_size": resolve_batch_size(args),
        "conditions": {
            trained.result.condition: {
                "params": trained.result.params,
                "val_loss": round(trained.result.val_loss, 6),
                "train_time_seconds": round(trained.result.train_time_seconds, 6),
                "gradient_norm_trace": gradient_trace_payload(trained.gradient_norm_trace),
                "recurrent_diagnostics": None
                if trained.recurrent_diagnostics is None
                else {
                    "per_iteration_val_loss": [
                        round(value, 6) for value in trained.recurrent_diagnostics.per_iteration_val_loss
                    ],
                    "activation_rms_by_iteration": [
                        round(value, 6) for value in trained.recurrent_diagnostics.activation_rms_by_iteration
                    ],
                },
            }
            for trained in results
        },
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {args.learning_rate}")

    config = ExperimentConfig()
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
            experiment_name="recurrent_depth_lm",
            variants=["distinct_4", "recurrent_4"],
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
            "config": asdict(config),
        },
    )

    train_dataset, val_dataset, vocab_size = load_dataset(context_size=config.context_size)
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

    steps = resolve_training_steps(args)
    batch_size = resolve_batch_size(args)
    conditions = ["distinct_4", "recurrent_4"]
    trained_conditions = [
        train_condition(
            condition,
            train_dataset,
            val_dataset,
            vocab_size=vocab_size,
            config=config,
            steps=steps,
            batch_size=batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=device,
            log_path=args.log_path,
        )
        for condition in conditions
    ]

    print("", flush=True)
    print_results_table([trained.result for trained in trained_conditions])
    recurrent = next(trained for trained in trained_conditions if trained.result.condition == "recurrent_4")
    if recurrent.recurrent_diagnostics is None:
        raise RuntimeError("recurrent_4 diagnostics missing")
    print_recurrent_tables(recurrent.recurrent_diagnostics)

    write_report(args.report_path, args=args, config=config, results=trained_conditions)
    append_log(
        args.log_path,
        {
            "stage": "run_completed",
            "report_path": str(args.report_path),
            "conditions": {
                trained.result.condition: {
                    "params": trained.result.params,
                    "val_loss": round(trained.result.val_loss, 6),
                    "train_time_seconds": round(trained.result.train_time_seconds, 6),
                }
                for trained in trained_conditions
            },
        },
    )


if __name__ == "__main__":
    main()
