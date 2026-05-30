from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import functional as F

from core.automaton import CellularAutomaton, ParameterSlice, count_parameters
from core.run_utils import (
    append_log,
    prepare_output_paths,
    redirect_sanity_check_paths,
    register_active_lock,
    resolve_device,
)
from core.training import current_git_sha, current_git_status_short, write_json


torch.backends.cuda.matmul.allow_tf32 = True

EAGER_BASELINE_STEP_TIME_S = 26.0
CUDA_GRAPH_WARMUP_STEPS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the cellular automaton model on TinyShakespeare with chunk-level CUDA graph capture.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--n-levels", type=int, default=8)
    parser.add_argument("--steps-per-token", type=int, default=8)
    parser.add_argument("--d-stream", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--loss-type", choices=("mse", "infonce"), default="mse")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--xblk-lambda", type=float, default=0.01)
    parser.add_argument("--sanity-check-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--use-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture the chunk-level forward/loss/backward pass in a CUDA graph.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("experiments/automaton/artifacts/report.json"),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("experiments/automaton/artifacts/run.jsonl"),
    )
    parser.add_argument("--no-lock", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}.")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}.")
    if args.seq_len <= 1:
        raise ValueError(f"--seq-len must be greater than 1, got {args.seq_len}.")
    if args.n_levels <= 0:
        raise ValueError(f"--n-levels must be positive, got {args.n_levels}.")
    if args.steps_per_token <= 0:
        raise ValueError(f"--steps-per-token must be positive, got {args.steps_per_token}.")
    if args.d_stream <= 0:
        raise ValueError(f"--d-stream must be positive, got {args.d_stream}.")
    if args.noise_std < 0.0:
        raise ValueError(f"--noise-std must be non-negative, got {args.noise_std}.")
    if args.lr <= 0.0:
        raise ValueError(f"--lr must be positive, got {args.lr}.")
    if args.chunk_size <= 0:
        raise ValueError(f"--chunk-size must be positive, got {args.chunk_size}.")
    if args.xblk_lambda < 0.0:
        raise ValueError(f"--xblk-lambda must be non-negative, got {args.xblk_lambda}.")
    if args.log_every <= 0:
        raise ValueError(f"--log-every must be positive, got {args.log_every}.")
    if args.seq_len % args.chunk_size != 0:
        raise ValueError(f"--seq-len must be divisible by --chunk-size, got seq_len={args.seq_len}, chunk_size={args.chunk_size}.")

    if args.sanity_check_only:
        args.steps = 5
    return args


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TinyShakespeareData:
    encoded: Int[Tensor, "tokens"]
    stoi: dict[str, int]
    itos: dict[int, str]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


@dataclass(frozen=True)
class ChunkStepResult:
    ce_loss: Float[Tensor, ""]
    prediction_loss_sums: Float[Tensor, "levels"]
    prediction_counts: Int[Tensor, "levels"]
    xblk_penalty: Float[Tensor, ""]


def load_tinyshakespeare() -> TinyShakespeareData:
    text_path = Path("experiments/corpora.ignore/tinyshakespeare_input.txt")
    text = text_path.read_text(encoding="utf-8")
    vocab = sorted(set(text))
    stoi = {char: index for index, char in enumerate(vocab)}
    itos = {index: char for index, char in enumerate(vocab)}
    encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    return TinyShakespeareData(encoded=encoded, stoi=stoi, itos=itos)


def sample_batch(
    encoded: Int[Tensor, "tokens"],
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"]]:
    if encoded.numel() <= seq_len:
        raise ValueError(f"Need corpus longer than seq_len={seq_len}, got {encoded.numel()} tokens.")
    max_start = encoded.numel() - seq_len - 1
    if max_start < 0:
        raise ValueError(
            f"Need at least seq_len + 1 tokens for next-token targets, got {encoded.numel()} and seq_len={seq_len}."
        )

    starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
    offsets = torch.arange(seq_len + 1, dtype=torch.long)
    windows = encoded[starts[:, None] + offsets]
    inputs = windows[:, :-1]
    targets = windows[:, 1:]
    pin_memory = device.type == "cuda"
    if pin_memory:
        inputs = inputs.pin_memory()
        targets = targets.pin_memory()
    return (
        inputs.to(device=device, dtype=torch.long, non_blocking=pin_memory),
        targets.to(device=device, dtype=torch.long, non_blocking=pin_memory),
    )


def make_log_payload(
    *,
    step: int,
    ce_loss: Tensor,
    prediction_losses: Tensor,
    prediction_counts: Tensor,
    xblk_penalty: Tensor,
    total_loss: Tensor,
    wall_time_s: float,
) -> dict[str, object]:
    return {
        "step": step,
        "ce_loss": round(ce_loss.item(), 6),
        "xblk_penalty": round(xblk_penalty.item(), 6),
        "total_loss": round(total_loss.item(), 6),
        "prediction_losses": [round(value, 6) for value in prediction_losses.detach().cpu().tolist()],
        "prediction_counts": [int(value) for value in prediction_counts.detach().cpu().tolist()],
        "wall_time_s": round(wall_time_s, 6),
    }


def make_level_optimizer(
    parameter_slices: list[ParameterSlice],
    *,
    lr: float,
) -> "LevelAdamW":
    return LevelAdamW(parameter_slices, lr=lr)


class LevelAdamW:
    def __init__(
        self,
        parameter_slices: list[ParameterSlice],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        self.parameter_slices = parameter_slices
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state: dict[tuple[int, int], dict[str, Tensor | int]] = {}

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for parameter_slice in self.parameter_slices:
            grad = parameter_slice.parameter.grad
            if grad is None:
                continue
            if set_to_none:
                grad[parameter_slice.level] = 0
            else:
                grad[parameter_slice.level].zero_()

    @torch.no_grad()
    def step(self) -> None:
        for parameter_slice in self.parameter_slices:
            parameter = parameter_slice.parameter
            grad = parameter.grad
            if grad is None:
                continue
            level = parameter_slice.level
            level_grad = grad[level]
            if self.weight_decay != 0.0:
                parameter[level].mul_(1 - (self.lr * self.weight_decay))

            state_key = (id(parameter), level)
            state = self.state.get(state_key)
            if state is None:
                state = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(parameter[level]),
                    "exp_avg_sq": torch.zeros_like(parameter[level]),
                }
                self.state[state_key] = state

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            step = int(state["step"]) + 1
            state["step"] = step

            exp_avg.mul_(self.beta1).add_(level_grad, alpha=1 - self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(level_grad, level_grad, value=1 - self.beta2)

            bias_correction1 = 1 - (self.beta1**step)
            bias_correction2 = 1 - (self.beta2**step)
            step_size = self.lr / bias_correction1
            denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(self.eps)
            parameter[level].addcdiv_(exp_avg, denom, value=-step_size)


class ChunkTrainer:
    def __init__(
        self,
        *,
        model: CellularAutomaton,
        batch_size: int,
        chunk_size: int,
        seq_len: int,
        vocab_size: int,
        device: torch.device,
        use_cuda_graph: bool,
        optimizers: list[LevelAdamW],
        xblk_lambda: float,
    ) -> None:
        if device.type != "cuda":
            raise RuntimeError(f"ChunkTrainer requires a CUDA device, got {device}.")

        self.model = model
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.device = device
        self.optimizers = optimizers
        self.xblk_lambda = xblk_lambda
        self.requested_cuda_graph = use_cuda_graph
        self.graph_active = False
        self.graph_failure_reason: str | None = None

        dtype = model.token_embedding.weight.dtype
        n_levels = model.n_levels
        d_stream = model.d_stream

        self.chunk_ce_scale = torch.tensor(chunk_size / seq_len, device=device, dtype=dtype)
        self.static_chunk_inputs = torch.zeros((batch_size, chunk_size), device=device, dtype=torch.long)
        self.static_chunk_targets = torch.zeros((batch_size, chunk_size), device=device, dtype=torch.long)
        self.static_states = torch.zeros((n_levels, batch_size, d_stream), device=device, dtype=dtype)
        self.static_lateral_buffers = torch.zeros_like(self.static_states)
        self.static_predictions = torch.zeros_like(self.static_states)
        self.static_has_predicted = torch.zeros((n_levels,), device=device, dtype=torch.bool)
        self.static_global_step_offset = torch.zeros((), device=device, dtype=torch.long)

        self.static_chunk_logits = torch.zeros((batch_size, chunk_size, vocab_size), device=device, dtype=dtype)
        self.static_next_states = torch.zeros_like(self.static_states)
        self.static_next_lateral_buffers = torch.zeros_like(self.static_states)
        self.static_next_predictions = torch.zeros_like(self.static_states)
        self.static_next_has_predicted = torch.zeros_like(self.static_has_predicted)
        self.static_ce_loss = torch.zeros((), device=device, dtype=dtype)
        self.static_prediction_loss_sums = torch.zeros((n_levels,), device=device, dtype=dtype)
        self.static_prediction_counts = torch.zeros((n_levels,), device=device, dtype=torch.long)
        self.static_prediction_losses = torch.zeros((n_levels,), device=device, dtype=dtype)
        self.static_xblk_penalty = torch.zeros((), device=device, dtype=dtype)

        self.graph: torch.cuda.CUDAGraph | None = None

        if use_cuda_graph:
            self._try_initialize_graph()

    def _zero_optimizer_grads(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def _copy_runtime_inputs(
        self,
        *,
        chunk_inputs: Int[Tensor, "batch chunk"],
        chunk_targets: Int[Tensor, "batch chunk"],
        states: Float[Tensor, "levels batch d_stream"],
        lateral_buffers: Float[Tensor, "levels batch d_stream"],
        predictions: Float[Tensor, "levels batch d_stream"],
        has_predicted: Bool[Tensor, "levels"],
        global_step_offset: int,
    ) -> None:
        with torch.no_grad():
            self.static_chunk_inputs.copy_(chunk_inputs)
            self.static_chunk_targets.copy_(chunk_targets)
            self.static_states.copy_(states)
            self.static_lateral_buffers.copy_(lateral_buffers)
            self.static_predictions.copy_(predictions)
            self.static_has_predicted.copy_(has_predicted)
            self.static_global_step_offset.fill_(global_step_offset)

    def _forward_backward_into_static_buffers(self) -> None:
        (
            chunk_logits,
            next_states,
            next_lateral_buffers,
            next_predictions,
            next_has_predicted,
            prediction_loss_sums,
            prediction_counts,
        ) = self.model.forward_chunk(
            self.static_chunk_inputs,
            self.static_states,
            self.static_lateral_buffers,
            self.static_predictions,
            self.static_has_predicted,
            self.static_global_step_offset,
        )
        prediction_losses = self.model.prediction_losses_from_sums(prediction_loss_sums, prediction_counts)
        ce_loss = (
            F.cross_entropy(
                chunk_logits.reshape(-1, self.vocab_size),
                self.static_chunk_targets.reshape(-1),
                reduction="mean",
            )
            * self.chunk_ce_scale
        )
        xblk_penalty = self._cross_block_covariance_penalty(next_states)
        total_loss = ce_loss + prediction_losses.sum() + xblk_penalty

        with torch.no_grad():
            self.static_chunk_logits.copy_(chunk_logits)
            self.static_next_states.copy_(next_states)
            self.static_next_lateral_buffers.copy_(next_lateral_buffers)
            self.static_next_predictions.copy_(next_predictions)
            self.static_next_has_predicted.copy_(next_has_predicted)
            self.static_ce_loss.copy_(ce_loss)
            self.static_prediction_loss_sums.copy_(prediction_loss_sums)
            self.static_prediction_counts.copy_(prediction_counts)
            self.static_prediction_losses.copy_(prediction_losses)
            self.static_xblk_penalty.copy_(xblk_penalty)

        total_loss.backward()

    def _cross_block_covariance_penalty(
        self,
        states: Float[Tensor, "levels batch d_stream"],
    ) -> Float[Tensor, ""]:
        if self.xblk_lambda == 0.0:
            return torch.zeros((), device=states.device, dtype=states.dtype)

        centered_states = states.float() - states.float().mean(dim=1, keepdim=True)
        covariance = torch.einsum("lbd,mbf->lmdf", centered_states, centered_states) / self.batch_size
        upper_triangle = torch.triu_indices(self.model.n_levels, self.model.n_levels, offset=1, device=states.device)
        pairwise_covariance = covariance[upper_triangle[0], upper_triangle[1]]
        penalty = pairwise_covariance.square().sum().to(dtype=states.dtype)
        return penalty * self.xblk_lambda

    def _try_initialize_graph(self) -> None:
        try:
            warmup_stream = torch.cuda.Stream(device=self.device)
            current_stream = torch.cuda.current_stream(device=self.device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream):
                for _ in range(CUDA_GRAPH_WARMUP_STEPS):
                    self._forward_backward_into_static_buffers()
                    self._zero_optimizer_grads()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self._forward_backward_into_static_buffers()
            self._zero_optimizer_grads()
            current_stream.wait_stream(warmup_stream)
            torch.cuda.synchronize(self.device)
            self.graph_active = True
        except Exception as exc:  # noqa: BLE001
            self.graph = None
            self.graph_active = False
            self.graph_failure_reason = f"{type(exc).__name__}: {exc}"
            self._zero_optimizer_grads()

    def run_chunk(
        self,
        *,
        chunk_inputs: Int[Tensor, "batch chunk"],
        chunk_targets: Int[Tensor, "batch chunk"],
        states: Float[Tensor, "levels batch d_stream"],
        lateral_buffers: Float[Tensor, "levels batch d_stream"],
        predictions: Float[Tensor, "levels batch d_stream"],
        has_predicted: Bool[Tensor, "levels"],
        global_step_offset: int,
    ) -> ChunkStepResult:
        self._copy_runtime_inputs(
            chunk_inputs=chunk_inputs,
            chunk_targets=chunk_targets,
            states=states,
            lateral_buffers=lateral_buffers,
            predictions=predictions,
            has_predicted=has_predicted,
            global_step_offset=global_step_offset,
        )

        if self.graph_active and self.graph is not None:
            try:
                self.graph.replay()
            except Exception as exc:  # noqa: BLE001
                self.graph_active = False
                self.graph_failure_reason = f"graph replay failed: {type(exc).__name__}: {exc}"
                self._forward_backward_into_static_buffers()
        else:
            self._forward_backward_into_static_buffers()

        with torch.no_grad():
            states.copy_(self.static_next_states)
            lateral_buffers.copy_(self.static_next_lateral_buffers)
            predictions.copy_(self.static_next_predictions)
            has_predicted.copy_(self.static_next_has_predicted)

        return ChunkStepResult(
            ce_loss=self.static_ce_loss,
            prediction_loss_sums=self.static_prediction_loss_sums,
            prediction_counts=self.static_prediction_counts,
            xblk_penalty=self.static_xblk_penalty,
        )


def maybe_print_cuda_graph_status(chunk_trainer: ChunkTrainer) -> None:
    print(
        json.dumps(
            {
                "stage": "cuda_graph_status",
                "requested": chunk_trainer.requested_cuda_graph,
                "active": chunk_trainer.graph_active,
                "failure_reason": chunk_trainer.graph_failure_reason,
            }
        ),
        flush=True,
    )


def maybe_print_speedup_estimate(*, step: int, wall_times_s: list[float], chunk_trainer: ChunkTrainer) -> None:
    if step != 3 or len(wall_times_s) < 3:
        return

    average_wall_time_s = sum(wall_times_s) / len(wall_times_s)
    estimated_speedup = EAGER_BASELINE_STEP_TIME_S / average_wall_time_s
    print(
        json.dumps(
            {
                "stage": "cuda_graph_speed_estimate",
                "active": chunk_trainer.graph_active,
                "avg_wall_time_s": round(average_wall_time_s, 6),
                "estimated_speedup_vs_26s_baseline": round(estimated_speedup, 3),
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            f"train_cuda_graph.py requires CUDA because this training path is designed for CUDA graph capture. Resolved device: {device}."
        )

    data = load_tinyshakespeare()

    if args.sanity_check_only:
        redirect_sanity_check_paths(args)
    else:
        prepare_output_paths(report_path=args.report_path, log_path=args.log_path)

    register_active_lock(
        experiment_name="automaton_cuda_graph",
        variants={
            "n_levels": args.n_levels,
            "steps_per_token": args.steps_per_token,
            "d_stream": args.d_stream,
            "loss_type": args.loss_type,
            "use_cuda_graph": args.use_cuda_graph,
            "xblk_lambda": args.xblk_lambda,
        },
        enabled=not args.no_lock,
        run_size="small" if args.sanity_check_only else "large",
    )

    model = CellularAutomaton(
        vocab_size=data.vocab_size,
        d_stream=args.d_stream,
        n_levels=args.n_levels,
        steps_per_token=args.steps_per_token,
        noise_std=args.noise_std,
        loss_type=args.loss_type,
    ).to(device)
    optimizers = [make_level_optimizer(model.level_parameters(level), lr=args.lr) for level in range(args.n_levels)]
    chunk_trainer = ChunkTrainer(
        model=model,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        seq_len=args.seq_len,
        vocab_size=data.vocab_size,
        device=device,
        use_cuda_graph=args.use_cuda_graph,
        optimizers=optimizers,
        xblk_lambda=args.xblk_lambda,
    )
    maybe_print_cuda_graph_status(chunk_trainer)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)

    last_payload: dict[str, object] | None = None
    wall_times_s: list[float] = []

    for step in range(1, args.steps + 1):
        inputs, targets = sample_batch(
            data.encoded,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
            generator=rng,
        )

        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

        states, lateral_buffers, predictions, has_predicted = model.initial_recurrent_state(args.batch_size, device=device)
        prediction_loss_sums = torch.zeros((args.n_levels,), device=device, dtype=model.token_embedding.weight.dtype)
        prediction_counts = torch.zeros((args.n_levels,), device=device, dtype=torch.long)
        ce_loss = torch.zeros((), device=device, dtype=model.token_embedding.weight.dtype)
        xblk_penalty = torch.zeros((), device=device, dtype=model.token_embedding.weight.dtype)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_started_at = time.perf_counter()

        for chunk_start in range(0, args.seq_len, args.chunk_size):
            chunk_stop = chunk_start + args.chunk_size
            chunk_result = chunk_trainer.run_chunk(
                chunk_inputs=inputs[:, chunk_start:chunk_stop],
                chunk_targets=targets[:, chunk_start:chunk_stop],
                states=states,
                lateral_buffers=lateral_buffers,
                predictions=predictions,
                has_predicted=has_predicted,
                global_step_offset=chunk_start * args.steps_per_token,
            )
            ce_loss = ce_loss + chunk_result.ce_loss
            prediction_loss_sums = prediction_loss_sums + chunk_result.prediction_loss_sums
            prediction_counts = prediction_counts + chunk_result.prediction_counts
            xblk_penalty = xblk_penalty + chunk_result.xblk_penalty

        for optimizer in optimizers:
            optimizer.step()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_time_s = time.perf_counter() - step_started_at
        wall_times_s.append(wall_time_s)
        maybe_print_speedup_estimate(step=step, wall_times_s=wall_times_s, chunk_trainer=chunk_trainer)

        if step % args.log_every != 0 and step != args.steps:
            continue

        prediction_losses = model.prediction_losses_from_sums(prediction_loss_sums, prediction_counts)
        total_loss = ce_loss + prediction_losses.sum() + xblk_penalty
        payload = make_log_payload(
            step=step,
            ce_loss=ce_loss,
            prediction_losses=prediction_losses,
            prediction_counts=prediction_counts,
            xblk_penalty=xblk_penalty,
            total_loss=total_loss,
            wall_time_s=wall_time_s,
        )
        last_payload = payload
        if not args.sanity_check_only:
            append_log(args.log_path, payload)
        else:
            print(json.dumps(payload), flush=True)

    if args.sanity_check_only:
        return

    if last_payload is None:
        raise RuntimeError("Training finished without producing any log payload.")

    report = {
        "git_sha": current_git_sha(),
        "git_status_short": current_git_status_short(),
        "parameter_count": count_parameters(model),
        "device": str(device),
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "n_levels": args.n_levels,
            "steps_per_token": args.steps_per_token,
            "d_stream": args.d_stream,
            "noise_std": args.noise_std,
            "loss_type": args.loss_type,
            "lr": args.lr,
            "chunk_size": args.chunk_size,
            "xblk_lambda": args.xblk_lambda,
            "use_cuda_graph": args.use_cuda_graph,
        },
        "cuda_graph": {
            "requested": chunk_trainer.requested_cuda_graph,
            "active": chunk_trainer.graph_active,
            "failure_reason": chunk_trainer.graph_failure_reason,
        },
        "final": last_payload,
    }
    write_json(args.report_path, report)
    print(json.dumps({"stage": "report_written", "report_path": str(args.report_path)}), flush=True)


if __name__ == "__main__":
    main()
