from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch import Tensor


DEFAULT_NUM_BLOCKS = [4, 8, 16]
DEFAULT_D_MODELS = [64, 128, 256, 512]
DEFAULT_BATCH_SIZE = 64
DEFAULT_SEQ_LEN = 128
DEFAULT_TIMESTEPS = 32
DEFAULT_WARMUP = 10
DEFAULT_ITERS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-blocks", type=int, nargs="*", default=DEFAULT_NUM_BLOCKS)
    parser.add_argument("--d-model", type=int, nargs="*", default=DEFAULT_D_MODELS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "results.json",
    )
    return parser.parse_args()


def ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for experiments.async_wallclock, but torch.cuda.is_available() is False.")


def validate_args(args: argparse.Namespace) -> None:
    values = [
        ("batch_size", args.batch_size),
        ("seq_len", args.seq_len),
        ("timesteps", args.timesteps),
        ("warmup", args.warmup),
        ("iters", args.iters),
    ]
    errors = [f"{name} must be positive, got {value}." for name, value in values if value <= 0]
    errors.extend(f"num_blocks entries must be positive, got {value}." for value in args.num_blocks if value <= 0)
    errors.extend(f"d_model entries must be positive, got {value}." for value in args.d_model if value <= 0)
    if errors:
        raise ValueError("Argument validation failed:\n" + "\n".join(errors))


def current_git_sha(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def current_git_status_short(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def make_block_parameters(num_blocks: int, d_model: int, device: torch.device) -> list[tuple[Tensor, Tensor, Tensor, Tensor]]:
    hidden = 4 * d_model
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    parameters: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    for _ in range(num_blocks):
        w1 = torch.randn((hidden, d_model), generator=generator, dtype=torch.float32)
        b1 = torch.randn((hidden,), generator=generator, dtype=torch.float32)
        w2 = torch.randn((d_model, hidden), generator=generator, dtype=torch.float32)
        b2 = torch.randn((d_model,), generator=generator, dtype=torch.float32)
        parameters.append(
            (
                (w1 / math.sqrt(d_model)).to(device=device),
                (b1 / math.sqrt(d_model)).to(device=device),
                (w2 / math.sqrt(hidden)).to(device=device),
                (b2 / math.sqrt(hidden)).to(device=device),
            )
        )
    return parameters


def make_initial_state(batch_size: int, seq_len: int, d_model: int, device: torch.device) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    state = torch.randn((batch_size, seq_len, d_model), generator=generator, dtype=torch.float32)
    return (state / math.sqrt(d_model)).to(device=device)


def block_delta(state: Tensor, parameters: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
    w1, b1, w2, b2 = parameters
    hidden = F.linear(state, w1, b1)
    activated = F.gelu(hidden, approximate="tanh")
    return F.linear(activated, w2, b2)


def prepare_sync_sequential(
    initial_state: Tensor,
    parameters: list[tuple[Tensor, Tensor, Tensor, Tensor]],
    timesteps: int,
) -> tuple[callable[[], torch.cuda.Event], dict[str, object]]:
    state = torch.empty_like(initial_state)
    final_event = torch.cuda.Event()

    def run_once() -> torch.cuda.Event:
        state.copy_(initial_state)
        for _ in range(timesteps):
            for block_parameters in parameters:
                state.add_(block_delta(state, block_parameters))
        final_event.record(torch.cuda.current_stream())
        return final_event

    return run_once, {"state": state}


def prepare_sync_parallel(
    initial_state: Tensor,
    parameters: list[tuple[Tensor, Tensor, Tensor, Tensor]],
    timesteps: int,
) -> tuple[callable[[], torch.cuda.Event], dict[str, object]]:
    num_blocks = len(parameters)
    state = torch.empty_like(initial_state)
    snapshot = torch.empty_like(initial_state)
    delta_buffers = [torch.empty_like(initial_state) for _ in range(num_blocks)]
    streams = [torch.cuda.Stream() for _ in range(num_blocks)]
    snapshot_ready = torch.cuda.Event()
    block_done = [torch.cuda.Event() for _ in range(num_blocks)]
    final_event = torch.cuda.Event()

    def run_once() -> torch.cuda.Event:
        default_stream = torch.cuda.current_stream()
        state.copy_(initial_state)
        for _ in range(timesteps):
            snapshot.copy_(state)
            snapshot_ready.record(default_stream)
            for block_index, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    stream.wait_event(snapshot_ready)
                    delta_buffers[block_index].copy_(block_delta(snapshot, parameters[block_index]))
                    block_done[block_index].record(stream)
            for event in block_done:
                default_stream.wait_event(event)
            state.copy_(snapshot)
            for delta_buffer in delta_buffers:
                state.add_(delta_buffer)
        final_event.record(default_stream)
        return final_event

    return run_once, {
        "state": state,
        "snapshot": snapshot,
        "delta_buffers": delta_buffers,
        "streams": streams,
    }


def prepare_async_pipelined(
    initial_state: Tensor,
    parameters: list[tuple[Tensor, Tensor, Tensor, Tensor]],
    timesteps: int,
) -> tuple[callable[[], torch.cuda.Event], dict[str, object]]:
    num_blocks = len(parameters)
    state = torch.empty_like(initial_state)
    snapshot_buffers = [torch.empty_like(initial_state) for _ in range(num_blocks)]
    delta_buffers = [torch.empty_like(initial_state) for _ in range(num_blocks)]
    block_streams = [torch.cuda.Stream() for _ in range(num_blocks)]
    commit_stream = torch.cuda.Stream()
    compute_done = [torch.cuda.Event() for _ in range(num_blocks)]
    commit_done = [torch.cuda.Event() for _ in range(num_blocks)]
    final_event = torch.cuda.Event()

    def run_once() -> torch.cuda.Event:
        default_stream = torch.cuda.current_stream()
        state.copy_(initial_state)
        for block_index in range(num_blocks):
            commit_done[block_index].record(default_stream)

        for _ in range(timesteps):
            for block_index, stream in enumerate(block_streams):
                with torch.cuda.stream(stream):
                    stream.wait_event(commit_done[block_index])
                    snapshot_buffers[block_index].copy_(state)
                    delta_buffers[block_index].copy_(block_delta(snapshot_buffers[block_index], parameters[block_index]))
                    compute_done[block_index].record(stream)

                with torch.cuda.stream(commit_stream):
                    commit_stream.wait_event(compute_done[block_index])
                    state.add_(delta_buffers[block_index])
                    commit_done[block_index].record(commit_stream)

        final_event.record(commit_stream)
        return final_event

    return run_once, {
        "state": state,
        "snapshot_buffers": snapshot_buffers,
        "delta_buffers": delta_buffers,
        "block_streams": block_streams,
        "commit_stream": commit_stream,
    }


def measure_variant(run_once: callable[[], torch.cuda.Event], *, warmup: int, iters: int) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            run_once().synchronize()

        samples_ms: list[float] = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream())
            final_event = run_once()
            torch.cuda.current_stream().wait_event(final_event)
            end.record(torch.cuda.current_stream())
            end.synchronize()
            samples_ms.append(start.elapsed_time(end))

    return {
        "mean_ms": statistics.fmean(samples_ms),
        "std_ms": statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
    }


def benchmark_config(
    *,
    num_blocks: int,
    d_model: int,
    batch_size: int,
    seq_len: int,
    timesteps: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, object]:
    parameters = make_block_parameters(num_blocks, d_model, device)
    initial_state = make_initial_state(batch_size, seq_len, d_model, device)

    sequential, _ = prepare_sync_sequential(initial_state, parameters, timesteps)
    parallel, _ = prepare_sync_parallel(initial_state, parameters, timesteps)
    async_pipelined, _ = prepare_async_pipelined(initial_state, parameters, timesteps)

    sequential_stats = measure_variant(sequential, warmup=warmup, iters=iters)
    parallel_stats = measure_variant(parallel, warmup=warmup, iters=iters)
    async_stats = measure_variant(async_pipelined, warmup=warmup, iters=iters)

    baseline_ms = parallel_stats["mean_ms"]
    return {
        "config": {
            "num_blocks": num_blocks,
            "d_model": d_model,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "timesteps": timesteps,
            "warmup": warmup,
            "iters": iters,
        },
        "variants": {
            "sync-sequential": {
                **sequential_stats,
                "speedup_vs_sync_parallel": baseline_ms / sequential_stats["mean_ms"],
            },
            "sync-parallel": {
                **parallel_stats,
                "speedup_vs_sync_parallel": 1.0,
            },
            "async-pipelined": {
                **async_stats,
                "speedup_vs_sync_parallel": baseline_ms / async_stats["mean_ms"],
            },
        },
    }


def format_config_report(result: dict[str, object]) -> str:
    config = result["config"]
    variants = result["variants"]
    sequential = variants["sync-sequential"]
    parallel = variants["sync-parallel"]
    async_pipelined = variants["async-pipelined"]
    return "\n".join(
        [
            f"Config: num_blocks={config['num_blocks']}, d_model={config['d_model']}",
            f"  sync-sequential:  {sequential['mean_ms']:8.2f} ms (±{sequential['std_ms']:.2f})  [speedup: {sequential['speedup_vs_sync_parallel']:.2f}x vs parallel]",
            f"  sync-parallel:    {parallel['mean_ms']:8.2f} ms (±{parallel['std_ms']:.2f})  [baseline]",
            f"  async-pipelined:  {async_pipelined['mean_ms']:8.2f} ms (±{async_pipelined['std_ms']:.2f})  [speedup: {async_pipelined['speedup_vs_sync_parallel']:.2f}x vs parallel]",
        ]
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    ensure_cuda()

    device = torch.device("cuda")
    repo_root = Path(__file__).resolve().parents[2]
    status_short = current_git_status_short(repo_root)

    torch.cuda.init()
    torch.cuda.synchronize(device)

    results = []
    for num_blocks in args.num_blocks:
        for d_model in args.d_model:
            result = benchmark_config(
                num_blocks=num_blocks,
                d_model=d_model,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                timesteps=args.timesteps,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            results.append(result)
            print(format_config_report(result))

    payload = {
        "environment": {
            "git_sha": current_git_sha(repo_root),
            "git_working_tree_clean": status_short == [],
            "git_status_short": status_short,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_device_name": torch.cuda.get_device_name(device),
        },
        "sweep": {
            "num_blocks": args.num_blocks,
            "d_model": args.d_model,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "timesteps": args.timesteps,
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "results": results,
    }
    write_json(args.output, payload)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
