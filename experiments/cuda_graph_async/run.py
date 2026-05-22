from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import statistics

import torch
from torch import Tensor, nn

from core.training import current_git_sha, current_git_status_short, write_json


@dataclass(frozen=True)
class SweepPoint:
    batch_size: int
    seq_len: int
    d_model: int
    num_blocks: int
    timesteps: int


@dataclass(frozen=True)
class TimingStats:
    mean_ms: float
    std_ms: float
    samples_ms: list[float]


@dataclass(frozen=True)
class CorrectnessStats:
    graph_sequential_max_abs_diff: float
    graph_parallel_max_abs_diff: float
    graph_sequential_allclose: bool
    graph_parallel_allclose: bool


DEFAULT_SWEEP = (
    # timesteps=8 to avoid OOM (CUDA Graphs pin all intermediates; ts=32 OOMs at 30GB)
    SweepPoint(batch_size=64, seq_len=128, d_model=128, num_blocks=4, timesteps=8),
    SweepPoint(batch_size=64, seq_len=128, d_model=512, num_blocks=4, timesteps=8),
    SweepPoint(batch_size=256, seq_len=128, d_model=256, num_blocks=4, timesteps=8),
)


class ResidualFeedForwardBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden_dim = 4 * d_model
        self.proj_in = nn.Linear(d_model, hidden_dim)
        self.activation = nn.GELU()
        self.proj_out = nn.Linear(hidden_dim, d_model)

    def forward(self, stream: Tensor) -> Tensor:
        return self.proj_out(self.activation(self.proj_in(stream)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("results.json"),
    )
    return parser.parse_args()


def round_float(value: float) -> float:
    return round(value, 6)


def clone_blocks(state_dict: dict[str, Tensor], *, num_blocks: int, d_model: int, device: torch.device) -> nn.ModuleList:
    blocks = nn.ModuleList([ResidualFeedForwardBlock(d_model) for _ in range(num_blocks)]).to(device)
    blocks.load_state_dict(state_dict)
    blocks.eval()
    return blocks


def build_block_state_dict(*, point: SweepPoint, device: torch.device) -> dict[str, Tensor]:
    seed = point.batch_size + (point.seq_len * 10) + (point.d_model * 100) + (point.num_blocks * 1_000)
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(seed)
        blocks = nn.ModuleList([ResidualFeedForwardBlock(point.d_model) for _ in range(point.num_blocks)]).to(device)
    return blocks.state_dict()


class EagerSequentialVariant:
    def __init__(self, *, blocks: nn.ModuleList, input_state: Tensor, timesteps: int) -> None:
        self.blocks = blocks
        self.input_state = input_state
        self.timesteps = timesteps
        self.output = torch.empty_like(input_state)

    def run(self) -> Tensor:
        state = self.input_state
        for _ in range(self.timesteps):
            step_input = state
            deltas = [block(step_input) for block in self.blocks]
            combined = torch.stack(deltas, dim=0).sum(dim=0)
            state = step_input + combined
        self.output.copy_(state)
        return self.output


class GraphedSequentialVariant:
    def __init__(self, *, blocks: nn.ModuleList, input_state: Tensor, timesteps: int) -> None:
        self.blocks = blocks
        self.input_state = input_state
        self.timesteps = timesteps
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream()
        self.state = torch.empty_like(input_state)
        self.output = torch.empty_like(input_state)
        self.delta_buffers = [torch.empty_like(input_state) for _ in range(len(blocks))]
        self.sum_buffer = torch.empty_like(input_state)
        self._warmup_and_capture()

    def _workload(self) -> None:
        self.state.copy_(self.input_state)
        for _ in range(self.timesteps):
            self.sum_buffer.zero_()
            for block, delta_buffer in zip(self.blocks, self.delta_buffers, strict=True):
                delta_buffer.copy_(block(self.state))
                self.sum_buffer.add_(delta_buffer)
            self.state.add_(self.sum_buffer)
        self.output.copy_(self.state)

    def _warmup_and_capture(self) -> None:
        self.capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.capture_stream):
            for _ in range(3):
                self._workload()
        torch.cuda.current_stream().wait_stream(self.capture_stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self._workload()
        torch.cuda.current_stream().wait_stream(self.capture_stream)
        torch.cuda.synchronize()

    def run(self) -> Tensor:
        self.graph.replay()
        return self.output


class GraphedParallelVariant:
    def __init__(self, *, blocks: nn.ModuleList, input_state: Tensor, timesteps: int) -> None:
        self.blocks = blocks
        self.input_state = input_state
        self.timesteps = timesteps
        self.graph = torch.cuda.CUDAGraph()
        self.capture_stream = torch.cuda.Stream()
        self.worker_streams = [torch.cuda.Stream() for _ in range(len(blocks))]
        self.state = torch.empty_like(input_state)
        self.output = torch.empty_like(input_state)
        self.delta_buffers = [torch.empty_like(input_state) for _ in range(len(blocks))]
        self.sum_buffer = torch.empty_like(input_state)
        self._warmup_and_capture()

    def _workload(self) -> None:
        self.state.copy_(self.input_state)
        for _ in range(self.timesteps):
            current_stream = torch.cuda.current_stream()
            for worker_stream in self.worker_streams:
                worker_stream.wait_stream(current_stream)
            for worker_stream, block, delta_buffer in zip(
                self.worker_streams,
                self.blocks,
                self.delta_buffers,
                strict=True,
            ):
                with torch.cuda.stream(worker_stream):
                    delta_buffer.copy_(block(self.state))
            for worker_stream in self.worker_streams:
                current_stream.wait_stream(worker_stream)
            self.sum_buffer.zero_()
            for delta_buffer in self.delta_buffers:
                self.sum_buffer.add_(delta_buffer)
            self.state.add_(self.sum_buffer)
        self.output.copy_(self.state)

    def _warmup_and_capture(self) -> None:
        self.capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.capture_stream):
            for _ in range(3):
                self._workload()
        torch.cuda.current_stream().wait_stream(self.capture_stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self._workload()
        torch.cuda.current_stream().wait_stream(self.capture_stream)
        torch.cuda.synchronize()

    def run(self) -> Tensor:
        self.graph.replay()
        return self.output


def measure_variant(variant, *, warmup_iters: int, measure_iters: int) -> TimingStats:
    for _ in range(warmup_iters):
        variant.run()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(measure_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        variant.run()
        end_event.record()
        end_event.synchronize()
        samples_ms.append(start_event.elapsed_time(end_event))

    mean_ms = statistics.fmean(samples_ms)
    std_ms = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    return TimingStats(
        mean_ms=round_float(mean_ms),
        std_ms=round_float(std_ms),
        samples_ms=[round_float(sample) for sample in samples_ms],
    )


def benchmark_point(point: SweepPoint, *, warmup_iters: int, measure_iters: int, device: torch.device) -> dict[str, object]:
    input_state = torch.randn(
        point.batch_size,
        point.seq_len,
        point.d_model,
        device=device,
    )
    block_state_dict = build_block_state_dict(point=point, device=device)

    eager_variant = EagerSequentialVariant(
        blocks=clone_blocks(block_state_dict, num_blocks=point.num_blocks, d_model=point.d_model, device=device),
        input_state=input_state,
        timesteps=point.timesteps,
    )
    graph_sequential_variant = GraphedSequentialVariant(
        blocks=clone_blocks(block_state_dict, num_blocks=point.num_blocks, d_model=point.d_model, device=device),
        input_state=input_state,
        timesteps=point.timesteps,
    )
    graph_parallel_variant = GraphedParallelVariant(
        blocks=clone_blocks(block_state_dict, num_blocks=point.num_blocks, d_model=point.d_model, device=device),
        input_state=input_state,
        timesteps=point.timesteps,
    )

    eager_output = eager_variant.run().clone()
    graph_sequential_output = graph_sequential_variant.run().clone()
    graph_parallel_output = graph_parallel_variant.run().clone()
    torch.cuda.synchronize()

    correctness = CorrectnessStats(
        graph_sequential_max_abs_diff=round_float((graph_sequential_output - eager_output).abs().max().item()),
        graph_parallel_max_abs_diff=round_float((graph_parallel_output - eager_output).abs().max().item()),
        graph_sequential_allclose=torch.allclose(graph_sequential_output, eager_output, rtol=1e-5, atol=1e-6),
        graph_parallel_allclose=torch.allclose(graph_parallel_output, eager_output, rtol=1e-5, atol=1e-6),
    )

    eager_stats = measure_variant(eager_variant, warmup_iters=warmup_iters, measure_iters=measure_iters)
    graph_sequential_stats = measure_variant(
        graph_sequential_variant,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
    )
    graph_parallel_stats = measure_variant(
        graph_parallel_variant,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
    )

    speedups = {
        "graph_sequential_vs_eager": round_float(eager_stats.mean_ms / graph_sequential_stats.mean_ms),
        "graph_parallel_vs_graph_sequential": round_float(
            graph_sequential_stats.mean_ms / graph_parallel_stats.mean_ms
        ),
        "graph_parallel_vs_eager": round_float(eager_stats.mean_ms / graph_parallel_stats.mean_ms),
    }

    return {
        "config": asdict(point),
        "correctness": asdict(correctness),
        "variants": {
            "eager_sequential": asdict(eager_stats),
            "cuda_graph_sequential": asdict(graph_sequential_stats),
            "cuda_graph_parallel": asdict(graph_parallel_stats),
        },
        "speedups": speedups,
    }


def print_results_table(results: list[dict[str, object]]) -> None:
    headers = [
        "shape",
        "eager seq (ms)",
        "graph seq (ms)",
        "graph par (ms)",
        "par/seq",
    ]
    rows: list[list[str]] = []
    for result in results:
        config = result["config"]
        variants = result["variants"]
        speedups = result["speedups"]
        eager = variants["eager_sequential"]
        graph_seq = variants["cuda_graph_sequential"]
        graph_par = variants["cuda_graph_parallel"]
        rows.append(
            [
                f"b={config['batch_size']} s={config['seq_len']} d={config['d_model']} k={config['num_blocks']} t={config['timesteps']}",
                f"{eager['mean_ms']:.3f} ± {eager['std_ms']:.3f}",
                f"{graph_seq['mean_ms']:.3f} ± {graph_seq['std_ms']:.3f}",
                f"{graph_par['mean_ms']:.3f} ± {graph_par['std_ms']:.3f}",
                f"{speedups['graph_parallel_vs_graph_sequential']:.3f}x",
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    divider = "  ".join("-" * width for width in widths)
    print(format_row(headers))
    print(divider)
    for row in rows:
        print(format_row(row))


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    with torch.inference_mode():
        results = [
            benchmark_point(
                point,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                device=device,
            )
            for point in DEFAULT_SWEEP
        ]

    payload = {
        "metadata": {
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "warmup_iterations": args.warmup_iters,
            "measure_iterations": args.measure_iters,
            "git_sha": current_git_sha(),
            "git_status_short": current_git_status_short(),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)
    print_results_table(results)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
