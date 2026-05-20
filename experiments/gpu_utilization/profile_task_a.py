from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from experiments.gpu_utilization import benchmark


DEFAULT_TEXT_FILE = Path("experiments") / "corpora.ignore" / "tinyshakespeare_input.txt"


@dataclass(frozen=True)
class ModeMetrics:
    mode: str
    batch_size: int
    warmup_steps: int
    timed_steps: int
    mean_wall_ms: float
    median_wall_ms: float
    p95_wall_ms: float
    mean_cuda_ms: float
    median_cuda_ms: float
    mean_host_wait_ms: float
    mean_h2d_copy_ms: float
    gpu_active_ratio: float
    tokens_per_second: float
    loss_mean: float
    loss_stdev: float
    samples_collected: int
    gpu_util_mean: float | None
    gpu_util_max: float | None
    power_draw_mean_watts: float | None
    power_draw_max_watts: float | None


@dataclass(frozen=True)
class TaskAResult:
    git_sha: str
    python_version: str
    torch_version: str
    cuda_device_name: str
    dtype: str
    family: str
    context_size: int
    target_parameters: int
    parameter_count: int
    config: dict[str, int]
    text_file: str
    synthetic: ModeMetrics
    real_dataloader: ModeMetrics


class SequenceWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, text: str, context_size: int) -> None:
        vocab = sorted(set(text))
        stoi = {char: index for index, char in enumerate(vocab)}
        encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)
        if len(vocab) != benchmark.VOCAB_SIZE:
            raise ValueError(
                f"Expected TinyShakespeare vocab size {benchmark.VOCAB_SIZE}, got {len(vocab)}."
            )
        if encoded.numel() <= context_size:
            raise ValueError(
                f"Need text longer than context size {context_size}, got {encoded.numel()} tokens."
            )
        self.inputs = torch.stack(
            [encoded[start : start + context_size] for start in range(encoded.numel() - context_size)]
        )
        self.targets = torch.stack(
            [
                encoded[start + 1 : start + context_size + 1]
                for start in range(encoded.numel() - context_size)
            ]
        )

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.inputs[index], self.targets[index]


class NvidiaSmiSampler:
    def __init__(self, sample_period_ms: int) -> None:
        self.sample_period_ms = sample_period_ms
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float, float]] = []

    def start(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
            "-lms",
            str(self.sample_period_ms),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            self._process = None
            return

        def reader() -> None:
            assert self._process is not None
            assert self._process.stdout is not None
            for line in self._process.stdout:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 2:
                    continue
                try:
                    util = float(parts[0])
                    power = float(parts[1])
                except ValueError:
                    continue
                self._samples.append((util, power))

        self._thread = threading.Thread(target=reader, daemon=True)
        self._thread.start()

    def stop(self) -> list[tuple[float, float]]:
        if self._process is None:
            return []
        self._process.terminate()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return list(self._samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--family", default="gru")
    parser.add_argument("--context-size", type=int, default=32)
    parser.add_argument("--target-parameters", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--timed-steps", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--telemetry-period-ms", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def cycle_loader(loader: DataLoader[tuple[Tensor, Tensor]]) -> Iterator[tuple[Tensor, Tensor]]:
    while True:
        for batch in loader:
            yield batch


def summarize_samples(samples: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples)
    mean_value = statistics.fmean(samples)
    median_value = statistics.median(ordered)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return mean_value, median_value, ordered[p95_index]


def summarize_gpu_samples(samples: list[tuple[float, float]]) -> tuple[float | None, float | None, float | None, float | None]:
    if not samples:
        return None, None, None, None
    utils = [util for util, _ in samples]
    power = [watts for _, watts in samples]
    return statistics.fmean(utils), max(utils), statistics.fmean(power), max(power)


def run_synthetic_mode(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    context_size: int,
    warmup_steps: int,
    timed_steps: int,
    telemetry_period_ms: int,
) -> ModeMetrics:
    inputs = torch.randint(
        benchmark.VOCAB_SIZE,
        (batch_size, context_size),
        device=benchmark.DEVICE,
        dtype=torch.long,
    )
    targets = torch.randint(
        benchmark.VOCAB_SIZE,
        (batch_size, context_size),
        device=benchmark.DEVICE,
        dtype=torch.long,
    )
    for _ in range(warmup_steps):
        benchmark.training_step(model, optimizer, inputs, targets)
    torch.cuda.synchronize()

    wall_ms: list[float] = []
    cuda_ms: list[float] = []
    losses: list[float] = []
    sampler = NvidiaSmiSampler(sample_period_ms=telemetry_period_ms)
    sampler.start()
    for _ in range(timed_steps):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        started_at = time.perf_counter()
        start_event.record()
        losses.append(benchmark.training_step(model, optimizer, inputs, targets))
        end_event.record()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - started_at) * 1000.0)
        cuda_ms.append(start_event.elapsed_time(end_event))
    gpu_samples = sampler.stop()
    mean_wall_ms, median_wall_ms, p95_wall_ms = summarize_samples(wall_ms)
    mean_cuda_ms, median_cuda_ms, _ = summarize_samples(cuda_ms)
    gpu_util_mean, gpu_util_max, power_mean, power_max = summarize_gpu_samples(gpu_samples)
    return ModeMetrics(
        mode="synthetic",
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
        mean_wall_ms=mean_wall_ms,
        median_wall_ms=median_wall_ms,
        p95_wall_ms=p95_wall_ms,
        mean_cuda_ms=mean_cuda_ms,
        median_cuda_ms=median_cuda_ms,
        mean_host_wait_ms=0.0,
        mean_h2d_copy_ms=0.0,
        gpu_active_ratio=mean_cuda_ms / mean_wall_ms,
        tokens_per_second=(batch_size * context_size * timed_steps) / (sum(wall_ms) / 1000.0),
        loss_mean=statistics.fmean(losses),
        loss_stdev=statistics.pstdev(losses) if len(losses) > 1 else 0.0,
        samples_collected=len(gpu_samples),
        gpu_util_mean=gpu_util_mean,
        gpu_util_max=gpu_util_max,
        power_draw_mean_watts=power_mean,
        power_draw_max_watts=power_max,
    )


def run_real_mode(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader[tuple[Tensor, Tensor]],
    batch_size: int,
    context_size: int,
    warmup_steps: int,
    timed_steps: int,
    telemetry_period_ms: int,
) -> ModeMetrics:
    batches = cycle_loader(loader)
    for _ in range(warmup_steps):
        inputs_cpu, targets_cpu = next(batches)
        inputs = inputs_cpu.to(benchmark.DEVICE, non_blocking=True)
        targets = targets_cpu.to(benchmark.DEVICE, non_blocking=True)
        benchmark.training_step(model, optimizer, inputs, targets)
    torch.cuda.synchronize()

    wall_ms: list[float] = []
    host_wait_ms: list[float] = []
    h2d_copy_ms: list[float] = []
    cuda_ms: list[float] = []
    losses: list[float] = []
    sampler = NvidiaSmiSampler(sample_period_ms=telemetry_period_ms)
    sampler.start()
    for _ in range(timed_steps):
        step_started_at = time.perf_counter()
        batch_fetch_started_at = time.perf_counter()
        inputs_cpu, targets_cpu = next(batches)
        host_wait_ms.append((time.perf_counter() - batch_fetch_started_at) * 1000.0)

        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)

        copy_start.record()
        inputs = inputs_cpu.to(benchmark.DEVICE, non_blocking=True)
        targets = targets_cpu.to(benchmark.DEVICE, non_blocking=True)
        copy_end.record()

        step_start.record()
        losses.append(benchmark.training_step(model, optimizer, inputs, targets))
        step_end.record()

        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - step_started_at) * 1000.0)
        h2d_copy_ms.append(copy_start.elapsed_time(copy_end))
        cuda_ms.append(step_start.elapsed_time(step_end))
    gpu_samples = sampler.stop()

    mean_wall_ms, median_wall_ms, p95_wall_ms = summarize_samples(wall_ms)
    mean_cuda_ms, median_cuda_ms, _ = summarize_samples(cuda_ms)
    gpu_util_mean, gpu_util_max, power_mean, power_max = summarize_gpu_samples(gpu_samples)
    return ModeMetrics(
        mode="real_dataloader",
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
        mean_wall_ms=mean_wall_ms,
        median_wall_ms=median_wall_ms,
        p95_wall_ms=p95_wall_ms,
        mean_cuda_ms=mean_cuda_ms,
        median_cuda_ms=median_cuda_ms,
        mean_host_wait_ms=statistics.fmean(host_wait_ms),
        mean_h2d_copy_ms=statistics.fmean(h2d_copy_ms),
        gpu_active_ratio=mean_cuda_ms / mean_wall_ms,
        tokens_per_second=(batch_size * context_size * timed_steps) / (sum(wall_ms) / 1000.0),
        loss_mean=statistics.fmean(losses),
        loss_stdev=statistics.pstdev(losses) if len(losses) > 1 else 0.0,
        samples_collected=len(gpu_samples),
        gpu_util_mean=gpu_util_mean,
        gpu_util_max=gpu_util_max,
        power_draw_mean_watts=power_mean,
        power_draw_max_watts=power_max,
    )


def render_summary(result: TaskAResult) -> str:
    lines = [
        f"family={result.family} context={result.context_size} params={result.parameter_count:,} dtype={result.dtype}",
        "mode             batch  wall_ms  cuda_ms  host_ms  h2d_ms  active  tok/s   gpu%   watts",
        "---------------  -----  -------  -------  -------  ------  ------  ------  -----  -----",
    ]
    for metrics in (result.synthetic, result.real_dataloader):
        gpu_util = "n/a" if metrics.gpu_util_mean is None else f"{metrics.gpu_util_mean:5.1f}"
        watts = (
            "n/a"
            if metrics.power_draw_mean_watts is None
            else f"{metrics.power_draw_mean_watts:5.1f}"
        )
        lines.append(
            f"{metrics.mode:<15}  {metrics.batch_size:>5}  {metrics.mean_wall_ms:>7.2f}  "
            f"{metrics.mean_cuda_ms:>7.2f}  {metrics.mean_host_wait_ms:>7.2f}  "
            f"{metrics.mean_h2d_copy_ms:>6.2f}  {metrics.gpu_active_ratio * 100:>5.1f}%  "
            f"{metrics.tokens_per_second:>6.0f}  {gpu_util:>5}  {watts:>5}"
        )
    return "\n".join(lines)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Task A profiling.")

    args = parse_args()
    if args.smoke:
        args.warmup_steps = 4
        args.timed_steps = 12

    repo_root = Path(__file__).resolve().parents[2]
    output_path = (
        args.output
        or repo_root
        / "experiments"
        / "gpu_utilization"
        / "artifacts"
        / "profiling"
        / "task_a_synthetic_vs_real.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_file = args.text_file or (repo_root / DEFAULT_TEXT_FILE)
    text = text_file.read_text(encoding="utf-8")

    config = benchmark.search_family_config(
        family=args.family,
        target_parameters=args.target_parameters,
        context_size=args.context_size,
    )
    parameter_count = benchmark.parameter_count_for_config(args.family, args.context_size, config)
    batch_size = args.batch_size or benchmark.tune_batch_size(args.family, config, args.context_size)

    synthetic_model = benchmark.build_model(args.family, args.context_size, config).to(benchmark.DEVICE)
    dtype = str(next(synthetic_model.parameters()).dtype)
    synthetic_optimizer = torch.optim.AdamW(synthetic_model.parameters(), lr=0.001)
    synthetic_metrics = run_synthetic_mode(
        model=synthetic_model,
        optimizer=synthetic_optimizer,
        batch_size=batch_size,
        context_size=args.context_size,
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
        telemetry_period_ms=args.telemetry_period_ms,
    )
    del synthetic_model
    del synthetic_optimizer
    benchmark.clear_cuda_memory()

    dataset = SequenceWindowDataset(text=text, context_size=args.context_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    real_model = benchmark.build_model(args.family, args.context_size, config).to(benchmark.DEVICE)
    real_optimizer = torch.optim.AdamW(real_model.parameters(), lr=0.001)
    real_metrics = run_real_mode(
        model=real_model,
        optimizer=real_optimizer,
        loader=loader,
        batch_size=batch_size,
        context_size=args.context_size,
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
        telemetry_period_ms=args.telemetry_period_ms,
    )
    del real_model
    del real_optimizer
    benchmark.clear_cuda_memory()

    result = TaskAResult(
        git_sha=benchmark.current_git_sha(repo_root),
        python_version=sys.version,
        torch_version=torch.__version__,
        cuda_device_name=torch.cuda.get_device_name(0),
        dtype=dtype,
        family=args.family,
        context_size=args.context_size,
        target_parameters=args.target_parameters,
        parameter_count=parameter_count,
        config={key: value for key, value in asdict(config).items() if value is not None},
        text_file=str(text_file),
        synthetic=synthetic_metrics,
        real_dataloader=real_metrics,
    )
    summary = render_summary(result)
    print(summary)
    output_path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
