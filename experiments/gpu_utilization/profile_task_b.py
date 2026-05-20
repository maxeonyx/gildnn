from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from experiments.gpu_utilization import benchmark


@dataclass(frozen=True)
class ProfileSpec:
    slug: str
    family: str
    target_parameters: int
    context_size: int


@dataclass(frozen=True)
class KernelSummary:
    name: str
    count: int
    self_device_time_us: float
    device_time_us: float


@dataclass(frozen=True)
class ConfigProfileSummary:
    slug: str
    family: str
    target_parameters: int
    context_size: int
    batch_size: int
    parameter_count: int
    dtype: str
    config: dict[str, int]
    active_steps: int
    mean_wall_ms_per_step: float
    median_wall_ms_per_step: float
    mean_loss: float
    cpu_self_time_us_total: float
    cuda_self_time_us_total: float
    cuda_time_share: float
    cpu_overhead_share: float
    kernel_launch_count: int
    kernel_launches_per_step: float
    kernel_launch_cpu_overhead_us_total: float
    kernel_launch_cpu_overhead_us_per_step: float
    cuda_kernel_event_count: int
    cuda_kernel_events_per_step: float
    top_kernels_by_self_device_time: list[KernelSummary]
    tensor_core_kernel_names: list[str]
    likely_tensor_core_usage: bool
    trace_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--active-steps", type=int, default=6)
    parser.add_argument("--max-top-kernels", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def profiler_specs() -> list[ProfileSpec]:
    return [
        ProfileSpec(
            slug="transformer_1m_ctx32",
            family="transformer",
            target_parameters=1_000_000,
            context_size=32,
        ),
        ProfileSpec(
            slug="gru_1m_ctx32",
            family="gru",
            target_parameters=1_000_000,
            context_size=32,
        ),
        ProfileSpec(
            slug="gru_10m_ctx32",
            family="gru",
            target_parameters=10_000_000,
            context_size=32,
        ),
    ]


def device_type_name(item: object) -> str | None:
    device_type = getattr(item, "device_type", None)
    return getattr(device_type, "name", None)


def kernel_time_us(item: object) -> float:
    return float(getattr(item, "self_device_time_total", 0.0))


def aggregate_times(items: list[object]) -> tuple[float, float]:
    cpu_total = sum(float(getattr(item, "self_cpu_time_total", 0.0)) for item in items)
    cuda_total = sum(kernel_time_us(item) for item in items if device_type_name(item) == "CUDA")
    return cpu_total, cuda_total


def summarize_top_kernels(items: list[object], limit: int) -> list[KernelSummary]:
    kernels = [item for item in items if device_type_name(item) == "CUDA"]
    kernels.sort(key=kernel_time_us, reverse=True)
    return [
        KernelSummary(
            name=str(item.key),
            count=int(item.count),
            self_device_time_us=kernel_time_us(item),
            device_time_us=float(getattr(item, "device_time_total", 0.0)),
        )
        for item in kernels[:limit]
    ]


def detect_tensor_core_kernels(kernels: list[KernelSummary]) -> list[str]:
    patterns = ("1688", "tensorop", "hmma", "wmma", "tf32", "bf16", "fp16")
    matches: list[str] = []
    for kernel in kernels:
        lowered = kernel.name.lower()
        if any(pattern in lowered for pattern in patterns):
            matches.append(kernel.name)
    return matches


def profile_config(
    spec: ProfileSpec,
    *,
    output_dir: Path,
    warmup_steps: int,
    active_steps: int,
    max_top_kernels: int,
) -> ConfigProfileSummary:
    benchmark.clear_cuda_memory()
    config = benchmark.search_family_config(
        family=spec.family,
        target_parameters=spec.target_parameters,
        context_size=spec.context_size,
    )
    parameter_count = benchmark.parameter_count_for_config(spec.family, spec.context_size, config)
    batch_size = benchmark.tune_batch_size(spec.family, config, spec.context_size)
    model = benchmark.build_model(spec.family, spec.context_size, config).to(benchmark.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    dtype = str(next(model.parameters()).dtype)
    inputs = torch.randint(
        benchmark.VOCAB_SIZE,
        (batch_size, spec.context_size),
        device=benchmark.DEVICE,
        dtype=torch.long,
    )
    targets = torch.randint(
        benchmark.VOCAB_SIZE,
        (batch_size, spec.context_size),
        device=benchmark.DEVICE,
        dtype=torch.long,
    )

    for _ in range(warmup_steps):
        benchmark.training_step(model, optimizer, inputs, targets)
    torch.cuda.synchronize()

    losses: list[float] = []
    wall_ms: list[float] = []
    trace_path = output_dir / f"{spec.slug}.trace.json"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(active_steps):
            started_at = time.perf_counter()
            losses.append(benchmark.training_step(model, optimizer, inputs, targets))
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - started_at) * 1000.0)
    prof.export_chrome_trace(str(trace_path))

    items = list(prof.key_averages())
    cpu_self_time_us_total, cuda_self_time_us_total = aggregate_times(items)
    total_recorded_time_us = cpu_self_time_us_total + cuda_self_time_us_total
    launch_item = next((item for item in items if str(item.key) == "cudaLaunchKernel"), None)
    launch_count = 0 if launch_item is None else int(launch_item.count)
    launch_cpu_overhead_us_total = 0.0 if launch_item is None else float(launch_item.self_cpu_time_total)
    cuda_items = [item for item in items if device_type_name(item) == "CUDA"]
    top_kernels = summarize_top_kernels(items, max_top_kernels)
    tensor_core_kernels = detect_tensor_core_kernels(top_kernels)

    del model
    del optimizer
    benchmark.clear_cuda_memory()

    return ConfigProfileSummary(
        slug=spec.slug,
        family=spec.family,
        target_parameters=spec.target_parameters,
        context_size=spec.context_size,
        batch_size=batch_size,
        parameter_count=parameter_count,
        dtype=dtype,
        config={key: value for key, value in asdict(config).items() if value is not None},
        active_steps=active_steps,
        mean_wall_ms_per_step=statistics.fmean(wall_ms),
        median_wall_ms_per_step=statistics.median(wall_ms),
        mean_loss=statistics.fmean(losses),
        cpu_self_time_us_total=cpu_self_time_us_total,
        cuda_self_time_us_total=cuda_self_time_us_total,
        cuda_time_share=0.0 if total_recorded_time_us == 0 else cuda_self_time_us_total / total_recorded_time_us,
        cpu_overhead_share=0.0 if total_recorded_time_us == 0 else cpu_self_time_us_total / total_recorded_time_us,
        kernel_launch_count=launch_count,
        kernel_launches_per_step=launch_count / active_steps,
        kernel_launch_cpu_overhead_us_total=launch_cpu_overhead_us_total,
        kernel_launch_cpu_overhead_us_per_step=launch_cpu_overhead_us_total / active_steps,
        cuda_kernel_event_count=sum(int(item.count) for item in cuda_items),
        cuda_kernel_events_per_step=sum(int(item.count) for item in cuda_items) / active_steps,
        top_kernels_by_self_device_time=top_kernels,
        tensor_core_kernel_names=tensor_core_kernels,
        likely_tensor_core_usage=bool(tensor_core_kernels),
        trace_path=str(trace_path),
    )


def render_summary(summary: ConfigProfileSummary) -> str:
    first_kernel = summary.top_kernels_by_self_device_time[0].name if summary.top_kernels_by_self_device_time else "n/a"
    tensor_core = ", ".join(summary.tensor_core_kernel_names[:3]) if summary.tensor_core_kernel_names else "none detected"
    return (
        f"{summary.slug}: batch={summary.batch_size} wall={summary.mean_wall_ms_per_step:.2f}ms/step "
        f"cuda_share={summary.cuda_time_share * 100:.1f}% launches/step={summary.kernel_launches_per_step:.1f} "
        f"launch_cpu_overhead={summary.kernel_launch_cpu_overhead_us_per_step / 1000.0:.2f}ms/step "
        f"top_kernel={first_kernel} tensor_core={tensor_core}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Task B profiling.")

    args = parse_args()
    if args.smoke:
        args.warmup_steps = 2
        args.active_steps = 2

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or (
        repo_root / "experiments" / "gpu_utilization" / "artifacts" / "profiling" / "task_b"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = profiler_specs()
    if args.smoke:
        specs = specs[:1]

    summaries = [
        profile_config(
            spec,
            output_dir=output_dir,
            warmup_steps=args.warmup_steps,
            active_steps=args.active_steps,
            max_top_kernels=args.max_top_kernels,
        )
        for spec in specs
    ]
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "git_sha": benchmark.current_git_sha(repo_root),
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_device_name": torch.cuda.get_device_name(0),
                "warmup_steps": args.warmup_steps,
                "active_steps": args.active_steps,
                "profiles": [asdict(summary) for summary in summaries],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for summary in summaries:
        print(render_summary(summary))


if __name__ == "__main__":
    main()
