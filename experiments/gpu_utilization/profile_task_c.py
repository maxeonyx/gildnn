from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from experiments.gpu_utilization import benchmark


@dataclass(frozen=True)
class SweepPoint:
    family: str
    target_parameters: int
    context_size: int
    parameter_count: int
    batch_size: int
    timed_steps: int
    config: dict[str, int]
    train: dict[str, float | int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-parameters", type=int, default=1_000_000)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--timed-steps", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def contexts(smoke: bool) -> list[int]:
    return [32, 128] if smoke else [32, 64, 128, 256, 512, 1024]


def measure_point(
    *,
    family: str,
    target_parameters: int,
    context_size: int,
    warmup_steps: int,
    timed_steps: int,
) -> SweepPoint:
    config = benchmark.search_family_config(
        family=family,
        target_parameters=target_parameters,
        context_size=context_size,
    )
    parameter_count = benchmark.parameter_count_for_config(family, context_size, config)
    batch_size = benchmark.tune_batch_size(family, config, context_size)
    train = benchmark.measure_training(
        family=family,
        config=config,
        context_size=context_size,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
    )
    return SweepPoint(
        family=family,
        target_parameters=target_parameters,
        context_size=context_size,
        parameter_count=parameter_count,
        batch_size=batch_size,
        timed_steps=timed_steps,
        config={key: value for key, value in asdict(config).items() if value is not None},
        train=asdict(train),
    )


def render_summary(points: list[SweepPoint]) -> str:
    lines = [
        "family       ctx   params      batch  train tok/s  ms/step  peak VRAM",
        "-----------  ----  ----------  -----  -----------  -------  ---------",
    ]
    for point in points:
        lines.append(
            f"{point.family:<11}  {point.context_size:>4}  {point.parameter_count:>10,}  "
            f"{point.batch_size:>5}  {point.train['tokens_per_second']:>11,.0f}  "
            f"{point.train['milliseconds_per_step']:>7.2f}  "
            f"{point.train['peak_vram_bytes'] / (1024**3):>8.2f}G"
        )
    return "\n".join(lines)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Task C profiling.")

    args = parse_args()
    if args.smoke:
        args.warmup_steps = 2
        args.timed_steps = 6

    repo_root = Path(__file__).resolve().parents[2]
    output_path = (
        args.output
        or repo_root
        / "experiments"
        / "gpu_utilization"
        / "artifacts"
        / "profiling"
        / "task_c_sequence_sweep.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points = [
        measure_point(
            family=family,
            target_parameters=args.target_parameters,
            context_size=context_size,
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
        )
        for family in ("transformer", "gru")
        for context_size in contexts(args.smoke)
    ]
    summary = render_summary(points)
    print(summary)
    output_path.write_text(
        json.dumps(
            {
                "git_sha": benchmark.current_git_sha(repo_root),
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_device_name": torch.cuda.get_device_name(0),
                "target_parameters": args.target_parameters,
                "warmup_steps": args.warmup_steps,
                "timed_steps": args.timed_steps,
                "results": [asdict(point) for point in points],
                "summary_table": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
