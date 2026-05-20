from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch import Tensor

from .model import DenseModuleBank, PrototypeConfig, count_parameters, make_initial_memory


@dataclass(frozen=True)
class RunOutput:
    version_previews: list[dict[str, object]]
    tick_summaries: list[dict[str, object]]
    full_version_history: list[Tensor]
    aggregate_deltas: list[Tensor]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def round_float(value: float) -> float:
    return round(float(value), 6)


def preview_tensor(tensor: Tensor) -> list[list[list[float]]]:
    detached = tensor.detach().cpu()
    return [
        [
            [round_float(value) for value in slot]
            for slot in batch
        ]
        for batch in detached.tolist()
    ]


def slot0_preview(tensor: Tensor) -> list[float]:
    return [round_float(value) for value in tensor.detach().cpu()[0, 0].tolist()]


def run_variant(
    *,
    module_bank: DenseModuleBank,
    initial_memory: Tensor,
    read_lags: list[int],
    label: str,
    config: PrototypeConfig,
) -> RunOutput:
    history: list[Tensor] = [initial_memory.clone()]
    tick_summaries: list[dict[str, object]] = []
    aggregate_deltas: list[Tensor] = []

    for tick in range(config.num_ticks):
        latest_version_id = len(history) - 1
        read_version_ids = [max(0, latest_version_id - lag) for lag in read_lags]
        visible_memory = torch.stack([history[version_id] for version_id in read_version_ids], dim=0)
        module_deltas = module_bank(visible_memory)
        aggregate_delta = module_deltas.sum(dim=0)
        aggregate_deltas.append(aggregate_delta)
        next_memory = history[-1] + aggregate_delta
        history.append(next_memory)

        module_rows = []
        for module_index in range(config.num_modules):
            read_version_id = read_version_ids[module_index]
            read_memory = history[read_version_id]
            module_rows.append(
                {
                    "module_index": module_index,
                    "read_version_id": read_version_id,
                    "latest_version_id": latest_version_id,
                    "read_slot0": slot0_preview(read_memory),
                    "delta_slot0": slot0_preview(module_deltas[module_index]),
                    "max_read_vs_latest_abs_diff": round_float(
                        (read_memory - history[-1]).abs().max().item()
                    ),
                }
            )

        tick_summaries.append(
            {
                "label": label,
                "tick": tick,
                "latest_version_id": latest_version_id,
                "read_version_ids": read_version_ids,
                "unique_read_version_ids": sorted(set(read_version_ids)),
                "aggregate_delta_slot0": slot0_preview(aggregate_delta),
                "committed_version_id": latest_version_id + 1,
                "committed_slot0": slot0_preview(next_memory),
                "module_rows": module_rows,
            }
        )

    version_previews = [
        {
            "version_id": version_id,
            "slot0": slot0_preview(memory),
            "full_memory": preview_tensor(memory),
        }
        for version_id, memory in enumerate(history)
    ]
    return RunOutput(
        version_previews=version_previews,
        tick_summaries=tick_summaries,
        full_version_history=history,
        aggregate_deltas=aggregate_deltas,
    )


def zero_staleness_equivalence(control: RunOutput, zero_lag_async: RunOutput) -> dict[str, float]:
    final_diff = (control.full_version_history[-1] - zero_lag_async.full_version_history[-1]).abs().max().item()
    history_diff = max(
        (left - right).abs().max().item()
        for left, right in zip(control.full_version_history, zero_lag_async.full_version_history, strict=True)
    )
    if history_diff > 1e-7:
        raise RuntimeError(f"Zero-staleness equivalence failed with max history diff {history_diff}.")
    return {
        "max_final_memory_abs_diff": round_float(final_diff),
        "max_all_versions_abs_diff": round_float(history_diff),
    }


def write_rule_check(run_output: RunOutput) -> dict[str, float]:
    max_diff = 0.0
    for before, aggregate_delta, after in zip(
        run_output.full_version_history[:-1],
        run_output.aggregate_deltas,
        run_output.full_version_history[1:],
        strict=True,
    ):
        diff = (after - (before + aggregate_delta)).abs().max().item()
        max_diff = max(max_diff, diff)
    if max_diff > 1e-7:
        raise RuntimeError(f"Write-rule check failed with max tensor diff {max_diff}.")
    return {"max_tensor_abs_diff": round_float(max_diff)}


def stale_read_witness(async_run: RunOutput) -> dict[str, object]:
    witness_tick = None
    for tick_summary in async_run.tick_summaries:
        if len(tick_summary["unique_read_version_ids"]) > 1:
            witness_tick = tick_summary
            break
    if witness_tick is None:
        raise RuntimeError("No stale-read witness tick found.")

    stale_modules = [
        row for row in witness_tick["module_rows"] if row["read_version_id"] < row["latest_version_id"]
    ]
    if not stale_modules:
        raise RuntimeError("No module read a stale version.")

    max_gap = max(row["max_read_vs_latest_abs_diff"] for row in stale_modules)
    if max_gap <= 0.0:
        raise RuntimeError("Stale-read witness did not differ numerically from the latest committed state.")

    return {
        "witness_tick": witness_tick["tick"],
        "latest_version_id": witness_tick["latest_version_id"],
        "read_version_ids": witness_tick["read_version_ids"],
        "unique_read_version_ids": witness_tick["unique_read_version_ids"],
        "stale_modules": stale_modules,
        "max_read_vs_latest_abs_diff": round_float(max_gap),
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = PrototypeConfig()
    output_dir = args.output_dir or (
        args.repo_root / "experiments" / "async_volatile_memory" / "artifacts" / "stage2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    module_bank = DenseModuleBank(config).to(device)
    initial_memory = make_initial_memory(config, device)

    control = run_variant(
        module_bank=module_bank,
        initial_memory=initial_memory,
        read_lags=[0, 0, 0],
        label="synchronous_control",
        config=config,
    )
    zero_lag_async = run_variant(
        module_bank=module_bank,
        initial_memory=initial_memory,
        read_lags=[0, 0, 0],
        label="async_zero_staleness",
        config=config,
    )
    stale_async = run_variant(
        module_bank=module_bank,
        initial_memory=initial_memory,
        read_lags=[0, 1, 2],
        label="async_stale_reads",
        config=config,
    )

    report = {
        "config": asdict(config),
        "environment": {
            "git_sha": current_git_sha(),
            "git_working_tree_clean": current_git_status_short() == [],
            "git_status_short": current_git_status_short(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_is_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "semantics": {
            "shared_memory_shape": [config.batch_size, config.num_slots, config.d_model],
            "num_modules": config.num_modules,
            "num_ticks": config.num_ticks,
            "commit_rule": "memory[t+1] = memory[t] + sum(module_deltas[t])",
            "synchronous_control_read_lags": [0, 0, 0],
            "async_read_lags": [0, 1, 2],
            "single_changed_variable": "committed-memory visibility",
            "all_modules_execute_every_tick": True,
            "per_token_branching": False,
            "gradients_in_scope": False,
        },
        "parameter_count": count_parameters(module_bank),
        "checks": {
            "zero_staleness_equivalence": zero_staleness_equivalence(control, zero_lag_async),
            "write_rule": {
                "synchronous_control": write_rule_check(control),
                "async_stale_reads": write_rule_check(stale_async),
            },
            "stale_read_witness": stale_read_witness(stale_async),
        },
        "initial_memory": preview_tensor(initial_memory),
        "control_trace": {
            "version_previews": control.version_previews,
            "tick_summaries": control.tick_summaries,
        },
        "async_trace": {
            "version_previews": stale_async.version_previews,
            "tick_summaries": stale_async.tick_summaries,
        },
    }
    output_path = output_dir / "mechanical_trace.json"
    write_json(output_path, report)
    print("PASS async volatile memory stage2")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
