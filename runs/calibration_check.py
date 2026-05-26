from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in (None, ""):
    raise RuntimeError(
        "Run from the repo root with `.\\.venv\\Scripts\\python.exe -m runs.calibration_check ...` so `core` imports resolve cleanly."
    )

import torch
from torch import Tensor

from core.fixed_window_char import load_dataset, set_seed
from core.run_utils import prepare_output_paths, resolve_device
from runs.halting_regression import (
    BEST_TRADEOFF_MAX_LOSS_HIT,
    EPSILON_SWEEP,
    ExperimentConfig,
    autocast_context,
    build_model,
    compute_actual_gains,
    compute_depth_outputs,
    dataset_to_device,
)


@dataclass(frozen=True)
class SplitIndices:
    train: Tensor
    calibration: Tensor
    test: Tensor


@dataclass(frozen=True)
class SweepEntry:
    epsilon: float
    avg_depth: float
    val_loss: float
    loss_hit_vs_full_depth: float
    learned_speedup: float
    oracle_efficiency: float
    oracle_speedup: float


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "tinyshakespeare" / "artifacts" / "calibration_check"
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=artifact_dir / "report.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--eval-batch-size", type=positive_int, default=2048)
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def split_indices(total_examples: int, *, seed: int) -> SplitIndices:
    if total_examples < 3:
        raise ValueError(f"Need at least 3 examples for train/calibration/test split, got {total_examples}.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(total_examples, generator=generator)
    train_count = total_examples // 3
    calibration_count = total_examples // 3
    test_count = total_examples - train_count - calibration_count
    if min(train_count, calibration_count, test_count) <= 0:
        raise ValueError(
            "Split produced an empty partition. "
            f"Counts: train={train_count}, calibration={calibration_count}, test={test_count}."
        )
    return SplitIndices(
        train=permutation[:train_count],
        calibration=permutation[train_count : train_count + calibration_count],
        test=permutation[train_count + calibration_count :],
    )


def fit_affine_parameters(predicted_gains: Tensor, actual_gains: Tensor) -> list[dict[str, float]]:
    if predicted_gains.shape != actual_gains.shape:
        raise ValueError(
            f"Predicted/actual gain shapes must match, got {tuple(predicted_gains.shape)} vs {tuple(actual_gains.shape)}."
        )
    parameters: list[dict[str, float]] = []
    for depth_index in range(predicted_gains.shape[1]):
        predictions = predicted_gains[:, depth_index].double()
        targets = actual_gains[:, depth_index].double()
        design = torch.stack([predictions, torch.ones_like(predictions)], dim=1)
        solution = torch.linalg.lstsq(design, targets).solution
        scale = float(solution[0].item())
        bias = float(solution[1].item())
        if not torch.isfinite(solution).all():
            raise RuntimeError(f"Affine calibration produced non-finite parameters at depth {depth_index + 1}.")
        parameters.append({"depth": depth_index + 1, "scale": scale, "bias": bias})
    return parameters


def apply_affine(predicted_gains: Tensor, parameters: list[dict[str, float]]) -> Tensor:
    if predicted_gains.shape[1] != len(parameters):
        raise ValueError(
            f"Parameter count must match depth count, got {len(parameters)} for shape {tuple(predicted_gains.shape)}."
        )
    calibrated_columns = []
    for depth_index, parameter in enumerate(parameters):
        calibrated_columns.append(predicted_gains[:, depth_index] * parameter["scale"] + parameter["bias"])
    return torch.stack(calibrated_columns, dim=1)


def choose_depth_indices(predicted_gains: Tensor, *, epsilon: float, full_depth_index: int) -> Tensor:
    early_halt_mask = predicted_gains < epsilon
    any_halt = early_halt_mask.any(dim=1)
    return torch.where(
        any_halt,
        early_halt_mask.float().argmax(dim=1),
        torch.full((predicted_gains.shape[0],), full_depth_index, dtype=torch.int64),
    )


def summarize_sweep(per_depth_losses: Tensor, predicted_gains: Tensor) -> dict[str, object]:
    actual_gains = compute_actual_gains(per_depth_losses)
    depth_count = per_depth_losses.shape[1]
    full_depth_index = depth_count - 1
    full_depth_val_loss = float(per_depth_losses[:, -1].mean().item())
    example_indices = torch.arange(per_depth_losses.shape[0], dtype=torch.int64)
    entries: list[SweepEntry] = []
    for epsilon in EPSILON_SWEEP:
        chosen_depth_indices = choose_depth_indices(predicted_gains, epsilon=epsilon, full_depth_index=full_depth_index)
        halted_losses = per_depth_losses[example_indices, chosen_depth_indices]
        avg_depth = float((chosen_depth_indices + 1).float().mean().item())
        val_loss = float(halted_losses.mean().item())
        learned_speedup = depth_count / avg_depth

        oracle_depth_indices = choose_depth_indices(actual_gains, epsilon=epsilon, full_depth_index=full_depth_index)
        oracle_avg_depth = float((oracle_depth_indices + 1).float().mean().item())
        oracle_speedup = depth_count / oracle_avg_depth
        oracle_efficiency = 0.0
        if oracle_speedup > 1.0:
            oracle_efficiency = (learned_speedup - 1.0) / (oracle_speedup - 1.0)

        entries.append(
            SweepEntry(
                epsilon=epsilon,
                avg_depth=avg_depth,
                val_loss=val_loss,
                loss_hit_vs_full_depth=val_loss - full_depth_val_loss,
                learned_speedup=learned_speedup,
                oracle_efficiency=oracle_efficiency,
                oracle_speedup=oracle_speedup,
            )
        )

    acceptable_entries = [entry for entry in entries if entry.loss_hit_vs_full_depth <= BEST_TRADEOFF_MAX_LOSS_HIT + 1e-12]
    best_tradeoff = None
    if len(acceptable_entries) > 0:
        best_tradeoff = max(
            acceptable_entries,
            key=lambda entry: (entry.learned_speedup, -entry.loss_hit_vs_full_depth, -entry.epsilon),
        )
    epsilon_0_02 = next(entry for entry in entries if abs(entry.epsilon - 0.02) < 1e-12)
    return {
        "full_depth_val_loss": round(full_depth_val_loss, 6),
        "epsilon_0_02": sweep_entry_payload(epsilon_0_02),
        "best_tradeoff": None if best_tradeoff is None else sweep_entry_payload(best_tradeoff),
        "epsilon_sweep": [sweep_entry_payload(entry) for entry in entries],
    }


def sweep_entry_payload(entry: SweepEntry) -> dict[str, float]:
    return {
        "epsilon": round(entry.epsilon, 6),
        "avg_depth": round(entry.avg_depth, 6),
        "val_loss": round(entry.val_loss, 6),
        "loss_hit_vs_full_depth": round(entry.loss_hit_vs_full_depth, 6),
        "learned_speedup": round(entry.learned_speedup, 6),
        "oracle_efficiency": round(entry.oracle_efficiency, 6),
        "oracle_speedup": round(entry.oracle_speedup, 6),
    }


@torch.inference_mode()
def collect_outputs(
    *,
    model: torch.nn.Module,
    dataset: tuple[Tensor, Tensor],
    eval_batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    model.eval()
    inputs, targets = dataset
    per_depth_loss_batches: list[Tensor] = []
    predicted_gain_batches: list[Tensor] = []
    for start in range(0, targets.shape[0], eval_batch_size):
        stop = min(start + eval_batch_size, targets.shape[0])
        batch_inputs = inputs[start:stop]
        batch_targets = targets[start:stop]
        with autocast_context(device):
            per_depth_losses, predicted_gains = compute_depth_outputs(model, batch_inputs, batch_targets)
        per_depth_loss_batches.append(per_depth_losses.float().cpu())
        predicted_gain_batches.append(predicted_gains.float().cpu())
    return torch.cat(per_depth_loss_batches, dim=0), torch.cat(predicted_gain_batches, dim=0)


def subset_rows(values: Tensor, indices: Tensor) -> Tensor:
    return values.index_select(0, indices)


def main() -> None:
    args = parse_args()
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    prepare_output_paths(report_path=args.report_path, log_path=args.report_path.parent / "run.jsonl")
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    config = ExperimentConfig(**checkpoint["config"])
    set_seed(int(checkpoint["seed"]))
    _, val_dataset, vocab_size = load_dataset(
        context_size=config.context_size,
        train_characters=int(checkpoint["train_characters"]),
        val_characters=int(checkpoint["val_characters"]),
    )
    val_dataset = dataset_to_device(val_dataset, device)
    model = build_model(vocab_size=int(checkpoint["vocab_size"]), config=config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    per_depth_losses, predicted_gains = collect_outputs(
        model=model,
        dataset=val_dataset,
        eval_batch_size=args.eval_batch_size,
        device=device,
    )
    actual_gains = compute_actual_gains(per_depth_losses)
    splits = split_indices(per_depth_losses.shape[0], seed=args.split_seed)
    affine_parameters = fit_affine_parameters(
        subset_rows(predicted_gains, splits.train),
        subset_rows(actual_gains, splits.train),
    )

    calibration_original = summarize_sweep(
        subset_rows(per_depth_losses, splits.calibration),
        subset_rows(predicted_gains, splits.calibration),
    )
    calibration_affine = summarize_sweep(
        subset_rows(per_depth_losses, splits.calibration),
        apply_affine(subset_rows(predicted_gains, splits.calibration), affine_parameters),
    )
    test_original = summarize_sweep(
        subset_rows(per_depth_losses, splits.test),
        subset_rows(predicted_gains, splits.test),
    )
    test_affine = summarize_sweep(
        subset_rows(per_depth_losses, splits.test),
        apply_affine(subset_rows(predicted_gains, splits.test), affine_parameters),
    )

    original_epsilon_0_02 = test_original["epsilon_0_02"]
    affine_epsilon_0_02 = test_affine["epsilon_0_02"]
    epsilon_speedup_gain = (
        float(affine_epsilon_0_02["learned_speedup"]) - float(original_epsilon_0_02["learned_speedup"])
    )
    material_improvement = float(affine_epsilon_0_02["learned_speedup"]) > 1.30

    payload = {
        "checkpoint_path": str(args.checkpoint_path),
        "device": device.type,
        "split_seed": args.split_seed,
        "config": asdict(config),
        "split_sizes": {
            "train": int(splits.train.numel()),
            "calibration": int(splits.calibration.numel()),
            "test": int(splits.test.numel()),
        },
        "affine_parameters": [
            {
                "depth": parameter["depth"],
                "scale": round(parameter["scale"], 6),
                "bias": round(parameter["bias"], 6),
            }
            for parameter in affine_parameters
        ],
        "calibration_split": {
            "original": calibration_original,
            "affine": calibration_affine,
        },
        "test_split": {
            "original": test_original,
            "affine": test_affine,
        },
        "epsilon_0_02_test_delta": {
            "original_speedup": round(float(original_epsilon_0_02["learned_speedup"]), 6),
            "affine_speedup": round(float(affine_epsilon_0_02["learned_speedup"]), 6),
            "speedup_gain": round(epsilon_speedup_gain, 6),
            "original_loss_hit": round(float(original_epsilon_0_02["loss_hit_vs_full_depth"]), 6),
            "affine_loss_hit": round(float(affine_epsilon_0_02["loss_hit_vs_full_depth"]), 6),
            "moves_above_1_30": material_improvement,
        },
    }
    args.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["epsilon_0_02_test_delta"], indent=2), flush=True)


if __name__ == "__main__":
    main()
