from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from base_experiments.transformer import build_fixed_length_split
from core.fixed_window_char import resolve_device, set_seed

from .model import (
    BlockState,
    ResidualLocalLearningModel,
    VariantSpec,
    compute_loss_bundle,
    count_parameters,
    rms,
)


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 32
    train_characters: int = 100_000
    val_characters: int = 20_000
    d_model: int = 72
    local_loss_weight: float = 1.0
    seed: int = 42
    check_batch_size: int = 8


PLANNED_VARIANTS = (
    VariantSpec(family="end_to_end", num_blocks=1, ff_hidden=1203),
    VariantSpec(family="end_to_end", num_blocks=3, ff_hidden=400),
    VariantSpec(family="end_to_end", num_blocks=6, ff_hidden=199),
    VariantSpec(family="local", num_blocks=1, ff_hidden=1167),
    VariantSpec(family="local", num_blocks=3, ff_hidden=364),
    VariantSpec(family="local", num_blocks=6, ff_hidden=163),
)


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


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def grad_norm(parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += parameter.grad.detach().float().pow(2).sum().item()
    return total**0.5


def assert_zero(value: float, *, label: str, tolerance: float = 1e-12) -> None:
    if abs(value) > tolerance:
        raise RuntimeError(f"Expected zero at {label}, got {value:.12f}.")


def assert_positive(value: float, *, label: str, threshold: float = 1e-9) -> None:
    if value <= threshold:
        raise RuntimeError(f"Expected positive value at {label}, got {value:.12f}.")


def block_snapshot(model: ResidualLocalLearningModel) -> list[dict[str, float | int]]:
    snapshots = []
    for block_index, block in enumerate(model.blocks, start=1):
        snapshots.append(
            {
                "block_index": block_index,
                "block_grad_norm": grad_norm(list(block.parameters())),
                "local_head_grad_norm": grad_norm(
                    list(model.local_heads[block_index - 1].parameters())
                )
                if model.local_heads is not None
                else 0.0,
            }
        )
    return snapshots


def tensor_preview(tensor: Tensor, *, take: int = 6) -> list[float]:
    flat = tensor.detach().reshape(-1)[:take].float().tolist()
    return [round(value, 6) for value in flat]


def parameter_table(vocab_size: int, config: RunConfig) -> list[dict[str, int | str]]:
    rows = []
    for variant in PLANNED_VARIANTS:
        model = ResidualLocalLearningModel(
            vocab_size=vocab_size,
            context_size=config.context_size,
            d_model=config.d_model,
            variant=variant,
        )
        rows.append(
            {
                "label": variant.label,
                "family": variant.family,
                "num_blocks": variant.num_blocks,
                "ff_hidden": variant.ff_hidden,
                "parameter_count": count_parameters(model),
            }
        )
    return rows


def forward_shape_check(
    model: ResidualLocalLearningModel,
    tokens: Tensor,
    *,
    vocab_size: int,
) -> dict[str, object]:
    run = model.run(tokens)
    expected_logits_shape = [tokens.shape[0], vocab_size]
    actual_logits_shape = list(run.logits.shape)
    if actual_logits_shape != expected_logits_shape:
        raise RuntimeError(
            f"Forward logits shape mismatch: expected {expected_logits_shape}, got {actual_logits_shape}."
        )
    block_shapes = []
    for block_index, block_state in enumerate(run.block_states, start=1):
        expected = list(block_state.input_residual.shape)
        actual_delta = list(block_state.delta.shape)
        if actual_delta != expected:
            raise RuntimeError(
                f"Block {block_index} delta shape mismatch: expected {expected}, got {actual_delta}."
            )
        predicted_shape = (
            list(block_state.predicted_delta.shape)
            if block_state.predicted_delta is not None
            else None
        )
        if predicted_shape is not None and predicted_shape != expected:
            raise RuntimeError(
                f"Block {block_index} predicted delta shape mismatch: expected {expected}, got {predicted_shape}."
            )
        block_shapes.append(
            {
                "block_index": block_index,
                "residual_shape": expected,
                "predicted_delta_shape": predicted_shape,
            }
        )
    return {
        "expected_logits_shape": expected_logits_shape,
        "actual_logits_shape": actual_logits_shape,
        "blocks": block_shapes,
    }


def local_target_checks(run_block_states: list[BlockState]) -> dict[str, object]:
    modules = []
    for block_index, block_state in enumerate(run_block_states, start=1):
        if block_state.predicted_delta is None:
            continue
        reconstructed = block_state.output_residual - block_state.input_residual
        own_delta_error = (reconstructed - block_state.delta).abs().max().item()
        if not torch.allclose(reconstructed, block_state.delta, atol=1e-6, rtol=1e-6):
            raise RuntimeError(
                f"Block {block_index} delta reconstruction mismatch; max abs error {own_delta_error:.12f}."
            )
        if block_state.local_target_delta is None:
            raise RuntimeError(f"Block {block_index} missing local target delta.")
        modules.append(
            {
                "block_index": block_index,
                "prediction_shape": list(block_state.predicted_delta.shape),
                "target_shape": list(block_state.local_target_delta.shape),
                "max_abs_error_output_minus_input_vs_own_delta": own_delta_error,
                "target_preview": tensor_preview(block_state.local_target_delta),
                "prediction_preview": tensor_preview(block_state.predicted_delta),
            }
        )
    return {"blocks": modules}


def residual_stats(block_states: list[BlockState]) -> list[dict[str, float | int | None]]:
    rows = []
    for block_index, block_state in enumerate(block_states, start=1):
        rows.append(
            {
                "block_index": block_index,
                "incoming_residual_rms": rms(block_state.input_residual),
                "predicted_delta_rms": rms(block_state.predicted_delta)
                if block_state.predicted_delta is not None
                else None,
                "true_target_delta_rms": rms(block_state.local_target_delta)
                if block_state.local_target_delta is not None
                else rms(block_state.delta),
            }
        )
    return rows


def local_gradient_checks(
    model: ResidualLocalLearningModel,
    tokens: Tensor,
    targets: Tensor,
    *,
    local_loss_weight: float,
) -> dict[str, object]:
    bundle = compute_loss_bundle(
        model,
        tokens,
        targets,
        local_loss_weight=local_loss_weight,
    )
    if len(bundle.local_losses) != len(model.blocks):
        raise RuntimeError(
            f"Expected {len(model.blocks)} local losses, got {len(bundle.local_losses)}."
        )

    local_loss_snapshots = []
    for local_loss_index, local_loss in enumerate(bundle.local_losses, start=1):
        model.zero_grad(set_to_none=True)
        local_loss.backward(retain_graph=True)
        snapshot = block_snapshot(model)
        for block_index, block_data in enumerate(snapshot, start=1):
            if block_index == local_loss_index:
                assert_positive(
                    float(block_data["block_grad_norm"]),
                    label=f"block_{block_index}_own_local_loss_block_grad",
                )
                assert_positive(
                    float(block_data["local_head_grad_norm"]),
                    label=f"block_{block_index}_own_local_loss_head_grad",
                )
            else:
                assert_zero(
                    float(block_data["block_grad_norm"]),
                    label=f"block_{block_index}_foreign_local_loss_block_grad",
                )
                assert_zero(
                    float(block_data["local_head_grad_norm"]),
                    label=f"block_{block_index}_foreign_local_loss_head_grad",
                )
        assert_zero(
            grad_norm(list(model.lm_head.parameters())),
            label=f"local_loss_{local_loss_index}_to_lm_head",
        )
        local_loss_snapshots.append(
            {
                "local_loss_block_index": local_loss_index,
                "block_gradients": snapshot,
                "lm_head_grad_norm": grad_norm(list(model.lm_head.parameters())),
            }
        )

    model.zero_grad(set_to_none=True)
    bundle.lm_loss.backward()
    lm_snapshot = block_snapshot(model)
    for block_index, block_data in enumerate(lm_snapshot, start=1):
        if block_index == len(lm_snapshot):
            assert_positive(
                float(block_data["block_grad_norm"]),
                label="lm_loss_to_last_block",
            )
        else:
            assert_zero(
                float(block_data["block_grad_norm"]),
                label=f"lm_loss_to_block_{block_index}",
            )
        assert_zero(
            float(block_data["local_head_grad_norm"]),
            label=f"lm_loss_to_local_head_{block_index}",
        )
    lm_head_grad_norm = grad_norm(list(model.lm_head.parameters()))
    assert_positive(lm_head_grad_norm, label="lm_loss_to_lm_head")
    return {
        "local_losses": local_loss_snapshots,
        "lm_loss": {
            "block_gradients": lm_snapshot,
            "lm_head_grad_norm": lm_head_grad_norm,
        },
    }


def end_to_end_gradient_checks(
    model: ResidualLocalLearningModel,
    tokens: Tensor,
    targets: Tensor,
) -> dict[str, object]:
    bundle = compute_loss_bundle(model, tokens, targets, local_loss_weight=0.0)
    model.zero_grad(set_to_none=True)
    bundle.lm_loss.backward()
    snapshot = block_snapshot(model)
    for block_index, block_data in enumerate(snapshot, start=1):
        assert_positive(
            float(block_data["block_grad_norm"]),
            label=f"end_to_end_lm_loss_to_block_{block_index}",
        )
        assert_zero(
            float(block_data["local_head_grad_norm"]),
            label=f"end_to_end_local_head_grad_block_{block_index}",
        )
    lm_head_grad_norm = grad_norm(list(model.lm_head.parameters()))
    assert_positive(lm_head_grad_norm, label="end_to_end_lm_head_grad")
    return {
        "lm_loss": {
            "block_gradients": snapshot,
            "lm_head_grad_norm": lm_head_grad_norm,
        }
    }


def known_tiny_batch_loss(logits: Tensor, targets: Tensor, *, vocab_size: int) -> dict[str, object]:
    loss = F.cross_entropy(logits, targets).item()
    expected_low = math.log(vocab_size) - 1.0
    expected_high = math.log(vocab_size) + 1.0
    if not expected_low <= loss <= expected_high:
        raise RuntimeError(
            f"Known tiny-batch loss {loss:.6f} fell outside expected range [{expected_low:.6f}, {expected_high:.6f}]."
        )
    return {
        "loss": loss,
        "expected_range": [expected_low, expected_high],
    }


def run_variant_checks(
    *,
    variant: VariantSpec,
    vocab_size: int,
    tokens: Tensor,
    targets: Tensor,
    config: RunConfig,
) -> dict[str, object]:
    model = ResidualLocalLearningModel(
        vocab_size=vocab_size,
        context_size=config.context_size,
        d_model=config.d_model,
        variant=variant,
    ).to(tokens.device)
    run = model.run(tokens)
    bundle = compute_loss_bundle(
        model,
        tokens,
        targets,
        local_loss_weight=config.local_loss_weight,
    )
    expected_total = bundle.lm_loss.item() + config.local_loss_weight * bundle.local_loss_total.item()
    if not math.isclose(bundle.total_loss.item(), expected_total, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(f"Loss wiring mismatch for {variant.label}.")
    checks: dict[str, object] = {
        "parameter_count": count_parameters(model),
        "forward_shapes": forward_shape_check(model, tokens, vocab_size=vocab_size),
        "known_tiny_batch_loss": known_tiny_batch_loss(run.logits, targets, vocab_size=vocab_size),
        "loss_wiring": {
            "lm_loss": bundle.lm_loss.item(),
            "local_loss_total": bundle.local_loss_total.item(),
            "weighted_total_loss": bundle.total_loss.item(),
        },
        "residual_stats": residual_stats(run.block_states),
    }
    if variant.family == "local":
        checks["local_target_checks"] = local_target_checks(run.block_states)
        checks["gradient_routing"] = local_gradient_checks(
            model,
            tokens,
            targets,
            local_loss_weight=config.local_loss_weight,
        )
    else:
        checks["gradient_routing"] = end_to_end_gradient_checks(model, tokens, targets)
    return checks


def dataset_round_trip(dataset, *, context_size: int) -> dict[str, object]:
    sample = dataset.text[:context_size]
    round_trip = dataset.decode(dataset.encode(sample))
    if round_trip != sample:
        raise RuntimeError("Dataset encode/decode round-trip failed.")
    return {"sample": sample.replace("\n", "\\n")}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig()
    text_file = args.text_file or (
        args.repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"
    )
    output_dir = args.output_dir or (
        args.repo_root / "experiments" / "local_learning_residual" / "artifacts" / "stage1_checks"
    )

    set_seed(config.seed)
    device = resolve_device(args.device)
    raw_text = text_file.read_text(encoding="utf-8")
    split = build_fixed_length_split(raw_text, config=config)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens = split.train_inputs[: config.check_batch_size].to(device)
    targets = split.train_targets[: config.check_batch_size].to(device)
    parameter_counts = parameter_table(split.train_dataset.vocab_size, config)

    report: dict[str, object] = {
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
        "corpus_summary": {
            "source_file": str(text_file),
            "source_total_characters": len(raw_text),
            "source_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "train_characters": len(split.train_text),
            "val_characters": len(split.val_text),
            "val_start": split.val_start,
            "val_stop": split.val_start + len(split.val_text),
            "vocab_size": split.train_dataset.vocab_size,
        },
        "dataset_round_trip": dataset_round_trip(split.train_dataset, context_size=config.context_size),
        "parameter_counts": parameter_counts,
        "variants": {},
    }

    for variant in PLANNED_VARIANTS:
        set_seed(config.seed)
        report["variants"][variant.label] = run_variant_checks(
            variant=variant,
            vocab_size=split.train_dataset.vocab_size,
            tokens=tokens,
            targets=targets,
            config=config,
        )
        print(f"PASS {variant.label}")

    write_json(output_dir / "mechanical_trust.json", report)

    print("\nParameter counts")
    for row in parameter_counts:
        print(
            f"{row['label']:>14}  blocks={row['num_blocks']}  ff_hidden={row['ff_hidden']}  params={row['parameter_count']}"
        )

    print("\nLocal residual stats")
    for variant in [spec for spec in PLANNED_VARIANTS if spec.family == "local"]:
        stats = report["variants"][variant.label]["residual_stats"]
        for row in stats:
            print(
                f"{variant.label} block={row['block_index']}  in_rms={row['incoming_residual_rms']:.6f}  "
                f"pred_rms={row['predicted_delta_rms']:.6f}  target_rms={row['true_target_delta_rms']:.6f}"
            )

    print(f"\nWrote {output_dir / 'mechanical_trust.json'}")


if __name__ == "__main__":
    main()
