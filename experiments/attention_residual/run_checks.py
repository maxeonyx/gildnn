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

from .model import AttentionResidualCharModel, VariantSpec, count_parameters


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 32
    train_characters: int = 100_000
    val_characters: int = 20_000
    d_model: int = 72
    num_heads: int = 4
    num_layers: int = 3
    seed: int = 42
    check_batch_size: int = 8
    memorization_batch_size: int = 32
    memorization_steps: int = 120
    memorization_learning_rate: float = 0.02
    perturbation_scale: float = 0.1


VARIANTS = {
    "external_control": VariantSpec(
        family="external_control",
        num_layers=3,
        num_heads=4,
        feedforward_dim=256,
    ),
    "depth_only": VariantSpec(
        family="depth_only",
        num_layers=3,
        num_heads=4,
        feedforward_dim=159,
    ),
    "internal_control": VariantSpec(
        family="internal_control",
        num_layers=3,
        num_heads=4,
        feedforward_dim=231,
    ),
}


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


def make_model(
    *,
    vocab_size: int,
    config: RunConfig,
    variant: VariantSpec,
    device: torch.device,
) -> AttentionResidualCharModel:
    return AttentionResidualCharModel(
        vocab_size=vocab_size,
        context_size=config.context_size,
        d_model=config.d_model,
        variant=variant,
    ).to(device)


def parameter_table(vocab_size: int, config: RunConfig) -> list[dict[str, int | str]]:
    baseline_count = count_parameters(
        make_model(
            vocab_size=vocab_size,
            config=config,
            variant=VARIANTS["external_control"],
            device=torch.device("cpu"),
        )
    )
    rows = []
    for variant in VARIANTS.values():
        model = make_model(
            vocab_size=vocab_size,
            config=config,
            variant=variant,
            device=torch.device("cpu"),
        )
        parameter_count = count_parameters(model)
        rows.append(
            {
                "label": variant.label,
                "feedforward_dim": variant.feedforward_dim,
                "parameter_count": parameter_count,
                "parameter_delta_vs_external_control": parameter_count - baseline_count,
            }
        )
    return rows


def dataset_round_trip(dataset, *, context_size: int) -> dict[str, object]:
    sample = dataset.text[:context_size]
    round_trip = dataset.decode(dataset.encode(sample))
    if round_trip != sample:
        raise RuntimeError("Dataset encode/decode round-trip failed.")
    return {"sample": sample.replace("\n", "\\n")}


def forward_shape_check(
    model: AttentionResidualCharModel,
    tokens: Tensor,
    *,
    vocab_size: int,
) -> dict[str, object]:
    run = model.run(tokens, capture_details=True)
    expected_logits_shape = [tokens.shape[0], tokens.shape[1], vocab_size]
    actual_logits_shape = list(run.full_logits.shape)
    if actual_logits_shape != expected_logits_shape:
        raise RuntimeError(
            f"Full logits shape mismatch: expected {expected_logits_shape}, got {actual_logits_shape}."
        )
    block_rows = []
    for trace in run.block_traces:
        block_rows.append(
            {
                "block_index": trace.block_index,
                "memory_length": trace.memory_length,
                "depth_update_rms": trace.depth_update_rms,
                "sequence_update_rms": trace.sequence_update_rms,
                "feedforward_update_rms": trace.feedforward_update_rms,
                "depth_attention_weight_shape": list(trace.depth_attention_weights.shape)
                if trace.depth_attention_weights is not None
                else None,
                "internal_mix_weight_shape": list(trace.internal_mix_weights.shape)
                if trace.internal_mix_weights is not None
                else None,
            }
        )
    return {
        "expected_logits_shape": expected_logits_shape,
        "actual_logits_shape": actual_logits_shape,
        "blocks": block_rows,
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


def grad_norm(parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += parameter.grad.detach().float().pow(2).sum().item()
    return total**0.5


def gradient_check(
    model: AttentionResidualCharModel,
    tokens: Tensor,
    targets: Tensor,
) -> dict[str, object]:
    logits = model(tokens)
    loss = F.cross_entropy(logits, targets)
    model.zero_grad(set_to_none=True)
    loss.backward()
    block_rows = []
    for block in model.blocks:
        block_row = {
            "block_index": block.block_index,
            "sequence_attention_grad_norm": grad_norm(list(block.sequence_attention.parameters())),
            "feedforward_grad_norm": grad_norm(list(block.feedforward.parameters())),
            "depth_path_grad_norm": grad_norm(list(block.depth_readout.parameters())),
        }
        block_rows.append(block_row)
    output_grad_norm = grad_norm(list(model.output.parameters()))
    if output_grad_norm <= 0.0:
        raise RuntimeError("Output projection gradient norm was zero.")
    for block_row in block_rows:
        if block_row["sequence_attention_grad_norm"] <= 0.0:
            raise RuntimeError(f"Sequence attention gradient was zero in block {block_row['block_index']}.")
        if block_row["feedforward_grad_norm"] <= 0.0:
            raise RuntimeError(f"Feedforward gradient was zero in block {block_row['block_index']}.")
    return {
        "loss": loss.item(),
        "output_grad_norm": output_grad_norm,
        "blocks": block_rows,
    }


def memorization_check(
    model: AttentionResidualCharModel,
    inputs: Tensor,
    targets: Tensor,
    *,
    config: RunConfig,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.memorization_learning_rate)
    trace: list[dict[str, float | int]] = []
    final_loss = float("nan")
    final_accuracy = 0.0
    for step in range(config.memorization_steps + 1):
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
        if step in {0, config.memorization_steps} or step % 20 == 0:
            trace.append(
                {
                    "step": step,
                    "loss": round(loss.item(), 6),
                    "accuracy": round(accuracy, 6),
                }
            )
        final_loss = loss.item()
        final_accuracy = accuracy
        if step == config.memorization_steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if not (final_loss < 0.05 and final_accuracy == 1.0):
        raise RuntimeError(
            "One-batch memorization check failed: expected loss < 0.05 and accuracy 1.0, "
            f"got loss={final_loss:.6f}, accuracy={final_accuracy:.6f}."
        )
    return {
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "trace": trace,
    }


def causal_mask_check(model: AttentionResidualCharModel, tokens: Tensor) -> dict[str, object]:
    run_original = model.run(tokens)
    edited_tokens = tokens.clone()
    edited_tokens[:, -1] = (edited_tokens[:, -1] + 1) % model.output.out_features
    run_edited = model.run(edited_tokens)
    earlier_max_diff = (
        run_original.full_logits[:, :-1, :] - run_edited.full_logits[:, :-1, :]
    ).abs().max().item()
    last_position_max_diff = (
        run_original.full_logits[:, -1, :] - run_edited.full_logits[:, -1, :]
    ).abs().max().item()
    if earlier_max_diff > 1e-6:
        raise RuntimeError(
            f"Causal mask check failed: earlier positions changed by {earlier_max_diff:.12f}."
        )
    if last_position_max_diff <= 1e-6:
        raise RuntimeError("Causal mask check failed: edited future token did not affect its own position.")
    return {
        "edited_position": tokens.shape[1] - 1,
        "earlier_positions_max_abs_diff": earlier_max_diff,
        "edited_position_max_abs_diff": last_position_max_diff,
    }


def depth_memory_access_check(
    model: AttentionResidualCharModel,
    tokens: Tensor,
    *,
    perturbation_scale: float,
) -> dict[str, object]:
    run = model.run(tokens, capture_details=True)
    rows: list[dict[str, object]] = []
    for trace in run.block_traces:
        block = model.blocks[trace.block_index]
        boundary_memory = run.boundary_states[: trace.block_index]
        current_state = run.boundary_states[trace.block_index]
        base_update, _, _ = block.depth_readout(
            current_state,
            boundary_memory=boundary_memory,
            capture_details=False,
        )
        block_row: dict[str, object] = {
            "block_index": trace.block_index + 1,
            "memory_length": trace.memory_length,
            "base_depth_update_rms": torch.sqrt(torch.mean(base_update.detach().float().square())).item(),
            "slots": [],
        }
        for slot_index, memory_state in enumerate(boundary_memory):
            same_token_memory = [state.clone() for state in boundary_memory]
            same_token_memory[slot_index][:, 0, 0] += perturbation_scale
            same_token_update, _, _ = block.depth_readout(
                current_state,
                boundary_memory=same_token_memory,
                capture_details=False,
            )

            other_token_memory = [state.clone() for state in boundary_memory]
            other_token_memory[slot_index][:, 1, 0] += perturbation_scale
            other_token_update, _, _ = block.depth_readout(
                current_state,
                boundary_memory=other_token_memory,
                capture_details=False,
            )

            same_token_position_diff = (
                same_token_update[:, 0, :] - base_update[:, 0, :]
            ).abs().max().item()
            unaffected_position_diff = (
                same_token_update[:, 1, :] - base_update[:, 1, :]
            ).abs().max().item()
            other_token_position_diff = (
                other_token_update[:, 0, :] - base_update[:, 0, :]
            ).abs().max().item()
            if model.variant.family == "external_control":
                if same_token_position_diff > 1e-8 or other_token_position_diff > 1e-8:
                    raise RuntimeError("External control depth path reacted despite having no depth readout.")
            else:
                if same_token_position_diff <= 1e-8:
                    raise RuntimeError(
                        f"{model.variant.family} block {trace.block_index + 1} did not react to accessible memory slot {slot_index}."
                    )
                if unaffected_position_diff > 1e-8:
                    raise RuntimeError(
                        f"{model.variant.family} block {trace.block_index + 1} leaked same-token perturbation to another token position."
                    )
                if other_token_position_diff > 1e-8:
                    raise RuntimeError(
                        f"{model.variant.family} block {trace.block_index + 1} read a different token position from the depth bank."
                    )
            block_row["slots"].append(
                {
                    "slot_index": slot_index,
                    "same_token_position_change": same_token_position_diff,
                    "unaffected_position_change": unaffected_position_diff,
                    "other_token_position_change": other_token_position_diff,
                }
            )
        if trace.depth_attention_weights is not None:
            block_row["attention_preview"] = (
                trace.depth_attention_weights[0, 0, :, :].tolist()
            )
        if trace.internal_mix_weights is not None:
            block_row["mix_weights"] = trace.internal_mix_weights.tolist()
        rows.append(block_row)
    return {"blocks": rows}


def rough_compute_table(config: RunConfig) -> list[dict[str, object]]:
    rows = []
    sequence_length = config.context_size
    d_model = config.d_model
    baseline_ff = VARIANTS["external_control"].feedforward_dim
    baseline_units = None
    for variant in VARIANTS.values():
        total_units = 0
        for block_index in range(variant.num_layers):
            sequence_units = 4 * sequence_length * d_model * d_model + 2 * sequence_length * sequence_length * d_model
            feedforward_units = 2 * sequence_length * d_model * variant.feedforward_dim
            depth_units = 0
            if variant.family == "depth_only" and block_index > 0:
                depth_units = 4 * sequence_length * d_model * d_model + 2 * sequence_length * block_index * d_model
            if variant.family == "internal_control" and block_index > 0:
                depth_units = sequence_length * block_index * d_model + sequence_length * d_model * d_model
            total_units += sequence_units + feedforward_units + depth_units
        row = {
            "label": variant.label,
            "feedforward_dim": variant.feedforward_dim,
            "rough_forward_units": total_units,
        }
        if variant.family == "external_control":
            baseline_units = total_units
            row["relative_to_external_control"] = 1.0
        else:
            row["relative_to_external_control"] = total_units / baseline_units
        rows.append(row)
    if baseline_units is None:
        raise RuntimeError("Missing external control compute anchor.")
    return rows


def run_variant_checks(
    *,
    variant: VariantSpec,
    vocab_size: int,
    tokens: Tensor,
    targets: Tensor,
    split,
    config: RunConfig,
    device: torch.device,
) -> dict[str, object]:
    set_seed(config.seed)
    model = make_model(vocab_size=vocab_size, config=config, variant=variant, device=device)
    return {
        "parameter_count": count_parameters(model),
        "forward_shapes": forward_shape_check(model, tokens, vocab_size=vocab_size),
        "known_tiny_batch_loss": known_tiny_batch_loss(model(tokens), targets, vocab_size=vocab_size),
        "gradient_flow": gradient_check(model, tokens, targets),
        "causal_mask": causal_mask_check(model, tokens),
        "depth_memory_access": depth_memory_access_check(
            model,
            tokens,
            perturbation_scale=config.perturbation_scale,
        ),
        "one_batch_memorization": memorization_check(
            make_model(vocab_size=vocab_size, config=config, variant=variant, device=device),
            split.train_inputs[: config.memorization_batch_size].to(device),
            split.train_targets[: config.memorization_batch_size].to(device),
            config=config,
        ),
    }


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
        args.repo_root / "experiments" / "attention_residual" / "artifacts" / "stage3_checks"
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
        "rough_compute": rough_compute_table(config),
        "variants": {},
    }

    for variant in VARIANTS.values():
        report["variants"][variant.label] = run_variant_checks(
            variant=variant,
            vocab_size=split.train_dataset.vocab_size,
            tokens=tokens,
            targets=targets,
            split=split,
            config=config,
            device=device,
        )
        print(f"PASS {variant.label}")

    write_json(output_dir / "mechanical_trust.json", report)
    print(f"Wrote {output_dir / 'mechanical_trust.json'}")


if __name__ == "__main__":
    main()
