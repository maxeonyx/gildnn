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

from .model import AsyncSelectiveCharModel, VariantSpec, budget_regularizer, count_parameters


@dataclass(frozen=True)
class RunConfig:
    context_size: int = 32
    train_characters: int = 100_000
    val_characters: int = 20_000
    d_model: int = 72
    seed: int = 42
    check_batch_size: int = 8
    target_open_rate: float = 0.5
    budget_weight: float = 1.0


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


def make_variant(name: str, *, config: RunConfig, target_open_rate: float | None = None) -> VariantSpec:
    return VariantSpec(
        name=name,  # type: ignore[arg-type]
        target_open_rate=target_open_rate,
        budget_weight=config.budget_weight,
    )


def make_model(
    *,
    vocab_size: int,
    config: RunConfig,
    variant: VariantSpec,
    device: torch.device,
) -> AsyncSelectiveCharModel:
    return AsyncSelectiveCharModel(
        vocab_size=vocab_size,
        context_size=config.context_size,
        d_model=config.d_model,
        variant=variant,
    ).to(device)


def parameter_table(vocab_size: int, config: RunConfig) -> list[dict[str, int | str | None]]:
    variants = [
        make_variant("synchronous_control", config=config),
        make_variant("forced_open", config=config),
        make_variant("learned_gate", config=config, target_open_rate=0.75),
        make_variant("learned_gate", config=config, target_open_rate=0.50),
        make_variant("random_skip", config=config, target_open_rate=0.75),
        make_variant("random_skip", config=config, target_open_rate=0.50),
    ]
    rows = []
    for variant in variants:
        model = make_model(
            vocab_size=vocab_size,
            config=config,
            variant=variant,
            device=torch.device("cpu"),
        )
        rows.append(
            {
                "label": variant.label,
                "target_open_rate": variant.target_open_rate,
                "feedforward_dims": list(variant.feedforward_dims),
                "parameter_count": count_parameters(model),
            }
        )
    return rows


def dataset_round_trip(dataset, *, context_size: int) -> dict[str, object]:
    sample = dataset.text[:context_size]
    round_trip = dataset.decode(dataset.encode(sample))
    if round_trip != sample:
        raise RuntimeError("Dataset encode/decode round-trip failed.")
    return {"sample": sample.replace("\n", "\\n")}


def forced_open_equivalence(
    *,
    vocab_size: int,
    config: RunConfig,
    tokens: Tensor,
    targets: Tensor,
    device: torch.device,
) -> dict[str, object]:
    set_seed(config.seed)
    control = make_model(
        vocab_size=vocab_size,
        config=config,
        variant=make_variant("synchronous_control", config=config),
        device=device,
    )
    forced_open = make_model(
        vocab_size=vocab_size,
        config=config,
        variant=make_variant("forced_open", config=config),
        device=device,
    )
    forced_open.load_state_dict(control.state_dict())
    control.eval()
    forced_open.eval()

    control_run = control.run(tokens)
    forced_open_run = forced_open.run(tokens)
    logits_max_abs_diff = (
        control_run.full_logits - forced_open_run.full_logits
    ).abs().max().item()
    if logits_max_abs_diff > 1e-8:
        raise RuntimeError(
            f"Forced-open equivalence failed: max logits diff {logits_max_abs_diff:.12f}."
        )

    control.zero_grad(set_to_none=True)
    forced_open.zero_grad(set_to_none=True)
    control_loss = F.cross_entropy(control_run.last_logits, targets)
    forced_open_loss = F.cross_entropy(forced_open_run.last_logits, targets)
    control_loss.backward()
    forced_open_loss.backward()
    grad_diffs = {}
    for name, parameter in control.named_parameters():
        forced_parameter = dict(forced_open.named_parameters())[name]
        if parameter.grad is None and forced_parameter.grad is None:
            continue
        if parameter.grad is None or forced_parameter.grad is None:
            raise RuntimeError(f"Gradient mismatch nullability for parameter {name}.")
        grad_diffs[name] = (parameter.grad - forced_parameter.grad).abs().max().item()
    max_grad_diff = max(grad_diffs.values(), default=0.0)
    if max_grad_diff > 1e-8:
        raise RuntimeError(
            f"Forced-open gradient equivalence failed: max grad diff {max_grad_diff:.12f}."
        )
    return {
        "logits_max_abs_diff": logits_max_abs_diff,
        "max_grad_abs_diff": max_grad_diff,
    }


def skip_semantics(
    *,
    vocab_size: int,
    config: RunConfig,
    tokens: Tensor,
    device: torch.device,
) -> dict[str, object]:
    model = make_model(
        vocab_size=vocab_size,
        config=config,
        variant=make_variant("forced_open", config=config),
        device=device,
    )
    model.eval()
    block_rows = []
    for block_index in (1, 2):
        mask = torch.ones(tokens.shape[0], tokens.shape[1], 1, device=device)
        mask[:, block_index::2, :] = 0.0
        run = model.run(tokens, gate_overrides={block_index: mask})
        input_to_block = run.hidden_states[block_index]
        output_of_block = run.hidden_states[block_index + 1]
        skipped_positions = mask.squeeze(-1) == 0.0
        executed_positions = mask.squeeze(-1) == 1.0
        skipped_identity_diff = (
            output_of_block[skipped_positions] - input_to_block[skipped_positions]
        ).abs().max().item()
        executed_change = (
            output_of_block[executed_positions] - input_to_block[executed_positions]
        ).abs().max().item()
        if skipped_identity_diff > 1e-8:
            raise RuntimeError(
                f"Skip semantics failed in block {block_index + 1}: skipped positions changed by {skipped_identity_diff:.12f}."
            )
        if executed_change <= 1e-8:
            raise RuntimeError(
                f"Skip semantics failed in block {block_index + 1}: executed positions did not change."
            )
        block_rows.append(
            {
                "block_index": block_index + 1,
                "skipped_identity_max_abs_diff": skipped_identity_diff,
                "executed_positions_max_abs_change": executed_change,
            }
        )
    return {"blocks": block_rows}


def causal_mask_check(model: AsyncSelectiveCharModel, tokens: Tensor) -> dict[str, object]:
    model.eval()
    original = model.run(tokens)
    edited_tokens = tokens.clone()
    edited_tokens[:, -1] = (edited_tokens[:, -1] + 1) % model.output.out_features
    edited = model.run(edited_tokens)
    earlier_max_diff = (
        original.full_logits[:, :-1, :] - edited.full_logits[:, :-1, :]
    ).abs().max().item()
    edited_position_diff = (
        original.full_logits[:, -1, :] - edited.full_logits[:, -1, :]
    ).abs().max().item()
    if earlier_max_diff > 1e-6:
        raise RuntimeError(
            f"Causality check failed: earlier positions changed by {earlier_max_diff:.12f}."
        )
    if edited_position_diff <= 1e-6:
        raise RuntimeError("Causality check failed: edited token did not affect its own position.")
    return {
        "edited_position": tokens.shape[1] - 1,
        "earlier_positions_max_abs_diff": earlier_max_diff,
        "edited_position_max_abs_diff": edited_position_diff,
    }


def grad_norm(parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += parameter.grad.detach().float().pow(2).sum().item()
    return total**0.5


def gradient_routing(
    *,
    vocab_size: int,
    config: RunConfig,
    tokens: Tensor,
    targets: Tensor,
    device: torch.device,
) -> dict[str, object]:
    model = make_model(
        vocab_size=vocab_size,
        config=config,
        variant=make_variant(
            "learned_gate",
            config=config,
            target_open_rate=config.target_open_rate,
        ),
        device=device,
    )
    model.train()
    run = model.run(tokens)
    loss = F.cross_entropy(run.last_logits, targets) + run.budget_loss
    model.zero_grad(set_to_none=True)
    loss.backward()
    block_rows = []
    for block in model.blocks:
        gate_grad = 0.0 if block.gate is None else grad_norm(list(block.gate.parameters()))
        block_row = {
            "block_index": block.block_index + 1,
            "attention_grad_norm": grad_norm(list(block.attention.parameters())),
            "feedforward_grad_norm": grad_norm(list(block.feedforward.parameters())),
            "gate_grad_norm": gate_grad,
        }
        if block_row["attention_grad_norm"] <= 0.0:
            raise RuntimeError(f"Zero attention gradient in block {block.block_index + 1}.")
        if block_row["feedforward_grad_norm"] <= 0.0:
            raise RuntimeError(f"Zero feedforward gradient in block {block.block_index + 1}.")
        if block.block_index > 0 and gate_grad <= 0.0:
            raise RuntimeError(f"Zero gate gradient in block {block.block_index + 1}.")
        block_rows.append(block_row)
    return {
        "loss": loss.item(),
        "budget_loss": run.budget_loss.item(),
        "blocks": block_rows,
    }


def budget_control(*, vocab_size: int, config: RunConfig, tokens: Tensor, device: torch.device) -> dict[str, object]:
    model = make_model(
        vocab_size=vocab_size,
        config=config,
        variant=make_variant(
            "learned_gate",
            config=config,
            target_open_rate=config.target_open_rate,
        ),
        device=device,
    )
    model.eval()
    zeros = torch.zeros(tokens.shape[0], tokens.shape[1], 1, device=device)
    ones = torch.ones(tokens.shape[0], tokens.shape[1], 1, device=device)
    half = ones.clone()
    half[:, ::2, :] = 0.0
    all_closed_loss = model.run(tokens, gate_overrides={1: zeros, 2: zeros}).budget_loss.item()
    exact_target_loss = model.run(tokens, gate_overrides={1: half, 2: half}).budget_loss.item()
    all_open_loss = model.run(tokens, gate_overrides={1: ones, 2: ones}).budget_loss.item()
    expected_all_closed = budget_regularizer(
        [zeros, zeros],
        target_open_rate=config.target_open_rate,
        budget_weight=config.budget_weight,
    ).item()
    expected_exact_target = budget_regularizer(
        [half, half],
        target_open_rate=config.target_open_rate,
        budget_weight=config.budget_weight,
    ).item()
    expected_all_open = budget_regularizer(
        [ones, ones],
        target_open_rate=config.target_open_rate,
        budget_weight=config.budget_weight,
    ).item()
    if not math.isclose(all_closed_loss, expected_all_closed, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("Budget control failed for all-closed gate override.")
    if not math.isclose(exact_target_loss, expected_exact_target, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("Budget control failed for exact-target gate override.")
    if not math.isclose(all_open_loss, expected_all_open, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError("Budget control failed for all-open gate override.")
    if exact_target_loss > 1e-8:
        raise RuntimeError(f"Budget control failed: expected exact target loss 0, got {exact_target_loss:.12f}.")
    return {
        "target_open_rate": config.target_open_rate,
        "all_closed_loss": all_closed_loss,
        "exact_target_loss": exact_target_loss,
        "all_open_loss": all_open_loss,
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
        args.repo_root / "experiments" / "async_selective" / "artifacts" / "stage1_checks"
    )

    set_seed(config.seed)
    device = resolve_device(args.device)
    raw_text = text_file.read_text(encoding="utf-8")
    split = build_fixed_length_split(raw_text, config=config)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens = split.train_inputs[: config.check_batch_size].to(device)
    targets = split.train_targets[: config.check_batch_size].to(device)
    vocab_size = split.train_dataset.vocab_size
    control_model = make_model(
        vocab_size=vocab_size,
        config=config,
        variant=make_variant("synchronous_control", config=config),
        device=device,
    )

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
            "vocab_size": vocab_size,
        },
        "dataset_round_trip": dataset_round_trip(split.train_dataset, context_size=config.context_size),
        "parameter_counts": parameter_table(vocab_size, config),
        "checks": {
            "forced_open_equivalence": forced_open_equivalence(
                vocab_size=vocab_size,
                config=config,
                tokens=tokens,
                targets=targets,
                device=device,
            ),
            "skip_semantics": skip_semantics(
                vocab_size=vocab_size,
                config=config,
                tokens=tokens,
                device=device,
            ),
            "causality": causal_mask_check(control_model, tokens),
            "gradient_routing": gradient_routing(
                vocab_size=vocab_size,
                config=config,
                tokens=tokens,
                targets=targets,
                device=device,
            ),
            "budget_control": budget_control(
                vocab_size=vocab_size,
                config=config,
                tokens=tokens,
                device=device,
            ),
        },
    }
    write_json(output_dir / "mechanical_trust.json", report)
    print("PASS mechanical trust")
    print(f"Wrote {output_dir / 'mechanical_trust.json'}")


if __name__ == "__main__":
    main()
