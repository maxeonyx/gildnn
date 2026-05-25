from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter

# Ensure repo root is importable regardless of how this script is launched
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from einops import rearrange, repeat
from jaxtyping import Int
from torch import Tensor, nn

from core.dataset import load_corpus
from core.model import ParallelDiagonalModel
from core.run_utils import append_log, checkpoint_metrics, release_memory, resolve_device
from runs.temporal_window_4block import (
    CONTEXT_SIZE,
    DEFAULT_SEEDS,
    EVAL_BATCH_SIZE,
    EVAL_SAMPLES,
    VariantSpec,
    build_model,
    variant_specs,
)

EXPECTED_VARIANT_KEY = "W8_all"
EXPECTED_VOCAB_SIZE = 4_980
AGE_SELECTIVE_KEEP_COUNTS = (1, 2, 4, 8)


type TemporalIntervention = Callable[[list[Tensor], int], None]


class TemporalInterventionModel(nn.Module):
    def __init__(self, *, model: ParallelDiagonalModel, intervention: TemporalIntervention) -> None:
        super().__init__()
        self.model = model
        self.intervention = intervention

    def forward(self, tokens: Int[Tensor, "batch context"]) -> Tensor:
        return eval_with_temporal_intervention(
            model=self.model,
            tokens=tokens,
            intervention=self.intervention,
        )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifact_dir = repo_root / "experiments" / "temporal-window" / "artifacts"
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--eval-samples", type=int, default=EVAL_SAMPLES)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--train-path",
        type=Path,
        default=repo_root / "data" / "wikitext-103-raw" / "wiki.train.raw",
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=repo_root / "data" / "wikitext-103-raw" / "wiki.valid.raw",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=artifact_dir / "trajectory_diagnostics.jsonl",
    )
    return parser.parse_args()


def resolve_w8_spec() -> VariantSpec:
    spec = variant_specs().get(EXPECTED_VARIANT_KEY)
    if spec is None:
        raise RuntimeError(f"Variant {EXPECTED_VARIANT_KEY!r} not found in runs.temporal_window_4block.variant_specs().")
    if spec.temporal_window != 8:
        raise RuntimeError(f"Expected W8_all temporal_window=8, got {spec.temporal_window}.")
    if spec.temporal_window_mode != "history":
        raise RuntimeError(
            f"Expected W8_all temporal_window_mode='history', got {spec.temporal_window_mode!r}."
        )
    return spec


def checkpoint_path_for_seed(*, checkpoint_dir: Path, seed: int) -> Path:
    return checkpoint_dir / f"{EXPECTED_VARIANT_KEY}_seed_{seed}.pt"


def validate_checkpoint_paths(*, checkpoint_dir: Path, seeds: Sequence[int]) -> list[Path]:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {checkpoint_dir}")

    checkpoint_paths = [checkpoint_path_for_seed(checkpoint_dir=checkpoint_dir, seed=seed) for seed in seeds]
    missing = [path for path in checkpoint_paths if not path.exists()]
    if len(missing) > 0:
        missing_rendered = "\n- ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing W8_all checkpoints for trajectory diagnostics. Expected files:\n- " + missing_rendered
        )
    return checkpoint_paths


def load_model_from_checkpoint(
    *,
    checkpoint_path: Path,
    device: torch.device,
    vocab_size: int,
    spec: VariantSpec,
) -> ParallelDiagonalModel:
    model = build_model(device=device, vocab_size=vocab_size, spec=spec)
    loaded_state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(loaded_state, dict):
        raise TypeError(f"Expected checkpoint at {checkpoint_path} to be a state_dict dict.")
    model.load_state_dict(loaded_state)
    return model


def validate_model_for_diagnostics(model: nn.Module) -> ParallelDiagonalModel:
    if not isinstance(model, ParallelDiagonalModel):
        raise TypeError(f"Trajectory diagnostics require ParallelDiagonalModel, got {type(model).__name__}.")
    if model.temporal_window <= 0:
        raise ValueError(f"Trajectory diagnostics require temporal_window > 0, got {model.temporal_window}.")
    if model.temporal_window_mode != "history":
        raise ValueError(
            "Trajectory diagnostics require temporal_window_mode='history'. "
            f"Got {model.temporal_window_mode!r}."
        )
    if model.window_proj is None:
        raise RuntimeError("Trajectory diagnostics require window_proj to be present.")
    return model


@torch.inference_mode()
def eval_with_temporal_intervention(
    *,
    model: ParallelDiagonalModel,
    tokens: Int[Tensor, "batch context"],
    intervention: TemporalIntervention,
) -> Tensor:
    model = validate_model_for_diagnostics(model)
    embeddings = model.embedded_tokens(tokens)
    batch_size = tokens.shape[0]
    previous_states = [
        torch.zeros(batch_size, model.d_model, device=tokens.device, dtype=embeddings.dtype)
        for _ in range(model.num_blocks)
    ]
    temporal_history = [
        torch.zeros(
            batch_size,
            model.temporal_window,
            model.d_model,
            device=tokens.device,
            dtype=embeddings.dtype,
        )
        for _ in range(model.num_blocks)
    ]

    for time_index in range(model.context_size):
        token_state = embeddings[:, time_index, :]
        if model.token_injection == "all":
            seeded_states = [
                token_mix(previous_state, token_state)
                for token_mix, previous_state in zip(model.token_mixes, previous_states, strict=True)
            ]
        else:
            seeded_states = list(previous_states)
            seeded_states[0] = model.token_mixes[0](previous_states[0], token_state)

        current_states = list(previous_states)
        for internal_step in range(model.internal_steps):
            next_states = list(current_states)
            for block_index, (block, block_mix, rate) in enumerate(
                zip(model.blocks, model.block_mixes, model.rates, strict=True)
            ):
                if time_index % rate != 0:
                    continue

                state_input = seeded_states[block_index] if internal_step == 0 else current_states[block_index]
                if model.topology == "isolated" or (block_index == 0 and model.topology == "upward"):
                    block_input = state_input
                else:
                    if block_index == 0:
                        lateral_source = previous_states[1] if internal_step == 0 else current_states[1]
                    else:
                        lateral_source = (
                            previous_states[block_index - 1]
                            if internal_step == 0
                            else current_states[block_index - 1]
                        )
                    current_lower = model._maybe_detach_lateral(lateral_source)
                    block_input = 0.5 * (state_input + current_lower)
                    if block_index > 0:
                        lower_history = model._maybe_detach_lateral(temporal_history[block_index - 1])
                        aux = model.window_proj(
                            rearrange(lower_history, "batch window d_model -> batch (window d_model)")
                        )
                        block_input = block_input + aux

                block_delta = block(block_input)
                next_states[block_index] = block_mix(block_input, block_delta)

            current_states = next_states

        previous_states = current_states
        for block_index, (state, rate) in enumerate(zip(previous_states, model.rates, strict=True)):
            if time_index % rate != 0:
                continue
            updated_history = torch.roll(temporal_history[block_index], shifts=-1, dims=1)
            updated_history[:, -1, :] = state
            temporal_history[block_index] = updated_history
        intervention(temporal_history, time_index)

    return model.output(model._readout_state(previous_states))


def no_op_intervention(temporal_history: list[Tensor], time_index: int) -> None:
    del temporal_history, time_index


def build_repeat_last_history_intervention(*, window: int) -> TemporalIntervention:
    def intervention(temporal_history: list[Tensor], time_index: int) -> None:
        del time_index
        latest_state = temporal_history[0][:, -1, :]
        temporal_history[0] = repeat(latest_state, "batch d_model -> batch window d_model", window=window)

    return intervention


def build_window_permutation_intervention(*, permutation: Tensor) -> TemporalIntervention:
    def intervention(temporal_history: list[Tensor], time_index: int) -> None:
        del time_index
        temporal_history[0] = temporal_history[0][:, permutation, :]

    return intervention


def build_age_selective_ablation_intervention(*, keep_count: int, window: int) -> TemporalIntervention:
    if keep_count <= 0:
        raise ValueError(f"keep_count must be positive, got {keep_count}.")
    if keep_count > window:
        raise ValueError(f"keep_count must be <= window ({window}), got {keep_count}.")

    def intervention(temporal_history: list[Tensor], time_index: int) -> None:
        del time_index
        if keep_count == window:
            return
        temporal_history[0][:, : window - keep_count, :] = 0.0

    return intervention


def fixed_non_identity_permutation(*, window: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(window, generator=generator)
    identity = torch.arange(window)
    if torch.equal(permutation, identity):
        permutation = torch.roll(identity, shifts=1)
    return permutation


def run_single_diagnostic(
    *,
    model: ParallelDiagonalModel,
    intervention_name: str,
    intervention_family: str,
    intervention: TemporalIntervention,
    val_inputs: Int[Tensor, "examples context"],
    val_targets: Int[Tensor, "examples"],
    eval_batch_size: int,
    baseline_loss: float | None,
) -> dict[str, object]:
    model_call = TemporalInterventionModel(model=model, intervention=intervention)
    metrics = checkpoint_metrics(
        model=model,
        model_call=model_call,
        val_inputs=val_inputs,
        val_targets=val_targets,
        batch_size=eval_batch_size,
        step=0,
        include_tokens_per_second=False,
    )
    val_loss = float(metrics["val_loss"])
    return {
        "diagnostic": intervention_name,
        "family": intervention_family,
        "val_loss": val_loss,
        "val_accuracy": float(metrics["val_accuracy"]),
        "delta_vs_baseline": None if baseline_loss is None else round(val_loss - baseline_loss, 6),
    }


def build_diagnostic_plan(*, window: int, seed: int) -> list[dict[str, object]]:
    permutation = fixed_non_identity_permutation(window=window, seed=seed + 10_000)
    diagnostics: list[dict[str, object]] = [
        {
            "diagnostic": "baseline",
            "family": "baseline",
            "intervention": no_op_intervention,
            "metadata": {},
        },
        {
            "diagnostic": "repeat_last_history",
            "family": "repeat_last_history",
            "intervention": build_repeat_last_history_intervention(window=window),
            "metadata": {},
        },
        {
            "diagnostic": "window_permutation",
            "family": "window_permutation",
            "intervention": build_window_permutation_intervention(permutation=permutation),
            "metadata": {"permutation": permutation.tolist()},
        },
    ]
    for keep_count in AGE_SELECTIVE_KEEP_COUNTS:
        diagnostics.append(
            {
                "diagnostic": f"age_selective_k{keep_count}",
                "family": "age_selective_ablation",
                "intervention": build_age_selective_ablation_intervention(keep_count=keep_count, window=window),
                "metadata": {"keep_count": keep_count},
            }
        )
    return diagnostics


def format_summary_table(rows: Sequence[dict[str, object]]) -> str:
    headers = ("seed", "diagnostic", "val_loss", "delta_vs_baseline")
    rendered_rows = [
        (
            str(row["seed"]),
            str(row["diagnostic"]),
            f"{float(row['val_loss']):.6f}",
            "-" if row["delta_vs_baseline"] is None else f"{float(row['delta_vs_baseline']):+.6f}",
        )
        for row in rows
    ]
    widths = [
        max(len(header), *(len(rendered_row[index]) for rendered_row in rendered_rows))
        for index, header in enumerate(headers)
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for rendered_row in rendered_rows:
        lines.append(
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(rendered_row))
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    spec = resolve_w8_spec()
    device = resolve_device(args.device)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    log_restart_payload = {
        "stage": "run_restarted",
        "checkpoint_dir": str(args.checkpoint_dir),
    }
    append_log(args.log_path, log_restart_payload)

    validate_checkpoint_paths(checkpoint_dir=args.checkpoint_dir, seeds=args.seeds)
    corpus = load_corpus(
        train_path=args.train_path,
        val_path=args.val_path,
        context_size=CONTEXT_SIZE,
        eval_samples=args.eval_samples,
    )
    if corpus.vocab_size != EXPECTED_VOCAB_SIZE:
        raise RuntimeError(
            f"Expected vocab size {EXPECTED_VOCAB_SIZE} for WikiText-103 raw char setup, got {corpus.vocab_size}."
        )

    val_inputs = corpus.val_inputs.to(device=device, dtype=torch.long)
    val_targets = corpus.val_targets.to(device=device, dtype=torch.long)
    append_log(
        args.log_path,
        {
            "stage": "trajectory_diagnostics_started",
            "checkpoint_dir": str(args.checkpoint_dir),
            "seeds": list(args.seeds),
            "eval_samples": args.eval_samples,
            "eval_batch_size": args.eval_batch_size,
            "device": str(device),
            "context_size": CONTEXT_SIZE,
            "vocab_size": corpus.vocab_size,
            "variant": EXPECTED_VARIANT_KEY,
        },
    )

    started_at = perf_counter()
    summary_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        checkpoint_path = checkpoint_path_for_seed(checkpoint_dir=args.checkpoint_dir, seed=seed)
        model = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            vocab_size=corpus.vocab_size,
            spec=spec,
        )
        diagnostics = build_diagnostic_plan(window=model.temporal_window, seed=seed)
        baseline_entry = run_single_diagnostic(
            model=model,
            intervention_name="baseline",
            intervention_family="baseline",
            intervention=no_op_intervention,
            val_inputs=val_inputs,
            val_targets=val_targets,
            eval_batch_size=args.eval_batch_size,
            baseline_loss=None,
        )
        baseline_loss = float(baseline_entry["val_loss"])
        baseline_log_row = {
            "stage": "diagnostic_result",
            "seed": seed,
            "checkpoint_path": str(checkpoint_path),
            **baseline_entry,
        }
        append_log(args.log_path, baseline_log_row)
        summary_rows.append({"seed": seed, **baseline_entry})

        for diagnostic in diagnostics[1:]:
            result = run_single_diagnostic(
                model=model,
                intervention_name=str(diagnostic["diagnostic"]),
                intervention_family=str(diagnostic["family"]),
                intervention=diagnostic["intervention"],
                val_inputs=val_inputs,
                val_targets=val_targets,
                eval_batch_size=args.eval_batch_size,
                baseline_loss=baseline_loss,
            )
            log_row = {
                "stage": "diagnostic_result",
                "seed": seed,
                "checkpoint_path": str(checkpoint_path),
                **result,
                **dict(diagnostic["metadata"]),
            }
            append_log(args.log_path, log_row)
            summary_rows.append({"seed": seed, **result})

        del model
        release_memory(device=device)

    wall_seconds = round(perf_counter() - started_at, 6)
    append_log(
        args.log_path,
        {
            "stage": "trajectory_diagnostics_finished",
            "seeds": list(args.seeds),
            "rows": len(summary_rows),
            "wall_seconds": wall_seconds,
        },
    )
    print(format_summary_table(summary_rows), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        import traceback

        args = parse_args()
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        append_log(args.log_path, {"stage": "crash", "traceback": traceback.format_exc()})
        raise
