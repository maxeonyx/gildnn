from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    raise RuntimeError(
        "Run from the repo root with `.\\.venv\\Scripts\\python.exe -m experiments.grammar_depth.analyze_depth ...` so package imports resolve cleanly."
    )

import torch
from jaxtyping import Int
from torch import Tensor

from core.recurrent_depth import RecurrentDepthConfig, RecurrentDepthLM
from experiments.grammar_depth.common import load_depth_annotations, token_role


ROLE_ORDER = {
    "opener": 0,
    "closer": 1,
    "separator": 2,
    "atom": 3,
    "newline": 4,
}


@dataclass(frozen=True)
class AggregateRow:
    token_role: str
    nesting_depth: int
    count: int
    mean_halt_depth: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(requested_device)


def load_checkpoint(path: Path, *, device: torch.device) -> tuple[dict[str, object], RecurrentDepthLM]:
    payload = torch.load(path, map_location=device)
    model_config = RecurrentDepthConfig(**payload["model_config"])
    model = RecurrentDepthLM(vocab_size=int(payload["vocab_size"]), config=model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return payload, model


def encode_text(text: str, *, char_to_idx: dict[str, int]) -> Int[Tensor, "tokens"]:
    return torch.tensor([char_to_idx[character] for character in text], dtype=torch.long)


def all_validation_windows(
    encoded_validation: Int[Tensor, "tokens"],
    *,
    context_size: int,
) -> tuple[Int[Tensor, "examples context"], Int[Tensor, "examples"]]:
    total_windows = encoded_validation.numel() - context_size
    if total_windows <= 0:
        raise ValueError(
            "Validation corpus is too short for full teacher-forced analysis. "
            f"Got length {encoded_validation.numel()} and context_size {context_size}."
        )
    inputs = encoded_validation.unfold(0, context_size, 1)[:total_windows].clone()
    targets = encoded_validation[context_size:].clone()
    return inputs, targets


@torch.inference_mode()
def collect_halting_depths(
    model: RecurrentDepthLM,
    *,
    inputs: Int[Tensor, "examples context"],
    epsilon: float,
    batch_size: int,
) -> list[int]:
    halt_depths: list[int] = []
    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        output = model.forward_with_halting(inputs[start:stop], epsilon=epsilon)
        halt_depths.extend(int(value) for value in output.halt_depths.detach().cpu().tolist())
    return halt_depths


def aggregate_rows(
    *,
    characters: str,
    nesting_depths: list[int],
    halt_depths: list[int],
) -> list[AggregateRow]:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for character, nesting_depth, halt_depth in zip(characters, nesting_depths, halt_depths, strict=True):
        grouped[(token_role(character), nesting_depth)].append(halt_depth)
    rows = [
        AggregateRow(
            token_role=token_role_name,
            nesting_depth=nesting_depth,
            count=len(values),
            mean_halt_depth=sum(values) / len(values),
        )
        for (token_role_name, nesting_depth), values in grouped.items()
    ]
    return sorted(rows, key=lambda row: (ROLE_ORDER[row.token_role], row.nesting_depth))


def print_table(rows: list[AggregateRow]) -> None:
    print("token_role | nesting_depth | count | mean_halt_depth", flush=True)
    for row in rows:
        print(
            f"{row.token_role:<10} | {row.nesting_depth:<13} | {row.count:<5} | {row.mean_halt_depth:.4f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_payload, model = load_checkpoint(args.checkpoint, device=device)
    experiment_config = checkpoint_payload["experiment_config"]
    data_dir = Path(str(experiment_config["data_dir"]))
    val_text = (data_dir / "val.txt").read_text(encoding="utf-8")
    val_depths = load_depth_annotations(data_dir / "val_depth.json")
    if len(val_text) != len(val_depths):
        raise ValueError("Validation text and depth annotations must have the same length")

    char_to_idx = {str(character): int(index) for character, index in checkpoint_payload["char_to_idx"].items()}
    encoded_validation = encode_text(val_text, char_to_idx=char_to_idx)
    context_size = int(checkpoint_payload["model_config"]["context_size"])
    inputs, _ = all_validation_windows(encoded_validation, context_size=context_size)
    halt_depths = collect_halting_depths(
        model,
        inputs=inputs.to(device),
        epsilon=float(experiment_config["halt_epsilon"]),
        batch_size=int(experiment_config["eval_batch_size"]),
    )

    target_characters = val_text[context_size:]
    target_depths = val_depths[context_size:]
    if not (len(target_characters) == len(target_depths) == len(halt_depths)):
        raise ValueError(
            "Teacher-forced targets, depth annotations, and halting outputs must align. "
            f"Got chars={len(target_characters)}, depths={len(target_depths)}, halts={len(halt_depths)}"
        )

    rows = aggregate_rows(characters=target_characters, nesting_depths=target_depths, halt_depths=halt_depths)
    summary = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(data_dir),
        "context_size": context_size,
        "examples": len(halt_depths),
        "rows": [
            {
                "token_role": row.token_role,
                "nesting_depth": row.nesting_depth,
                "count": row.count,
                "mean_halt_depth": round(row.mean_halt_depth, 6),
            }
            for row in rows
        ],
    }

    output_path = args.output_path or (args.checkpoint.parent / "analysis_summary.json")
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_table(rows)
    print(f"json_summary={output_path}", flush=True)


if __name__ == "__main__":
    main()
