from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    raise RuntimeError(
        "Run from the repo root with `.\\.venv\\Scripts\\python.exe -m runs.generate_text ...` so `core` imports resolve cleanly."
    )

import torch

from core.fixed_window_char import FixedWindowCharDataset, resolve_device, set_seed
from core.generation import generate_with_halting, halting_depth_histogram
from core.recurrent_depth import RecurrentDepthConfig, RecurrentDepthLM


DEFAULT_CORPUS_PATH = Path("experiments") / "corpora.ignore" / "tinyshakespeare_input.txt"
REPO_ROOT = Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--length", type=non_negative_int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--top-k", type=positive_int, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--calibration-report", type=Path, default=None)
    parser.add_argument("--show-token-table", action="store_true")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path, *, device: torch.device) -> dict[str, object]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")
    return checkpoint


def build_model_from_checkpoint(checkpoint: dict[str, object], *, device: torch.device) -> RecurrentDepthLM:
    config_payload = checkpoint["config"]
    if not isinstance(config_payload, dict):
        raise TypeError(f"Expected checkpoint config dict, got {type(config_payload).__name__}")
    config = RecurrentDepthConfig(
        context_size=int(config_payload["context_size"]),
        d_model=int(config_payload["d_model"]),
        n_heads=int(config_payload["n_heads"]),
        ff_dim=int(config_payload["ff_dim"]),
        iterations=int(config_payload["recurrent_iterations"]),
        temperature=float(config_payload["temperature"]),
        dropout=float(config_payload.get("dropout", 0.0)),
    )
    model = RecurrentDepthLM(vocab_size=int(checkpoint["vocab_size"]), config=config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_vocab_from_corpus(*, corpus_path: Path, train_characters: int, context_size: int) -> FixedWindowCharDataset:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    corpus_text = corpus_path.read_text(encoding="utf-8")
    train_text = corpus_text[:train_characters]
    return FixedWindowCharDataset(train_text, context_size=context_size)


def load_calibration(report_path: Path | None, *, checkpoint_path: Path) -> dict[str, list[float]] | None:
    if report_path is None:
        return None
    if not report_path.exists():
        raise FileNotFoundError(f"Calibration report not found: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint_entry = payload.get("checkpoint_path")
    if checkpoint_entry is not None:
        resolved_report_checkpoint = (REPO_ROOT / Path(checkpoint_entry)).resolve()
        if resolved_report_checkpoint != checkpoint_path.resolve():
            raise ValueError(
                f"Calibration report targets {resolved_report_checkpoint}, not requested checkpoint {checkpoint_path.resolve()}"
            )
    affine_parameters = payload.get("affine_parameters")
    if not isinstance(affine_parameters, list) or len(affine_parameters) == 0:
        raise ValueError(f"Calibration report missing affine_parameters: {report_path}")
    return {
        "scale": [float(entry["scale"]) for entry in affine_parameters],
        "bias": [float(entry["bias"]) for entry in affine_parameters],
    }


def render_visible_char(char: str) -> str:
    if char == "\n":
        return "\\n"
    if char == "\t":
        return "\\t"
    if char == " ":
        return "<sp>"
    return char


def format_annotated_tokens(text: str, depths: tuple[int, ...]) -> str:
    tokens = [f"{render_visible_char(char)}[{depth}]" for char, depth in zip(text, depths, strict=True)]
    rows = []
    row_width = 20
    for start in range(0, len(tokens), row_width):
        rows.append(" ".join(tokens[start : start + row_width]))
    return "\n".join(rows)


def format_token_table(text: str, depths: tuple[int, ...]) -> str:
    rows = ["index token depth"]
    for index, (char, depth) in enumerate(zip(text, depths, strict=True)):
        rows.append(f"{index:>5} {render_visible_char(char):>5} {depth:>5}")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    if args.temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {args.temperature}")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {args.top_p}")

    device = resolve_device(args.device)
    set_seed(args.seed)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    model = build_model_from_checkpoint(checkpoint, device=device)
    calibration = load_calibration(args.calibration_report, checkpoint_path=args.checkpoint)
    dataset = build_vocab_from_corpus(
        corpus_path=args.corpus_path,
        train_characters=int(checkpoint["train_characters"]),
        context_size=model.config.context_size,
    )
    result = generate_with_halting(
        model,
        prompt=args.prompt,
        stoi=dataset.stoi,
        itos=dataset.itos,
        length=args.length,
        threshold=args.threshold,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        calibration=calibration,
    )

    depth_histogram = halting_depth_histogram(result.halting_depths)
    summary = {
        "checkpoint": str(args.checkpoint),
        "device": device.type,
        "prompt": args.prompt,
        "prompt_length": len(args.prompt),
        "generated_length": len(result.generated_text),
        "temperature": args.temperature,
        "threshold": args.threshold,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "calibrated": calibration is not None,
        "depth_histogram": depth_histogram,
        "mean_depth": round(sum(result.halting_depths) / max(len(result.halting_depths), 1), 6),
    }

    print("=== Summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("=== Generated text ===", flush=True)
    print(result.generated_text, flush=True)
    print("=== Annotated generated tokens ===", flush=True)
    print(format_annotated_tokens(result.generated_text, result.halting_depths), flush=True)
    if args.show_token_table:
        print("=== Token table ===", flush=True)
        print(format_token_table(result.generated_text, result.halting_depths), flush=True)


if __name__ == "__main__":
    main()
