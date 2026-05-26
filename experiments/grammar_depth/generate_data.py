from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in (None, ""):
    raise RuntimeError(
        "Run from the repo root with `.\\.venv\\Scripts\\python.exe -m experiments.grammar_depth.generate_data` so package imports resolve cleanly."
    )

from experiments.grammar_depth.common import (
    ATOMS,
    RULES,
    DatasetBundle,
    grammar_data_dir,
    load_depth_annotations,
    generate_dataset,
    validate_annotation,
    write_annotated_split,
)


DEFAULT_SEED = 42
DEFAULT_MAX_DEPTH = 5
DEFAULT_TRAIN_CHARACTERS = 2_000_000
DEFAULT_VAL_CHARACTERS = 100_000


@dataclass(frozen=True)
class SplitSummary:
    characters: int
    min_depth: int
    max_depth: int


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-depth", type=positive_int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--train-characters", type=positive_int, default=DEFAULT_TRAIN_CHARACTERS)
    parser.add_argument("--val-characters", type=positive_int, default=DEFAULT_VAL_CHARACTERS)
    return parser.parse_args()


def summarize_split(text: str, depths: list[int]) -> SplitSummary:
    return SplitSummary(
        characters=len(text),
        min_depth=min(depths),
        max_depth=max(depths),
    )


def write_bundle(output_dir: Path, *, bundle: DatasetBundle) -> dict[str, SplitSummary]:
    write_annotated_split(output_dir, split_name="train", annotated=bundle.train)
    write_annotated_split(output_dir, split_name="val", annotated=bundle.val)
    return {
        "train": summarize_split(bundle.train.text, bundle.train.depths),
        "val": summarize_split(bundle.val.text, bundle.val.depths),
    }


def verify_written_split(output_dir: Path, *, split_name: str) -> None:
    text = (output_dir / f"{split_name}.txt").read_text(encoding="utf-8")
    depths = load_depth_annotations(output_dir / f"{split_name}_depth.json")
    validate_annotation(text=text, depths=depths)


def print_summary(label: str, output_dir: Path, summaries: dict[str, SplitSummary], *, configured_max_depth: int) -> None:
    print(f"[{label}] output_dir={output_dir}", flush=True)
    print(
        f"[{label}] vocab={sorted({*ATOMS, *{symbol for rule in RULES for symbol in rule}, chr(10)})}",
        flush=True,
    )
    print(f"[{label}] configured_max_depth={configured_max_depth}", flush=True)
    for split_name, summary in summaries.items():
        print(f"[{label}] {split_name}={asdict(summary)}", flush=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    grammar_bundle = generate_dataset(
        seed=args.seed,
        max_depth=args.max_depth,
        train_characters=args.train_characters,
        val_characters=args.val_characters,
    )
    flat_bundle = generate_dataset(
        seed=args.seed,
        max_depth=1,
        train_characters=args.train_characters,
        val_characters=args.val_characters,
    )

    grammar_dir = grammar_data_dir(repo_root, flat=False)
    flat_dir = grammar_data_dir(repo_root, flat=True)
    grammar_summaries = write_bundle(grammar_dir, bundle=grammar_bundle)
    flat_summaries = write_bundle(flat_dir, bundle=flat_bundle)

    for output_dir in (grammar_dir, flat_dir):
        verify_written_split(output_dir, split_name="train")
        verify_written_split(output_dir, split_name="val")

    print_summary("grammar", grammar_dir, grammar_summaries, configured_max_depth=args.max_depth)
    print_summary("grammar-flat", flat_dir, flat_summaries, configured_max_depth=1)


if __name__ == "__main__":
    main()
