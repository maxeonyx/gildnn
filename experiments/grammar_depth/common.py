from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


ATOMS = ("a", "b", "c")
RULES = (
    ("(", ",", ")"),
    ("[", ";", "]"),
    ("{", ":", "}"),
)
OPEN_TO_CLOSE = {opener: closer for opener, _, closer in RULES}
TOKEN_ROLE_BY_CHARACTER = {
    "(": "opener",
    "[": "opener",
    "{": "opener",
    ")": "closer",
    "]": "closer",
    "}": "closer",
    ",": "separator",
    ";": "separator",
    ":": "separator",
    "a": "atom",
    "b": "atom",
    "c": "atom",
    "\n": "newline",
}


@dataclass(frozen=True)
class AnnotatedText:
    text: str
    depths: list[int]


@dataclass(frozen=True)
class DatasetBundle:
    train: AnnotatedText
    val: AnnotatedText
    max_depth: int
    seed: int


def grammar_data_dir(repo_root: Path, *, flat: bool) -> Path:
    return repo_root / "data" / ("grammar-flat" if flat else "grammar")


def token_role(character: str) -> str:
    try:
        return TOKEN_ROLE_BY_CHARACTER[character]
    except KeyError as exc:
        raise ValueError(f"Unsupported grammar character: {character!r}") from exc


def _should_emit_atom(rng: random.Random, *, current_depth: int, max_depth: int) -> bool:
    if current_depth >= max_depth:
        return True
    base_probability = 0.28 + (0.12 * current_depth)
    return rng.random() < min(base_probability, 0.92)


def generate_expression(rng: random.Random, *, current_depth: int, max_depth: int) -> AnnotatedText:
    if _should_emit_atom(rng, current_depth=current_depth, max_depth=max_depth):
        atom = rng.choice(ATOMS)
        return AnnotatedText(text=atom, depths=[current_depth])

    opener, separator, closer = rng.choice(RULES)
    wrapper_depth = current_depth + 1
    left = generate_expression(rng, current_depth=wrapper_depth, max_depth=max_depth)
    right = generate_expression(rng, current_depth=wrapper_depth, max_depth=max_depth)
    return AnnotatedText(
        text=f"{opener}{left.text}{separator}{right.text}{closer}",
        depths=[wrapper_depth, *left.depths, wrapper_depth, *right.depths, current_depth],
    )


def generate_dataset(
    *,
    seed: int,
    max_depth: int,
    train_characters: int,
    val_characters: int,
) -> DatasetBundle:
    rng = random.Random(seed)
    train = generate_split(rng, target_characters=train_characters, max_depth=max_depth)
    val = generate_split(rng, target_characters=val_characters, max_depth=max_depth)
    return DatasetBundle(train=train, val=val, max_depth=max_depth, seed=seed)


def generate_split(rng: random.Random, *, target_characters: int, max_depth: int) -> AnnotatedText:
    parts: list[str] = []
    depths: list[int] = []
    while len(depths) < target_characters:
        expression = generate_expression(rng, current_depth=0, max_depth=max_depth)
        parts.append(expression.text)
        depths.extend(expression.depths)
        parts.append("\n")
        depths.append(0)
    text = "".join(parts)
    validate_annotation(text=text, depths=depths)
    return AnnotatedText(text=text, depths=depths)


def validate_annotation(*, text: str, depths: list[int]) -> None:
    if len(text) != len(depths):
        raise ValueError(f"Annotation length mismatch: text has {len(text)} chars, depths has {len(depths)} entries")

    expected_closers: list[str] = []
    for index, (character, depth) in enumerate(zip(text, depths, strict=True)):
        if depth < 0:
            raise ValueError(f"Negative depth at position {index}: {depth}")

        role = token_role(character)
        if role == "opener":
            expected_closers.append(OPEN_TO_CLOSE[character])
            expected_depth = len(expected_closers)
        elif role == "closer":
            if len(expected_closers) == 0:
                raise ValueError(f"Unexpected closer {character!r} at position {index}")
            expected_closer = expected_closers.pop()
            if character != expected_closer:
                raise ValueError(
                    f"Mismatched closer at position {index}: expected {expected_closer!r}, got {character!r}"
                )
            expected_depth = len(expected_closers)
        elif role == "newline":
            if len(expected_closers) != 0:
                raise ValueError(f"Newline at position {index} while expression is still open")
            expected_depth = 0
        else:
            expected_depth = len(expected_closers)

        if depth != expected_depth:
            raise ValueError(
                f"Depth mismatch at position {index}: character={character!r}, expected {expected_depth}, got {depth}"
            )

    if len(expected_closers) != 0:
        raise ValueError("Annotation ended with unclosed expressions")


def write_annotated_split(output_dir: Path, *, split_name: str, annotated: AnnotatedText) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{split_name}.txt").write_text(annotated.text, encoding="utf-8")
    with (output_dir / f"{split_name}_depth.json").open("w", encoding="utf-8") as handle:
        json.dump(annotated.depths, handle, separators=(",", ":"))


def load_depth_annotations(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not all(isinstance(value, int) for value in payload):
        raise ValueError(f"Depth annotation file must contain a JSON array of integers: {path}")
    return payload
