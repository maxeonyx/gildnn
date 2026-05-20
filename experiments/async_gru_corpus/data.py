from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen
import zipfile


ENWIK8_URL = "https://mattmahoney.net/dc/enwik8.zip"
BASELINE_TINYSHAKESPEARE_VOCAB_CHARACTERS = 100_000


def current_git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def current_git_status_short() -> list[str]:
    result = subprocess.run(["git", "status", "--short"], check=True, capture_output=True, text=True)
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def tinyshakespeare_path(repo_root: Path) -> Path:
    return repo_root / "experiments" / "corpora.ignore" / "tinyshakespeare_input.txt"


def enwik8_text_path(repo_root: Path) -> Path:
    return repo_root / "experiments" / "async_gru_corpus" / "data.ignore" / "enwik8.txt"


def ensure_enwik8(repo_root: Path) -> Path:
    text_path = enwik8_text_path(repo_root)
    if text_path.exists():
        return text_path

    text_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = text_path.with_suffix(".zip")
    with urlopen(ENWIK8_URL) as response:
        zip_path.write_bytes(response.read())

    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        if "enwik8" not in members:
            raise RuntimeError(f"Downloaded archive missing enwik8 payload: {members}")
        text_path.write_bytes(archive.read("enwik8"))

    return text_path


def baseline_tinyshakespeare_vocab(repo_root: Path) -> set[str]:
    text = tinyshakespeare_path(repo_root).read_text(encoding="utf-8")
    return set(text[:BASELINE_TINYSHAKESPEARE_VOCAB_CHARACTERS])


def preprocess_corpus_text(*, repo_root: Path, corpus: str, text_file: Path, raw_text: str, allow_vocab_growth: bool) -> tuple[str, dict[str, object]]:
    if corpus != "tinyshakespeare" or allow_vocab_growth:
        return raw_text, {
            "allow_vocab_growth": allow_vocab_growth,
            "characters_removed": 0,
            "removed_characters": [],
        }

    allowed_chars = baseline_tinyshakespeare_vocab(repo_root)
    removed_counts: dict[str, int] = {}
    filtered_characters: list[str] = []
    for char in raw_text:
        if char in allowed_chars:
            filtered_characters.append(char)
            continue
        removed_counts[char] = removed_counts.get(char, 0) + 1
    filtered_text = "".join(filtered_characters)
    return filtered_text, {
        "allow_vocab_growth": allow_vocab_growth,
        "characters_removed": sum(removed_counts.values()),
        "removed_characters": [
            {"character": character, "count": count}
            for character, count in sorted(removed_counts.items(), key=lambda item: item[0])
        ],
        "baseline_vocab_source_file": str(tinyshakespeare_path(repo_root)),
        "baseline_vocab_characters": BASELINE_TINYSHAKESPEARE_VOCAB_CHARACTERS,
        "source_file_before_preprocessing": str(text_file),
        "source_total_characters_before_preprocessing": len(raw_text),
    }


def resolve_text_file(*, repo_root: Path, corpus: str, text_file: Path | None, download_missing: bool) -> Path:
    if text_file is not None:
        return text_file
    if corpus == "tinyshakespeare":
        return tinyshakespeare_path(repo_root)
    if corpus == "enwik8":
        if download_missing:
            return ensure_enwik8(repo_root)
        return enwik8_text_path(repo_root)
    raise ValueError(f"Unsupported corpus {corpus!r}.")


def corpus_summary(*, corpus: str, text_file: Path, raw_text: str, train_characters: int, val_characters: int, vocab_size: int, preprocessing: dict[str, object]) -> dict[str, object]:
    return {
        "corpus": corpus,
        "source_file": str(text_file),
        "source_total_characters": len(raw_text),
        "source_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "train_characters": train_characters,
        "val_characters": val_characters,
        "vocab_size": vocab_size,
        "preprocessing": preprocessing,
    }


def artifact_stem(*, corpus: str, train_characters: int, val_characters: int) -> str:
    return f"{corpus}_train{train_characters}_val{val_characters}"
