from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from shutil import copyfileobj
from urllib.error import HTTPError
from urllib.request import urlopen
from zipfile import ZipFile

import torch
import pyarrow.parquet as pq
from torch import Tensor
from torch.utils.data import Dataset


WIKITEXT_103_RAW_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
WIKITEXT_103_RAW_FALLBACK_URL = (
    "https://research.metamind.io.s3.us-west-2.amazonaws.com/wikitext/wikitext-103-raw-v1.zip"
)
WIKITEXT_103_RAW_ARCHIVE_PREFIX = "wikitext-103-raw/"
WIKITEXT_103_RAW_FILES = (
    "wiki.train.raw",
    "wiki.valid.raw",
    "wiki.test.raw",
)
UNKNOWN_CHAR_TOKEN = "<unk>"
WIKITEXT_103_RAW_HF_PARQUET_URLS = {
    "wiki.train.raw": (
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
        "wikitext-103-raw-v1/train-00000-of-00002.parquet?download=true",
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
        "wikitext-103-raw-v1/train-00001-of-00002.parquet?download=true",
    ),
    "wiki.valid.raw": (
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
        "wikitext-103-raw-v1/validation-00000-of-00001.parquet?download=true",
    ),
    "wiki.test.raw": (
        "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
        "wikitext-103-raw-v1/test-00000-of-00001.parquet?download=true",
    ),
}


class RandomWindowCharDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, encoded_corpus: Tensor, *, context_size: int) -> None:
        if encoded_corpus.ndim != 1:
            raise ValueError(
                f"RandomWindowCharDataset expects a 1D encoded corpus tensor, got shape {tuple(encoded_corpus.shape)}."
            )
        if encoded_corpus.dtype != torch.long:
            raise ValueError(
                f"RandomWindowCharDataset expects torch.long tokens, got {encoded_corpus.dtype}."
            )
        if encoded_corpus.numel() <= context_size:
            raise ValueError(
                "RandomWindowCharDataset needs more encoded tokens than context_size. "
                f"Got corpus length {encoded_corpus.numel()} and context_size {context_size}."
            )

        self.encoded_corpus = encoded_corpus
        self.context_size = context_size

    def __len__(self) -> int:
        return self.encoded_corpus.numel() - self.context_size

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(f"Window start {index} out of range for dataset of length {len(self)}.")
        stop = index + self.context_size
        return self.encoded_corpus[index:stop], self.encoded_corpus[stop]


@dataclass(frozen=True)
class CorpusData:
    vocab_size: int
    context_size: int
    train_dataset: Dataset[tuple[Tensor, Tensor]]
    val_inputs: Tensor
    val_targets: Tensor
    char_to_idx: dict[str, int]
    idx_to_char: dict[int, str]

    def encode_text(self, text: str) -> list[int]:
        return [self.char_to_idx[char] for char in text]

    def decode_tokens(self, tokens: list[int] | Tensor) -> str:
        if isinstance(tokens, Tensor):
            resolved_tokens = tokens.tolist()
        else:
            resolved_tokens = tokens
        return "".join(self.idx_to_char[int(token)] for token in resolved_tokens)


def _format_character_list(characters: set[str]) -> str:
    return ", ".join(repr(character) for character in sorted(characters))


def _read_text(path: Path, *, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if len(text) == 0:
        raise ValueError(f"{label} file is empty: {path}")
    return text


def _validate_load_corpus_args(
    *,
    context_size: int,
    val_ratio: float,
    eval_samples: int,
) -> None:
    errors: list[str] = []
    if context_size <= 0:
        errors.append(f"context_size must be positive, got {context_size}")
    if not 0.0 < val_ratio < 1.0:
        errors.append(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if eval_samples <= 0:
        errors.append(f"eval_samples must be positive, got {eval_samples}")
    if errors:
        raise ValueError("load_corpus argument validation failed:\n- " + "\n- ".join(errors))


def _split_single_corpus(text: str, *, val_ratio: float, context_size: int) -> tuple[str, str]:
    split_index = int(len(text) * (1.0 - val_ratio))
    if split_index <= context_size:
        raise ValueError(
            "Single-file split leaves too little training text for the requested context. "
            f"Got split_index {split_index}, context_size {context_size}, corpus length {len(text)}."
        )
    if len(text) - split_index <= context_size:
        raise ValueError(
            "Single-file split leaves too little validation text for the requested context. "
            f"Got validation length {len(text) - split_index}, context_size {context_size}, corpus length {len(text)}."
        )
    return text[:split_index], text[split_index:]


def _build_vocab(train_text: str) -> tuple[dict[str, int], dict[int, str]]:
    vocabulary = sorted(set(train_text))
    if UNKNOWN_CHAR_TOKEN in vocabulary:
        raise ValueError(
            f"Training corpus contains reserved token {UNKNOWN_CHAR_TOKEN!r}; choose a different unknown token strategy."
        )
    vocabulary.append(UNKNOWN_CHAR_TOKEN)
    char_to_idx = {char: index for index, char in enumerate(vocabulary)}
    idx_to_char = {index: char for char, index in char_to_idx.items()}
    return char_to_idx, idx_to_char


def _encode_text(text: str, *, char_to_idx: dict[str, int], split_name: str) -> Tensor:
    missing_characters = set(text) - set(char_to_idx)
    if len(missing_characters) > 0 and UNKNOWN_CHAR_TOKEN not in char_to_idx:
        raise ValueError(
            f"{split_name} text contains characters that are not present in the training vocabulary, but no unknown token is configured: "
            f"{_format_character_list(missing_characters)}"
        )
    unknown_index = char_to_idx[UNKNOWN_CHAR_TOKEN]
    return torch.tensor([char_to_idx.get(char, unknown_index) for char in text], dtype=torch.long)


def _build_fixed_eval_set(
    encoded_validation: Tensor,
    *,
    context_size: int,
    eval_samples: int,
) -> tuple[Tensor, Tensor]:
    max_start = encoded_validation.numel() - context_size - 1
    if max_start < 0:
        raise ValueError(
            "Validation corpus is too short for next-token evaluation. "
            f"Got validation length {encoded_validation.numel()} and context_size {context_size}."
        )

    start_positions = torch.linspace(0, max_start, steps=eval_samples, dtype=torch.float64)
    start_positions = start_positions.round().to(dtype=torch.long)
    eval_inputs = torch.stack(
        [encoded_validation[start : start + context_size] for start in start_positions.tolist()]
    )
    eval_targets = encoded_validation[start_positions + context_size]
    return eval_inputs, eval_targets


def load_corpus(
    train_path: Path,
    val_path: Path | None = None,
    context_size: int = 32,
    val_ratio: float = 0.1,
    eval_samples: int = 1024,
    seed: int = 42,
) -> CorpusData:
    del seed
    _validate_load_corpus_args(
        context_size=context_size,
        val_ratio=val_ratio,
        eval_samples=eval_samples,
    )

    if val_path is None:
        combined_text = _read_text(train_path, label="Training corpus")
        train_text, val_text = _split_single_corpus(
            combined_text,
            val_ratio=val_ratio,
            context_size=context_size,
        )
    else:
        train_text = _read_text(train_path, label="Training corpus")
        val_text = _read_text(val_path, label="Validation corpus")
        if len(train_text) <= context_size:
            raise ValueError(
                "Training corpus is too short for next-token sampling. "
                f"Got training length {len(train_text)} and context_size {context_size}."
            )
        if len(val_text) <= context_size:
            raise ValueError(
                "Validation corpus is too short for next-token evaluation. "
                f"Got validation length {len(val_text)} and context_size {context_size}."
            )

    char_to_idx, idx_to_char = _build_vocab(train_text)
    train_encoded = _encode_text(train_text, char_to_idx=char_to_idx, split_name="Training")
    val_encoded = _encode_text(val_text, char_to_idx=char_to_idx, split_name="Validation")
    train_dataset = RandomWindowCharDataset(train_encoded, context_size=context_size)
    val_inputs, val_targets = _build_fixed_eval_set(
        val_encoded,
        context_size=context_size,
        eval_samples=eval_samples,
    )
    return CorpusData(
        vocab_size=len(char_to_idx),
        context_size=context_size,
        train_dataset=train_dataset,
        val_inputs=val_inputs,
        val_targets=val_targets,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
    )


def _download_url_bytes(*, urls: tuple[str, ...], timeout_seconds: int) -> bytes:
    failures: list[str] = []
    for url in urls:
        try:
            with urlopen(url, timeout=timeout_seconds) as response:
                return response.read()
        except HTTPError as error:
            failures.append(f"{url} -> HTTP {error.code} {error.reason}")
        except OSError as error:
            failures.append(f"{url} -> {error}")

    raise RuntimeError(
        "Failed to download WikiText-103 raw archive from all known URLs. "
        "Tried:\n- "
        + "\n- ".join(failures)
    )


def _write_rows_from_parquet_urls(
    *,
    parquet_urls: tuple[str, ...],
    destination_path: Path,
    timeout_seconds: int,
) -> None:
    with destination_path.open("w", encoding="utf-8", newline="") as destination:
        for parquet_url in parquet_urls:
            parquet_bytes = _download_url_bytes(urls=(parquet_url,), timeout_seconds=timeout_seconds)
            table = pq.read_table(BytesIO(parquet_bytes), columns=["text"])
            text_values = table.column("text").to_pylist()
            destination.write("".join(text_values))


def _download_wikitext_from_huggingface(
    *,
    expected_paths: dict[str, Path],
    timeout_seconds: int,
) -> None:
    for file_name, output_path in expected_paths.items():
        parquet_urls = WIKITEXT_103_RAW_HF_PARQUET_URLS[file_name]
        _write_rows_from_parquet_urls(
            parquet_urls=parquet_urls,
            destination_path=output_path,
            timeout_seconds=timeout_seconds,
        )


def download_wikitext_103_raw(
    destination_dir: Path | None = None,
    *,
    force: bool = False,
    timeout_seconds: int = 120,
) -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    resolved_destination = destination_dir or (repo_root / "data" / "wikitext-103-raw")
    resolved_destination.mkdir(parents=True, exist_ok=True)

    expected_paths = {
        file_name: resolved_destination / file_name for file_name in WIKITEXT_103_RAW_FILES
    }
    if not force and all(path.exists() for path in expected_paths.values()):
        return expected_paths

    try:
        archive_bytes = _download_url_bytes(
            urls=(WIKITEXT_103_RAW_URL, WIKITEXT_103_RAW_FALLBACK_URL),
            timeout_seconds=timeout_seconds,
        )

        with ZipFile(BytesIO(archive_bytes)) as archive:
            for file_name, output_path in expected_paths.items():
                archive_member = f"{WIKITEXT_103_RAW_ARCHIVE_PREFIX}{file_name}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(archive_member) as source, output_path.open("wb") as destination:
                    copyfileobj(source, destination)
    except RuntimeError:
        _download_wikitext_from_huggingface(
            expected_paths=expected_paths,
            timeout_seconds=timeout_seconds,
        )

    missing_files = [str(path) for path in expected_paths.values() if not path.exists()]
    if len(missing_files) > 0:
        raise RuntimeError(
            "WikiText-103 raw download completed but expected files are missing from the destination: "
            + ", ".join(missing_files)
        )

    return expected_paths
