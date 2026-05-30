#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fcntl
import subprocess
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TextIO


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run shell commands from a queue file until it is empty.",
    )
    parser.add_argument("queue_file", help="Path to the queue file")
    return parser.parse_args(argv)


def pop_next_command(queue_path: Path) -> str | None:
    with queue_path.open("r+", encoding="utf-8") as queue_file:
        fcntl.flock(queue_file.fileno(), fcntl.LOCK_EX)
        try:
            lines = queue_file.readlines()
            for index, line in enumerate(lines):
                stripped = line.strip()
                if stripped == "" or stripped.startswith("#"):
                    continue

                remaining_lines = lines[:index] + lines[index + 1 :]
                queue_file.seek(0)
                queue_file.writelines(remaining_lines)
                queue_file.truncate()
                return line.rstrip("\n")

            return None
        finally:
            fcntl.flock(queue_file.fileno(), fcntl.LOCK_UN)


def format_timestamp_for_filename(now: datetime) -> str:
    return now.strftime("%Y%m%d-%H%M%S-%f")


def format_timestamp_for_log(now: datetime) -> str:
    return now.isoformat(timespec="seconds")


def queue_name_for_log(queue_path: Path) -> str:
    return queue_path.stem


def open_log_file(log_dir: Path, queue_name: str, started_at: datetime) -> tuple[Path, TextIO]:
    log_path = log_dir / f"{format_timestamp_for_filename(started_at)}-{queue_name}.log"
    handle = log_path.open("w", encoding="utf-8")
    return log_path, handle


def write_log_header(handle: TextIO, *, command: str, queue_name: str, started_at: datetime) -> None:
    handle.write(f"command: {command}\n")
    handle.write(f"timestamp: {format_timestamp_for_log(started_at)}\n")
    handle.write(f"queue: {queue_name}\n")
    handle.write("\n")
    handle.flush()


def write_log_footer(handle: TextIO, *, exit_code: int, elapsed_seconds: float) -> None:
    handle.write("\n")
    handle.write(f"exit code: {exit_code}\n")
    handle.write(f"elapsed time: {elapsed_seconds:.3f}s\n")
    handle.flush()


def run_command(command: str, *, queue_name: str, log_dir: Path) -> int:
    started_at = datetime.now().astimezone()
    started_monotonic = time.monotonic()
    _, log_handle = open_log_file(log_dir, queue_name, started_at)

    with log_handle:
        write_log_header(
            log_handle,
            command=command,
            queue_name=queue_name,
            started_at=started_at,
        )

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                executable="/bin/bash",
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            exit_code = process.wait()
        except OSError as exc:
            log_handle.write(
                f"failed to start command: {exc.__class__.__name__}: {exc}\n",
            )
            exit_code = 127

        elapsed_seconds = time.monotonic() - started_monotonic
        write_log_footer(log_handle, exit_code=exit_code, elapsed_seconds=elapsed_seconds)

    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    queue_path = Path(args.queue_file)
    queue_name = queue_name_for_log(queue_path)
    log_dir = Path("runs/queue-logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    if not queue_path.exists():
        raise FileNotFoundError(
            f"Queue file does not exist: {queue_path}. Create it first, then rerun queue_runner.",
        )
    if not queue_path.is_file():
        raise ValueError(
            f"Queue path is not a file: {queue_path}. Pass a plain text queue file path.",
        )

    while True:
        command = pop_next_command(queue_path)
        if command is None:
            return 0

        run_command(command, queue_name=queue_name, log_dir=log_dir)


if __name__ == "__main__":
    raise SystemExit(main())
