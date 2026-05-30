from __future__ import annotations

import argparse
import atexit
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch

try:
    import psutil
except ImportError:
    psutil = None


def append_log(log_path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def redirect_sanity_check_paths(args: argparse.Namespace) -> None:
    """Redirect artifact paths to a temp directory in sanity-check mode.

    Prevents sanity checks from interfering with live experiment runs
    that write to the same artifact directory.
    """
    import shutil
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="gildnn_sanity_"))
    args.report_path = tmp_dir / "report.json"
    args.log_path = tmp_dir / "run.jsonl"

    def _cleanup() -> None:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    atexit.register(_cleanup)


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(requested_device)


def register_active_lock(
    *,
    experiment_name: str,
    variants: object,
    enabled: bool = True,
    started_at: str | None = None,
    lock_path: Path | None = None,
    run_size: Literal["large", "small"] = "large",
) -> None:
    if not enabled:
        return

    resolved_lock_path = lock_path
    lock_description = "active experiment lock"
    if resolved_lock_path is None:
        resolved_lock_path = Path(f"runs/active-{run_size}.lock")
        lock_description = f"{run_size} experiment lock"

    def _parse_lock_metadata(lock_content: str) -> tuple[int, str]:
        fields: dict[str, str] = {}
        for line in lock_content.splitlines():
            if ": " not in line:
                continue
            key, value = line.split(": ", 1)
            fields[key] = value
        pid_text = fields.get("PID")
        experiment = fields.get("Experiment")
        if pid_text is None or experiment is None:
            raise ValueError(
                f"Active lock at {resolved_lock_path} is malformed: expected PID and Experiment lines, found {lock_content!r}."
            )
        try:
            pid = int(pid_text)
        except ValueError as exc:
            raise ValueError(
                f"Active lock at {resolved_lock_path} is malformed: PID must be an integer, found {pid_text!r}."
            ) from exc
        return pid, experiment

    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if psutil is not None:
            return psutil.pid_exists(pid)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return True
        return True

    resolved_lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_content = (
        f"PID: {os.getpid()}\n"
        f"Experiment: {experiment_name}\n"
        f"Variants: {variants}\n"
        f"Started: {started_at or datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    )

    while True:
        try:
            lock_handle = resolved_lock_path.open("x+", encoding="utf-8")
            break
        except FileExistsError:
            existing_content = resolved_lock_path.read_text(encoding="utf-8")
            existing_pid, existing_experiment = _parse_lock_metadata(existing_content)
            if _pid_is_alive(existing_pid):
                raise RuntimeError(
                    f"Cannot acquire the {lock_description} because another experiment is still running. "
                    f"Lock path: {resolved_lock_path}. Experiment: {existing_experiment}. PID: {existing_pid}. "
                    "If this invocation is only a safe test, rerun with --no-lock so you do not interfere with the live run."
                )
            try:
                resolved_lock_path.unlink()
            except FileNotFoundError:
                continue
            except PermissionError:
                continue

    lock_handle.write(lock_content)
    lock_handle.flush()

    def _remove_lock() -> None:
        try:
            current_content = resolved_lock_path.read_text(encoding="utf-8")
        except OSError:
            current_content = None
        try:
            lock_handle.close()
        except OSError:
            pass
        if current_content is None:
            return
        try:
            current_pid, _ = _parse_lock_metadata(current_content)
        except ValueError:
            return
        if current_pid != os.getpid():
            return
        try:
            resolved_lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    atexit.register(_remove_lock)


def prepare_output_paths(*, report_path: Path, log_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
