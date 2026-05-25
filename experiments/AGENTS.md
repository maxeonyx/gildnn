# experiments/

Per-experiment directories with scripts, configs, and all grungy local artifacts.

## Structure

Each experiment gets its own subdirectory: `experiments/<experiment-name>/`. The directory contains:

- Python scripts (the experiment entry point and any helpers)
- Configs/hyperparameters
- Raw outputs: prediction dumps, loss logs, debug traces, text samples
- Any artifact that is too messy or verbose for the Max-readable report

Run experiments from the repo root: `.\.venv\Scripts\python.exe -m runs.<name>` (e.g. `runs.closed_loop_prediction`, `runs.transformer_baseline`). The `experiments/` directory holds artifacts, not entry-point scripts.

**Why `-m`:** When Python runs a script directly (`python runs/foo.py`), it puts `runs/` on sys.path — not the repo root. This means `from core import ...` fails with ImportError. Using `-m` puts the current working directory on sys.path instead, so `core` is importable. The scripts also have a `sys.path.insert` fallback, so direct execution works too — but `-m` is the canonical method.

## What does NOT go here

- Polished reports (those go in `research/questions/<name>/README.md`)
- Shared reusable code (that goes in `core/`)
- Standard baseline training (that goes in `base-experiments/`)

## Conventions

- Record git SHA, hyperparameters, and seed for meaningful runs
- Save enough config that a result can be recreated without guesswork
- Delete dead experiments that taught nothing and are only clutter
- If two experiments share logic, move that logic into `core/`

## Experiment visibility

Before running any experiment expected to take more than ~30 seconds, stop and report back to the orchestrating agent with: what you're about to run, expected duration, and what it will produce. Wait to be resumed before proceeding.

This is per run, not per task. If you fix something and want to retry a 10-20 minute run, stop and report back again first. Do not chain long runs without returning first. The orchestrator logs a fresh visibility note with the current time before each such run.

## Background execution (>5 minutes)

Experiments expected to take more than ~5 minutes use a **two-phase delegation**:

1. **Phase 1 (launch only):** Start the process in the background (`Start-Process` on Windows). Return the PID and log path. Stop immediately — do NOT wait, poll, or check results.
2. **Phase 2 (check):** The orchestrator resumes you later to analyse results. The orchestrator decides when; you don't.

Set `PYTHONUNBUFFERED=1` before launching. The experiment scripts manage `runs/active.lock` automatically (create on startup, remove on exit via `atexit`). Failing to split the two-phase delegation is a process failure.

**⚠️ NEVER run experiment scripts against real artifact directories while a long experiment is active.** See root `AGENTS.md` for the full rule. Short version: always use `--sanity-check-only` (routes to temp dir) for testing code changes. The lock file now refuses acquisition when a live PID holds it, but `--no-lock` bypasses this — if you use `--no-lock`, you MUST also redirect `--report-path` and `--log-path` to a separate temp directory. Violating this corrupts checkpoints.
