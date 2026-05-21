# experiments/

Per-experiment directories with scripts, configs, and all grungy local artifacts.

## Structure

Each experiment gets its own subdirectory: `experiments/<experiment-name>/`. The directory contains:

- Python scripts (the experiment entry point and any helpers)
- Configs/hyperparameters
- Raw outputs: prediction dumps, loss logs, debug traces, text samples
- Any artifact that is too messy or verbose for the Max-readable report

Run experiments from the repo root: `python -m experiments.<dir-name>`

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
