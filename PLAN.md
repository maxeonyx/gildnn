# Plan

Working file. Rewritten as state changes — not a changelog.

---

## Current state

Rust codebase archived to `rust-archive/`. Starting Python from scratch. No Python code yet.

`loop.ps1` was failing immediately because this OpenCode build rejects bare `--no-ephemeral` in `opencode run`. The script has been updated to use `--no-ephemeral=true` instead. This still needs real verification from a fresh terminal after the current session exits.

---

## Immediate priorities

1. **Verify the outer loop** — from a fresh terminal after this session exits, run `./loop.ps1` and confirm OpenCode starts/resumes instead of printing help and exiting.
2. **Environment setup** — choose and validate Python backend + CUDA (prefer JAX + XLA; PyTorch acceptable). UV, local venv. Pin versions once confirmed.
3. **Sanity-check experiment** — 5-char context, feedforward network, next-token prediction on character-level text. Prove the pipeline end to end.
4. **Ordinary transformer baseline** — standard attention + FFN, same task. Reference point for all comparisons.
5. **Ordinary RNN baseline** — basic recurrent model, same task. Simplest stateful baseline.
6. **Experiment harness** — logging, checkpointing, config locking, reproducibility. Once, done right.

## Next up

- Mix-Add implementation and comparison against plain residual (see `research/questions/mix-add/`)
- Dynamic depth experiment on standard transformer (Thread 2 — independent of cortical columns)
- First cortical column prototype — start with synchronous, fixed graph, no async. This is tentative; design depends on resolving open questions first.

## Open questions driving current work

See `research/questions/` for all open question folders. Currently none are actively being investigated (no code yet).

## Deferred

Longer-horizon ideas live in `VISION.md`. Nothing deferred yet that affects the next few handovers.
