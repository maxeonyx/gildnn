# Plan

Working file. Rewritten as state changes — not a changelog.

---

## Current state

Rust codebase archived to `rust-archive/`. Starting Python from scratch. No Python source code yet.

Process scaffolding exists: root docs, `loop.ps1`, and question-folder structure under `research/questions/`.

One minimal local backend path is now verified on this machine: UV + local `.venv` + CPython 3.12.12 + `torch==2.11.0+cu128`, with a tiny tensor op succeeding on CUDA. See `research/questions/backend-validation/`.

That is an operational probe, not a project-wide backend decision. The backend/framework choice remains open. Experimental readiness is still only partial: no model code, no sanity-check run, no baselines.

---

## Immediate priorities

1. **Use the currently proven path for the first sanity-check experiment** — start with the proven PyTorch path, while keeping backend choice open and only splitting out another backend probe if the work stops being a small discriminating step.
2. **Sanity-check experiment** — 5-char context, feedforward network, or something like that, next-token prediction on character-level text. Prove the pipeline end to end.
3. **Ordinary transformer baseline** — standard attention + FFN, same task. Reference point for all comparisons.
4. **Ordinary RNN baseline (or something like it)** — simplest stateful baseline on the same task.
5. **Experiment harness** — logging, checkpointing, config locking, reproducibility. Tighten this once the first runnable stack exists.

## After the first runnable stack exists

- Mix-Add implementation and comparison against plain residual (see `research/questions/mix-add/`). This is interesting, but it is not ahead of proving the basic stack and baselines.
- Dynamic depth experiment on a standard model. This remains separate from the cortical-column work for now and is still provisional.
- A first cortical-column prototype is still tentative. Do not treat the graph/update design as settled before the open questions narrow.

## Open questions driving current work

See `research/questions/` for the open question folders and current write-ups. Their existence does not mean those threads are experimentally active yet; right now they are scoping and hypothesis documents, not result folders.

## Deferred

Longer-horizon ideas live in `VISION.md`. Nothing deferred yet that affects the next few handovers.
