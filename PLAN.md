# Plan

Working file. Rewritten as state changes — not a changelog.

---

## Current state

Rust codebase archived to `rust-archive/`. Python work has now started in `experiments/`, with one minimal sanity-check script and evidence folder.

Process scaffolding exists: root docs, `loop.ps1`, and question-folder structure under `research/questions/`.

One minimal local backend path is verified on this machine: UV + local `.venv` + CPython 3.12.12 + `torch==2.11.0+cu128`, with a tiny tensor op succeeding on CUDA. See `research/questions/backend-validation/`.

The first bounded experiment rung is also now proven on that path: a tiny PyTorch character-level next-token pipeline can overfit one batch, run end to end on tiny data, and produce inspectable saved outputs. See `research/questions/pytorch-char-sanity-check/`.

An ordinary transformer baseline on the same tiny task is now also proven on that same path, with the task framing held fixed enough that the main changed variable is model family. See `research/questions/pytorch-char-transformer-baseline/`.

These are still bounded proofs, not project-wide decisions. The backend/framework choice remains open. The first ordinary transformer baseline now exists, but no RNN baseline exists yet, and the experiment harness is still only whatever these tiny bounded paths genuinely needed.

---

## Immediate priorities

1. **Ordinary RNN baseline (or something like it)** — simplest stateful baseline on the same tiny character-level next-token task, so the repo has the first three model-family reference points on one bounded problem.
2. **Compare the tiny bounded references honestly** — feedforward sanity check, ordinary transformer baseline, and then the RNN baseline once it exists. Keep claims narrow and artifact-backed.
3. **Experiment harness** — logging, checkpointing, config locking, reproducibility. Tighten this once the first few runnable baselines exist beyond the current tiny bounded path.
4. **Another backend probe only if it becomes the sharper uncertainty reducer** — keep backend choice open, but do not branch into it unless it is more discriminating than the next baseline step.

## After the first runnable stack exists

- Mix-Add implementation and comparison against plain residual (see `research/questions/mix-add/`). This is interesting, but it is not ahead of proving the basic stack and baselines.
- Dynamic depth experiment on a standard model. This remains separate from the cortical-column work for now and is still provisional.
- A first cortical-column prototype is still tentative. Do not treat the graph/update design as settled before the open questions narrow.

## Open questions driving current work

See `research/questions/` for the open question folders and current write-ups. Their existence does not mean those threads are experimentally active yet; right now they are scoping and hypothesis documents, not result folders.

## Deferred

Longer-horizon ideas live in `VISION.md`. Nothing deferred yet that affects the next few handovers.
