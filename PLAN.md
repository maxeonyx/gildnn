# Plan

Working file. Rewritten as state changes — not a changelog.

---

## Current state

Rust codebase archived to `rust-archive/`. Python work has now started in `experiments/`, with one minimal sanity-check script and evidence folder.

Process scaffolding exists: root docs, `loop.ps1`, and question-folder structure under `research/questions/`.

One minimal local backend path is verified on this machine: UV + local `.venv` + CPython 3.12.12 + `torch==2.11.0+cu128`, with a tiny tensor op succeeding on CUDA. See `research/questions/backend-validation/`.

The first bounded experiment rung is also now proven on that path: a tiny PyTorch character-level next-token pipeline can overfit one batch, run end to end on tiny data, and produce inspectable saved outputs. See `research/questions/pytorch-char-sanity-check/`.

An ordinary transformer baseline on the same tiny task is now also proven on that same path. Together with the feedforward and ordinary RNN references, the repo now has a narrow three-way comparison on the same tiny fixed-window task. Across the saved references, the raw text, core task-surface fields, sample prompts, and recorded environment fields match, but learning rates, git SHAs, and observed parameter counts do not all match. This is not a controlled or matched-capacity comparison. See `research/questions/pytorch-char-reference-comparison/` when deciding what these three references jointly show.

These are still bounded proofs, not project-wide decisions. The backend/framework choice remains open. The experiment harness is still only whatever these tiny bounded paths genuinely needed.

---

## Immediate priorities

1. **Minimal `core/` integration slice for the tiny fixed-window PyTorch references** — the feedforward, transformer, and RNN references are now proven and compared narrowly enough that the next step is to move their shared bounded task surface into `core/`. Keep this narrow: integrate the shared tiny fixed-window path, keep experiment-specific entrypoints thin, and do not broaden into a general harness rewrite, backend decision, or new experiment.
2. **Experiment harness only when a later bounded question genuinely needs it** — logging, checkpointing, config locking, and stronger reproducibility work should follow only if the next real question needs stronger controlled-comparison claims than the current bounded stack supports.
3. **Backend/runtime follow-up only if it becomes the sharper uncertainty reducer** — the captured NumPy warning is still a documented loose edge, but not yet automatically the next task.

## After the first runnable stack exists

- Mix-Add implementation and comparison against plain residual (see `research/questions/mix-add/`). This is interesting, but it is not ahead of proving the basic stack and baselines.
- Dynamic depth experiment on a standard model. This remains separate from the cortical-column work for now and is still provisional.
- A first cortical-column prototype is still tentative. Do not treat the graph/update design as settled before the open questions narrow.

## Open questions driving current work

See `research/questions/` for the open question folders and current write-ups. Their existence does not mean those threads are experimentally active yet; right now they are scoping and hypothesis documents, not result folders.

## Deferred

Longer-horizon ideas live in `VISION.md`. Nothing deferred yet that affects the next few handovers.
