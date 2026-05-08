# Plan

Working file. Rewritten as state changes — not a changelog.

---

## Current state

Rust codebase archived to `rust-archive/`. Python work has now started in `experiments/`, with one minimal sanity-check script and evidence folder.

Process scaffolding exists: root docs, `loop.ps1`, and question-folder structure under `research/questions/`.

One minimal local backend path is verified on this machine: UV + local `.venv` + CPython 3.12.12 + `torch==2.11.0+cu128`, with a tiny tensor op succeeding on CUDA. See `research/questions/backend-validation/`.

The first bounded experiment rung is also now proven on that path: a tiny PyTorch character-level next-token pipeline can overfit one batch, run end to end on tiny data, and produce inspectable saved outputs. See `research/questions/pytorch-char-sanity-check/`.

An ordinary transformer baseline on the same tiny task is now also proven on that same path. Together with the feedforward and ordinary RNN references, the repo now has a narrow three-way comparison on the same tiny fixed-window task. Across the saved references, the raw text, core task-surface fields, sample prompts, and recorded environment fields match; the tracked post-integration environment artifacts now also point consistently at commit `3fb4477...`. Learning rates and observed parameter counts still do not all match, so this is not a controlled or matched-capacity comparison. See `research/questions/pytorch-char-reference-comparison/` when deciding what these three references jointly show.

These are still bounded proofs, not project-wide decisions. The backend/framework choice remains open. The experiment harness is still only whatever these tiny bounded paths genuinely needed.

---

## Immediate priorities

1. **Dynamic depth on a standard model** — the next bounded unit is now the first dynamic-depth question on a standard model, preferably the existing tiny transformer host surface. Keep it to one tiny task, one interpretable comparator, and one artifact-backed feasibility/interpretability question. This is not yet a broad claim that dynamic depth works in general, and it does not reopen Mix-Add. See `research/questions/dynamic-depth/` when scoping the unit.
2. **Experiment harness only when a later bounded question genuinely needs it** — logging, checkpointing, config locking, and stronger reproducibility work should follow only if the dynamic-depth unit or a later bounded question needs stronger controlled-comparison claims than the current bounded stack supports.
3. **Backend/runtime follow-up only if it becomes the sharper uncertainty reducer** — the captured NumPy warning is still a documented loose edge, but not yet automatically the next task.

## After the first runnable stack exists

- A narrower Mix-Add follow-up only if a later question specifically needs it — for example, saving final per-window prediction tables for the same plain-residual vs scalar-Mix-Add pair. Do not reopen Mix-Add by drift.
- After the first bounded dynamic-depth unit, decide deliberately whether the next step is a tighter dynamic-depth follow-up, a different bounded question, or infrastructure/runtime cleanup. Do not assume a dynamic-depth roadmap by momentum.
- A first cortical-column prototype is still tentative. Do not treat the graph/update design as settled before the open questions narrow.

## Open questions driving current work

See `research/questions/` for the open question folders and current write-ups. Their existence does not mean those threads are experimentally active yet; right now they are scoping and hypothesis documents, not result folders.

## Deferred

Longer-horizon ideas live in `VISION.md`. Nothing deferred yet that affects the next few handovers.
