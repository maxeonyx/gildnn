# Plan

Working file. Rewrite it as the state changes.

---

## Current state

- The repo has a working minimal Python path on this machine: UV + local `.venv` + CPython 3.12.12 + PyTorch CUDA. See `research/questions/backend-validation/`.
- The tiny character-level sanity stack exists and has working reference experiments, including overfit-style proofs and baseline comparisons. See `research/questions/pytorch-char-sanity-check/` and related question folders.
- The process became too bureaucratic and started blocking actual research. That is now being simplified.
- Backend choice is still open.
- The cortical-column architecture is still an open research thread, not a settled design.

## Immediate priorities

1. Clean up stale planning language and remove records of failed selection bureaucracy from the repo.
2. Integrate or delete any experimental code that is already understood enough to stop living as drift.
3. Then resume research by picking one small concrete next experiment from the live question folders.

## How to pick the next experiment

Prefer the next step that is:

- small
- runnable now
- likely to produce inspectable outputs
- useful for narrowing an open question
- unlikely to grow the codebase much

Redoing from scratch is allowed if that is cheaper than untangling the current version.

## Live constraints

- Integrate before starting broad new branches of experimentation.
- Keep the codebase small.
- Use the experiment ladder: overfit one batch, tiny end-to-end run, inspect outputs, then scale.
- Reports should stay high quality and evidence-backed.
- Open questions should stay open until experiments actually narrow them.

## Next handover

If nothing is actively in flight, read the live question folders and pick the cheapest experiment that could produce new evidence without requiring framework work first.
