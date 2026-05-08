# Plan

Working file. Rewritten as state changes — not a changelog.

---

## Current state

Project is being set up from scratch. No working code yet.

---

## Immediate priorities

1. **Environment setup** — Python + JAX + CUDA working. Pin versions, document in setup notes. Use UV, local venv.
2. **Sanity-check experiment** — simplest possible: 5-char context, feedforward network, next-token prediction on character-level text. Prove the pipeline works end to end.
3. **Ordinary transformer baseline** — standard attention + FFN, same task. This is the reference point everything else is compared against.
4. **Ordinary RNN baseline** — a basic recurrent model on the same task. Simplest stateful baseline.
5. **Experiment harness** — logging, checkpointing, config locking, reproducibility infrastructure. Do this once, do it right.
6. **Arbitrary-order image patch pipeline** — data loader, positional encoding, verify manually before any model touches it.

## Next up

- Mix-Add implementation and comparison against plain residual (see `research/questions/mix-add/`)
- Dynamic depth experiment on standard transformer (Thread 2 — independent of cortical columns)
- First cortical column prototype — start with synchronous, fixed graph, no async. This is tentative; design depends on resolving open questions first.

## Open questions driving current work

See `research/questions/` for all open question folders. Currently none are actively being investigated (no code yet).

## Deferred

Longer-horizon ideas live in `VISION.md`. Nothing deferred yet that affects the next few handovers.
