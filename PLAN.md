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

## Open questions driving experiments

See `research/questions/` for each question's dedicated folder.

Active questions:
- What is a column's residual stream?
- What is the global communication channel? (broadcast router vs all-to-all)
- What exactly is unhooked from gradients?
- What triggers a column update? (surprisal semantics)
- What is the I/O contract for the architecture?
- What does bidirectional propagation look like?
- Does the graph structure matter, or does the global channel dominate?
- How does Mix-Add behave under real conditions?
- Does dynamic depth work on a standard model before combining with columns?

## Deferred / stretch

- Bidirectional propagation (open question, not yet designed)
- True async execution (may not be feasible on GPU — explore semi-async approximation first)
- Structured prompt-space text dataset
- Image-text unified model
- Combining Thread 1 (cortical columns) and Thread 2 (dynamic depth)
- Noise gate / modularity loss (side quest)

## Standing priorities (always in effect)

- Integrate finished experimental code before starting new experiments
- Keep `core/` small and clean
- If `research/` output can't be backed by a real artifact, run the experiment first
- When something breaks and gets fixed, turn it into a question folder with a regression test
