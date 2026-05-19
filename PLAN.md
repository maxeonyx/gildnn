# Plan

Working file. Rewrite it as the state changes.

---

## Current state

- Working Python path: UV + `.venv` + CPython 3.12.12 + PyTorch CUDA on RTX 3090.
- Baseline experiments exist: feedforward, transformer, RNN — all on tiny char next-token task, all hitting ~0.979 accuracy.
- Mix-Add and dynamic-depth experiments exist but produced near-null results on the tiny task (tied with baselines).
- **First predictive chain experiment is done.** 3-node recurrent line graph (A→B→C) with local next-input prediction losses. Key finding: task head overfits perfectly, but auxiliary predictive losses remain non-trivial (A: 0.51, B: 0.20, C: 0.04). Predicting a neighbor's next message is harder than the downstream task at this scale.
- Process simplified. Stale bureaucracy artifacts deleted.
- Backend choice still open. Cortical-column architecture still open.

## What the predictive chain result means

The aux losses decreasing A→B→C (0.51 → 0.20 → 0.04) suggests information gets progressively easier to predict deeper in the chain — B's output is more predictable from C's state than A's output is from B's state, which makes sense because A faces raw token variation while deeper nodes see increasingly processed signals. But all aux losses remain above zero even at overfit, meaning the local prediction task is genuinely non-trivial.

## Possible next steps (pick one)

- **Vary aux weight / try detached gradients** — does the chain learn different representations when aux weight is higher (forcing nodes to be more predictable to their neighbors)? Or with stop_grad on messages?
- **Add attention between neighbors** — instead of passing raw hidden state as the message, let B attend over A's recent outputs. Does this help the local prediction tasks?
- **Increase number of nodes** — go from 3 to 6-8 nodes. Does the pattern (decreasing aux loss deeper in chain) continue?
- **Try bidirectional messages** — A↔B↔C instead of A→B→C. Does backward information flow help?
- **Stronger aux weight experiment** — what happens if aux loss is weighted equally with task loss? Does the task head still learn? Do representations become more predictable?

Prefer whichever is cheapest and most informative about the core question: does local predictive learning produce useful distributed computation?

## Live constraints

- Integrate before starting broad new branches.
- Keep the codebase small.
- Experiment ladder: overfit one batch, tiny end-to-end, inspect outputs, scale.
- Open questions stay open.
