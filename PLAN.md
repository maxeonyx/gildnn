# Plan

Working file. Rewrite it as the state changes.

---

## Current state

- Working Python path: UV + `.venv` + CPython 3.12.12 + PyTorch CUDA on RTX 3090.
- **Thread 1 (Predictive Chain) — well-characterized.** 9 experiments complete. Key conclusion: the architecture works (local predictive learning, detached gradients, multi-hop chain all viable) but its only measurable advantage is a regularization effect that disappears at scale. At 100K chars / 200K params, RNN and transformer match the chain. The chain is also 10-80× slower.
- Remaining Thread 1 value is in properties that don't show up as val loss: async execution, interpretability, modularity, graceful degradation.
- All results documented in `research/questions/predictive-chain/README.md`.

## Active work: Thread 2 — Dynamic Computation Depth

From VISION.md: "use the same model weights multiple times per output token, iterating on internal state, with a learned halting criterion."

The idea: easy tokens need one forward pass, hard tokens need more. A loss-prediction head estimates when additional computation isn't helping. This is well-trodden (ACT, Universal Transformers, PonderNet) — the goal is understanding, not novelty.

### Next steps

1. **Research existing approaches** — ACT (Graves 2016), Universal Transformer (Dehghani 2018), PonderNet (Banino 2021). Understand what works and what doesn't.
2. **Implement on standard RNN** — start with the Shakespeare task, a simple GRU that iterates K times per token, with a halting mechanism.
3. **Train the loss-prediction head** — Max's specific idea: run multiple rollouts per token at different depths, record loss at each depth, train the head to predict those losses.
4. **Compare** — does adaptive depth beat fixed depth? On what tokens does it choose to think longer?

### Why this next

- Explicitly flagged in VISION.md as simpler and independent
- Well-trodden territory — clear baselines to compare against
- Eventually composes with Thread 1
- Clean experiment that can produce a clear result

## Deferred

- Thread 1 remaining questions (async execution, internal specialization analysis)
- Image patches dataset
- Mix-Add operation
- Graph topology experiments (already shown to hurt at this scale)

## Live constraints

- Integrate before starting broad new branches.
- Keep the codebase small.
- Experiment ladder: overfit one batch, tiny end-to-end, inspect outputs, scale.
- Open questions stay open.
