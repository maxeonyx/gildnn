# Plan

Working file. Rewrite it as the state changes.

---

## Current state

- Working Python path: UV + `.venv` + CPython 3.12.12 + PyTorch CUDA on RTX 3090.
- **Thread 1 (Predictive Chain) — concluded.** 9 experiments, clear answer: the architecture works but its only advantage is regularization that disappears at scale. With 100K chars / 200K params, RNN/transformer match it while being 10-80× faster. See `research/questions/predictive-chain/README.md`.
- **Thread 2 (Dynamic Depth) — first experiment done, promising.** Multi-exit weight-shared GRU with loss-prediction head. The model learns meaningful depth allocation (harder characters get more compute). Gets near-fixed-8 quality with 1.35 average depth. But halting criterion needs calibration work.

## Next steps for Thread 2

The dynamic depth mechanism works but the loss-prediction head doesn't generalize to validation. Improvements to try:

1. **Better halting criterion** — instead of absolute threshold from training loss, try:
   - Relative improvement: stop when predicted_loss(d+1) / predicted_loss(d) > 0.99
   - Or: train the predictor on VALIDATION-style data (using a held-out calibration set)
2. **Larger corpus** — the 7K Shakespeare excerpt may not have enough variety. Try the 100K corpus.
3. **Compare against standard ACT** — implement Graves-style ACT and compare halting patterns.
4. **Inspect what "hard" means** — correlate depth with character entropy, word frequency, position-in-word, etc.

## Bigger picture — what's worth pursuing?

Now that Thread 1 is concluded (architecture works, no performance advantage at scale), and Thread 2 is started (dynamic depth works in principle), the project needs a direction decision:

- **Double down on Thread 2** — get dynamic depth working well, then combine with the chain
- **Image patches** — VISION.md's second primary dataset. Completely different task that may exercise different properties
- **Scaling study** — much larger model + data on the chain to see if the qualitative properties (interpretability, modularity) emerge at scale even without loss advantage
- **Integration** — clean up what works into `core/`, make it reusable

## Live constraints

- Keep the codebase small.
- Experiment ladder: overfit one batch, tiny end-to-end, inspect outputs, scale.
- Open questions stay open.
