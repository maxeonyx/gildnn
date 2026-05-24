# Dynamic Depth / Early Exit (Pathway 5)

> **Roadmap connection:** Pathway 5 — "Can a loss predictor decide when recurrent iterations are done?"

---

## Question

Do different tokens benefit differently from additional tied-depth iterations? If so, dynamic depth (early exit) could improve both speed and quality.

## Exploratory probe (2026-05-25)

Trained depth=8 tied-depth transformer (d_model=72, TinyShakespeare ctx=32, seed=42). Measured per-token cross-entropy at all 8 depths across all validation window positions.

### What was measured

- Per-token marginal improvement (Δloss from depth d to d+1) across 638K position instances
- Distribution of "earliest depth within ε of depth-8 loss"
- Fraction of tokens with negative total improvement (depth-8 worse than depth-1)

### Raw numbers

Mean total improvement d1→d8: 0.112 nats (std 1.413)
Earliest-done depth mean: 3.1 (std 2.57) at ε=0.1

Marginal improvements (mean / std):
- d1→d2: 0.062 / 0.584
- d2→d3: 0.009 / 0.466
- d3→d4: -0.020 / 0.340
- d4→d5: 0.005 / 0.273
- d5→d6: 0.015 / 0.167
- d6→d7: 0.023 / 0.127
- d7→d8: 0.019 / 0.133

45.5% of token instances have negative total improvement (depth-8 worse than depth-1).

### Honest interpretation

**Supported:** Substantial token-level heterogeneity exists. Marginal gains are dispersed (std >> mean at every depth transition).

**Not supported:**
- "Most tokens are done early" — ε=0.1 is nearly the size of the mean total improvement (0.112), so the "done" threshold is too loose to be meaningful
- "Dynamic depth would improve quality" — haven't shown a halting mechanism would help, and self-attention means tokens can't be independently halted
- "45.5% are harmed by more depth" — more likely normal probability redistribution (model improves average loss by helping some tokens more than it hurts others)

### Methodological concerns

1. **Non-standard evaluation frame** — all positions × overlapping windows ≠ standard next-token val (mean depth-8 loss here is 3.37 vs standard val 1.65)
2. **Single seed** — percentages could shift substantially across seeds
3. **Self-attention coupling** — can't independently halt tokens in this architecture
4. **Train/val comparison not done** — haven't ruled out overfitting as explanation for late-depth degradation

### Status

**Interesting hint, not evidence.** Pathway 5 remains plausible but unvalidated. Worth revisiting with cleaner methodology when dynamic depth becomes the active question.

### What would constitute real evidence

1. Clean evaluation (one prediction per raw validation position with maximal context)
2. Oracle best-depth measurement (not "within ε")
3. Train vs val comparison to rule out overfitting
4. Multi-seed replication
5. Test whether shallow state can predict sign/magnitude of future marginal improvement
