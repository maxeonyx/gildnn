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

---

## Clean measurement design (pre-registered 2026-05-26)

**Substrate:** Existing trained tied-depth transformer (d_model=72, 4 heads, TinyShakespeare ctx=32, depth=8, 3 seeds). Eval-only — no new training required. GPU time: minutes.

### Methodology

**Standard next-token eval only.** For each corpus position t, build exactly one example with maximum available left context (up to 32 chars). Predict next token ONCE. Record logits at the FINAL position after every tied iteration d=1..8. Compute per-token cross-entropy L_t(d). Do this on val AND a matched train subset of equal size, for all 3 seeds.

**Self-attention coupling addressed:** Depth is defined per PREDICTION EVENT (whole forward pass uses same depth), not per-token within a single pass. This is valid — you can't halt different positions independently, but you CAN choose different depths for different examples.

### Metrics to compute

1. **Mean loss by depth** E[L(d)] — recover normal depth-vs-quality curve
2. **Oracle-best loss** E[min_d L_t(d)] — if this beats depth-8, some tokens are HARMED by extra depth
3. **Oracle depth histogram** — distribution of d*_t = argmin_d L_t(d)
4. **No-regret shallowest depth** d_t^δ = min{d: L_t(d) ≤ L_t(8) + δ} with δ=0.01 (NOT 0.1 which was too loose)
5. **Oracle speedup** 8 / E[d_t^δ]
6. **Predictability** — can shallow features (depth-1 or depth-2 state) predict d*_t or marginal improvement?

### Decision criteria

| Result | Interpretation |
|---|---|
| Oracle speedup > 1.3× AND oracle-best improves over depth-8 | **Worth pursuing** — real headroom exists |
| d*=8 almost always, speedup 1.0-1.1× | **Not worth it** — depth-8 is near-optimal for all tokens |
| Oracle-best substantially better but not predictable from shallow state | Headroom exists but halting is hard to learn (still interesting for Pathway 5) |
| Train speedup >> val speedup | Overfitting artifact — depth isn't helping on new data |

### Implementation notes

- The existing `per_token_depth_analysis.py` script in `runs/` does something similar but with flawed methodology (overlapping windows). Could be adapted.
- Key change: use the standard eval function, not sliding windows. One prediction per position.
- Save raw L_t(d) matrix for all positions — enables later analysis without re-running.
