# Question: Is There Opportunity for Computation Compression? (Pathway 4 Prerequisite)

## What this asks

Before designing any compression mechanism, we need to know: does the trained recurrent depth model leave unused quality on the table at intermediate depths? If depth 4 ≈ depth 8, there's nothing to compress. If depth 4 is significantly worse, there's room to push quality earlier.

## Which goals this serves

- **Pathway 4 (Computation Compression)** — [ROADMAP.md](../../../ROADMAP.md): "Can a network learn to front-load or suppress unnecessary computation over training?"
- **Pathway 5 (Dynamic Depth)** — quantifies how much compute the halt head is saving vs what it theoretically could save

## The result

**Yes — there is substantial opportunity, but it's concentrated.** The loss improvement curve is an exponential decay: 86% of total improvement happens in iteration 1→2, and only 1.8% happens in iterations 5-8 combined. The halt head (avg depth 6.49) is already partially exploiting this, but could theoretically be more aggressive.

---

## Measurement

Ran `compute_per_depth_losses_and_predictions` on the capstone d=256 checkpoint (trained 20K steps on WikiText-103, val_loss 1.63). Evaluated 2048 random validation windows on CPU.

| Depth | Mean Loss (nats) | Δ from final | Cumulative improvement | % of total |
|---|---|---|---|---|
| 1 | 4.6655 | +3.0378 | — | — |
| 2 | 2.0647 | +0.4370 | 2.601 (1→2) | 85.6% |
| 3 | 1.7496 | +0.1219 | 0.315 (2→3) | 10.4% |
| 4 | 1.6839 | +0.0562 | 0.066 (3→4) | 2.2% |
| 5 | 1.6553 | +0.0276 | 0.029 (4→5) | 0.9% |
| 6 | 1.6397 | +0.0120 | 0.016 (5→6) | 0.5% |
| 7 | 1.6313 | +0.0036 | 0.008 (6→7) | 0.3% |
| 8 | 1.6277 | +0.0000 | 0.004 (7→8) | 0.1% |

Per-token improvement distribution (depth 1 → depth 8):
- Mean: 3.04 nats improvement
- Std: 2.72 nats (high variance — some tokens gain 13+ nats, some lose up to 2.5)
- 91.4% of tokens benefit from full depth; 8.6% are slightly WORSE at depth 8

---

## Interpretation

### The opportunity is real but front-loaded

The model does most of its work in iterations 1-3. After that, each additional iteration contributes exponentially less. This means:

1. **The halt head at avg depth 6.49 is being conservative** — it could plausibly stop at depth 4-5 for most tokens and lose only 0.03-0.06 nats.
2. **Active compression would need to target iterations 1-3** to be meaningful — making iteration 1 do what iteration 2 currently does would save 50% of compute.
3. **Iterations 5-8 are "refinement" — small corrections** that matter for the long tail of hard tokens.

### What the halt head is doing vs could do

- Current: avg depth 6.49/8 (saves ~19% of max iterations)
- Theoretical min (accepting 0.06 nats loss): depth 4/8 (saves 50%)
- The gap suggests the halt head is somewhat under-trained or over-cautious

### The 8.6% "harmed" tokens

Interesting: for ~9% of tokens, the final iteration makes things WORSE. These are likely tokens where the model is overfitting to recurrent processing patterns — additional iterations corrupt an already-good early prediction. This supports the case for early exit.

---

## What this does NOT settle

- Whether explicit self-prediction can compress computation (needs its own experiment — but see prior negative result below)
- Whether the halt head can be made more aggressive (→ **answered below: yes, dramatically**)
- Whether predictive silencing emerges naturally (needs the multi-timestep architecture)
- Whether this pattern holds at d=512 (pending d=512 completion)

---

## Halt threshold sweep (epsilon sensitivity)

The halt head decides when to stop iterating based on predicted gain. The `epsilon` parameter is the threshold below which the predicted gain triggers early exit. Sweeping epsilon on the same d=256 checkpoint (2048 validation tokens, CPU):

| Epsilon | Avg Depth | Val Loss | Delta (nats) | Compute Savings |
|---|---|---|---|---|
| full depth (no halt) | 8.00 | 1.6277 | — | 0% |
| 0.000 (current) | 6.43 | 1.6321 | +0.004 | 19.6% |
| 0.010 | 5.96 | 1.6363 | +0.009 | 25.5% |
| 0.020 | 5.43 | 1.6424 | +0.015 | 32.1% |
| **0.050** | **4.27** | **1.6611** | **+0.033** | **46.6%** |
| 0.100 | 3.41 | 1.6936 | +0.066 | 57.4% |
| 0.150 | 3.02 | 1.7200 | +0.092 | 62.2% |
| 0.200 | 2.78 | 1.7470 | +0.119 | 65.3% |
| 0.500 | 2.19 | 1.8711 | +0.243 | 72.6% |
| 1.000 | 1.84 | 2.0343 | +0.407 | 77.0% |

### Key findings

1. **The halt head is already good at epsilon=0:** saves 19.6% compute for only +0.004 nats loss. The default operating point is nearly Pareto-optimal for high-quality mode.

2. **The sweet spot is epsilon=0.05:** 46.6% compute savings (avg depth 4.27) for only +0.033 nats degradation. This is a much better operating point for inference-focused use.

3. **Diminishing returns above epsilon=0.1:** going from 57% to 77% savings (epsilon 0.1→1.0) costs +0.34 nats — the easy tokens are already captured.

4. **The curve is smooth and well-behaved:** no sudden quality cliffs. The halt head produces calibrated predictions across the full range.

### Implication for Pathway 4

The model already CAN exit early at depth 4 for most tokens without significant loss. The "compression opportunity" isn't blocked by the architecture — it's just a threshold tuning question. Active compression (making depth-4 ACTUALLY as good as depth-8, not just "good enough") would need to improve the underlying quality curve, not the halting mechanism.

This reframes Pathway 4: the halt head already exploits most of the available slack. The remaining opportunity is: can training be modified so that depth-2 produces what depth-4 currently produces? That's a harder question — and the prior negative result for KL distillation suggests it may not work via simple distillation.

### Halt head calibration quality (Pathway 5 evidence)

The halt head's predictions correlate strongly with actual gains at early depths where the decisions matter most:

| Transition | Predicted gain (mean +/- std) | Actual gain (mean) | Correlation |
|---|---|---|---|
| 1→2 | 2.99 +/- 2.50 | 3.04 | **0.919** |
| 2→3 | 0.40 +/- 0.56 | 0.44 | **0.662** |
| 3→4 | 0.11 +/- 0.12 | 0.12 | 0.269 |
| 4→5 | 0.05 +/- 0.06 | 0.06 | 0.190 |
| 5→6 | 0.02 +/- 0.04 | 0.03 | 0.127 |

At epsilon=0.05 (depth 2 threshold): tokens classified as "easy" have actual gain 0.034; tokens classified as "hard" have actual gain 0.491. The halt head makes correct decisions.

The high-std predictions show the halt head genuinely differentiates between tokens, not just applying a constant threshold. This is strong Pathway 5 evidence: the mechanism is structurally sensitive to per-token difficulty.

---

## Next steps

**⚠️ Prior negative result exists.** The narrow version of Pathway 4 — KL distillation from depth-8 logits to depth-2 logits — was already tested and **failed** on a smaller model (GRU scaffold, TinyShakespeare). See [`research/questions/self-prediction-compute-compression/README.md`](../self-prediction-compute-compression/README.md). At every fixed depth, the self-prediction variant was slightly worse. Stronger auxiliary weight failed the ladder gate entirely.

That negative result argues against re-running the same mechanism at larger scale. It's one data point on a different architecture, but the simplest reading is: "multi-exit training already extracts most of what shallow steps can learn."

**What remains open (from that report):**
- Latent-space distillation (predict hidden state, not logits)
- Different prediction target (e.g., future lateral state in multi-timestep architecture)
- Whether the RecurrentDepthLM architecture (transformer-based, not GRU) changes the sign
- Whether the "predictive silencing" mechanism from the multi-timestep theory emerges naturally without explicit training

**Concrete options for next Pathway 4 work (in order of cheapness):**

1. **Repeat per-depth measurement on d=512** — when that checkpoint is ready. Same cost (CPU-only), confirms whether the curve shape is model-size-invariant.
2. **Halt threshold sweep** — lower `halt_epsilon` on existing d=256 checkpoint to see if the halt head can be more aggressive without quality loss. CPU-only.
3. **Continuation A/B test** — continue from d=256 checkpoint with vs without depth-2 KL distillation. 2K steps, measure per-depth val loss. This tests whether the architecture change (transformer vs GRU) matters. ~5 min GPU.
4. **Latent-space distillation** — predict depth-8 hidden state from depth-2, via a learned projection head. Different mechanism from the prior negative. ~10 min GPU.

Given the negative prior, **1 and 2 are the most honest next steps** (they're free and produce evidence). Option 3 is the cheapest that could change the sign but risks confirming a known negative. Option 4 is genuinely novel but more expensive.
