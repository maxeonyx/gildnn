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

- Whether explicit self-prediction can compress computation (needs its own experiment)
- Whether the halt head can be made more aggressive (needs lower halt_weight or curriculum)
- Whether predictive silencing emerges naturally (needs the multi-timestep architecture)
- Whether this pattern holds at d=512 (pending d=512 completion)

## Next steps

1. **Repeat on d=512** — when that checkpoint is ready, run same analysis. Does larger model show same or different depth curve?
2. **Design compression experiment** — given the opportunity exists, test: explicit self-prediction at depth 2 predicting depth 8's output. Does it improve depth-2 quality?
3. **Halt threshold sweep** — lower `halt_epsilon` to see if the halt head can be made more aggressive without quality loss
