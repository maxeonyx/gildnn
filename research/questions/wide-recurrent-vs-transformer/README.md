# Wide Recurrent vs Transformer (Pathway 1)

> **Roadmap connection:** Pathway 1 — THE fundamental comparison. "Can a wide shallow network run many times match a deep transformer?"

> **Dictation grounding:** [2026-05-24-8](../../../dictations/2026-05-24-8.md) — Max restructured the project around pathways; this is pathway 1. [2026-05-22-12](../../../dictations/2026-05-22-12.md) — backend choice settled (PyTorch + torch.compile).

---

## Question

At matched compute (FLOPs per forward pass), how much does tying transformer block weights across depth cost in quality?

This is the cheapest honest test of the core thesis: can a single set of weights applied recurrently substitute for many distinct layers?

## What this experiment settles

- The quality gap between weight-shared and weight-distinct depth at this scale
- Whether iterative refinement happens (does later depth improve over earlier depth?)
- Whether standard training hyperparameters work or need adaptation for weight sharing

## What this experiment does NOT settle

- Whether width scaling can close the gap (next experiment if results are promising)
- Whether pure-FFN recurrence without attention works (separate question)
- Whether dynamic depth / early exit helps (Pathway 5)
- Anything about longer context, larger models, or WikiText-103
- Whether async or local learning interact with weight sharing

---

## Design

### Architecture: Tied-Depth Transformer

Identical to the baseline transformer, with one change: the 3 depth applications share MHA + FFN weights.

```
h0 = token_embedding(x) + position_embedding(pos)

For k = 1..3:
    h = LN_k(h)            # separate LN per depth (NOT shared)
    h = h + MHA(h)         # shared MHA weights
    h = LN_ffn_k(h)        # separate LN per depth
    h = h + FFN(h)         # shared FFN weights

logits = LMHead(LN_final(h))
```

**Critical design choice:** LayerNorms are NOT shared — each depth gets its own. This isolates the "weight sharing" question from "shared normalization statistics" which is a different (harsher) constraint.

### Matching

| | Baseline Transformer | Tied-Depth |
|---|---|---|
| d_model | 72 | 72 |
| ff_dim | 256 | 256 |
| heads | 4 | 4 |
| depth applications | 3 (distinct) | 3 (shared) |
| FLOPs per forward | ~same | ~same |
| Parameters | ~186K | ~70K (est.) |
| Context | 32 | 32 |

We match **FLOPs**, not parameters. The parameter reduction is an outcome to report, not a confound to fix.

### Training

- Dataset: TinyShakespeare, 100K train / 20K val
- Optimizer: AdamW
- **Two learning rates:** 0.003 (baseline LR) and 0.001 (lower, because shared weights accumulate gradients from 3 depths)
- Batch size: 256
- Gradient clip: 1.0
- Epochs: 13
- Seed: 42

### Instrumentation

- Best-of-run val_loss (primary metric)
- Per-depth eval loss (after depth 1, 2, 3) — logged at eval time, not trained
- Parameter count
- Runtime
- Fixed-prompt text sample

---

## Hypotheses

1. **Primary:** Tied-depth will be worse than baseline (1.643) but competitive — expected range 1.67–1.71
2. **LR sensitivity:** Lower LR (0.001) may work better for tied-depth because gradients accumulate across depth uses
3. **Iterative refinement:** Per-depth loss should decrease with depth (depth 3 < depth 2 < depth 1) — if it doesn't, the model isn't using the extra iterations productively

## Decision criteria (best val_loss at better of 2 LRs)

| Result | Interpretation | Next step |
|---|---|---|
| ≤ 1.67 | Strong positive: sharing mostly works | Width scaling test |
| 1.67–1.70 | Promising: real tax but plausibly closeable | Width scaling test |
| 1.70–1.711 | Mixed: competitive with RNN but not transformer | Record, consider Muon (Pathway 9) |
| > 1.711 | Negative: can't even beat vanilla RNN | Record as negative, redirect |

## Stop criterion

Run 2 learning rates × 1 seed. Report. Do NOT tune further. If the result is ambiguous at this point, the question at this scale is answered ("not clearly better or worse") and the meaningful test requires a larger scale or different approach.

---

## Results

### Rung 1: Single-seed, 2 LRs (seed 42)

| Model | Params | Best val_loss | Δ vs baseline |
|---|---|---|---|
| Baseline transformer (3 distinct layers) | 186K | 1.643 | — |
| **Tied-depth (LR=0.003)** | **70K** | **1.654** | **+0.011** |
| Tied-depth (LR=0.001) | 70K | 1.667 | +0.024 |
| RNN baseline | 186K | 1.711 | +0.068 |

Per-depth eval losses (LR=0.003, final epoch):

| Depth | Val loss | Δ from previous |
|---|---|---|
| 1 | 2.188 | — |
| 2 | 1.706 | -0.482 |
| 3 | 1.656 | -0.050 |

**Interpretation (honest):** On this single seed, tying transformer weights across 3 depth applications increased best val_loss by only +0.011 relative to the distinct-layer baseline, while using 63% fewer parameters. The per-depth losses decrease monotonically — all 3 iterations are used, with depth 2 doing most of the recovery from depth 1's crude representation.

This is encouraging evidence that weight sharing is not catastrophic at matched compute. But the effect size (+0.011) is small enough relative to run-to-run variance that **single-seed is underpowered for this claim**. Multi-seed confirmation needed.

The text sample (below) shows heavy repetition — typical of small models on short context, not specific to weight sharing (the baseline shows similar patterns).

```
First Citizen:
Before we proceed the people, and they deserves the consul,
They say the people the people, and they deserves the consul, [repeats]
```

**Status:** Keeps Pathway 1 alive. Does not yet validate "width + recurrence substitutes for depth." Next: multi-seed confirmation.

### Rung 2: Multi-seed confirmation (seeds 42, 43, 44 at LR=0.003)

| Seed | Tied-depth val_loss | Gap vs baseline (1.643) |
|---|---|---|
| 42 | 1.654 | +0.011 |
| 43 | 1.642 | -0.001 |
| 44 | 1.651 | +0.007 |
| **Mean** | **1.649** | **+0.006** |
| **Std** | **0.005** | |

**Interpretation:** The weight-sharing penalty is negligible at matched FLOPs. Mean gap +0.006 ± 0.005 nats — confidence interval includes zero. One seed (43) slightly beats the 186K-param baseline with only 70K params.

The original single-seed estimate (+0.011) was pessimistic — seed 42 happened to be the worst of three. The stable result is: **tied-depth ≈ baseline at matched compute, with 63% fewer parameters.**

**What this settles:** At this tiny rung (ctx=32, TinyShakespeare, 186K baseline FLOPs), weight sharing across 3 depth iterations is essentially free in quality.

**What this does NOT settle:** Whether the thesis scales. Whether parameter-matched width (giving the tied model the same budget) actually wins. Whether it works beyond 3 iterations. Whether it works at longer context or larger scale.

**Next:** Parameter-matched width scaling — widen the tied model until it has ~186K params (same as baseline). If the wider tied model BEATS the baseline, that's strong evidence for the core thesis.

---

## Non-goals for this experiment

- Hyperparameter sweeps beyond the 2 LRs
- WikiText-103 or longer context (that's a later scale-up)
- Any auxiliary mechanisms (local learning, prediction heads, async)
