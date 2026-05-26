# Can shared-weight recurrence match distinct-layer depth?

**Pathway:** 1 (Wide Recurrent vs Deep Transformer) — THE fundamental thesis
**Dictation context:** The entire project premise; see VISION.md
**Connection:** If recurrence works, everything else follows — dynamic depth, parallelism, local learning composition

## The question

A standard N-layer transformer has N sets of unique weights, each applied once. A recurrent transformer has 1 set of weights applied N times. At matched compute (same number of block applications per token), can recurrence match distinct-layer quality?

This is the simplest possible test of the project's core thesis.

## Why now

Pathway 3 (Local Learning) proved that locally-trained blocks work. Pathway 8 (Multi-Rate) is fully closed. The project needs to test its fundamental premise directly — does recurrence substitute for depth?

## Simplifications

- Char-level LM on TinyShakespeare (fast feedback, known baselines)
- d_model=128, ctx=128, matching our established baseline scale
- N=4 (enough iterations to be non-trivial, not so many that stability dominates)
- No laterals, no local learning, no multi-rate — pure recurrence test (isolation before composition)
- Separate LayerNorms per iteration (so normalization isn't the confound)

## Hypotheses

**H1 (optimistic):** Recurrent-4 matches distinct-4 within ~0.02-0.03 nats despite having 1/4 the parameters. Weight sharing is free or nearly free.

**H2 (pessimistic):** Recurrent-4 is significantly worse (>0.05 nats). Same weights can't learn diverse layer-specific features.

**H3 (instability):** Recurrent-4 training is unstable (gradient explosion through 4 iterations). Would need optimizer changes (Muon, orthogonal init).

## What this does NOT settle

- Whether recurrence works at larger scale (d=512+, many more iterations)
- Whether width can compensate for the gap (if any)
- Whether Muon/orthogonal parameterization improves stability at higher iteration counts
- How local learning (Pathway 3) composes with recurrence
- Whether the gap shrinks or grows with more training

## Planned evidence

| Measurement | Purpose |
|---|---|
| val_loss for both conditions | Primary comparison |
| Per-iteration eval loss (recurrent) | Does each iteration help? Diminishing returns? |
| Parameter count | Quantify the efficiency gain |
| Wall-clock time | Practical speed comparison |
| Gradient norms | Stability evidence |
| Activation RMS by iteration | Distribution health through iterations |

## Decision rule

| Gap (distinct - recurrent) | Interpretation | Next step |
|---|---|---|
| ≤ 0.03 | Strong evidence for Pathway 1 | Width scaling: can wider recurrent beat distinct at same params? |
| 0.03 – 0.05 | Moderate gap, worth investigating | Width scaling to see if more params closes the gap |
| > 0.05 | Significant tax on recurrence | Optimizer/stability investigation, or confidence in Pathway 1 drops |
| Unstable | Stability is the bottleneck | Muon, orthogonal init, gradient clipping experiments |

## Script

`runs/recurrent_depth_lm.py` — two conditions (distinct_4, recurrent_4), same training budget, same readout.

## Results

### Combined comparison (4 and 8 iterations)

| Config | Params | Val Loss | Δ from matched distinct | Time |
|---|---|---|---|---|
| distinct_4 | 817K | 1.723 | — | 402s |
| recurrent_4 | 224K | 1.632 | -0.092 | 319s |
| distinct_8 | 1,610K | 1.787 | — | 838s |
| recurrent_8 | 226K | 1.615 | -0.172 | 554s |

Recurrent wins at both depths, and the gap grows from 0.092 nats at 4 iterations to 0.172 nats at 8 iterations.

### Per-iteration eval (recurrent_4)

| Iteration | Val Loss |
|---|---|
| 1 | 1.9032 |
| 2 | 1.7247 |
| 3 | 1.6632 |
| 4 | 1.6315 |

Each iteration improves monotonically. By iteration 2, the recurrent model already matches the distinct model's final result (1.7247 vs 1.7234). No diminishing returns at 4 iterations — more iterations might help further.

### Per-iteration eval (recurrent_8)

| Iteration | Val Loss | Δ from previous |
|---|---|---|
| 1 | 4.055 | — |
| 2 | 2.562 | -1.493 |
| 3 | 1.992 | -0.570 |
| 4 | 1.776 | -0.216 |
| 5 | 1.685 | -0.091 |
| 6 | 1.642 | -0.043 |
| 7 | 1.622 | -0.020 |
| 8 | 1.615 | -0.007 |

The first four iterations do most of the work. Iterations 5-8 still help, but only by 0.07 nats total, so returns are clearly diminishing by this point.

### Stability: no issues

Gradient norms for both conditions stay in range 2.5–4.2 throughout training. No spikes, no explosion. 4 iterations of shared weights is completely stable with standard AdamW + residual connections + LayerNorm.

### Activation RMS (recurrent, sampled eval)

| Iteration | RMS |
|---|---|
| 1 | 0.4677 |
| 2 | 0.6976 |
| 3 | 0.9531 |
| 4 | 1.2135 |

Growth factor: ~1.5× per iteration. At 8 iterations this would be ~3.6, at 16 iterations ~11. This is linear-ish growth, not exponential — stable for now but may become a concern at higher iteration counts.

### Activation RMS growth (recurrent_8, sampled eval)

| Iteration | RMS |
|---|---|
| 1 | 0.491 |
| 2 | 0.741 |
| 3 | 1.008 |
| 4 | 1.278 |
| 5 | 1.545 |
| 6 | 1.819 |
| 7 | 2.099 |
| 8 | 2.385 |

Growth is linear at roughly 0.27 RMS per iteration, not exponential. A naive extrapolation to 16 iterations would put RMS around 4.3, which is elevated but still far from catastrophic blowup.

### Overfitting analysis

| Condition | Train Loss | Val Loss | Gap |
|---|---|---|---|
| distinct_4 | ~1.1 | 1.72 | ~0.62 |
| recurrent_4 | ~1.3 | 1.63 | ~0.23 |

The distinct model is heavily overfitting (train-val gap of 0.62 nats). Weight sharing acts as strong regularization, preventing the recurrent model from overfitting as badly (gap of 0.23).

**Important caveat:** This means the recurrent advantage is partly (maybe largely) a regularization effect at this data scale. With more training data (where overfitting is less of a concern), the distinct model's extra capacity might close or reverse the gap. This needs testing at larger data scale.

### Key findings from the 8-iteration run

1. **Recurrent advantage grows with iteration depth:** Δ=-0.092 at N=4, Δ=-0.172 at N=8.
2. **Distinct gets worse with more layers:** 1.723 → 1.787, consistent with stronger overfitting as parameters increase.
3. **Recurrent gets better with more iterations:** 1.632 → 1.615, so extra shared-weight computation still helps.
4. **Training stays stable at 8 iterations:** gradient norms remained in the 3.0-4.2 range with no instability signal.
5. **Returns diminish after about 5 iterations:** the big gains are in iterations 1-4; iterations 5-8 add only 0.07 nats total.
6. **Activation RMS growth is linear and predictable:** per-iteration LayerNorms prevent exponential blowup.

## Interpretation

**H1 (optimistic) is confirmed** at this scale: recurrence not only matches but beats distinct layers, and the margin grows from 0.092 nats at 4 iterations to 0.172 nats at 8 iterations.

The growing gap strongly supports the overfitting explanation. The distinct 8-layer model has 1.6M parameters on only 100K training characters and gets worse as depth increases, while the recurrent 8-iteration model stays at 226K parameters and keeps improving with more computation.

**H3 (instability) is rejected:** 8 iterations are still completely stable with standard training. Activation RMS growth is linear, not exponential.

This raises the critical next question: **does the recurrent advantage persist once there is enough data that neither model can overfit heavily?** If the distinct model stops memorizing, its extra capacity might close or reverse the gap.

## What this settles

- At this data scale (100K chars, d=128, ctx=128), shared-weight recurrence is BETTER than distinct layers at matched compute
- 8 iterations of the same block are stable with no special techniques needed
- Each iteration provides clear monotonic improvement, with diminishing returns appearing after roughly iteration 5
- Weight sharing provides implicit regularization that is beneficial at this scale

## What this does NOT settle

- Whether the advantage persists at larger data scale (where overfitting matters less)
- Whether wider recurrent models (same param count as distinct) are even better
- How many iterations before instability appears (8 is still stable, and RMS growth remains linear)
- Whether local learning (Pathway 3) composes well with recurrence
- Whether this result generalizes beyond TinyShakespeare

## Next steps (per decision rule)

Gap is ≤ 0.03 (actually recurrent is BETTER): **strong evidence for Pathway 1.**

Immediate follow-ups:
1. **Larger data:** Run the same comparison on 900K train chars / 100K val chars to test whether the current advantage is mostly regularization.
2. **Width scaling:** Make recurrent model as wide as distinct (same param count). Does it dominate even further?
3. **Iteration scaling:** Try 16 iterations. Where does instability appear beyond the now-verified stable 8-iteration point?
4. **Composition with Pathway 3:** Add local CE loss per iteration (= the proven lateral mechanism applied to temporal iterations)

## Artifacts

- Report JSON: `experiments/tinyshakespeare/artifacts/recurrent_depth_lm/report.json`
- Log: `runs/recurrent_depth_run.log`
- Script: `runs/recurrent_depth_lm.py`
