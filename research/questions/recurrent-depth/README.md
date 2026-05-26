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

## Larger-data control: advantage is regularization

### Results

| Condition | Params | Val Loss (100K train) | Val Loss (900K train) |
|---|---|---|---|
| distinct_4 | 817K | 1.723 | **1.615** |
| recurrent_4 | 224K | **1.632** | 1.684 |

- On 100K chars: recurrent wins by Δ=-0.092. Weight sharing prevents overfitting.
- On 900K chars: distinct wins by Δ=+0.069. Extra capacity pays off when data is sufficient.

### Per-iteration eval (recurrent_4, large data)

| Iteration | Val Loss |
|---|---|
| 3 | 1.707 |
| 4 | 1.684 |

Iteration 4 still helps, but the larger-data recurrent model stops at 1.684 instead of reaching the 1.615 achieved by the distinct model.

### Activation RMS (large data)

0.42 → 0.72 → 1.07 → 1.44

The recurrent dynamics still look healthy. This run does not show a stability failure; it shows a capacity ceiling.

### Key findings

1. **The recurrent advantage is entirely regularization from weight sharing.**
2. At 100K chars with 817K params, the distinct model memorizes (train loss ~1.0, val 1.72).
3. At 900K chars, the distinct model stops overfitting (train ~1.1-1.4, val 1.62) and wins.
4. The recurrent model's parameter-efficient design becomes a disadvantage when data is plentiful — it lacks the capacity to exploit the additional training signal.
5. Weight sharing is not a free lunch for quality — it is a strong regularizer that helps when data is scarce.

### Implication for Pathway 1

This does not kill Pathway 1, but it changes the question. The original question was "can shared weights match distinct layers?" The answer is: not at matched depth with fewer params.

The real question is now:

**At matched parameter count (wider recurrent model vs narrow distinct), which architecture is better?**

If a recurrent model with ~817K params (wider `d_model`) still matches or beats a 4-layer distinct model with ~817K params, then iteration provides useful inductive bias beyond regularization.

## Interpretation

The original 100K-char result was real, but its cause was misidentified. Recurrent-4 beats distinct-4 on small data because weight sharing regularizes the model. Once training data increases to 900K characters, the distinct model's extra capacity stops being a liability and becomes an advantage.

So the answer to the original question is narrower than it first appeared: shared-weight recurrence can beat distinct depth at matched compute when the distinct model is over-parameterized for the dataset, but not when the same narrow recurrent model is asked to compete with a larger-capacity distinct model on sufficient data.

**H3 (instability) is still rejected:** both the 8-iteration run and the larger-data recurrent control remain stable. Stability is not the blocker here.

The next question is now a fairer Pathway 1 test: if recurrence gets the same parameter budget as distinct depth, does the iterative inductive bias still help?

## What this settles

- At 100K chars, shared-weight recurrence beats distinct depth because weight sharing regularizes an otherwise overfit comparison
- At 900K chars, the 817K-parameter distinct model beats the 224K-parameter recurrent model by 0.069 nats
- 8 iterations of the same block are stable with no special techniques needed
- Weight sharing provides useful regularization, but it is not free quality once data is plentiful

## What this does NOT settle

- Whether a wider recurrent model at the same parameter count as distinct still wins or matches
- How many iterations before instability appears (8 is still stable, and RMS growth remains linear)
- Whether local learning (Pathway 3) composes well with recurrence
- Whether this result generalizes beyond TinyShakespeare

## Width-scaling: matched-param comparison (PROVISIONAL)

### Setup

Widen the recurrent model to d=256 so it has ~842K params (vs distinct_4_d128's ~817K). Same data (900K chars), same training (20K steps). This tests whether recurrence helps at matched parameter count.

### Results

| Model | d_model | Params | Val Loss | Time |
|---|---|---|---|---|
| recurrent_4 (d=256) | 256 | 842K | **1.594** | 505s |
| distinct_4 (d=128, from largedata) | 128 | 817K | 1.615 | 402s |
| distinct_4 (d=256, same run) | 256 | 3,208K | 1.567 | 511s |

**Matched-param comparison: recurrent wins by Δ=-0.021** (842K vs 817K params, 1.594 vs 1.615).

### Per-iteration diagnostics (recurrent_4_d256)

| Iteration | Val Loss | Activation RMS |
|---|---|---|
| 1 | 2.362 | 0.770 |
| 2 | 1.760 | 1.634 |
| 3 | 1.616 | 2.349 |
| 4 | 1.594 | 3.074 |

### Important caveats

1. **Single seed.** The effect (0.021) is small enough that seed variance could explain it. Multi-seed verification needed.
2. **Compute mismatch.** At d=256, the recurrent model does ~4× more FLOPs per token than distinct at d=128 (due to d² scaling). Wall-clock difference is only 25% (GPU not saturated), but the raw compute is higher.
3. **Cross-run comparison.** The distinct_4_d128 baseline is from a different experiment run, introducing potential non-determinism.

### Two-metric interpretation

| Comparison | Winner | Δ | What it means |
|---|---|---|---|
| **Per-parameter** (842K rec vs 817K dist) | Recurrent | -0.021 | Sharing makes better use of params |
| **Per-FLOP** (same d=256, same compute) | Distinct | +0.027 | More unique weights exploit same compute better |

**Practical implication:** If limited by memory/storage (e.g., phone deployment), recurrence wins. If limited by compute, distinct wins.

### Status: PROVISIONAL

This result is encouraging but unconfirmed. The matched-param advantage needs multi-seed verification before Pathway 1 can be declared confirmed at this scale.

## Next steps

1. **Multi-seed verification (IMMEDIATE):** Run 3 seeds of both recurrent_4_d256 and distinct_4_d128 in the same experiment to get confidence intervals.
2. **Iteration scaling:** Try 8/16 iterations at d=256. Activation RMS growth is steeper at d=256 (~2× first step, then ~1.4× per step) — at what iteration count does it become problematic?
3. **Composition with Pathway 3:** Add local CE loss per iteration (the proven lateral mechanism applied to temporal iterations).

## Artifacts

- Report JSON: `experiments/tinyshakespeare/artifacts/recurrent_depth_lm/report.json`
- Report JSON (8 iterations): `experiments/tinyshakespeare/artifacts/recurrent_depth_lm/report_8iter.json`
- Report JSON (900K control): `experiments/tinyshakespeare/artifacts/recurrent_depth_lm/report_largedata.json`
- Report JSON (width-scaling): `experiments/tinyshakespeare/artifacts/recurrent_depth_lm/report_wide.json`
- Log: `runs/recurrent_depth_run.log`
- Log (8 iterations): `runs/recurrent_depth_8iter_run.log`
- Log (900K control): `runs/recurrent_depth_largedata_run.log`
- Log (width-scaling): `runs/recurrent_depth_wide_run.log`
- Script: `runs/recurrent_depth_lm.py`
