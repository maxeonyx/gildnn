# Hierarchical prediction via bottleneck latents

Serves [dictation 2026-05-23-5](../../../dictations/2026-05-23-5.md): "Block one tries to predict the next word, but block two tries to predict the features that predict the distribution over the next word."

## Status

**Mechanism works. Net performance benefit tentative.** V3 closed-loop with additive zero-init gate: both seeds show predictive coding mode (negative gain), deep integration (ablation gap +0.22), but the end-task improvement over 1-block baseline is -0.006 — barely past the preregistered threshold of -0.005, n=2 only.

**Strict-local COLLAPSED.** Phase 2 tested whether the mechanism survives without CE flowing through the feedback path. It does not — both seeds collapse catastrophically (D=3.05 vs A=1.67). Full-state cosine prediction without task gradient is a bad local objective. See `research/questions/local-learning-variants/README.md` for the detailed analysis.

**E_grounded resolved: stable but no benefit.** Strict-local with local CE for block 1 prevents collapse (confirms the target caused collapse, not locality itself) but doesn't improve task performance (+0.026 vs A). The feedback gradient through the interface is what teaches the predictor WHAT to predict — that's the real value. See `research/questions/local-learning-variants/README.md` for the complete analysis. **Currently testing: N=3 star topology** (width scaling).

## Phase 1: aux-only (NULL)

Tested with `token_injection="all"` (wrong architecture per [dictation 2026-05-23-7](../../../dictations/2026-05-23-7.md)). Every block saw tokens directly; predicting features between blocks added nothing.

| Seed | A: baseline | B: probe-only | C: hierarchical | C − A |
|------|-------------|---------------|-----------------|-------|
| 42 | 1.738 | 1.734 | 1.734 | -0.005 |
| 43 | 1.761 | 1.755 | 1.755 | -0.006 |
| 44 | 1.725 | 1.743 | 1.744 | +0.019 |
| **Avg** | **1.741** | **1.744** | **1.744** | **+0.003** |

The hierarchical loss DOES learn (0.94 → 0.41 cosine distance) — blocks CAN predict each other's future states. But this doesn't improve the task.

Artifact: [`experiments/fixed_multi_rate/artifacts/hierarchical_prediction/`](../../../experiments/fixed_multi_rate/artifacts/hierarchical_prediction/)

## Why aux-only is insufficient

The spectator problem on corrected architecture (`token_injection="block0"`) is NOT about gradient signal, information access, readout competition, or short context. Six follow-up experiments confirm this (all NULL or HURTS).

**Root cause:** Same-objective identical blocks have no reason to specialize. Block 0 already does the task optimally because it touches every token. Any auxiliary loss that doesn't affect the forward pass is just "blocks learn something on the side."

The spectator problem is a forward-pass architecture problem, not a training signal problem.

## Phase 2: closed-loop hierarchical prediction

**Key difference from Phase 1:** Block 1's predictions feed BACK into block 0's computation. Block 0 learns to USE the predictions. This gives block 1 a genuine forward role.

### Architecture (all versions)

Two blocks: block 0 (rate=1), block 1 (rate=2). Block 0 only sees tokens.

- Block 1 receives block 0's state, processes via FFN, produces prediction of block 0's next 2 states
- Predictions stored in a 2-slot buffer
- At each timestep, block 0 receives the prediction for THIS timestep via a gating mechanism
- Block 1 input: `x1 = 0.5 * (s1 + s0.detach())` (detach cuts pred_loss → block 0 path)

**Loss:** CE (next-token from final state) + 0.1 × pred_loss (cosine distance on LayerNorm'd states).

**Controls:**
- A_single: 1 block only (baseline)
- B_spectator: 2 blocks, shared CE, weighted readout (reconfirms spectator)
- C_closed_loop: 2 blocks, block 1 predicts, predictions fed back

### V1: catastrophic collapse

Block 0 feedback gate was MixAdd. In v1, `s0` was NOT detached from block 1's input, so pred_loss could train block 0 to be predictable. Collapse to constant state — val_loss stuck at 3.14 while A converges to 1.67.

### V2: collapse + NaN (detach alone insufficient)

Added `s0.detach()` to block 1's input. Still collapsed, then NaN at step 10K.

The problem was not the gradient path from pred_loss. It was MixAdd's initialization formula.

### Root cause: MixAdd sqrt formula

`MixAdd(init_keep=0.9)` computes `sqrt(mix)*keep + sqrt(1-mix)*add`.

At initialization: `sqrt(0.9)=0.949` keep, `sqrt(0.1)=0.316` add. The "prior" signal gets **31.6% influence**, not 10%.

Combined with final-only CE over 128 recurrent steps, this creates a stable collapsed fixed point: a random/garbage prediction signal at 31.6% influence at EVERY recurrent step overwhelms the single CE signal at the end. Block 0 can't recover → collapses → block 1 trivially predicts constant → pred_loss→0.

### V3: additive zero-init gate (the fix that works)

Replaced MixAdd with:

```python
x0 = seed0 + gain * LayerNorm(prior)
```

`gain` is a scalar parameter initialized to **0**. Block 0 starts identical to A_single — zero contamination at init. Gain grows or shrinks only if CE benefits.

### V3 results (2 seeds)

| Variant | Mean val_loss | C−A |
|---------|--------------|-----|
| A_single | 1.6703 | — |
| B_spectator | 1.6766 | +0.006 |
| **C_closed_loop** | **1.6647** | **-0.006** |

Per-seed: Seed 42 C−A = -0.009, Seed 43 C−A = -0.003. Mean: **-0.0056**.

Key metrics at convergence (averaged across seeds):

| Metric | Value | Meaning |
|--------|-------|---------|
| gain | -0.065 | Negative = predictive coding (subtract expected, process surprise) |
| pred_loss | ~0.29 | Non-trivial predictions (not collapsed) |
| ablation gap | +0.22 | Zeroing predictions hurts — deeply integrated |
| Wall time C vs A | ~980s vs ~545s | Prediction head + feedback loop overhead |

### Honest assessment

**Mechanism evidence: strong.** Both seeds show negative gain, large ablation gap, non-trivial pred_loss. The model spontaneously discovers predictive coding — it subtracts the expected and processes surprise. This is stable and deeply integrated.

**Performance evidence: tentative.** n=2, barely past the -0.005 threshold, ~30-40% chance a third seed flips it. 14% parameter confound (C has more params than A due to prediction head). B being worse than A (+0.006) suggests extra params alone don't help — the closed-loop mechanism specifically does — but this is weak evidence with n=2.

## What this settles

- The corrected architecture CAN support predictive feedback without collapse, given the right gating (additive zero-init, not MixAdd).
- The model spontaneously discovers predictive coding (negative gain = subtract expected, process surprise).
- The spectator problem is solved by role differentiation — prediction task gives block 1 a unique function that contributes to the forward pass.
- MixAdd's sqrt formula creates pathological initialization for recurrent feedback (31.6% influence at "10%" setting).

## What this does not settle

- Whether the net performance benefit is real (could be noise/params — needs more seeds or matched capacity).
- ~~Whether the mechanism works under truly local gradients~~ → **ANSWERED: NO.** Strict-local with full-state cosine prediction collapses. CE through the feedback interface is load-bearing. Currently testing whether task-grounded local objectives (E_grounded: block 1 has its own CE) can rescue strict-local.
- Whether this scales beyond 2 blocks.
- Whether the prediction quality is sufficient to help at larger model scale (currently 263K backbone params).
