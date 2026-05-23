# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## Active — strict-local test (Phase 2) NEXT

The closed-loop v3 experiment is complete. Mechanism works. Next: test if it survives without CE flowing through the feedback path. See `research/questions/local-learning-variants/README.md` for the full analysis.

**Implementation:** One extra `.detach()` on the feedback path — either detach `prior_t` before it enters block 0, or detach `pred_pair` before `.copy_()`. This cuts the CE→block1 gradient. Block 1 trains ONLY on pred_loss. Block 0 trains ONLY on CE. No gradient crosses the boundary.

Same run script, same controls (A/B/C), same seeds, same 20K steps.

### Decision rules (strict-local)

- **Strict-local C < A:** LOCAL LEARNING WORKS. Block 1 finds task-relevant predictions without any task gradient. Training can genuinely parallelize.
- **Strict-local C ≈ A but semi-local C < A:** CE shaping through the feedback path is load-bearing. Not fully local yet.
- **Strict-local C > A:** Without CE guidance, predictor learns wrong things. Gate collapses to zero or predictions mislead.

## Closed-loop v3 DONE — mechanism works, net benefit tentative

### Final results (2 seeds averaged)

| Variant | Mean val_loss | C−A |
|---------|--------------|-----|
| A_single | 1.6703 | — |
| B_spectator | 1.6766 | +0.006 |
| **C_closed_loop** | **1.6647** | **-0.006** |

Per-seed: Seed 42 C-A = -0.009, Seed 43 C-A = -0.003. Mean: **-0.0056** (barely past -0.005 threshold).

### Honest assessment

- **Mechanism evidence: strong.** Both seeds show negative gain (-0.07, -0.06), huge ablation gap (+0.23, +0.22), non-trivial pred_loss (~0.29). Model uses predictions deeply.
- **Performance evidence: tentative.** n=2, barely past threshold, ~30-40% chance a third seed would flip it back below threshold. Parameter confound: C has 14% more params than A.
- **B confirms spectator problem:** B is WORSE than A (+0.006 averaged). Extra params alone don't help; the closed-loop mechanism specifically does.

The key takeaway: v3 consistently learned and used the closed-loop prediction pathway, but its end-task gain over the 1-block baseline is tiny, barely over the preregistered threshold, and still plausibly explained by seed noise and unmatched capacity. What IS clear: the mechanism is active, stable, and deeply integrated. That's sufficient to proceed with the strict-local test — which is the actual question (can this work without global backprop?).

### V3 architecture summary

- `x0 = seed0 + gain * LayerNorm(prior)`, gain=0 at init
- Block 1 input: `x1 = 0.5 * (s1 + s0.detach())`
- Gain goes NEGATIVE (-0.06 to -0.07) = predictive coding: subtract expected, process surprise
- Ablation gap +0.22: predictions deeply embedded despite tiny net benefit

### Prior failures and their fixes

- **V1:** Catastrophic collapse (val_loss stuck at 3.14). pred_loss gradient into block 0.
- **V2:** Collapse + NaN. MixAdd `sqrt(0.1)` = 31.6% prior coefficient (not 10%). Stable collapsed fixed point with final-only CE.
- **V3 fix:** Additive zero-init gate. No contamination at init.

## Critical findings (carry forward)

1. **MixAdd sqrt formula at init=0.9 gives 31.6% coefficient, not 10%.** Root cause of v1/v2 collapse.
2. **Additive zero-init gate works.** No collapse. Model learns gain automatically. Negative gain = predictive coding.
3. **Semi-local is NOT genuinely local.** CE flows through feedback. Current v3 is global backprop through a narrow interface.
4. **Predictions provide early-learning acceleration** (C beats A by 0.03 at step 2K) that narrows at convergence.
5. **The spectator problem is real and persistent.** B hurts on average (+0.006). Extra blocks with shared objective don't help.

## Queue

- Transformer matched-compute baseline (unfair at total params — backbone 263K vs 1.53M)
- Named/typed tensor dimensions
- Loop management tooling
- Graph architecture from dictation

## Key completed findings

| Experiment | Result | Notes |
|---|---|---|
| **Closed-loop v3** | **C < A by 0.006** (2 seeds) | Mechanism active, net benefit tentative. Predictive coding mode. |
| **Closed-loop v1** | **COLLAPSE** (+1.47) | pred_loss trained block 0 to be constant |
| **Closed-loop v2** | **COLLAPSE → NaN** | MixAdd 31.6% prior = stable collapsed fixed point |
| ctx=128 corrected | **HURTS** (+0.014) | Spectator worse at longer context |
| ctx=128 ensemble | Tiny benefit (-0.020) | Was -0.101 at ctx=32; collapses |
| ctx=32 baseline | B wins (-0.101), C≈A | Spectator on corrected arch |
| ctx=32 local aux loss | **NULL** (+0.003) | Gradient isn't the problem |
| ctx=32 equal readout | **HURTS** (+0.024) | Information poverty confirmed |
| ctx=32 temporal window | **NULL** (+0.003) | Learned projection of history doesn't help |
| Bidirectional top-down | **HURTS** | TinyShakespeare |
| CUDA Graph training | 10.86× speedup | GraphTrainer in core/ |
| Self-prediction | NULL | Zero effect |
