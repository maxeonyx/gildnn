# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## Active — closed-loop prediction v3 (additive gain=0) RUNNING

**PID 20572**, log: `experiments/wikitext_103/artifacts/closed_loop_prediction/run_v3.jsonl`

### Seed 42 COMPLETE — provisional POSITIVE result

| Variant | val_loss | accuracy |
|---------|----------|----------|
| A_single | 1.669 | 0.523 |
| B_spectator | 1.668 | 0.532 |
| **C_closed_loop** | **1.660** | **0.529** |

C beats A by **0.009 nats**. First positive result on corrected architecture.

Key C metrics at convergence:
- `prior_gain`: -0.071 (negative = predictive coding: subtract prediction, process residual)
- `pred_loss`: 0.294 (non-trivial — block 1 making real predictions, not collapsed)
- `ablation gap`: +0.231 (removing predictions hurts significantly)

The gain went NEGATIVE, which means the model learned: "subtract the predicted state from my input → process only what differs from prediction." This is classic predictive coding — unexpected inputs get amplified, expected inputs get suppressed.

### Seed 43 IN PROGRESS

Started A_single. ETA ~75 minutes for all 3 variants. Same decision rules apply.

### What happened in v1 and v2 (prior failures)

- **V1:** Catastrophic collapse (val_loss stuck at 3.14). Root cause: pred_loss gradient into block 0.
- **V2:** Collapse + NaN. Root cause: MixAdd `sqrt(0.1) = 0.316` prior coefficient (31.6% influence, not 10% as intended). Combined with final-only CE over 128 recurrent steps = stable collapsed fixed point.
- **V3 fix:** Replace MixAdd with `x0 = seed0 + gain * LayerNorm(prior)`, gain=0 at init. Block 0 starts identical to A_single. Gain grows/shrinks only if CE benefits.

### Decision rules (after seed 43)

Average C-A across both seeds:
- **Mean(C-A) < -0.005:** Positive result. Proceed to strict-local test (Phase 2).
- **Mean(C-A) ∈ [-0.005, +0.005]:** Borderline null. Predictions used but marginal benefit. Consider larger scale or different prediction target.
- **Mean(C-A) > +0.005:** Null/negative. Predictions don't help despite being used. Change prediction target.

## If positive → next experiments (see research/questions/local-learning-variants/README.md)

1. **Strict-local:** Also detach feedback path. Block 1 trained ONLY by pred_loss. Tests if pure local learning works.
2. **Per-dimension gate:** Replace scalar gain with vector gate (Linear layer, zero init). More expressive.
3. **Different prediction target:** Predict next TOKEN embedding instead of s0. Provides genuinely new information.
4. **N=3 chain:** Three blocks, adjacent prediction, tests multi-hop grounding.

## Critical findings this session

1. **MixAdd sqrt formula at init=0.9 gives 31.6% coefficient, not 10%.** Root cause of v1/v2 collapse.
2. **Final-only CE over 128 recurrent steps can't overcome strong prior contamination.** The MixAdd + recurrence created a stable collapsed fixed point.
3. **Additive zero-init gate works.** No collapse. Model learns gain automatically. Negative gain = predictive coding.
4. **Predictions provide early-learning acceleration** (C beats A by 0.03 at step 2K) that **narrows but persists at convergence** (C beats A by 0.009 at step 20K, seed 42).
5. **Semi-local is NOT genuinely local.** CE flows through the feedback path. The current setup is global backprop through a narrow interface. Strict-local (detach feedback too) is the real test.

## Queue

- Transformer matched-compute baseline (unfair at total params — backbone 263K vs 1.53M)
- Named/typed tensor dimensions
- Loop management tooling
- Graph architecture from dictation

## Key completed findings

| Experiment | Result | Notes |
|---|---|---|
| **Closed-loop v3** | **C < A by 0.009** (1 seed) | First positive! Predictive coding mode, gain=-0.07 |
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
