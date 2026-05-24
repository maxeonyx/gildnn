# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## What just happened

**E_grounded: stable but no benefit.** The full locality experiment series is now complete:

| Variant | Mean val_loss | vs A | Conclusion |
|---------|--------------|------|-----------|
| C_closed_loop (semi-local) | 1.665 | **-0.006** | CE through interface HELPS |
| A_single | 1.670 | — | baseline |
| B_spectator | 1.677 | +0.006 | extra params alone hurt |
| **E_grounded (strict-local + local CE)** | **1.696** | **+0.026** | stable but hurts |
| D_strict_local | 3.047 | +1.377 | COLLAPSED |

**What this settles:**
1. D's collapse was caused by the ungrounded prediction target, NOT by locality itself (E proves this).
2. Task-grounded strict-local IS viable (no collapse) but doesn't add value — predictions slightly hurt.
3. The CE-through-interface gradient specifically teaches the predictor WHAT to predict. That's the mechanism that makes semi-local C beat A.
4. **Neighborhood-local is the correct architecture.** Not because strict-local is impossible, but because the feedback gradient IS the value.

The feedback gradient does two things: (a) prevents collapse (which local CE also does ✓), and (b) teaches which prediction dimensions are useful (which local CE does NOT do ✗). Only (b) actually helps performance.

## What's next: N=3 semi-local width test

The local-learning-variants question is answered (neighborhood-local is the minimum viable locality, strict-local doesn't add value). Now test whether the mechanism scales.

**N=3 semi-local star topology** (from think agent analysis):
- Block 0: processes tokens, has CE loss
- Block 1 (rate=2): reads s0.detach(), predicts s0 future, feeds predictions to block 0. CE flows through interface.
- Block 2 (rate=4): reads s0.detach(), predicts s0 future, feeds predictions to block 0 via SEPARATE gain. CE flows through interface.
- Star topology: both helpers predict block 0 directly (not a chain)
- Different rates give different time horizons (short-horizon vs long-horizon predictions)

**Why star, not chain:**
- Chain (0←1←2) means block 2 is one hop from CE — same grounding-drift problem
- Star means both helpers get direct CE shaping through their interfaces
- Matches Max's "many more parallel blocks" vision better (scale by adding more spokes)
- Max's original vision note: "graph" not "chain"

**Decision rules (N=3 star):**
- **N=3 star helps more than N=2:** Width composition works! The architecture scales. Move quickly toward N=8.
- **N=3 star ≈ N=2:** Two helpers don't combine usefully. Maybe they're redundant (both predict same thing). Need to differentiate their roles more.
- **N=3 star hurts or collapses:** Something about adding a second helper breaks the mechanism. Investigate.

## Critical findings (carry forward)

1. **MixAdd sqrt formula at init=0.9 gives 31.6% coefficient, not 10%.** Root cause of v1/v2 collapse.
2. **Additive zero-init gate works.** No collapse. Model learns gain automatically. Negative gain = predictive coding.
3. **Semi-local IS neighborhood-local.** CE flows through feedback interface. Each block pair is a "neighborhood."
4. **Full-state cosine prediction is a bad LOCAL objective.** Ungrounded by task. Collapses under strict-local. Under semi-local, CE shapes it to be useful.
5. **Task-grounded strict-local is stable but neutral.** Local CE prevents collapse but doesn't make predictions useful. The feedback gradient teaches WHAT to predict — that's the value.
6. **Neighborhood-local is the correct architecture.** The minimum viable locality that actually helps.
7. **The spectator problem is solved by role differentiation.** B hurts; closed-loop gives block 1 a unique function.
8. **Predictive coding emerges spontaneously.** Gain goes negative — model subtracts predicted, processes surprise.

## Queue

- N=3 semi-local star (NEXT)
- Transformer matched-compute baseline
- Named/typed tensor dimensions
- Graph architecture exploration (from dictation 2026-05-24-1)
- Hierarchical dynamic tokenization (from dictation 2026-05-24-3) — queued, not active
- Loop management tooling

## Key completed findings

| Experiment | Result | Notes |
|---|---|---|
| **E_grounded** | **STABLE, +0.026 vs A** | Task-grounded strict-local doesn't collapse but doesn't help |
| **D_strict_local** | **COLLAPSE** (+1.38) | Full-state local prediction fails |
| **C_closed_loop (v3)** | **C < A by 0.006** (2 seeds) | Semi-local mechanism works |
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
