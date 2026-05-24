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

## What's next: discriminate rate-4 viability

### N=3 interim result (seed 42 complete, seed 43 in progress)

**F_star_3block HURTS.** Not just redundant — actively worse than both A and C:

| Variant | val_loss (seed 42) | vs A |
|---------|-------------------|------|
| C_closed_loop | 1.660 | **-0.009** |
| A_single | 1.669 | — |
| **F_star_3block** | **1.702** | **+0.033** |

Helper 2 (rate=4) was actively driven to zero by CE:

| Step | gain_1 (rate=2) | gain_2 (rate=4) |
|------|-----------------|-----------------|
| 5K   | -0.057          | -0.013          |
| 9K   | -0.058          | -0.007          |
| 15K  | -0.068          | -0.0004         |
| 20K  | -0.082          | -0.0002         |

gain_2 → 0 means helper 2 doesn't affect the forward pass. Yet F is still 0.033 worse than A. **Training-time interference through the shared prediction loss** — helper 2's noise in `prior_t_1 + prior_t_2` corrupts the gradient that shapes helper 1's predictions.

Supporting evidence: gain_1 in F (-0.082) is larger in magnitude than in C (-0.071), suggesting block 0 became MORE dependent on predictions in F — possibly because block 0 itself learned less effectively (degraded representations = more reliance on helpers).

**Provisional until seed 43 confirms.** Expected variance is ~0.005; the F-A gap (0.033) is much larger, so likely real.

### Next discriminating experiment: G_rate4_only

**Question:** Is rate-4 intrinsically harmful, or does the damage only occur when combined with rate-2 under a shared prediction objective?

**Test:** Run C_closed_loop but with a rate-4 helper instead of rate-2. Same everything else: 2 blocks, star topology, CE flows through interface.

**Decision rules:**
- **G_rate4_only ≈ C (both help vs A):** Rate-4 is viable in isolation! The N=3 failure was purely the shared-loss interference. Fix: per-helper prediction losses to remove the coupling.
- **G_rate4_only ≈ A (neutral):** Rate-4 predictions are too stale to help but don't hurt when alone. The N=3 hurt was from the shared-loss corrupting helper 1.
- **G_rate4_only > A (hurts):** Rate-4 is intrinsically bad — even in isolation it degrades training. Don't use high rates; try phase offsets instead.
- **G_rate4_only collapses:** Something specific about rate-4 breaks the mechanism entirely. Investigate.

**Code changes needed:**
- Make block 1's rate configurable in VariantSpec (currently hardcoded: fires `time_index % 2 == 0`, prediction head = `Linear(d_model, 2*d_model)`)
- New spec: `rates=(1, 4)` → block 1 fires every 4 steps, predicts 4 future states
- Safe to implement after current run completes (Python loaded code at start; file modification won't affect running process, but avoid in case of crash/restart)

### After G_rate4_only

If G works (rate-4 viable in isolation), test **per-helper prediction loss** at N=3:
- Each helper gets its own cosine loss against its own target segment
- Removes the identifiability problem
- If this makes both helpers contribute, we have a scaling path

If G fails (rate-4 intrinsically bad), test **phase offsets** at N=3:
- Two helpers both at rate=2, but on different phases (t%2==0 vs t%2==1)
- Removes the "harder task" problem while still testing width composition

## Critical findings (carry forward)

1. **MixAdd sqrt formula at init=0.9 gives 31.6% coefficient, not 10%.** Root cause of v1/v2 collapse.
2. **Additive zero-init gate works.** No collapse. Model learns gain automatically. Negative gain = predictive coding.
3. **Semi-local IS neighborhood-local.** CE flows through feedback interface. Each block pair is a "neighborhood."
4. **Full-state cosine prediction is a bad LOCAL objective.** Ungrounded by task. Collapses under strict-local. Under semi-local, CE shapes it to be useful.
5. **Task-grounded strict-local is stable but neutral.** Local CE prevents collapse but doesn't make predictions useful. The feedback gradient teaches WHAT to predict — that's the value.
6. **Neighborhood-local is the correct architecture.** The minimum viable locality that actually helps.
7. **The spectator problem is solved by role differentiation.** B hurts; closed-loop gives block 1 a unique function.
8. **Predictive coding emerges spontaneously.** Gain goes negative — model subtracts predicted, processes surprise.
9. **Summed-prior prediction loss creates interference at N>2.** When multiple helpers' predictions are summed before computing the loss, the secondary helper's noise corrupts the gradient for the primary helper — even after the secondary helper's gain goes to zero. F_star is 0.033 WORSE than A despite gain_2 ≈ 0. Fix requires per-helper supervision or removing the shared prediction sum.

## Queue

- **G_rate4_only discriminator** — NEXT (implement when current run completes)
- Transformer matched-param baseline — **READY TO LAUNCH** (`runs/transformer_baseline.py`, 2.856M params, CPU-verified)
- Per-helper prediction loss (if G shows rate-4 is viable)
- Phase offsets at N=3 (if G shows rate-4 is intrinsically bad)
- Named/typed tensor dimensions
- Graph architecture exploration (from dictation 2026-05-24-1) — see `research/questions/graph-architecture/README.md` for N=8 scaling analysis
- Hierarchical dynamic tokenization (from dictation 2026-05-24-3) — queued, not active

## After current run (F_star_3block) completes

1. Commit log, write up full analysis in local-learning-variants README
2. Implement G_rate4_only variant (configurable rate for block 1)
3. Run G alongside transformer baseline (both are 2-seed × 20K, can run sequentially)
4. Analyze G results per decision rules above

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
