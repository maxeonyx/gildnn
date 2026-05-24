# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## What just happened

**Phase 5 J: TARGET WAS THE BOTTLENECK. Older-window prediction works.**

| Variant | Seed 42 | Seed 43 | Mean | Δ vs A | std |
|---------|---------|---------|------|--------|-----|
| A_single | 1.669 | 1.672 | 1.670 | — | 0.0015 |
| C_closed_loop | 1.660 | 1.669 | 1.665 | -0.006 | 0.0047 |
| **J_older_window** | **1.660** | **1.660** | **1.660** | **-0.010** | **0.00009** |

Key metrics at 20K steps (both seeds):
- **pred_loss**: 0.165 (vs C's 0.29 — target is 2× more learnable)
- **mix_coeff**: -0.065 (predictive coding — same mechanism as C)
- **ablation_gap**: 0.182/0.193 (load-bearing at inference)
- **std across seeds**: 0.00009 (vs C's 0.005 — 53× more reliable)

**What this means:** Predicting "something block 0 couldn't already know" (mean of older states) IS genuinely more useful than predicting block 0's current state. The improvement is modest in absolute terms (-0.010 vs -0.006) but the mechanism is dramatically more reliable. C's apparent -0.006 mean was depressed by seed variance; J eliminates that variance entirely.

**Decision rule triggered:** J < C → Scale further.

## Current: Next experiment planning

The older-window target (positions 5-8) works. Now: what scales it?

**Natural scaling axes** (from think agent analysis):
1. **Temporal separation** — try farther-back windows (9-16, 17-32). Find where target becomes stale vs remains useful.
2. **Context length** — older-memory should matter more at longer context (128, 256). Currently ctx=32.
3. **Helper diversity** — multiple helpers with different windows (e.g. one at 5-8, one at 17-32)
4. **Model size** — only after the above are explored

**Immediate next steps:**
1. Run transformer matched-param baseline (2.856M params) — independent comparison point
2. Implement + run J_far_window variant (positions 17-32 instead of 5-8) — tests temporal separation
3. After that: context length scaling with the winning target

## Critical findings (carry forward)

1. **MixAdd sqrt formula at init=0.9 gives 31.6% coefficient, not 10%.** Root cause of v1/v2 collapse.
2. **Additive zero-init gate works.** No collapse. Model learns gain automatically. Negative gain = predictive coding.
3. **Semi-local IS neighborhood-local.** CE flows through feedback interface. Each block pair is a "neighborhood."
4. **Full-state cosine prediction is a bad LOCAL objective.** Ungrounded by task. Collapses under strict-local. Under semi-local, CE shapes it to be useful.
5. **Task-grounded strict-local is stable but neutral.** Local CE prevents collapse but doesn't make predictions useful. The feedback gradient teaches WHAT to predict — that's the value.
6. **Neighborhood-local is the correct architecture.** The minimum viable locality that actually helps.
7. **The spectator problem is solved by role differentiation.** B hurts; closed-loop gives block 1 a unique function.
8. **Predictive coding emerges spontaneously.** Gain goes negative — model subtracts predicted, processes surprise.
9. **N=3 with shared prediction loss is seed-sensitive / unstable.** Coupling between helpers under shared aux loss. Rate-4 helper always dies; instability comes from how quickly.
10. **Rate-4 is intrinsically too stale (G ≈ A).** pred_loss rises over training. Ablation gap = 0.
11. **Target matters more than width.** I (two helpers, full-state) = -0.006. J (one helper, older-window) = -0.010. Better target > more helpers.
12. **Older-window target is dramatically seed-robust.** J std=0.00009 vs C std=0.005. The less-redundant target creates a more reliable learning signal.

## Queue

- Transformer matched-param baseline — `runs/transformer_baseline.py`, 2.856M params, GPU free now
- J_far_window (positions 17-32) — implement and run
- Context length scaling with older-window target
- Graph architecture exploration (from dictation 2026-05-24-1) — see `research/questions/graph-architecture/README.md`
- Hierarchical dynamic tokenization (from dictation 2026-05-24-3) — queued, not active

## Key completed findings

| Experiment | Result | Notes |
|---|---|---|
| **J_older_window** | **J < A by 0.010, std 0.00009** | Target was the bottleneck; older memory works |
| **I_phase_offset** | **I ≈ A - 0.006** | Width saturates at same target |
| **G_rate4_only** | **G ≈ A, ablation gap 0** | Rate-4 too stale; auxiliary loss regularizes slightly |
| **F_star_3block** | **SEED-SENSITIVE** (range 0.044) | Shared loss coupling, not rate-4 per se |
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
