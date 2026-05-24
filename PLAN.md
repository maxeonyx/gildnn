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

## Current: Next experiment — J_far_window (offset 9-12)

All Phase 1-5 experiments ran at **ctx=128** (not ctx=32 — that was the old spectator architecture). The older-window target (positions 5-8 back) works. Now: what scales it?

**Natural scaling axes:**
1. **Temporal separation** — try farther-back windows (9-12, then 17-24). At ctx=128, plenty of room. Find where target becomes stale vs remains useful.
2. **Context length** — already at 128. Scaling to 256/512 is the next tier but changes many things. Later.
3. **Helper diversity** — multiple helpers with different windows (e.g. one at 5-8, one at 17-24)
4. **Model size** — only after the above are explored

**Immediate next steps:**
1. ✅ Transformer matched-param baseline — partial (13K steps, killed). val_loss=1.683 at step 13K. Rerun later.
2. ✅ J_far_window implemented (offset=12, size=4) — sanity check passes
3. ✅ **J_far_window full run LAUNCHED** — A_single + J_far_window, 2 seeds, 20K steps (PID 4516, ETA ~00:30)

## Queue

- **J_far_window full run** — RUNNING (active.lock set, ETA ~22:35)
- J_fixed_embedding — already in code, tests external vs self-generated target. **Most discriminating next experiment regardless of J_far outcome.**
- J_strict_local — NEW: older-window target + strict-local (no CE through interface). Tests whether a good target rescues strict-local. If yes → true parallelism possible.
- Wider temporal separation (offset=20 or 24) — if J_far shows plateau or improvement
- Multi-helper with different offsets (N=3, helpers at offset 5-8 and 9-12) — tests temporal band composition
- Transformer baseline (proper full run) — partial run killed at 13K, val_loss 1.683 on track
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
