# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## What just happened

**G_rate4_only: rate-4 is intrinsically too stale.**

| Variant | Seed 42 | Seed 43 | Mean | Δ vs A |
|---------|---------|---------|------|--------|
| A_single | 1.669 | 1.672 | 1.670 | — |
| G_rate4_only | 1.662 | 1.666 | 1.664 | -0.006 |

Ablation gap = 0 at both seeds. Helper predictions are not contributing at inference. The -0.006 advantage over A is from the auxiliary loss acting as a mild regularizer during training. pred_loss *rises* over training (0.23 → 0.49) — the staleness signature. Rate-2 (C) can track its target; rate-4 cannot.

**This settles the G decision rules:** G ≈ A (neutral). Rate-4 predictions are too stale. F's instability was the multi-helper interaction under shared loss, not rate-4 being harmful per se.

## Current: Phase 5 implementation (J — older-window target)

**I result (seed 42 complete, seed 43 interrupted at step 13K by GPU use):**
- I_phase_offset ≈ I_control ≈ A - 0.006. Width saturates. Target is the bottleneck.
- Seed 43 A_single confirmed (1.672). I_phase_offset seed 43 was tracking identically before interruption.
- Pattern matches ALL prior experiments (C, G, I all give -0.006). Effectively resolved.
- Formal seed 43 completion deferred to next available GPU window.

**Phase 5 J:** Change the prediction target from "full block-0 state" (which block 0 already knows) to "mean of block-0 states from 5-8 steps ago" (older context block 0 may not preserve). This directly tests Max's framing: "predict something block 0 couldn't already know — information from a longer time ago."

**Implementation:** Add `prediction_target` field to VariantSpec. Compute `target_history` from older window of `state0_history`. Pass to same `prediction_loss_terms`. Adjust valid mask (first 8 positions invalid).

**Decision rules:**
- **J < C (beats -0.006):** Target WAS the bottleneck. Older memory is genuinely useful. Scale further (K, L, then width + good target).
- **J ≈ C:** Older-window target at 5-8 char lag is too short to help at ctx=32→128. Try L (future chunk, different hypothesis family).
- **J > A (hurts) or collapses:** Self-generated target creates co-adaptation. Try J' (fixed embedding target).

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
10. **Rate-4 is intrinsically too stale (G ≈ A).** pred_loss rises over training. Ablation gap = 0. The staleness limit is somewhere between rate-2 (works) and rate-4 (too slow to track).

## Queue

- **I (phase offsets)** — DONE (seed 42 complete; seed 43 interrupted at step 13K by GPU use, deferred)
- ~~Transformer matched-param baseline~~ — blocked on GPU (Max gaming)
- **Phase 5: prediction target change** (J first) — IMPLEMENTING NOW (code changes, no GPU needed)
- Transformer matched-param baseline — `runs/transformer_baseline.py`, 2.856M params, launch when GPU free
- I seed 43 completion — rerun when GPU free (low priority, pattern already clear)
- Named/typed tensor dimensions
- Graph architecture exploration (from dictation 2026-05-24-1) — see `research/questions/graph-architecture/README.md`
- Hierarchical dynamic tokenization (from dictation 2026-05-24-3) — queued, not active

## Key completed findings

| Experiment | Result | Notes |
|---|---|---|
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
