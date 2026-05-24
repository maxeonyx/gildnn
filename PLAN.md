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

## Current experiment: I (phase offsets) — RUNNING

**Question:** Can width help if both helpers operate at rate-2 but at different phases?

**Architecture:** 3 blocks. Block 0 at rate 1. Block 1 at rate 2, phase 0 (updates on even steps). Block 2 at rate 2, phase 1 (updates on odd steps). Per-helper prediction losses (no shared objective — the known coupling fix).

**Control:** I_control — same as I_phase_offset but both helpers at phase 0. Isolates whether phase diversity matters or whether "two rate-2 helpers" is sufficient.

**Decision rules:**
- **I_phase_offset < A (helps) AND < I_control:** Phase offset creates useful role differentiation. Width + temporal diversity = the scaling path.
- **I_phase_offset ≈ I_control < A:** Both help, offset doesn't matter. Width alone scales (just add more rate-2 helpers).
- **I_phase_offset ≈ I_control ≈ A:** Per-helper losses removed the instability but helpers still get pruned under CE/gate competition. Would need alternating optimization.
- **I_phase_offset > A (hurts):** Something about per-helper losses + multi-helper is intrinsically bad. Investigate.

**Running:** A_single + I_phase_offset + I_control, 2 seeds × 20K steps each. Started ~3:49pm. Expected ~60 min (~4:50pm).
Log: `experiments/wikitext_103/artifacts/closed_loop_prediction/run_i.jsonl`

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

- **I (phase offsets)** — RUNNING NOW
- Transformer matched-param baseline — `runs/transformer_baseline.py`, 2.856M params, ready to launch after I
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
