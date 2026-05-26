# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current operational state (2026-05-26, 16:30 NZST)

**GPU: FREE.** No active experiment.

**Daily report 2026-05-26:** WRITTEN. See `research/daily/2026-05-26.md`.

---

## What Max wants tested (dictation 2026-05-26-2)

**The question:** Does adding predictive-loss-trained interior blocks improve over a single CE-connected block?

**Design constraint (non-negotiable):** Interior blocks receive ONLY local signals. No task gradient (CE) flows to them. The output block gets CE. Interior blocks predict what arrives laterally.

---

## Today's findings (local predictive loss, 2K-step short runs)

| Condition | Val loss @ 2K | Δ from baseline | Notes |
|---|---|---|---|
| Baseline (1 block, CE) | 2.304 | — | 2.85M params |
| Control (2 blocks, λ=0, block 1 untrained) | 2.501 | +0.197 | Mixing harm only |
| Treatment (2 blocks, λ=1, block 1 trained) | 2.467 | +0.163 | Local loss helps by 0.034 |
| Gated (lateral_mix=0.1 for block 0) | DIVERGED | — | Instability at steps 1000-1500 |

**Key findings:**
1. The local predictive loss WORKS — reduces mixing damage by 0.034 vs untrained block
2. The 50/50 lateral mixing is the dominant harm (+0.197)
3. Reducing the mix causes instability (divergence)
4. Block 1's local loss INCREASES over training (0.109→0.151): moving-target problem

---

## What to do next

The mechanism works but the architecture needs a different mixing strategy. The hardcoded `0.5 * (own_state + lateral)` is the bottleneck. Options to try:

1. **Additive lateral (residual):** `block_input = own_state + scale * projection(lateral)`. Block 0 keeps its FULL state, block 1's contribution is added as a small residual. Scale could be learnable or fixed small. This avoids the 50/50 dilution entirely.

2. **Layer-norm before mixing:** Normalize both sides before combining. Prevents scale explosions (fixes the gated instability) and makes MSE less sensitive to representation drift (partially fixes moving-target problem).

3. **Cosine / normalized prediction loss:** Predict direction not magnitude. More stable target.

4. **Lower lambda (0.1 instead of 1.0) with 0.5 mix:** Let block 1 learn slowly, don't let local loss dominate.

**Recommended:** Try (1) additive lateral first — it's the cleanest architectural fix and directly addresses the dominant harm (dilution). If that works, the mechanism has room to breathe.

---

## Decision state (what's been decided)

| Decision | Outcome | Constraint it imposes |
|---|---|---|
| Temporal_window 2-block | Branch 1 confirmed (Δ=+0.042, 3 seeds) | Trajectory info uniquely helps intended arch |
| 4-block follow-up | **Stop-loss fired** (conditions 4+5 fail) | No more intended-architecture rescue |
| Bridge_detach | Clearly worse (gap +0.027, 3 seeds) | Gradient IS needed for block specialization |
| **Dictation 2026-05-26-2** | **Redirect** | **Only compare against legitimate alternatives** |
| **Local predictive loss v1** | **Mechanism works, architecture harms it** | **Fix mixing before scaling** |
| Timebox | ~4 days remaining | Earn GPU time. Short runs first. |

**Traps to avoid:**
- Running longer with the same broken mixing (won't help)
- Multi-hour runs before the mixing problem is solved
- Symmetric lateral_mix init (block 1 NEEDS to see block 0)
- Changing too many things at once (one fix per run)

---

## Key references

- `VISION.md` — stakeholder requirements
- `ROADMAP.md` — Pathway 3 (local learning) is active
- `dictations/2026-05-26-2.md` — the redirect
- `research/questions/multi-timestep-architecture/README.md` — Max's architecture vision (distributional, Wasserstein)
- `runs/local_predictive.py` — the experiment script (supports --lateral-mix-init, --lambda-local)
- `research/daily/2026-05-26.md` — today's full write-up

---

## Completed work (reference)

| What | Pathway | Result | Key finding |
|---|---|---|---|
| Transformer baseline (WT-103) | all | 1.592 ± 0.003, 2.86M params | External anchor |
| Tied-depth tiny rung | 1 | Mean diff 0.0002, 3 seeds | Weight sharing free at tiny scale |
| Tied-depth WT-103 | 1 | Seed-sensitive, confounded | Cannot isolate (clean test cancelled) |
| Dynamic-depth oracle | 5 | Speedup 1.96× | Worth pursuing but not now |
| Temporal_window 2-block | 3 | Δ=+0.042, 3 seeds | Branch 1: trajectory uniquely helps |
| 4-block follow-up | 3 | Stop-loss fires | Block 1 rescues, blocks 2-3 spectators |
| C_old lateral ablation | 3 | Δ=+0.030, 2 seeds | Laterals load-bearing |
| Bridge_detach | 3 | +0.027, 3 seeds | U-shaped vs front-loaded readout |
| Warmup→detach | 3 | Scenario A (2/3 seeds) | Gradient needed continuously (moot now) |
| **Local predictive loss v1** | **3** | **Mechanism works, mixing harms** | **50/50 is the bottleneck, not the loss** |

---

## The agent's working loop

Follow PROCESS.md. Current position: **back at EXPERIMENT DESIGN**. The conceptual mechanism is validated. Next: fix the architecture (mixing strategy), then re-test.
