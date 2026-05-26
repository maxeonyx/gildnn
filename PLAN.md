# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current operational state (2026-05-26, 14:15 NZST)

**GPU: FREE.** Warmup_detach killed per dictation 2026-05-26-2. No active experiment.

**Daily report 2026-05-26:** NOT YET WRITTEN. Due after 4pm NZST.

**Direction change (dictation 2026-05-26-2):** Stop gradient-removal experiments. Implement actual local predictive loss. Short runs first.

---

## What Max wants tested (dictation 2026-05-26-2)

The three comparison points that matter:

| Condition | Val_loss | Notes |
|---|---|---|
| Transformer baseline | 1.592 | 2.86M params, `base-experiments/transformer_wt103/` |
| Single block with CE | ~1.832 | 2.85M params, from tied-depth experiment |
| **Multi-block with local predictive loss** | **?** | **Interior blocks: predict-what-arrives-laterally only. No CE on interior.** |

**The question:** Does adding predictive-loss-trained interior blocks improve over a single CE-connected block? By how much? How does it compare to the transformer?

**Design constraint (non-negotiable):** Interior blocks receive ONLY local signals. No task gradient (CE) flows to them. The output block gets CE. Interior blocks predict what arrives laterally. This is the biologically plausible predictive processing architecture.

**STOPPED (per dictation):**
- ~~warmup_detach~~ — answered irrelevant question ("does removing gradient make things worse?" when gradient isn't an option)
- ~~tied_sharing rerun~~ — lower priority than the actual local loss question
- ~~predictive-residual (old design)~~ — kept shared adjoint, which Max explicitly rejects

---

## What to do next

**Earn GPU time.** Per new process rule: prove mechanism in minutes first. The ladder:

1. **Design the local predictive loss.** What exactly does "predict what arrives laterally" mean in our architecture?
   - Block i predicts block (i-1)'s output? Or predicts the lateral signal it will receive next step?
   - Loss function: MSE? Cosine? Wasserstein (per dictation 2026-05-25-1)?
   - Conceptual clarification FIRST. Theory before code.

2. **Implement minimally.** Modify ParallelDiagonalModel to support per-block local losses with no shared adjoint on interior blocks. Only output block gets CE.

3. **Short run (~2K steps, 1 seed, <5 min).** Does the multi-block model with local loss beat single-block? If not, fix before scaling.

4. **Scale only if mechanism works.** 20K steps, 3 seeds — only if short run shows clear improvement.

---

## Decision state (what's been decided)

| Decision | Outcome | Constraint it imposes |
|---|---|---|
| Temporal_window 2-block | Branch 1 confirmed (Δ=+0.042, 3 seeds) | Trajectory info uniquely helps intended arch |
| 4-block follow-up | **Stop-loss fired** (conditions 4+5 fail) | No more intended-architecture rescue |
| Bridge_detach | Clearly worse (gap +0.027, 3 seeds) | Gradient IS needed for block specialization |
| **Dictation 2026-05-26-2** | **Redirect** | **Only compare against legitimate alternatives. Local predictive loss is the experiment now.** |
| Timebox | ~5 days remaining | Earn GPU time. Short runs first. |

**Traps to avoid:**
- Comparing against "full gradient" (not a legitimate alternative — can't have it in async arch)
- Multi-hour multi-seed runs before mechanism is demonstrated at short scale
- Custom CUDA kernels (infra not findings)
- Starting new pathways (stick to Pathway 3 local learning)

---

## Key references

- `VISION.md` — stakeholder requirements
- `ROADMAP.md` — Pathway 3 (local learning) is active
- `dictations/2026-05-26-2.md` — the redirect that created this plan
- `dictations/2026-05-25-1.md` — multi-timestep architecture with Wasserstein local loss
- `research/questions/local-learning/README.md` — prior work (bridge_detach, warmup results still valid as mechanism understanding)
- `research/questions/multi-timestep-architecture/README.md` — Max's latest architecture thinking

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
| Warmup→detach | 3 | Scenario A (2/3 seeds) | Gradient needed continuously (but question is now moot) |

---

## The agent's working loop

Follow PROCESS.md. The short version:

```
PROCESS CHECK → PATHWAY SELECTION → [adversarial gate] →
CONCEPTUAL CLARIFICATION → [adversarial gate] →
EXPERIMENT DESIGN → [adversarial gate] →
RUN & ANALYZE → [adversarial gate] →
INTEGRATE / REPORT / UPDATE PLAN → back to PATHWAY SELECTION
```

**Current position:** CONCEPTUAL CLARIFICATION. Must design the local predictive loss before implementing.
