# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current operational state (2026-05-26, 17:00 NZST)

**GPU: FREE.** No active experiment.

**Daily report 2026-05-26:** WRITTEN. See `research/daily/2026-05-26.md`.

**New dictation 2026-05-26-3:** PROCESSED. Process corrections applied to PROCESS.md. Work direction updated in this file.

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

## What to do next (per dictation 2026-05-26-3: decompose, don't iterate on the assembly)

The last experiment tested too many interacting variables at once: 50/50 mixing, MSE on raw activations, the topology, the prediction target. Per dictation 2026-05-26-3, the next step is NOT "try additive lateral on the same combined setup." The next step is: **decompose into pieces, test each in isolation.**

### The pieces to test independently (seconds each, not minutes):

1. **Does the MSE prediction objective converge at all?** — Strip away mixing entirely. One block, one linear head predicting a FIXED target (e.g. the embedding layer output, or a random but fixed projection). Does the local MSE loss decrease? This isolates whether the loss function itself works.

2. **Does mixing destroy signal?** — Baseline: 1 block with CE. Treatment: same block, but feed it `0.5 * (own_input + noise)`. How much does 50/50 mixing with random noise hurt? This isolates the mixing harm without ANY second block.

3. **Does the prediction objective track a moving target?** — One block, one predictor, target is the output of a SEPARATELY trained block (trained with CE on different data, frozen). Does the predictor converge to predicting the frozen block's output? This isolates the prediction mechanism from the moving-target problem.

4. **What mixing ratio preserves signal?** — Same as (2) but sweep ratios: 0.9/0.1, 0.8/0.2, etc. Find the point where mixing stops hurting.

Once pieces are understood, compose them. Only then does the full 2-block local-loss experiment make sense.

### Feedback loop optimization

The previous cycle took ~30 minutes per condition (13 min run + corpus loading + analysis). Per dictation: if a 2-minute run on a simpler task answers the same question, use that instead.

Options for faster feedback:
- **TinyShakespeare** or equivalent small dataset (no 90-second corpus load)
- **Synthetic data** (e.g. random sequences, or a CFG grammar) — can't be overparameterized, fast to generate
- **Fewer steps** — if the mechanism signal appears in 200 steps, don't run 2000
- **Smaller model** — d_model=64 or 128 instead of 256

### Synthetic task investigation

Per dictation 2026-05-26-3: investigate what synthetic task would be good for this project. Key property needed: data isn't the bottleneck, so architectural differences are visible. CFG/grammar tasks are interesting (strict nesting, hierarchical structure) but not prescribed. This is an investigation, not a commitment.

---

## Decision state (what's been decided)

| Decision | Outcome | Constraint it imposes |
|---|---|---|
| Temporal_window 2-block | Branch 1 confirmed (Δ=+0.042, 3 seeds) | Trajectory info uniquely helps intended arch |
| 4-block follow-up | **Stop-loss fired** (conditions 4+5 fail) | No more intended-architecture rescue |
| Bridge_detach | Clearly worse (gap +0.027, 3 seeds) | Gradient IS needed for block specialization |
| **Dictation 2026-05-26-2** | **Redirect** | **Only compare against legitimate alternatives** |
| **Local predictive loss v1** | **Mechanism works, architecture harms it** | **Decompose before iterating** |
| **Dictation 2026-05-26-3** | **Process: build from pieces** | **No more assembled-system experiments until pieces are tested** |
| Timebox | ~4 days remaining | Use fastest possible feedback loops |

**Traps to avoid:**
- Testing the whole assembly again with one tweak (violates "build from pieces")
- Using WikiText-103 when a 2-second synthetic dataset would answer the same question
- Running 2000 steps when 200 steps would show the signal
- Defaulting to what we were doing before (no inertia)
- Changing too many things at once (one variable per piece-test)

---

## Key references

- `VISION.md` — stakeholder requirements
- `ROADMAP.md` — Pathway 3 (local learning) is active
- `dictations/2026-05-26-3.md` — decompose, synthetic tasks, optimize feedback loops
- `dictations/2026-05-26-2.md` — the redirect to local predictive loss
- `research/questions/multi-timestep-architecture/README.md` — Max's architecture vision (distributional, Wasserstein)
- `runs/local_predictive.py` — the experiment script (has the confounds, but pieces can be extracted)
- `research/daily/2026-05-26.md` — today's write-up of the confounded experiment

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

Follow PROCESS.md. Current position: **CONCEPTUAL CLARIFICATION — decomposition.** Must design the isolated piece-tests before implementing. First: figure out the fastest experimental setup (dataset, model size, step count) that can prove each piece in seconds.
