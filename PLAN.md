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

## What to do next

### Key finding: staged training is required (2026-05-26 evening)

The full experimental chain this session:

| Experiment | Result | Insight |
|---|---|---|
| Piece tests (4 isolated) | All PASS | MSE, mixing, tracking, topology all work alone |
| Same-input composed (WikiText α=0.5) | +0.163 worse | Block 1 has no info advantage |
| Same-input composed (synthetic α=0.1) | +0.021 worse | Same problem, less mixing damage |
| Split-input (block 0 sees A, block 1 sees B) | 100% | Lateral works when info is split |
| Context-asymmetry co-trained | 12% (chance) | Co-training fails — lateral can't track unstable source |
| Context-asymmetry recon co-trained | 12% (chance) | Better local loss doesn't fix co-training |
| **Context-asymmetry staged** | **100%** | **Block 1 first → freeze → train lateral = works perfectly** |

**The architecture concept IS validated.** All three requirements are confirmed:
1. Information asymmetry (block 1 must have info block 0 lacks) ✓
2. Local self-supervised loss (reconstruction, not "predict block 0") ✓
3. Staged training (block 1 stabilizes BEFORE lateral learns) ✓

### What's next: apply staged training to a real LM task

The next honest test toward the actual architecture:
1. **Short-context block 0** (e.g., ctx=4-8) trained with CE on next-token prediction
2. **Long-context block 1** (e.g., ctx=32-64) trained with reconstruction loss independently
3. **Connect and fine-tune:** freeze block 1, connect lateral, train lateral_proj + output_head to use block 1's context

Task: any character-level LM where longer context helps (even TinyShakespeare — natural language inherently benefits from more context). Synthetic tasks could also work.

Key question this test answers: **On a real LM task, does a locally-trained context block improve short-context predictions?** This is the actual multi-timestep architecture in miniature.

### Alternative: curriculum warmup instead of hard staging

Instead of freeze-then-connect, try:
- `lateral_scale = 0` for first 300 steps (block 1 trains reconstruction)
- Ramp `lateral_scale` from 0 to target over steps 300-500
- This allows continuous training without hard phase boundaries
- More biologically plausible (connections strengthen as representations stabilize)

### Feedback loop status

All experiments complete in <15 seconds. Inline Python tests for rapid iteration. The feedback loop is tight.


---

## Decision state (what's been decided)

| Decision | Outcome | Constraint it imposes |
|---|---|---|
| **Staged training required** | **Co-training fails, staged works (100%)** | **Block 1 must stabilize before lateral is used** |
| **Information asymmetry required** | **Same-input always harms, split-input works** | **Blocks must see different information** |
| **Reconstruction > predict-block-0** | **Predict-block-0 pushes redundancy** | **Local loss should encode own input, not imitate other block** |
| Dictation 2026-05-26-2 | Redirect | Only compare against legitimate alternatives |
| Dictation 2026-05-26-3 | Process: build from pieces | Decompose, fast feedback, synthetic tasks |
| Timebox | ~4 days remaining | Use fastest possible feedback loops |

**Traps to avoid:**
- Co-training blocks from scratch (use staged/warmup instead)
- "Predict block 0's state" as local loss (pushes toward redundancy)
- Same-input-same-timestep experiments (no information advantage possible)
- Testing the full complex architecture before the minimal staged version works on real LM

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
| **Piece tests** | **3** | **All 4 PASS** | **MSE, mixing, tracking, topology all work individually** |
| **Split-input** | **3** | **100% accuracy** | **Lateral works perfectly with info asymmetry** |
| **Context-asymmetry co-trained** | **3** | **FAIL (12% = chance)** | **Co-training dynamics prevent learning** |
| **Context-asymmetry staged** | **3** | **100% accuracy** | **Block 1 first → freeze → lateral works** |

---

## The agent's working loop

Follow PROCESS.md. Current position: **EXPERIMENT DESIGN — real LM task.** The synthetic mechanism is fully validated. Next: apply staged training to a real character-level LM task with different context sizes (short-ctx block 0 + long-ctx block 1). This is the minimum bridge to the actual multi-timestep architecture.
