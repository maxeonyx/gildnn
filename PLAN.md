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

### Critical conceptual finding (2026-05-26 afternoon)

**The "same input, same timestep" composed test is fundamentally broken.** When both blocks see the same tokens at the same time, and block 1 is trained to predict block 0's output, block 1 has NO information advantage. Its lateral can only ever be a redundant imperfect copy of what block 0 already computes. Mixing that in (at any α > 0) is always harmful — confirmed at α=0.5 (WikiText: +0.163) and α=0.1 (synthetic: +0.021).

This is not a hyperparameter problem. It's a DESIGN problem. The local MSE objective pushes toward redundancy, not complementarity.

### What's needed: information asymmetry

For block 1 to help block 0, block 1 must know something block 0 does not. This aligns with Max's actual architecture vision (dictation 2026-05-25-1): different blocks run at different temporal rates, so higher blocks have "bigger picture" context that lower blocks lack.

### The correct next experiment: split-input test

**Cleanest honest composed test (< 60 seconds):**
- Task depends on TWO pieces of information: block 0 sees piece A, block 1 sees piece B
- Block 0 cannot solve the task alone; it needs block 1's lateral
- Conditions:
  1. Block 0 alone (should fail at the part requiring B)
  2. Block 0 + block 1 with lateral (should improve)
  3. Block 0 + block 1 with SHUFFLED lateral (should be same as alone — control)

**Concrete design options:**
- (a) Token stream where next-token depends on (current_token, context_summary). Block 0 sees current token. Block 1 sees context summary.
- (b) Generate sequences from a grammar; block 0 sees current position, block 1 sees stack state / nesting depth.
- (c) Simple XOR task: y = f(a, b), block 0 sees a, block 1 sees b. Minimal but tests the mechanism clearly.

(c) is the fastest to implement but doesn't test language modeling. (a) or (b) are closer to the real use case. Recommend (a) since it's a natural LM task with split information.

### Pieces already verified (done)

All 4 piece-tests pass (`runs/piece_tests.py`):
1. Fixed-target MSE: PASS (drop 274x, cosine 0.9986)
2. Mixing damage: PASS (α=0.5 catastrophic, α=0.1 preserves)
3. Moving target: PASS (only 17% worse than frozen)
4. Topology: PASS (correct wiring 31x better)

### Composed test that failed (design was broken)

`runs/composed_local_loss.py`: baseline 0.666, treatment 0.686 (+0.021). Block 1 learned (MSE 0.034→0.004) but couldn't help because it had no unique information.

### Feedback loop status

Piece-tests: 0.2–1.6 seconds each ✓  
Composed test: 25–53 seconds each ✓  
**Feedback loop is now FAST.** The bottleneck is conceptual clarity, not runtime.

---

## Decision state (what's been decided)

| Decision | Outcome | Constraint it imposes |
|---|---|---|
| Temporal_window 2-block | Branch 1 confirmed (Δ=+0.042, 3 seeds) | Trajectory info uniquely helps intended arch |
| 4-block follow-up | **Stop-loss fired** (conditions 4+5 fail) | No more intended-architecture rescue |
| Bridge_detach | Clearly worse (gap +0.027, 3 seeds) | Gradient IS needed for block specialization |
| **Dictation 2026-05-26-2** | **Redirect** | **Only compare against legitimate alternatives** |
| **Local predictive loss v1** | **Mechanism works, design is broken** | **Block 1 needs information advantage over block 0** |
| **Dictation 2026-05-26-3** | **Process: build from pieces** | **Decompose, fast feedback, synthetic tasks** |
| **Same-input composed test** | **Fundamentally cannot show benefit** | **Must give block 1 different/extra information** |
| Timebox | ~4 days remaining | Use fastest possible feedback loops |

**Traps to avoid:**
- Same-input-same-timestep experiments (block 1 can never help without unique info)
- Using WikiText-103 when a synthetic dataset would answer the same question faster
- Running longer hoping the signal appears (it can't if the design is broken)
- Defaulting to what we were doing before (no inertia)
- Treating the "prediction → redundancy" problem as a hyperparameter issue

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
| **Local predictive loss v1** | **3** | **Mechanism works, design is broken** | **Same-input → redundancy → can't help** |
| **Piece tests** | **3** | **All 4 PASS** | **MSE, mixing, tracking, topology all work individually** |
| **Composed (same input)** | **3** | **Baseline 0.666, treatment 0.686 (+0.021)** | **No information advantage = always harmful** |

---

## The agent's working loop

Follow PROCESS.md. Current position: **CONCEPTUAL CLARIFICATION → redesign.** The pieces work but the composition requires information asymmetry. Next: implement the split-input test (block 0 sees part of the input, block 1 sees the rest). This directly tests whether local predictive lateral communication can transfer useful information.
