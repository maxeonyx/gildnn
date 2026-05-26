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

### Key finding: LOCAL OBJECTIVE QUALITY is the key variable (2026-05-26 late evening)

Two iterations of the staged char LM experiment (`runs/staged_char_lm.py`):

**Iteration 1 — reconstruction at all positions (FAILED):**

| Condition | Val loss | Delta |
|---|---|---|
| block0_alone | 1.8736 | — |
| staged_lateral | 1.8875 | +0.0140 (worse) |
| cotrained_lateral | 1.8644 | -0.0092 (noise) |
| shuffled_staged | 1.8927 | +0.0191 (worse) |

Diagnosis: reconstruction converged to near-zero (0.0002) because self-attention + positional embeddings trivially copies each input token to its position. The last-position state doesn't encode useful long-context info because there's no pressure for it to.

**Iteration 2 — last-position-only CE (WORKS):**

| Condition | Val loss | Delta |
|---|---|---|
| block0_alone | 1.8736 | — |
| staged_lateral | 1.8606 | **-0.0129** |
| cotrained_lateral | 1.8563 | **-0.0172** |
| shuffled_staged | 1.8848 | +0.0112 (confirms content matters) |

**Conclusions:**
1. The mechanism WORKS on real language when the local objective pressures the EXPORTED vector
2. **Staging is NOT required** — co-training with gradient isolation works equally well (slightly better even)
3. The earlier synthetic result (staging required) was actually compensating for a WEAK objective
4. Gradient isolation (detach lateral from block 1's perspective) IS required — block 1 never receives output-head gradient

### Revised three requirements

1. **Information asymmetry** — block 1 must have info block 0 lacks (longer context) ✓
2. **Last-position pressure** — local loss must target the EXPORTED representation, not all positions ✓
3. **Gradient isolation** — lateral detached; no task gradient flows into interior blocks ✓

**Staging is NOT a requirement.** Remove from decision table.

### What's next: increase effect size and confirm with seeds

The current effect is -0.013 to -0.017 nats (1 seed). Options to increase confidence and effect size:

1. **Multi-seed run** (3 seeds) — confirm the -0.017 is real, not noise
2. **Increase context asymmetry** — try block 0 ctx=4, block 1 ctx=128 (more info gap)
3. **Larger models** — d_model=128 (more capacity to use the lateral info)
4. **Longer training** — 2000+ steps (may not have converged yet)
5. **Curriculum warmup** — lateral_scale ramps 0→0.2 over first 200 steps (alternative to hard connect)

The cheapest discriminating next step: **multi-seed (3 seeds) on the current setup** to confirm the finding is real.

### Feedback loop status

All experiments complete in <12 seconds per condition. The feedback loop is very tight.


---

## Decision state (what's been decided)

| Decision | Outcome | Constraint it imposes |
|---|---|---|
| **Objective must pressure exported vector** | **All-position recon trivial; last-pos CE works** | **Local loss targets the last-position state specifically** |
| **Staging NOT required** | **Co-training works with good objective + gradient isolation** | **No hard phase boundary needed** |
| **Information asymmetry required** | **Same-input always harms, split-input works** | **Blocks must see different information** |
| **Gradient isolation required** | **Lateral detached from block 1; no output-head gradient flows back** | **Interior blocks optimize own head only** |
| **"Predict block 0" is wrong** | **Pushes redundancy** | **Local loss should be self-contained** |
| Dictation 2026-05-26-2 | Redirect | Only compare against legitimate alternatives |
| Dictation 2026-05-26-3 | Process: build from pieces | Decompose, fast feedback, synthetic tasks |
| Timebox | ~4 days remaining | Use fastest possible feedback loops |

**Traps to avoid:**
- Reconstruction at all positions (trivially solved, doesn't pressure exported vector)
- "Predict block 0's state" as local loss (pushes redundancy)
- Same-input-same-timestep experiments (no information advantage possible)
- Assuming staging is required (it's not — was compensating for bad objective)
- Using shuffled controls without confirming they differ from baseline (content must matter)

---

## Key references

- `VISION.md` — stakeholder requirements
- `ROADMAP.md` — Pathway 3 (local learning) is active
- `dictations/2026-05-26-3.md` — decompose, synthetic tasks, optimize feedback loops
- `dictations/2026-05-26-2.md` — the redirect to local predictive loss
- `research/questions/multi-timestep-architecture/README.md` — Max's architecture vision (distributional, Wasserstein)
- `runs/staged_char_lm.py` — the working char-level LM experiment (last-pos CE, gradient isolation)
- `research/daily/2026-05-26.md` — today's write-up

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
| **Staged char LM (recon)** | **3** | **+0.014 (worse)** | **All-position reconstruction is trivial — doesn't pressure exported vector** |
| **Staged char LM (last-pos CE)** | **3** | **-0.017 (better!)** | **Last-position CE works; co-training works too; staging not required** |

---

## The agent's working loop

Follow PROCESS.md. Current position: **CONFIRM FINDING — multi-seed.** The mechanism works on real char-level LM. Next: run 3 seeds on the co-trained last-pos CE condition to confirm the -0.017 is real.
