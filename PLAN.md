# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current operational state (2026-05-27, 01:00 NZST)

**GPU: BUSY.** Running larger-data experiment (PID 11788). 4 distinct vs 4 recurrent on 900K training chars (9× more data). Log: `runs/recurrent_depth_largedata_run.log`. Report: `experiments/tinyshakespeare/artifacts/recurrent_depth_lm/report_largedata.json`. Expected ~25 min.

**Session results so far:**
1. 8-iter scaling COMPLETE: recurrent_8 val_loss=1.615, distinct_8 val_loss=1.787. Gap GROWS from -0.092 (N=4) to -0.172 (N=8). Stability perfect. Documented in README.
2. Larger-data experiment launched to test whether advantage is purely regularization.

**Key question being answered right now:** Does the recurrent advantage persist when there's enough data that the distinct model can't easily overfit?

**Daily report 2026-05-27:** NOT YET WRITTEN (it's only 01:00).

**Integration:** `core/tied_readout.py` contains validated model architecture.

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

### Multi-block result and resolution (2026-05-26, 18:30–19:00)

**Initial finding (800 steps):** blocks redundant at 800 steps.

| Condition (800 steps) | val_loss | Δ |
|---|---|---|
| block0_alone | 1.8561 | — |
| one_block_mid (ctx=32) | 1.8189 | -0.037 |
| one_block_long (ctx=128) | 1.8458 | -0.010 |
| two_blocks (32+128) | 1.8198 | -0.036 |

**Pretrain+freeze ablation (4000 pretrain + 800 finetune):**

| Condition | val_loss | Δ from 800-step baseline |
|---|---|---|
| one_block_long pretrained | 1.7043 | **-0.152** |
| one_block_mid pretrained | 1.6930 | **-0.163** |
| two_blocks pretrained | 1.6567 | **-0.199** |

**Co-train 4800 steps (same total compute, no freezing):**

| Condition | val_loss | Δ from 4800-step baseline |
|---|---|---|
| block0_alone @ 4800 | 1.7115 | — |
| two_blocks co-trained @ 4800 | **1.6435** | **-0.068** |

**Resolution:** Co-train 4800 beats pretrain+freeze (1.6435 < 1.6567). The redundancy finding was purely a training budget artifact. With enough steps, co-training self-organizes and blocks become complementary. **No staging, no freezing required.**

Key insights:
1. Interior blocks need ~4000+ steps to mature (800 is severely undertrained)
2. More width (d_model=128) partially helps but doesn't fully fix it (-0.024 vs -0.010)
3. More layers (2) doesn't help (-0.010 → -0.010)
4. Horizon specialization (offset=4) doesn't help (local task too hard)
5. Simply training longer is the best solution — co-training naturally discovers complementary representations

**Architecture implication:** The multi-block architecture WORKS. The key variable is training budget, not architecture tricks. Give interior blocks enough optimization and they develop useful, complementary representations naturally.

### Architecture validation + scale-up COMPLETE

**Architecture (from dictation 2026-05-26-5):**
- Shared normalized token embeddings (weight-tied readout)
- Normalized block outputs (L2-norm before addition)
- Addition-based lateral combination (lateral_scale=0.2 needed even with normalization)
- CE local loss for interior blocks (cosine/L2 worse)
- Temperature = 0.07–0.10

**Results at both scales:**

| Scale | block0_alone | two_blocks | Δ | Notes |
|---|---|---|---|---|
| Tiny (d=64, 1L, 4800 steps) | 1.7522 | 1.6646 | -0.088 | normalized, no scale needed |
| **Large (d=128, 2L, 20K steps)** | **1.6342** | **1.5893** | **-0.045** | normalized, scale=0.2 needed |

Text generation confirms qualitative difference: two_blocks maintains play structure (speaker labels, dialogue rhythm), block0_alone drifts into non-words.

Script: `runs/tied_readout_lm.py`

### What's next

Options (choose one):
1. **Longer context** — increase output block from 4→16 chars, interior to 64/256. Does the effect grow?
2. **Integrate into core/** — clean reusable architecture. "Integrate before experimenting."
3. **Move toward recurrent** — the vision's actual architecture: blocks fire at different rates, state persists
4. **Test noise on laterals** — only meaningful with temporal structure (option 3)

### Recurrent lateral experiment — STALE LATERALS DON'T WORK (2026-05-26, late evening)

Script: `runs/recurrent_lateral_lm.py`. Fixed lateral timing per dictation 2026-05-26-6.

**Finding: Stale laterals (one-position delay) from detached recurrent state provide NO benefit.**

| Setup | Result |
|---|---|
| lateral_scale=1.0, 1200 steps, 3 seeds | ALL WORSE (Δ = +0.12 to +0.56) |
| lateral_scale=0.2, 4800 steps, 1 seed | Neutral (Δ = +0.017) |

**Why it fails:** Block 1 is trained for local CE (predict its own next-char). It has no incentive to produce state that's useful for block 0 one step later. Without BPTT or temporal credit assignment, the state captures patterns that overfit the training data (train loss 1.48 < block0's 1.55) but don't generalize (val loss 1.63 > block0's 1.61).

**Previous "positive" result was timing bug:** Same-timestep lateral = extra sequential depth, which trivially helps. Parallel (stale) lateral = no benefit.

**Fixed-window still works** because it provides IMMEDIATE info asymmetry (128 chars vs 4), requiring no temporal accumulation or communication learning.

### What's next (revised)

The stale recurrent lateral as implemented doesn't work. But this doesn't kill multi-rate — it redirects it. Multi-rate needs to provide info asymmetry through **context accumulation at firing time** (the slow block processes a buffer of accumulated positions when it fires), not through step-by-step state compression.

Options:
1. ~~**Multi-rate with context buffer** — slow block fires every K positions, processes all K windows at once (= K*4 effective context). This is the fixed-window approach but amortized over time.~~ **TESTED — DOES NOT WORK.** Sequential regime is incompatible with laterals. See below.
2. **Truncated BPTT** — allow gradient through a few state steps. Tests whether temporal credit assignment unlocks the mechanism.
3. **Communication objective** — train block 1 to predict something useful for block 0, not just its own local CE.

### Multi-rate with context buffer — SEQUENTIAL REGIME INCOMPATIBLE (2026-05-26 night)

Script: `runs/multirate_buffer_lm.py`. Tests whether a slow block can help block 0 when processing sequences position-by-position.

**Finding: The sequential training regime is fundamentally incompatible with the lateral mechanism, regardless of context size.**

| Condition | slow_ctx | val_loss | Δ | Read |
|---|---|---|---|---|
| block0_alone (baseline) | — | 1.6154 | — | — |
| direct_ctx32_refresh1 (control) | 32 | 1.6154 | 0.000 | Lateral load-bearing (ablation +0.29) but net-zero benefit |
| buffered_ctx32_refresh8 | 32 | 2.2051 | +0.590 | Catastrophically worse |
| direct_ctx128_refresh1 (rescue attempt) | 128 | 1.6795 | +0.039 | Even 128-char context fails in sequential mode |

**Why:** In sequential processing, adjacent positions overlap 127/128 (or 31/32). The slow lateral is nearly constant between positions → block 0 treats it as a static bias → no content-specific information extracted.

**Contrast:** The same mechanism with 128-char context gives Δ=-0.088 in the window-based regime (tied_readout_lm.py), where each sample is independently drawn and the lateral varies meaningfully.

**Age-bucket analysis** (buffered condition): age 0 (fresh) = 1.94, age 1 = 2.19, age 2-7 ≈ 2.22-2.26. Even freshly-fired is catastrophically bad — problem is training dynamics (block 0 trained mostly with stale signal), not just inference staleness.

Full write-up: `research/questions/multirate-buffer/README.md`

### CRITICAL: All experiments use wrong sequence length (2026-05-27, Max)

**Problem:** Every experiment uses `SHORT_CONTEXT = 4` (4-char input windows). The architecture is an RNN — context length doesn't affect per-step compute, only batch size and weight size matter. We should be testing with **thousands of tokens** of sequence length.

**Why this matters:**
- With 4 chars there's no temporal structure to discover — recurrence is invisible
- Truncated BPTT means: forward through entire sequence (10K+ tokens), backprop through K steps (32-64). Hidden state carries long-range information even without full BPTT gradient.
- The "sequential regime incompatible with laterals" finding may be an artifact of tiny sequences where adjacent positions overlap 3/4 chars
- The "stale laterals don't work" finding may reverse when there's actually thousands of tokens of history to compress

**What was wrong about the transformer comparison:** We hobbled the RNN to match the transformer's interface (short windows). The correct comparison is: transformer at its sweet spot (128-256 token attention) vs RNN at its sweet spot (thousands of tokens, truncated BPTT) at matched parameter count.

**Action:** Redesign experiments around long-sequence unrolling with truncated BPTT. This is the priority.

---

### What's actually next (2026-05-26 night, post-multirate)

The complete picture of negative results:
1. **Stale recurrent laterals** — don't work (no temporal credit assignment)
2. **Buffered/held laterals in sequential mode** — don't work (near-constant signal, training dynamics)
3. **Sequential training regime** — incompatible with laterals entirely (even fresh per-position fails)

**The only working configuration: window-based training (random independent samples) with fresh per-position slow block computation.**

Remaining options:
1. **Truncated BPTT** — would provide temporal credit assignment. But would still be in sequential mode, which we just showed is incompatible. Would need to be combined with window-based training.
2. **Integration** — the fixed-window mechanism (tied_readout_lm.py) WORKS. Integrate it into `core/` as the proven baseline. "Integrate before experimenting."
3. **Different pathway** — the multi-rate / temporal reuse direction has been thoroughly tested and doesn't work without temporal credit assignment. Consider switching to a different roadmap pathway (e.g. Pathway 4: Computation Compression, Pathway 5: Dynamic Depth, Pathway 7: Hierarchical Tokenization).
4. **Longer context for fixed-window** — test whether increasing block 0 from 4→16 chars and slow from 128→512 grows the effect.

**Recommendation:** Option 2 (integrate) is highest priority per AGENTS.md rule "Integrate before experimenting." The fixed-window architecture works reliably; it should be in `core/` before starting new experiments.

### Inference-time caching test — IMMEDIATE COLLAPSE (2026-05-26 night)

Even the "inference only" version doesn't work. Trained the model normally (window-based, fresh), then at eval varied slow-block cache interval K:

| K | val_loss | Δ from K=1 |
|---|---|---|
| 1 | 1.6618 | — |
| 2 | 2.0918 | +0.43 |
| 4 | 2.4155 | +0.75 |
| 8 | 2.5797 | +0.92 |

**Architectural insight:** CE local objective produces prediction-specific output (tuned to predict ONE specific next char). This output is useless for any other position. Multi-rate is fundamentally incompatible with CE-trained laterals — not because of training dynamics, but because of what the objective produces.

This FULLY closes multi-rate for Pathway 8 with local CE objectives. Any form of temporal reuse (training or inference) fails immediately.

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
| **Asymmetry scaling** | **3** | **ctx 4/128: -0.023** | **Effect grows with more info asymmetry** |
| **3-seed confirmation** | **3** | **mean -0.037 ± 0.028** | **All 3 seeds positive; mechanism confirmed on real language** |
| **Multi-block (32+128)** | **3** | **two_blocks ≈ one_mid** | **Blocks redundant at this scale; long block underfits (capacity mismatch)** |
| **Pretrain+freeze ablation** | **3** | **two_blocks: -0.199** | **Interior blocks need more training; pretrained blocks are COMPLEMENTARY** |
| **Co-train 4800 steps** | **3** | **two_blocks: 1.6435** | **BEATS pretrain+freeze. No staging needed, just more steps.** |
| **Recurrent lateral (stale)** | **3→1** | **Stale laterals: NEUTRAL at best (Δ=+0.017 with scale=0.2, 4800 steps)** | **Detached state doesn't learn useful communication. Fixed-window approach still best for info asymmetry.** |
| **Multi-rate buffer (sequential)** | **8→3** | **Sequential regime fails: Δ=0.00 at ctx32, Δ=+0.04 at ctx128** | **Sequential training incompatible with laterals — near-constant signal from overlapping windows. Window-based training required.** |
| **Inference-time caching** | **8** | **Immediate collapse: K=2 gives Δ=+0.43** | **CE-trained blocks produce prediction-specific output, useless for any other position. Multi-rate fundamentally incompatible with CE local objective.** |
| **Recurrent depth (4 iter vs 4 distinct)** | **1** | **Recurrent WINS: 1.6315 vs 1.7234 (Δ=-0.092)** | **Weight sharing = regularization at this scale. 3.6× fewer params, 20% faster. Stable through 4 iterations.** |

---

## The agent's working loop

Follow PROCESS.md. Current position: **PATHWAY 1 CONFIRMED at this scale.** Recurrent depth beats distinct layers (1.63 vs 1.72). Multi-rate (Pathway 8) fully closed. Integration done. Next: iteration scaling (8, 16 iters), width scaling, or composition with Pathway 3 (local learning + recurrence).
