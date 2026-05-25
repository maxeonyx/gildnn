# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-25)

**Tied-depth experiment COMPLETE.** All 4 variants × 2 seeds finished. Full results:

| Variant | Seed 42 | Seed 43 | Mean | Std | Params |
|---------|---------|---------|------|-----|--------|
| A_single | 1.832 | 1.845 | 1.838 | 0.007 | 2,850,422 |
| tied_8iter | 1.798 | 2.549 | 2.173 | 0.375 | 2,850,422 |
| distinct_matched | 1.818 | 1.817 | 1.817 | 0.001 | 2,847,908 |
| distinct_rich | 1.750 | 1.744 | 1.747 | 0.003 | 4,690,820 |

Transformer baseline: 1.592 ± 0.003 (2.86M params, 4 layers)

**⚠️ Honest interpretation (post adversarial review):**

The comparison between tied_8iter and distinct_matched is **confounded** by multiple differences:
1. Token injection: tied_8iter=block0 (tokens enter once), distinct_matched=all (fresh tokens every block)
2. Topology: 1 block × 8 internal steps vs 8 blocks × 1 step
3. Block shape: one d=256 block vs eight d=146 blocks

**What we CAN say:**
- tied_8iter shows severe seed sensitivity under this recipe (1 of 2 seeds failed badly — plateaued at 2.55 from step 5K)
- distinct_matched is extremely stable (std 0.001) and beats A_single
- distinct_rich is the best overall (1.747) — more params help, expected
- On seed 42 where tied_8iter DOES converge, it actually beats distinct_matched (1.798 vs 1.818)
- The gap to transformer remains large: best variant (distinct_rich) is still 0.155 nats behind

**What we CANNOT say:**
- "Weight sharing causes instability" — not isolated (token routing confound)
- "Tied depth has a fundamental stability problem" — 2 seeds is not enough, and the tiny-rung result used a different architecture (actual tied-depth transformer, not ParallelDiagonalModel)
- "This contradicts the tiny-scale finding" — different model class, different dataset, can't compare directly

**What this teaches:**
- The specific configuration "single block, 8 internal iterations, token_injection=block0" is NOT robust
- The failure may be about lack of anchoring signal (no fresh tokens after iteration 1) rather than weight sharing per se
- A clean weight-sharing isolation test would need: shared weights WITH token_injection=all (8 blocks with tied weights, each getting tokens)

**C_old ablation COMPLETE (2026-05-25, 22:27 NZST).** Lateral connections are clearly load-bearing.

| Variant | Seed 42 | Seed 43 | Mean |
|---------|---------|---------|------|
| C_lateral (upward) | 1.765 | 1.756 | 1.760 |
| C_isolated (isolated) | 1.794 | 1.785 | 1.790 |
| **Δ** | **+0.029** | **+0.029** | **+0.030** |

Pre-registered outcome: **"Clearly lateral helps"** (Δ ≥ 0.015, both seeds concordant).
Readout ablation: ALL 4 blocks are load-bearing (block0_only degrades by +1.5 nats).
Next step per pre-registration: **bridge experiment** (detach_lateral, priority 4).

**Gated experiment — prior negative result.** B_gated (4-block, zero-init gates) was +0.245 nats WORSE than A_single at WikiText-103 ctx=128. Cold-start problem: zero-init gates starve upper blocks of information.

**What we have:**
- Working experiment infrastructure (training loop, eval, multi-seed, ablation, JSONL logs, CUDA graphs)
- Transformer baseline: val_loss 1.643, 186K params, TinyShakespeare ctx=32 (in base-experiments/)
- RNN baseline: val_loss 1.711, 186K params, TinyShakespeare ctx=32 (in base-experiments/)
- Transformer baseline: val_loss 1.592 ± 0.003, 2.86M params, WikiText-103 ctx=128
- **Pathway 1 (tiny rung):** tied-depth = distinct-layer at matched params (mean diff 0.0002 nats, 3 seeds) — confirmed for tied-depth transformer at TinyShakespeare
- **Pathway 1 (WikiText-103):** ParallelDiagonalModel tied_8iter shows seed sensitivity (1/2 seeds failed). Comparison confounded by token injection routing. See honest interpretation above.
- **distinct_matched (8 blocks, token_injection=all):** stable and effective, 1.817 ± 0.001
- Evidence: strict-local collapses, semi-local barely helps, prediction target matters more than topology
- Evidence: multi-rate [1,2,4,8] provides inductive bias (better per-step val_loss)
- Evidence: CUDA Graph concurrency gives 28% speedup; stale reads don't hurt quality
- 10.8x CUDA graph training speedup in core/
- **Gated WikiText-103 result** — cold-start problem with zero-init gates, blocks useless
- `.gitignore` now blocks `*.pt` files (model weights never committed)

**What we DON'T have:**
- **Clean weight-sharing isolation test** — RUNNING (PID 22508, expected ~02:30 NZST). tied weights WITH token_injection=all to separate sharing from routing
- Any custom CUDA concurrency beyond the Graph approach
- Self-prediction (computation compression)
- Clean dynamic-depth measurement (the probe was methodologically flawed)
- ~~Shared experiment runner~~ — DONE (f8579d5). Extracted to `core/run_utils.py`, net -1089 lines.

**Pathway 3 status:**
- **In surrogate architecture (token_injection=all):**
  - Multi-block works with all-injection (C_old is -0.021 better than single-block)
  - Lateral-only (token_injection=block0) fails with hardcoded 0.5 (+0.014 worse) AND with zero-init gates (+0.245 worse)
  - **C_old ablation POSITIVE: laterals are load-bearing (Δ=+0.030, all 4 blocks contribute)**
  - Bridge experiment (detach_lateral) is the next surrogate-architecture Pathway 3 test
- **In intended architecture (token_injection=block0):**
  - ALL configurations tested so far FAIL (spectators, cold-start)
  - Temporal_window is the fix hypothesis — gives upper blocks exclusive trajectory information
  - **This is the critical gap.** Max's intended architecture doesn't work yet. Fixing it is upstream of local learning.

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

Every gate is a separate subagent review that can send you back. See PROCESS.md for full details.

---

## What should happen next

Pick from this list based on cheapest honest test. These connect to specific roadmap pathways.

**⚠️ Surrogate vs intended architecture:** Experiments using `token_injection=all` test a SURROGATE architecture (all blocks get fresh tokens). Max's corrected intended architecture is `token_injection=block0` — only block 0 gets tokens, upper blocks depend on propagation delay ([dictation 2026-05-23-7](dictations/2026-05-23-7.md)). Surrogate results are useful for isolating mechanisms but do NOT validate the intended architecture. Guard against surrogate findings silently becoming "the architecture."

| Priority | Experiment | Pathway | Architecture | Why |
|---|---|---|---|---|
| 1 | **C_old ablation** | 3 | surrogate | **COMPLETE. Laterals load-bearing (Δ=+0.030).** |
| 2 | **Clean weight-sharing isolation** — shared vs distinct feedforward, both with all-injection | 1 | surrogate | Isolates weight sharing from routing. **RUNNING (PID 22508).** |
| 3 | **Temporal window** — 2-block, readout_mode="last", B0/H8/C8 conditions | 3 | **intended** | Does trajectory info create a niche for upper blocks in block0-only? The cheapest direct test of Max's corrected architecture. **READY TO LAUNCH** — `runs/temporal_window.py` verified on CPU. Pre-registered in `research/questions/temporal-window/`. |
| 4 | **Iteration-benefit measurement** — eval shared model at 1,2,4,8 iterations | 1/5 | surrogate | Only if tied_sharing positive. Tests dynamic depth in simplified regime. |
| 5 | **Bridge experiment (detach_lateral)** — full-backprop vs detached-lateral | 3 | surrogate | Can blocks learn useful laterals without cross-block gradient? Pre-registered in local-learning README. |
| 6 | **Custom CUDA concurrency** — persistent kernels or fused dispatch | 2 (async) | infra | Next step after the 28% CUDA Graph result. |
| 7 | **Dynamic depth (clean measurement)** | 5 | TBD | Preliminary probe methodology was flawed. Needs clean redo. |

**Priority rationale:** Temporal_window (priority 3) is ranked ABOVE iteration-benefit (priority 4) because it directly tests the corrected intended architecture — block0-only with propagation delay. Making this work is upstream of everything else: if upper blocks have no forward role in the intended architecture, local learning and dynamic depth in that architecture are moot. Iteration-benefit is still valuable but tests only the surrogate regime.

**Tied_sharing caveat:** The clean test shares only feedforward block weights — per-block `token_mixes` and `block_mixes` remain distinct. A positive result proves "shared feedforward processing is viable with position-specific routing," NOT "same weights applied N times with identical routing" (the full Pathway 1 hypothesis). It is a reasonable first step, not a final answer.

**After tied_sharing finishes:** Temporal_window is next REGARDLESS of tied_sharing outcome. Tied_sharing is a Pathway 1 test; temporal_window is a Pathway 3 test of the intended architecture. They answer different questions. Record tied_sharing results in `research/questions/wide-recurrent-vs-transformer/README.md`, update this file, then proceed to temporal_window implementation.

### Why token_injection="block0" fails (theory, 2026-05-25)

Upper blocks are downstream of a stale bottleneck controlled by block 0, while block 0 already solves the task directly. They're **redundant delayed decoders**, not complementary experts. This is a forward-architecture problem, not a training/gradient problem.

**The fix hypothesis:** Give upper blocks a temporal WINDOW of lower-block states (`temporal_window=4,8` — already exists in model.py). This provides exclusive trajectory information (how block 0's state has been moving) that block 0 can't easily exploit in a single step. With `readout_mode="last"` on a 2-block model, block 1 is forced to be load-bearing (no collapse possible). See full pre-registration: `research/questions/temporal-window/README.md`.

**Grounding:** Max in [dictation 2026-05-24-5](dictations/2026-05-24-5.md): "Block one should learn to predict something about block zero that block zero couldn't already know or wouldn't need to know therefore. For example, you know maybe block one's output is dependent on input from a longer time ago?"

**Design insight:** Use 2 blocks with `readout_mode="last"` (NOT 4 blocks with `readout_mode="all"`). This eliminates collapse, chain-scaling, and multi-block interaction confounds. The question reduces to: does trajectory info help the forced-readout upper block produce better predictions?

**This supersedes "local learning in working config" as priority 5** — fixing the forward architecture is upstream of fixing the learning rule. Local learning is meaningless on blocks that have no forward role.

---

## Completed work (reference)

| What | Pathway | Result | Interpretation |
|---|---|---|---|
| Transformer baseline | all | val_loss 1.643, 186K params, TinyShakespeare ctx=32 | Done, in base-experiments/ |
| RNN baseline | all | val_loss 1.711, 186K params | Done, in base-experiments/ |
| **Tied-depth vs transformer (tiny rung)** | **1** | **Identical: mean diff 0.0002 nats** | **Weight sharing is free. Viable architecture.** |
| **Iteration scaling (6, 8, 12)** | **1** | **No instability on single seed; quality peaks ~8** | **Tiny-rung only (TinyShakespeare, tied-depth transformer). Does NOT directly transfer to ParallelDiagonalModel at WikiText-103 scale.** |
| **Tied-depth WikiText-103 (4 variants × 2 seeds)** | **1** | **tied_8iter seed-sensitive (1/2 failed); distinct_matched stable (1.817 ± 0.001)** | **Confounded comparison: token_injection differs. Cannot isolate weight sharing as cause. On successful seed, tied beats distinct (1.798 vs 1.818). Clean isolation test needed.** |
| Per-token depth heterogeneity | 5 | Inconclusive | Heterogeneity exists but methodology flawed (non-standard eval frame, ε too loose). Hint only. |
| **Propagation-delay 2-block (tiny)** | **3** | **Spectator** | **Block B adds nothing at TinyShakespeare ctx=32. Hardcoded 0.5 mixing harmful; zero-init gate fixes ceiling but B stays closed.** |
| **Multi-block corrected at WikiText-103 ctx=128** | **3** | **Spectator** | **4-block corrected (hardcoded 0.5) is +0.014 worse than single-block. 4-block old (token_injection=all) is -0.021 better. Blocks help when fed fresh tokens; lateral-only with 0.5 mixing fails.** |
| **Gated multi-block at WikiText-103 ctx=128** | **3** | **Cold-start failure** | **4-block with zero-init gates is +0.245 worse than single-block. Gates starve upper blocks of signal — worse than hardcoded 0.5. Gate 3 opened negatively (suppressive). Experiment uninformative about original question due to cold-start confound.** |
| **C_old lateral ablation (WikiText-103 ctx=128)** | **3** | **Laterals clearly load-bearing (Δ=+0.030, 2 seeds)** | **Prerequisite for local learning met. All 4 blocks contribute. Bridge experiment unlocked.** |
| Multi-rate [1,2,4,8] | 8 | Spectator problem | Block 0 sees all tokens → no specialization |
| Closed-loop variants A-J | — | Marginal (-0.006 to -0.012) | Architecturally incoherent. Not on any pathway. Done. |
| CUDA graph training | infra | 10.8x speedup | In core/ |
| CUDA Graph concurrency | 2 | 28% speedup | Hardware CAN do concurrent execution |
| Stale-read quality cost | 2 | +0.005 ± 0.005 nats | Negligible (95% CI crosses zero) |
| Backend choice | infra | PyTorch + torch.compile | Decided |
| **Transformer baseline (WikiText-103 ctx=128)** | **1** | **val_loss 1.592 ± 0.003, 2.86M params, 2 seeds** | **External anchor for Pathway 1. 0.240 nats better than A_single.** |

---

## Roadmap evidence to suggest to Max

(Agent records evidence here; Max decides whether to update ROADMAP.md)

- **ROADMAP factual correction needed:** Baselines table says "val_loss 1.643, 187K params, WikiText-103 char-level" — but 1.643/187K is the **TinyShakespeare** baseline. The actual WikiText-103 transformer baseline is **1.592 ± 0.003, 2.86M params, 4 layers, 2 seeds**. Suggest updating the baselines table.
- **Pathway 1 (Wide Recurrent):** Weight sharing is free at tiny scale (tied-depth transformer, TinyShakespeare, 3 seeds). At WikiText-103 scale, tested a DIFFERENT architecture (ParallelDiagonalModel with tied_8iter): 1 of 2 seeds failed badly. However, the comparison is confounded — tied_8iter uses token_injection=block0 while the comparator (distinct_matched) uses token_injection=all. **Cannot attribute the failure to weight sharing specifically.** A clean isolation test is needed: shared weights with token_injection=all. On the successful seed, tied_8iter actually beat distinct_matched (1.798 vs 1.818), suggesting the inductive bias of weight sharing CAN help — it's the robustness that's the problem. Pathway remains alive but needs a cleaner test.
- **Pathway 2 (Async):** Prior positive — 28% concurrency via CUDA Graphs, stale reads don't hurt. Next: custom CUDA.
- **Pathway 3 (Local Learning):** The forward architecture problem (spectator blocks) is now explained: lateral-only multi-block fails because upper blocks are redundant delayed decoders. However, **C_old ablation (token_injection=all) proves laterals are genuinely load-bearing (Δ=+0.030, 2 seeds concordant, all 4 blocks contribute to readout).** Pathway is alive. Next: bridge experiment (detach_lateral) — can blocks learn useful communication without cross-block gradient? Also: temporal_window experiment (pre-registered) — can trajectory info create a niche for upper blocks even with token_injection=block0?
- **Pathway 5 (Dynamic Depth):** Now enabled by Pathway 1 iteration scaling. Per-depth losses show clear variation — some tokens probably benefit more from extra iterations than others. First measurement (per-token variance) not yet done.
- **Pathway 8 (Multi-Rate):** Prior weak-positive — inductive bias confirmed at ctx=32. Needs longer context to be meaningful.
- **Pathway 9 (Norm-Preserving):** May be relevant if the tied_8iter seed failure turns out to be a gradient stability issue through many iterations. Muon or orthogonal parameterization might fix it. But first need to isolate whether it's actually a stability issue vs a routing issue.

---

## Fading strands to watch (from first-principles re-derivation, 2026-05-25)

Max's original vision (dictation 2026-05-24-1) has two mechanisms that are currently not on the active experiment ladder:

1. **Attention-based routing between modules** — the original note says "how do inputs get aggregated by a node? They use attention." Current architecture uses fixed mix-add, not attention. This is fine as simplification while testing basic block viability, but should be revisited once the intended architecture works at all.

2. **Loss prediction as decision mechanism** — "a prediction head that is designed to predict the loss of another prediction head." Central to Pathways 4/5/6, not yet active. Correctly deprioritized (needs working architecture first) but should not be forgotten.

Neither is actionable now, but they represent significant chunks of the vision that need to become active once temporal_window (or equivalent) validates the intended architecture.
