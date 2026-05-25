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

**C_old ablation RUNNING.** PID 20548, launched 20:46 NZST. Trains C_lateral (upward topology) vs C_isolated (no lateral connections, same params). Tests whether lateral communication is actually used in the multi-block-with-token-injection architecture.

**Seed 42 COMPLETE (both variants):**
- C_lateral: val_loss **1.765**, accuracy 0.503
- C_isolated: val_loss **1.794**, accuracy 0.491
- **Δ = +0.029 nats (lateral better)** — above 0.015 "clearly lateral helps" threshold

**Seed 43 C_lateral COMPLETE:** val_loss **1.756** (better than seed 42)
Seed 43 C_isolated now running. Expected finish ~22:17 NZST.

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
- **Clean weight-sharing isolation test** — need tied weights WITH token_injection=all to separate sharing from routing
- C_old lateral ablation (IN PROGRESS — C_old ablation running now)
- Any custom CUDA concurrency beyond the Graph approach
- Self-prediction (computation compression)
- Clean dynamic-depth measurement (the probe was methodologically flawed)

**Pathway 3 status:**
- Only token_injection=all (C_old) makes multi-block useful at this scale
- Lateral-only (token_injection=block0) fails with hardcoded 0.5 (+0.014 worse) AND with zero-init gates (+0.245 worse)
- Local learning is untestable until we have a regime where blocks help under full backprop
- Key open question: does C_old actually USE lateral connections, or is it just an ensemble?
- **C_old ablation running now** — will answer this

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

Pick from this list based on cheapest honest test. These connect to specific roadmap pathways:

| Priority | Experiment | Pathway | Why |
|---|---|---|---|
| 1 | **C_old ablation** — C_lateral vs C_isolated, train-time structural comparison | 3 | **IN PROGRESS (PID 20548).** Does the multi-block architecture USE lateral connections, or is it just an ensemble? |
| 2 | **Clean weight-sharing isolation** — 8 blocks with SHARED weights + token_injection=all vs distinct_matched | 1 | The tied_8iter comparison was confounded. Need same routing, only sharing differs. |
| 3 | **Temporal window** — 2-block, readout_mode="last", temporal_window∈{0,4,8} | 3 | Does trajectory information create a niche for upper blocks? **PRE-REGISTERED** in `research/questions/temporal-window/`. |
| 4 | **Bridge experiment (detach_lateral)** — if C_old shows laterals used | 3 | Full-backprop vs detached-lateral in C_old regime. Pre-registered in local-learning README. |
| 5 | **Custom CUDA concurrency** — persistent kernels or fused dispatch | 2 (async) | Next step after the 28% CUDA Graph result. |
| 6 | **Dynamic depth (clean measurement)** | 5 | Preliminary probe showed heterogeneity but methodology was flawed. Needs clean redo. |

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
- **Pathway 3 (Local Learning):** Confidence **decreased**. Every attempt at lateral-only multi-block has failed: hardcoded 0.5 (+0.014 worse), zero-init gates (+0.245 worse, cold-start trap). Only token_injection=all (C_old, -0.021 better) makes blocks useful — but that may not involve lateral communication at all (might just be an ensemble). **C_old ablation now running** to determine this. Pathway is blocked pending that result.
- **Pathway 5 (Dynamic Depth):** Now enabled by Pathway 1 iteration scaling. Per-depth losses show clear variation — some tokens probably benefit more from extra iterations than others. First measurement (per-token variance) not yet done.
- **Pathway 8 (Multi-Rate):** Prior weak-positive — inductive bias confirmed at ctx=32. Needs longer context to be meaningful.
- **Pathway 9 (Norm-Preserving):** May be relevant if the tied_8iter seed failure turns out to be a gradient stability issue through many iterations. Muon or orthogonal parameterization might fix it. But first need to isolate whether it's actually a stability issue vs a routing issue.
