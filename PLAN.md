# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-26)

**Temporal_window experiment COMPLETE ✅ — BRANCH 1 CONFIRMED (strongest result in the project).**

| Variant | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|---------|---------|---------|---------|------------|
| B0 | 2.501 | 2.545 | 2.471 | 2.506 ± 0.037 |
| H8 | 2.370 | 2.369 | 2.390 | 2.376 ± 0.012 |
| C8 | 2.404 | 2.414 | 2.435 | 2.418 ± 0.016 |

| Delta | Seed 42 | Seed 43 | Seed 44 | Mean |
|-------|---------|---------|---------|------|
| Δ_trajectory (C8-H8) | +0.034 | +0.045 | +0.045 | **+0.042** |
| Δ_augmented (B0-H8) | +0.132 | +0.176 | +0.081 | +0.130 |

All 3 seeds concordant. Mean Δ_trajectory (+0.042) is nearly 3× the pre-registered threshold (0.015).

**4-block follow-up COMPLETE ✅ — STOP-LOSS FIRES (conditions 4+5 fail).**

| Variant | Seed 42 | Seed 43 | Seed 44 | 3-seed mean |
|---------|---------|---------|---------|-------------|
| A_all | 1.836 | 1.839 | 1.859 | 1.845 |
| C8_all | 1.822 | 1.852 | 1.901 | 1.858 |
| W8_all | 1.803 | 1.825 | 1.830 | 1.820 |

| Condition | Result | Details |
|---|---|---|
| 1. mean(W8)−mean(A) ≤ -0.015 | **PASS** (-0.025) | |
| 2. mean(W8)−mean(C8) ≤ -0.015 | **PASS** (-0.039) | |
| 3. All seeds concordant | **PASS** | W8 < both controls on all 3 seeds |
| 4. W8 upper blocks load-bearing | **FAIL** | Block 1: +0.42 ✅, Block 2: +0.02/+0.017 ⚠️, Block 3: ~0 ❌ |
| 5. A_all upper blocks spectator-like | **FAIL** | Block 1: +0.07 (load-bearing without temporal window!) |

**Interpretation:** Temporal window STRONGLY amplifies block 1 (0.07→0.42, 6x improvement) but doesn't rescue deeper blocks. Block 3 remains spectator in both conditions. This is "1-2 blocks found a niche, not broad architectural rescue." The hierarchy of usefulness degrades with distance from token source.

**Unexpected finding:** A_all block 1 is load-bearing (+0.07) even WITHOUT temporal window. This means the intended architecture isn't as broken as the "universal spectator" theory suggested — at least the nearest block can find some role. Investigate why (likely: even a single stale-state lateral from block 0 provides some information for readout_mode="all").

**C8 extreme seed sensitivity (std=0.039 vs W8 std=0.014)** confirms C8 is mathematically degenerate (diagonal subspace collapse). Real trajectory provides robust signal; fake capacity provides noise.

**Action:** Per decision tree → surrogate bridge_detach launched. No intended-architecture follow-up (criterion not met). Trajectory diagnostics NOT run (would be for strong-positive only).

**Bridge_detach experiment COMPLETE ✅** (2026-05-26). 3 seeds concordant "clearly worse" (mean gap +0.027 nats). Readout pattern perfectly reproduced: full_backprop U-shaped (block 3 cost +0.35-0.42); detached front-loaded (block 3 cost +0.07 all seeds).

**Warmup→detach RUNNING** (PID 20952, launched 10:13 NZST 2026-05-26). ~2.5 hours GPU time (6 runs × 20K steps). Reads bridge_detach baselines from report.json. ETA ~12:35 NZST.

**Results (3 seeds COMPLETE):**

| Seed | full_backprop | detached | Gap |
|---|---|---|---|
| 42 | 1.765 | 1.794 | **+0.029** |
| 43 | 1.756 | 1.794 | **+0.038** |
| 44 | 1.796 | 1.811 | **+0.016** |
| **Mean** | **1.772 ± 0.017** | **1.799 ± 0.008** | **+0.027** |

**Concordant: all 3 seeds "clearly worse."** Mean gap +0.027 nats. Full variance (std 0.017) higher than detached (std 0.008). Seed 44 shifted up for both variants but same pattern.

- Trajectory: identical for 12K steps → gap opens at 15K → accelerates through 20K

**Readout ablation pattern (most revealing finding):**

| Block | Full s42 | Full s43 | Full s44 | Det s42 | Det s43 | Det s44 |
|---|---|---|---|---|---|---|
| 0 | +1.12 | +0.93 | +0.96 | +1.11 | +0.84 | +1.05 |
| 1 | +0.10 | +0.20 | +0.10 | **+0.44** | **+0.68** | **+0.41** |
| 2 | +0.05 | +0.03 | +0.05 | +0.16 | +0.11 | +0.16 |
| 3 | **+0.42** | **+0.35** | **+0.42** | +0.07 | +0.07 | +0.07 |

**Story (perfectly consistent across all 3 seeds):**
- Full_backprop: "U-shaped" — block 3 develops large contribution (0.35-0.42). Lateral gradient enables useful upper-block computation.
- Detached: "front-loaded" — block 3 nearly useless (exactly 0.07 all seeds), block 1 picks up partial slack (+0.41-0.68 vs 0.10-0.20). Without lateral gradient, model settles for shallower solution.
- Hypothesis (not proven): senders don't learn what to send without gradient → receivers get uninformative lateral input → can't develop useful specialization.

**⚠️ Methodological learning:** The pre-registered "lateral-zeroing cost as discriminator" is uninformative in this regime. Both variants produce catastrophic values (10^11-10^15) because both were trained WITH forward laterals — eval-time removal is architectural mutilation, not the gentle ablation assumed. C_old's 0.030 gap compared two SEPARATELY TRAINED models; this is fundamentally different. The readout pattern is the actual discriminator.

**Decision per pre-registration:** "Clearly worse" (3/3 concordant) → **warmup→detach diagnostic** (pre-registered in local-learning README). Launched 2026-05-26 10:10 NZST.

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
- **Clean weight-sharing isolation test** — DATA LOST. Experiment ran but artifacts disappeared during refactoring sanity-check. Provisional 1-seed result: tied_shared=1.910 vs distinct_matched≈1.85 (~0.06 nats gap, above 0.05 "struggles" threshold). Needs 3-seed rerun to confirm — queued as follow-up after temporal_window.
- Any custom CUDA concurrency beyond the Graph approach
- Self-prediction (computation compression)
- ~~Clean dynamic-depth measurement~~ — DONE (f51db7c). Oracle speedup 1.96×, 71.5% tokens harmed by depth-8. Pathway 5 worth pursuing.
- ~~Shared experiment runner~~ — DONE (f8579d5). Extracted to `core/run_utils.py`, net -1089 lines.
- **Sanity-check safety** — DONE (e3e0620). Sanity checks now write to temp dirs, preventing interference with live runs.

**Pathway 3 status:**
- **In surrogate architecture (token_injection=all):**
  - Multi-block works with all-injection (C_old is -0.021 better than single-block)
  - Lateral-only (token_injection=block0) fails with hardcoded 0.5 (+0.014 worse) AND with zero-init gates (+0.245 worse)
  - **C_old ablation POSITIVE: laterals are load-bearing (Δ=+0.030, all 4 blocks contribute)**
  - **Bridge_detach COMPLETE ✅: 3 seeds concordant "clearly worse" (mean gap +0.027). Readout: full U-shaped, detached front-loaded.**
  - **Warmup→detach RUNNING** (PID 20952, ETA ~12:35 NZST). Tests whether gradient needed continuously or just for bootstrapping.
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

## GPU queue (explicit ordering)

1. ~~bridge_detach~~ — **COMPLETE ✅** (3 seeds concordant, mean gap +0.027)
2. **warmup_detach** — RUNNING (PID 20952, launched 10:13 NZST). ETA ~12:35 NZST. Then →
3. **tied_sharing rerun** — `runs/tied_sharing.py`. Independent core question, data lost. ~45 min. Then →
4. **What the data says** — budget permitting, escalation depends on warmup→detach outcome

## What should happen next

Pick from this list based on cheapest honest test. These connect to specific roadmap pathways.

**⚠️ Surrogate vs intended architecture:** Experiments using `token_injection=all` test a SURROGATE architecture (all blocks get fresh tokens). Max's corrected intended architecture is `token_injection=block0` — only block 0 gets tokens, upper blocks depend on propagation delay ([dictation 2026-05-23-7](dictations/2026-05-23-7.md)). Surrogate results are useful for isolating mechanisms but do NOT validate the intended architecture. Guard against surrogate findings silently becoming "the architecture."

| Priority | Experiment | Pathway | Architecture | Why |
|---|---|---|---|---|
| 1 | **C_old ablation** | 3 | surrogate | **COMPLETE. Laterals load-bearing (Δ=+0.030).** |
| 2 | **Clean weight-sharing isolation** — shared vs distinct feedforward, both with all-injection | 1 | surrogate | Isolates weight sharing from routing. **DATA LOST — needs 3-seed rerun.** Provisional 1-seed: tied_shared=1.910 vs distinct≈1.85 (~0.06 gap). |
| 3 | **Temporal window** — 2-block, readout_mode="last", B0/H8/C8 conditions | 3 | **intended** | **COMPLETE ✅ — BRANCH 1 CONFIRMED** (Δ_trajectory=+0.042). 4-block follow-up COMPLETE — STOP-LOSS (conditions 4+5 fail). |
| 4 | **Iteration-benefit measurement** — eval shared model at 1,2,4,8 iterations | 1/5 | surrogate | Only if tied_sharing positive. Tests dynamic depth in simplified regime. |
| 5 | **Bridge experiment (detach_lateral)** — full-backprop vs detached-lateral | 3 | surrogate | **COMPLETE ✅** — clearly worse (gap +0.027, 3 seeds concordant). Readout pattern confirmed. |
| 6 | **Custom CUDA concurrency** — persistent kernels or fused dispatch | 2 (async) | infra | Next step after the 28% CUDA Graph result. |
| 7 | **Dynamic depth (clean measurement)** | 5 | TBD | **COMPLETE.** Oracle speedup 1.96×, oracle-best beats depth-8 by 0.285 nats. Pathway 5 worth pursuing. Next: predictability test. |

**Priority rationale:** Temporal_window (priority 3) is ranked ABOVE iteration-benefit (priority 4) because it directly tests the corrected intended architecture — block0-only with propagation delay. Making this work is upstream of everything else: if upper blocks have no forward role in the intended architecture, local learning and dynamic depth in that architecture are moot. Iteration-benefit is still valuable but tests only the surrogate regime.

**Tied_sharing caveat:** The clean test shares only feedforward block weights — per-block `token_mixes` and `block_mixes` remain distinct. A positive result proves "shared feedforward processing is viable with position-specific routing," NOT "same weights applied N times with identical routing" (the full Pathway 1 hypothesis). It is a reasonable first step, not a final answer. Tied_sharing needs a 3-seed rerun (data lost) — queued for when GPU is free and if it's still worth the timebox budget.

### Temporal_window decision tree (pre-planned)

When results arrive, use pre-registered thresholds from `research/questions/temporal-window/README.md` (Δ_trajectory ≥ 0.015 + all seeds concordant = "clear H8 > C8"; |Δ| < 0.005 = "≈"). Branches 4–5 are overlays on 1–3.

1. **H8 > B0 AND H8 > C8** (trajectory uniquely helps): Proves temporal diversity creates a niche. Next: scale to 4 blocks with window + `readout_mode="all"`, ablate upper blocks — test voluntary usefulness without forced readout.
2. **H8 > B0 BUT H8 ≈ C8** (capacity/interface, not history): Extra projected input helps but it's not temporal diversity. Next: test simpler capacity/interface fixes on intended architecture. Deprioritize window-size sweeps.
3. **H8 ≈ B0 ≈ C8** (nothing helps): Current rescue mechanism fails. Pivot back to surrogate-architecture work: bridge/detach_lateral (Pathway 3) or iteration-benefit (Pathway 1/5). Park intended-architecture rescue as open design problem.
4. **B0 < 2.0** (overlay): Intended architecture isn't fundamentally broken → less urgency, can go straight to learning-signal tests on B0 if no unique H8 win.
5. **Seed instability** (overlay): Effect not decision-grade. Rerun with more seeds before theorizing. May be a Pathway 9 (stability) issue.

### ⚠️ Stop-loss rule (timebox discipline)

**Note: This is agent-imposed timebox discipline, not Max's expressed preference.** Max's dictations treat the intended architecture as central to the project. This rule exists because with ~5 days remaining and one GPU, unlimited intended-architecture experiments crowd out other high-information work. Max may override this if he disagrees with the budget allocation.

**The intended-architecture rescue budget is fixed at two experiments total:**
1. `temporal_window` forced-readout test ✅ (complete, positive)
2. `4-block` voluntary-readout scaling test (running)

**No further intended-architecture rescue-mechanism experiments after the 4-block test, regardless of outcome.**

**Exception — predeclared and narrowly scoped:** A single intended-architecture `bridge_detach` follow-up is allowed **only** if the 4-block test meets the full "trajectory-specific voluntary rescue" criterion (all 5 conditions below). This is allowed because it's not another rescue-mechanism search — it's a test of a different hypothesis (learning rule) on a predeclared-positive substrate. If the criterion is not met, stop-loss fires immediately.

Rationale: The surrogate architecture already has load-bearing laterals (Δ=+0.030), all blocks contribute, and multiple high-information experiments are immediately available (bridge/detach_lateral, gradient radius sweep, iteration-benefit). With ~10–12 experiments remaining in the timebox, spending more than 2 on intended-architecture rescue is not justified unless evidence is strong.

### Post-4-block decision tree (revised after adversarial review, 2026-05-26)

**Threshold note:** The 0.015 val_loss threshold was calibrated in a 2-block forced-readout setting. In this 4-block voluntary-readout setting, it is used as a **discipline threshold** (stop/go), not as a precise scientific validation boundary. It may be too strict for detecting a real voluntary-usefulness effect AND too lenient for claiming full architecture validation. Use it as a coarse guard.

**Ablation limitation:** Per-block zeroing at eval proves "the trained model depends on this block's contribution." It does NOT prove the block learned trajectory-specific information (vs capacity/copy/noise). To prove trajectory learning specifically, would need history-order shuffle or temporal-branch zeroing — not available in this experiment. Interpret ablation as "used," not "learned from history."

#### Advance to intended `bridge_detach` ONLY IF ALL are true:

1. `mean(W8_all) - mean(A_all) <= -0.015` (W8 clearly beats spectator baseline)
2. `mean(W8_all) - mean(C8_all) <= -0.015` (W8 clearly beats capacity control)
3. Both comparisons concordant across all 3 seeds (same sign)
4. All upper blocks in W8_all are load-bearing (`abl_k >= 0.02` for blocks 1, 2, 3)
5. A_all remains spectator-like (upper blocks NOT load-bearing in A_all)

**If all 5 met → STRONG POSITIVE:**
- Run trajectory-specificity diagnostics on W8_all checkpoints (eval-only, no new training):
  1. **Repeat-last-history**: replace window buffer with most-recent state repeated ×8. If performance barely changes → capacity trick, not trajectory learning. (trivial)
  2. **Window permutation**: shuffle temporal order of history slots. If big degradation → model learned from temporal order specifically. (easy)
  3. **Age-selective ablation**: keep only recent-1, recent-2, drop-oldest-half. If recent-1 recovers most benefit → recency shortcut, not broad trajectory use. (easy)
- If diagnostics confirm trajectory learning → intended-architecture bridge_detach (one experiment, predeclared)
- If diagnostics suggest capacity trick → treat as D1 (capacity helps, not trajectory). Pivot to surrogate.
- Eval-only dynamic-depth measurement on trained W8_all (cheap, no new training)
- Report as "strong provisional positive" — note threshold was ported from different regime

#### If ANY of those 5 conditions fails → stop-loss fires:

**Specific interpretations for common failure modes:**

| Outcome | Interpretation | Action |
|---------|---------------|--------|
| W8 > A_all, W8 ≈ C8_all | Capacity/interface helps, NOT trajectory specifically. Causal claim unsupported. | Pivot to surrogate bridge_detach. Future intended work must test interface design, not temporal info. |
| C8_all > W8_all | Trajectory actively worse than capacity control. Hypothesis directly disfavored. | Hard stop. Surrogate bridge_detach. |
| W8 blocks load-bearing but W8 ≈ A_all or W8 worse | Blocks learned something they depend on, but it doesn't help the task. "Used but not useful." | Negative for rescue. Surrogate bridge_detach. |
| W8 > both controls but only 1-2 blocks load-bearing | One block found a niche, not broad architectural escape from spectator collapse. | Partial. Surrogate bridge_detach. |
| W8 > both controls, concordant, blocks useful, BUT A_all also not spectator | Control validity broken — experiment can't test "rescue from spectator." | Reframe. Investigate why A_all escaped spectators before running mechanism work. |
| Mean effects positive but non-concordant (seeds disagree) | With n=3, non-concordance = insufficient stability evidence. | Partial/unstable. Surrogate bridge_detach. |
| Everything between thresholds (all |Δ| in 0.005-0.015 range) | Ambiguous. Below preregistered bar. Exactly where self-serving interpretation risk is highest. | Treat as null. Surrogate bridge_detach. |

**Default action when stop-loss fires:** Surrogate bridge_detach → bootstrap-vs-signal discriminator (warmup→detach) → remaining budget on other pathways.

**Recommended remaining-budget split (after bridge_detach, ~5 days left):**
- 3–4: surrogate local-learning ladder (warmup→detach → predictive coding → scale/Wasserstein)
- 1: weight-sharing rerun (independent core question, data lost, cheap)
- 1: concurrency sweep (strengthens Pathway 2 story beyond single data point)
- 1: buffer for reruns/surprises

**If bridge_detach POSITIVE (Scenario B — detached ≈ full):** Local-learning ladder collapses. Redirect budget:
- 1: gradient-radius sweep (detach / k=1 / k=2 / full — measures minimum useful nonlocality)
- 1: scale detached to more blocks / longer context (robustness vs tiny-surrogate luck)
- 1: async + local composition test (the raison d'être: does local learning survive stale/concurrent execution?)
- Weight-sharing + dynamic-depth + concurrency still run unconditionally.

**Traps to avoid:** more intended-architecture rescue (stop-loss fired), custom CUDA kernels (infra not finding), starting new pathways (4/6/7/9/10/11 can't meaningfully start in 5 days).

### Why token_injection="block0" fails (theory, 2026-05-25)

Upper blocks are downstream of a stale bottleneck controlled by block 0, while block 0 already solves the task directly. They're **redundant delayed decoders**, not complementary experts. Evidence suggests this is primarily a forward-architecture problem (full backprop didn't help — spectators with full gradients still spectate), though Max keeps the learning-signal question explicitly open ([dictation 2026-05-23-5](dictations/2026-05-23-5.md): "we need to try many variants").

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
| **Dynamic-depth clean measurement** | **5** | **Oracle speedup 1.96×, oracle-best 0.285 nats better than depth-8** | **Worth pursuing. 71.5% of tokens harmed by forcing depth-8. Val speedup > train → not overfitting. Surprisingly flat oracle histogram. Single seed (42).** |
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
- **Pathway 3 (Local Learning):** The forward architecture problem (spectator blocks) is now explained: lateral-only multi-block fails because upper blocks are redundant delayed decoders. However, **C_old ablation (token_injection=all) proves laterals are genuinely load-bearing (Δ=+0.030, 2 seeds concordant, all 4 blocks contribute to readout).** And **temporal_window proves trajectory info uniquely rescues the intended architecture (Δ=+0.042, 3 seeds concordant)** — upper blocks become useful when given exclusive temporal information. Pathway is alive and advancing. Next: bridge experiment (detach_lateral) — can blocks learn useful communication without cross-block gradient? 4-block follow-up (running) tests voluntary usefulness at scale.
- **Pathway 5 (Dynamic Depth):** **WORTH PURSUING.** Clean measurement (2026-05-26): oracle speedup 1.96× at δ=0.01, oracle-best beats depth-8 by 0.285 nats (17.3%), 71.5% of tokens harmed by forcing depth-8. Val speedup > train speedup → NOT overfitting. Oracle histogram surprisingly flat (not dominated by any single depth — 15.2% best at d1, 28.5% best at d8). Next: predictability test (can shallow state predict optimal depth?).
- **Pathway 8 (Multi-Rate):** Prior weak-positive — inductive bias confirmed at ctx=32. Needs longer context to be meaningful.
- **Pathway 9 (Norm-Preserving):** May be relevant if the tied_8iter seed failure turns out to be a gradient stability issue through many iterations. Muon or orthogonal parameterization might fix it. But first need to isolate whether it's actually a stability issue vs a routing issue.

---

## Fading strands to watch (from first-principles re-derivation, 2026-05-25)

Max's original vision (dictation 2026-05-24-1) has two mechanisms that are currently not on the active experiment ladder:

1. **Attention-based routing between modules** — the original note says "how do inputs get aggregated by a node? They use attention." Current architecture uses fixed mix-add, not attention. This is fine as simplification while testing basic block viability, but should be revisited once the intended architecture works at all.

2. **Loss prediction as decision mechanism** — "a prediction head that is designed to predict the loss of another prediction head." Central to Pathways 4/5/6, not yet active. Correctly deprioritized (needs working architecture first) but should not be forgotten.

Neither is actionable now, but they represent significant chunks of the vision that need to become active once temporal_window (or equivalent) validates the intended architecture.

3. **Graph topology** — Max's original vision is a graph, not a stack: "I don't know if that graph helps, but I kind of think it might" ([dictation 2026-05-23-5](dictations/2026-05-23-5.md)). Current architecture is a fixed stack. Graph topology is a major design dimension that hasn't been explored yet — correctly deprioritized while basic block viability is uncertain, but should not be forgotten.

4. **Hierarchical dynamic tokenization** (Pathway 7) — stacked autoencoders with learned chunk boundaries, loss prediction determining where to cut ([dictation 2026-05-24-4](dictations/2026-05-24-4.md)). Bottleneck "useful not just as context for the level above, but also useful as context for the following chunk" at the same level. Correctly deprioritized (needs working base architecture) but represents a major piece of the multi-timescale vision.

---

## Multi-timestep architecture — future theoretical direction (dictation 2026-05-25-1)

Max's LATEST design thinking (from a 2026-05-25 conversation) significantly evolves the architecture beyond the current ParallelDiagonalModel. Full write-up: `research/questions/multi-timestep-architecture/README.md`.

Key departures from current implementation:
- **Stream carries distributions** (diagonal Gaussians: μ, σ per dimension), not point vectors
- **Local loss = Wasserstein distance** (block predicts left neighbor's next distribution)
- **No cross-block gradients needed** by design (each block trains on its own prediction error)
- **Blocks are temporal edges** on a 2D grid (lateral positions × timesteps)
- **Combining function is shared/tied** across all positions (evolved wiring, not per-block learning)

**Relationship to current experiments:** The dictation says "This doesn't invalidate current work. It provides the theoretical direction for what comes AFTER the current experiments confirm the basics." Current experiments confirm:
- ✅ Lateral connections carry useful information — **surrogate architecture** (C_old: Δ=+0.030)
- ✅ Temporal trajectory is uniquely useful information — **intended architecture, forced-readout 2-block** (temporal_window: Δ=+0.042)
- ✅ Whether blocks can be voluntarily useful with trajectory — **intended architecture, 4-block** (stop-loss fired: block 1 rescues but blocks 2-3 stay spectators)
- ✅ Whether blocks can learn without cross-block gradient — **surrogate** (bridge_detach: "clearly worse," gap +0.027, 3 seeds)

**If bridge_detach (shared-adjoint) fails:** The Wasserstein distributional local loss is the natural escalation — it provides a RICHER local training signal (predict left neighbor) rather than relying on the weak shared adjoint alone. This is the most important connection between current work and the multi-timestep direction.
