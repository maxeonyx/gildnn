# Local learning variants

## Core question

Serves [dictation 2026-05-23-5](../../../dictations/2026-05-23-5.md): "how can I get local learning, i.e. enabling parallelism?"

Max's framing: "It doesn't have to be totally local. We can be using backpropagation through a local neighborhood of blocks. Then if we can do that, we can parallelize training significantly more."

## Prediction target: current simplification vs Max's intent

Per [dictation 2026-05-24-5](../../../dictations/2026-05-24-5.md): "Block one should learn to predict something about block zero that block zero couldn't already know or wouldn't need to know therefore. For example, maybe block one's output is dependent on input from a longer time ago?"

**All experiments below use a simplified prediction target** — cosine similarity to block 0's full hidden state. This is adequate for testing the *mechanism* (does feedback help? does locality kill? does width compose?) but is NOT what Max ultimately wants. The current target violates Max's framing because block 0 already knows its own state.

After the mechanism questions are resolved (locality, width, coupling), the next stage is changing WHAT is predicted:
- Block 0's future state (temporal look-ahead)
- A compression of block 0's history from further back (longer-range context)
- A higher-level pattern spanning multiple timesteps

See [`research/questions/local-learning-theory/README.md`](../local-learning-theory/README.md) for the full literature/theory analysis of prediction targets.

---

## Status: ANSWERED for N=2; N=3 seed-sensitive, investigating

**Neighborhood-local (CE flows through block interfaces) is the minimum viable locality.** The full N=2 experiment series proved this:

| Variant | Mean val_loss | vs A | Conclusion |
|---------|--------------|------|-----------|
| C_closed_loop (semi-local) | 1.665 | -0.006 | CE through interface HELPS |
| A_single | 1.670 | — | baseline |
| B_spectator | 1.677 | +0.006 | extra params alone hurt |
| E_grounded (strict-local + local CE) | 1.696 | +0.026 | stable but hurts |
| D_strict_local | 3.047 | +1.377 | COLLAPSED |

**The mechanism:** The CE-through-interface gradient does two jobs:
1. **Prevents collapse** — keeps predictions grounded (local CE can also do this ✓)
2. **Selects useful prediction components** — teaches predictor WHAT to predict for the task (local CE cannot do this ✗)

Only job (2) actually improves performance. E_grounded proves this: it solves collapse (via local CE) but can't solve selection (no feedback gradient). Result: stable but slightly worse than baseline.

**Now testing:** I (phase offsets) — two rate-2 helpers at different temporal phases with per-helper prediction losses. Tests whether width scales when coupling and staleness are both removed. See Phase 4 section below.

## Key insight: "parallel" means three different things

Three properties get conflated under "local learning." They're independent:

1. **Forward parallelism** — can blocks compute their forward passes concurrently? Requires that block i's input doesn't depend on block i-1's output within the same timestep. Our architecture already has this: block 1 reads a *detached copy* of block 0's previous state, not its current one.

2. **Backward locality** — do parameter updates require gradients flowing through distant blocks? This is what "local learning rule" usually means. Semi-local (current v3) fails here: CE backprops through the feedback path into block 1. Strict-local passes: one extra detach and no gradient crosses the boundary.

3. **Update latency** — can block i's optimizer step happen before downstream blocks finish their backward pass? This requires backward locality plus no shared optimizer state. With strict-local + per-block optimizers, block 1 can step as soon as its own pred_loss backward completes — it doesn't wait for block 0's CE backward.

Most "local learning" papers address only (2). Genuine training parallelism — Max's goal of "parallelize training significantly more" — requires all three. The architecture already gives us (1). **We now know (2) costs performance** — strict-local hurts or collapses. The viable path is neighborhood-local: backprop through a small neighborhood (the interface), which limits but doesn't eliminate parallelism.

## Experimental results

### Phase 1: v3 semi-local — DONE ✓

Both seeds confirm C < A (mean -0.006). Mechanism works under semi-local conditions.

Evidence: [`report_v3.json`](../../../experiments/wikitext_103/artifacts/closed_loop_prediction/report_v3.json)

### Phase 2a: strict-local — DONE ✓ (COLLAPSE)

Same architecture, one extra `detach()` on the feedback path. Result: **catastrophic collapse**, both seeds.

| Variant | Mean val_loss | vs A |
|---------|--------------|------|
| A_single | 1.670 | — |
| **D_strict_local** | **3.047** | **+1.377** |

Collapse trajectory:
1. Steps 1–2K: D was *better* than A (early predictions helpful regardless of alignment)
2. Steps 3–13K: D much worse, slowly improving (predictions drift, co-adaptation)
3. Steps 14–20K: Collapse (pred_loss → 0, accuracy frozen 0.19, gain → -0.17)

**Root cause:** full-state cosine prediction treats all hidden dimensions equally. Without CE shaping, block 1 predicts "whatever's easiest" rather than "what helps the task." Block 0 co-adapts (gain goes deeply negative → hard dependency on predictions), then when predictions drift from task-relevance, block 0 can't recover.

Evidence: [`report_strict_local.json`](../../../experiments/wikitext_103/artifacts/closed_loop_prediction/report_strict_local.json)

### Phase 2b: task-grounded strict-local (E_grounded) — DONE ✓ (STABLE BUT HURTS)

Same strict-local architecture as D, but block 1 gets its own next-token CE head (local CE loss). Tests whether D's collapse was the target or the locality.

| Variant | Mean val_loss | vs A |
|---------|--------------|------|
| A_single | 1.670 | — |
| **E_grounded** | **1.696** | **+0.026** |

Key metrics (seed 42 final):
- gain: -0.052 (predictive coding, same sign as C)
- ablation gap: +0.233 (predictions deeply integrated — block 0 depends on them)
- block1 accuracy: 0.346 (well above random — block 1 genuinely learned the task)
- pred_loss: 0.178 (stable, non-degenerate)

**What this proves:** D's collapse was caused by the ungrounded target, not by locality. Local CE grounds block 1 successfully — no collapse. But predictions still slightly hurt because the predictor doesn't know which dimensions block 0 NEEDS. It predicts "full hidden state" uniformly, and the noise in irrelevant dimensions outweighs the signal in relevant ones.

**Decisive finding:** The CE-through-interface gradient is the mechanism that provides selection pressure on prediction dimensions. Without it, you get stable predictions that are slightly worse than no predictions at all. With it (variant C), you get predictions that actively help.

Evidence: [`report_grounded.json`](../../../experiments/wikitext_103/artifacts/closed_loop_prediction/report_grounded.json)

### Phase 3: N=3 semi-local star — DONE ✓ (SEED-SENSITIVE)

**What changed from the original Phase 3 plan:** Originally planned as a chain (0←1←2). Changed to star topology because:
- Chain means block 2 is one hop from CE — same grounding-drift problem we've now proven matters
- Star means both helpers get direct CE shaping through their own interface
- Matches Max's "many more parallel blocks" vision better (scale by adding spokes, not lengthening chains)
- Max's original vision note says "graph" not "chain"

**Architecture (F_star_3block):**
- Block 0 (rate=1): processes tokens, has CE loss
- Block 1 (rate=2): reads s0.detach(), predicts s0's next 2 states, feeds via gain_1. CE flows back through interface.
- Block 2 (rate=4): reads s0.detach(), predicts s0's next 4 states, feeds via gain_2. CE flows back through interface.
- Both helpers have independent gradient paths to CE (star = parallel spokes, not serial chain)
- Different rates give different time horizons — block 1 sees 2-step patterns, block 2 sees 4-step patterns
- Shared auxiliary prediction loss supervises `layernorm(prior_1 + prior_2)` against state0 history

**Result (2 seeds):**

| Variant | Seed 42 | Seed 43 | Mean | Range |
|---------|---------|---------|------|-------|
| A_single | 1.669 | 1.672 | 1.670 | 0.003 |
| C_closed_loop | 1.660 | 1.669 | 1.665 | 0.009 |
| **F_star_3block** | **1.702** | **1.659** | **1.681** | **0.044** |

**Seeds disagree in direction:** F-A is +0.034 on seed 42 (hurts) but -0.013 on seed 43 (helps).

**What IS established:**
- Rate-4 helper is always rejected: gain_2 → 0 in both seeds
- F creates 15× more variance than A across seeds
- Outcome tracks death speed: fast pruning (seed 43, zero by step 5K) → good; slow pruning (seed 42, zero by step 15K) → bad
- C remains robustly helpful (both seeds agree)

**What is NOT established:**
- That F consistently hurts or helps
- The specific objective-mismatch mechanism (plausible but not proven)
- Whether the damage is "residual" vs just "behind at eval time"

**Next discriminator: G_rate4_only** — rate-4 helper in isolation (no shared loss, no second helper). Tests whether rate-4 predictions are intrinsically viable. If G ≈ C, the N=3 problem was coupling; if G ≈ A, rate-4 is just too stale; if G > A, rate-4 is bad.

Evidence: [`run_star.jsonl`](../../../experiments/wikitext_103/artifacts/closed_loop_prediction/run_star.jsonl)

### Phase 3b: G_rate4_only — DONE ✓ (G ≈ A, rate-4 too stale)

Rate-4 helper in isolation. Same architecture as C, but rate-4 instead of rate-2.

| Variant | Seed 42 | Seed 43 | Mean | Δ vs A |
|---------|---------|---------|------|--------|
| A_single | 1.669 | 1.672 | 1.670 | — |
| G_rate4_only | 1.662 | 1.666 | 1.664 | -0.006 |

**Ablation gap = 0** at both seeds. Predictions not contributing at inference. The -0.006 is a training regularization effect from the auxiliary loss, not prediction value.

The staleness signature: pred_loss *rises* over training (0.23 → 0.49 for seed 42). Rate-2 (C) converges at ~0.29; rate-4 can't track the evolving target.

**Conclusion:** Rate-4 predictions are intrinsically too stale. The staleness limit is between rate-2 (works) and rate-4 (doesn't). This is NOT a coupling problem — it's fundamental.

**Implication for F:** F_star's instability was the multi-helper *interaction* (shared loss coupling + slow dying), not rate-4 being harmful per se.

Evidence: [`report_g.json`](../../../experiments/wikitext_103/artifacts/closed_loop_prediction/report_g.json)

## How to interpret N=3 results

When the run finishes, look at these diagnostics in this order:

### 1. Did it collapse?
- F_star val_loss > 2.0 → collapse. Investigate which helper caused it (check gain magnitudes and pred_loss trajectories).
- Both helpers' gains trending to zero together → mutual interference.

### 2. How does F compare to C and A?
- **F < C < A:** Width helps! Two predictors compose. Move toward N=8.
- **F ≈ C < A:** Second helper redundant. Both predict the same thing. Need role differentiation (different targets, different temporal windows, different prediction modalities).
- **C < F < A or C < A < F:** Second helper hurts. Maybe it interferes with the first helper via the shared block 0 (conflicting gradient directions through the interface).

### 3. Are the helpers differentiated?
- `gain_1` ≈ `gain_2` → model treats them interchangeably. Rates aren't differentiating.
- `gain_1` >> `gain_2` or vice versa → model found one useful, one not. Rate difference matters.
- One gain positive, one negative → interesting! Different roles (one adds predictions, one subtracts noise).

### 4. Ablation structure
- Ablation gap (all priors off) vs C's ablation gap: how much are the combined predictions worth?
- Individual ablation (not yet implemented): would tell us if helpers contribute independently or only together.

### 5. Redundancy signature
- If N=3 ≈ N=2 AND one helper's gain → 0: the architecture self-prunes the redundant block. Need to give blocks genuinely different roles.
- If N=3 ≈ N=2 AND both gains active: they're splitting the same job between them (ensemble over same signal, marginal benefit).

## Variants analyzed (from most to least local)

### ~~Strict-local~~ — DEAD END

~~One additional `detach()` on the feedback path.~~ Proven to either collapse (D, ungrounded) or be neutral-to-harmful (E, grounded). The fundamental problem: without CE feedback, the predictor can't learn what's useful to predict.

### Semi-local / neighborhood-local — MAINLINE

CE flows through the feedback interface. Each block pair forms a "neighborhood." The gradient path is unbroken from CE through the gate into block 1's prediction head. This is supervised local learning where the supervisor is the task loss.

**Locality claim:** Weak in the strict-local sense (can't step block 1 independently). But strong in the practical sense: gradient only flows through a narrow bottleneck (one additive vector, gated by a single scalar per helper). The computational graph is shallow even though gradient flows.

### Alternating optimization — CONTINGENCY

Block coordinate descent: train block 0 for K steps with predictions frozen, then train block 1 for K steps with targets frozen. Genuinely parallel within each phase.

**When this becomes relevant:** If per-helper aux losses (H) remove instability but helpers STILL get pruned under CE/gate competition. The diagnosis would be: "the gates compete because they're all trained by the same CE signal — whichever helper gets a slight early advantage pulls the CE gradient toward itself, starving the others." Alternating optimization breaks this by freezing gate competition during helper-training phases.

**Not yet needed** because we haven't exhausted simpler fixes (separate aux losses, phase offsets). Only escalate here if H/I show that CE competition — not aux-loss coupling — is the binding constraint on width.

### N-block topologies

**Star (current):** All helpers predict block 0 directly. Each gets direct CE gradient through its interface. Scales by adding spokes. This is the mainline topology.

**Chain:** Block N predicts block N-1, which predicts... eventually reaching block 0. Task relevance degrades with hop count (proven by E_grounded analysis: even one hop of indirection loses selection pressure). Only viable if each link has its own CE head — but that's just star with extra hops.

**Hybrid:** Some blocks in a star around block 0, others forming short chains. Might be needed at large N if the interface bottleneck limits how many signals block 0 can use simultaneously. Not explored yet.

## Phase 4: post-G experiments

G confirmed rate-4 is intrinsically too stale (G ≈ A). The question shifts: can we get width without staleness?

### H: per-helper prediction losses — DEPRIORITIZED

H was designed for the "G ≈ C" outcome (rate-4 viable alone, coupling was the problem). Since G ≈ A instead, fixing the coupling won't help — even with perfect per-helper losses, rate-4 predictions are simply too stale to add value.

H remains relevant only if I shows that even rate-2 helpers get pruned under CE competition. Then per-helper losses might help by removing the last coupling mechanism. But it's no longer the primary path.

### I: phase offsets — RUNNING NOW

**Hypothesis:** Width can help if both helpers operate at rate-2 (proven to work) with different temporal phases. Phase offset gives temporal diversity without staleness.

**Architecture (I_phase_offset):** Two rate-2 helpers, one at phase 0 (updates on steps 0,2,4,...) and one at phase 1 (updates on steps 1,3,5,...). Per-helper prediction losses (removes known coupling confound). On any given step, one helper is maximally fresh and one is 1-step stale.

**Control (I_control):** Two rate-2 helpers, both at phase 0. Same per-helper losses. Isolates whether phase diversity is the key factor or whether two rate-2 helpers compose regardless.

**Decision rules:**
- **I_phase_offset < I_control < A:** Phase offset creates genuine role differentiation. Width + temporal diversity = scaling path.
- **I_phase_offset ≈ I_control < A:** Both help, offset doesn't matter. Width alone scales.
- **I_phase_offset ≈ I_control ≈ A:** Helpers get pruned under CE competition even with separate losses. Would need alternating optimization.
- **One helps, one ≈ A:** Interesting asymmetry. Investigate which configuration survives and why.

**Open question:** If I works, is the mechanism temporal ensembling (two different-age predictions averaged) or phase specialization (each helper learns a different function because it sees the stream at different offsets)? Would need per-helper ablation to distinguish.

## Phase 5: prediction target change (planned, after I resolves mechanism questions)

Per [dictation 2026-05-24-5](../../../dictations/2026-05-24-5.md): the current full-state cosine prediction target is a simplification for mechanism testing. The real goal is "predict something block 0 couldn't already know" — specifically, information from further back in time.

**Caveat:** "Block 0 can't compute this" is too strong information-theoretically. With ctx=32 and a recurrent-style rollout, block 0 could in principle encode past information. The real test is operational: can a dedicated helper objective make block 1 carry a **cleaner long-range signal** than block 0 bothers to preserve in its fast single state?

All variants below use the **same gated additive interface** (change one thing: the target). Same architecture, same d=256, same ctx=32, same rate-2 helper.

### J: Older-window memory summary

**The direct test of Max's "information from a longer time ago" framing.**

Target: predict a pooled summary of block 0's older history, not its current state.

```
z_t = mean(h_{t-8}, h_{t-7}, h_{t-6}, h_{t-5})   # 4-step old window
pred_loss = 1 - cos(LN(p_t), LN(z_t))
```

Block 1's role: dedicated "older memory channel." No new inputs needed — block 1 already has a 1-step-stale view of block 0 and accumulates context over its rate-2 schedule. It just needs to retain older information rather than tracking current state.

**Why block 0 may not do this on its own:** Block 0 has no explicit pressure to preserve a clean 4–8-step-old summary while simultaneously doing immediate next-char CE. The helper gives a dedicated "remember this" role.

### K: Nonlocal residue (old - recent)

**Explicitly complementary: predict what block 0 likely doesn't already emphasize.**

Target: the *difference* between older and recent context summaries.

```
m_old_t = mean(h_{t-8}, ..., h_{t-5})
m_recent_t = mean(h_{t-4}, ..., h_{t-1})
z_t = m_old_t - m_recent_t
pred_loss = 1 - cos(LN(p_t), LN(z_t))
```

Block 1's role: carry what's different about older context vs what block 0 likely already knows (recent state). The subtraction explicitly pushes block 1 away from redundant short-range copies.

**Discriminates against J:** If K > J, the raw older summary has too much overlap with what block 0 already retains. If J > K, the subtraction throws away useful overlap.

### L: Future chunk code

**Different hypothesis family: helper as slow predictor, not memory store.**

Target: compressed representation of the next few tokens (not just next-1).

```
z_t = mean(E(x_{t+1}), E(x_{t+2}), E(x_{t+3}), E(x_{t+4}))   # E = token embedding
pred_loss = 1 - cos(LN(p_t), LN(z_t))
```

Block 1's role: forecast a 4-token chunk code. This connects to the hierarchical dynamic prediction direction from [dictation 2026-05-24-3](../../../dictations/2026-05-24-3.md).

**Why block 0 can't do this:** Block 0 is trained only on immediate next-token CE. It has no dedicated multi-step forecast objective. The helper provides a "what's coming next in aggregate" signal.

**Note:** Must mask last 4 positions of context window (no future available there).

### What would NOT work (and why)

- **Predicting block 0's full state (current approach):** Block 0 already knows this. Useful for mechanism testing only.
- **Predicting next-token logits:** Recreates E_grounded (a second next-token model, not a distinct channel). Already tested — stable but doesn't help.
- **Predicting gradients/errors:** Available in training, not at inference. Block 1 learns something it can't use when generating.
- **Very long horizons (12+ steps):** Rate-4 was already too stale. With ctx=32 and rate-2 helper, start with 4-step windows.
- **Adding encoder/decoder before testing simple targets:** Too confounded. If it helps, you won't know if the win came from the target or the extra machinery.

### Decision framework

- **J helps, K doesn't:** Block 0 wants a coarse older-memory summary. Simple is best.
- **K helps more than J:** The right signal is specifically *nonlocal complement*, not raw memory.
- **L helps most:** Helper should be a slower predictive latent / chunk forecaster. Points toward hierarchical autoregressive direction.
- **None beat current C:** The binding issue is interface alignment / selection pressure (the CE-through-interface gradient), not target semantics alone. Current mechanism IS the right one.
- **All help but only marginally:** The helper channel at d=256 with ctx=32 may be too small/short for long-range context to matter. Would need to scale before concluding.

### Prerequisites

Phase 5 should only start AFTER:
1. I resolves whether width (multiple helpers) works mechanistically
2. Results are integrated and understood
3. Prediction target change is clearly the next binding question (not architecture/coupling)

---

## What this does NOT cover

- Whether any local variant matches a transformer at matched compute. That's a separate question about absolute performance, not about whether locality works.
- Continuous-time or rate-coding variants of the update rule.
- Discrete communication channels between blocks — relevant only if the continuous interface bottleneck limits scaling.
- The hierarchical dynamic tokenization direction (from [dictation 2026-05-24-3](../../../dictations/2026-05-24-3.md)) — a separate thread.

Scope is: can we train hierarchical prediction blocks with local-ish gradients and still get task benefit? And does it scale with width?
