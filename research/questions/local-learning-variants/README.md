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

## Status: Phase 5 J CONFIRMED — older-window target breaks the -0.006 ceiling.

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

**Phase 4 resolved (I):** Two rate-2 helpers (with or without phase offsets) saturate at the same -0.006 benefit as one helper. Per-helper losses fix coupling (both helpers survive). Width doesn't help because the prediction target is the bottleneck — full-state prediction provides redundant info block 0 already has. Phase 5 (prediction target change) directly motivated by this finding. Code implemented, awaiting GPU for sanity check and run.

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

### I: phase offsets — DONE ✓ (width saturates, target is bottleneck)

**Hypothesis:** Width can help if both helpers operate at rate-2 (proven to work) with different temporal phases. Phase offset gives temporal diversity without staleness.

**Architecture (I_phase_offset):** Two rate-2 helpers, one at phase 0 (updates on steps 0,2,4,...) and one at phase 1 (updates on steps 1,3,5,...). Per-helper prediction losses (removes known coupling confound). On any given step, one helper is maximally fresh and one is 1-step stale.

**Control (I_control):** Two rate-2 helpers, both at phase 0. Same per-helper losses. Isolates whether phase diversity is the key factor or whether two rate-2 helpers compose regardless.

**Result (seed 42 complete; seed 43 A_single confirmed 1.672, I variants interrupted by GPU use):**

| Variant | val_loss (s42) | Δ vs A | ablation gap | gain_1 | gain_2 |
|---------|----------------|--------|-------------|--------|--------|
| A_single | 1.669 | — | — | — | — |
| I_phase_offset | 1.663 | -0.006 | 0.351 | -0.050 | -0.033 |
| I_control (same phase) | 1.663 | -0.006 | 0.274 | -0.039 | -0.040 |

**Verdict: decision rule #3.** I_phase_offset ≈ I_control ≈ A - 0.006. Width adds no benefit beyond one helper. Both helpers survive (per-helper losses fix coupling). Ablation gap is much larger than C's 0.22 (helpers are deeply integrated) but net val benefit saturates.

**Why width doesn't help:** The prediction target is full block-0 state — which block 0 already knows. More helpers predicting the same redundant thing can't add information. The gate saturates at the point where "subtract prediction, process surprise" extracts the small useful signal from any single predictor.

**Phase offset does break symmetry** (I_phase_offset gains are asymmetric: -0.050, -0.033; I_control gains are symmetric: -0.039, -0.040) but this doesn't increase net benefit. Differentiation without new information is cosmetic.

## Phase 5: prediction target change — J CONFIRMED

Per [dictation 2026-05-24-5](../../../dictations/2026-05-24-5.md): the current full-state cosine prediction target is a simplification for mechanism testing. The real goal is "predict something block 0 couldn't already know" — specifically, information from further back in time.

### J_older_window: DONE ✓ — TARGET WAS THE BOTTLENECK

Block 1 predicts `mean(h_{t-8}, ..., h_{t-5})` instead of current `h_t`. Tests: "does predicting older context block 0 may not preserve actually help?"

| Variant | Seed 42 | Seed 43 | Mean | Δ vs A | std |
|---------|---------|---------|------|--------|-----|
| A_single | 1.669 | 1.672 | 1.670 | — | 0.0015 |
| C_closed_loop (full-state) | 1.660 | 1.669 | 1.665 | -0.006 | 0.0047 |
| **J_older_window** | **1.660** | **1.660** | **1.660** | **-0.010** | **0.00009** |

Full metrics at step 20000:

| Metric | Seed 42 | Seed 43 | Interpretation |
|--------|---------|---------|---------------|
| pred_loss | 0.168 | 0.163 | Target is 2× more learnable than C's 0.29 |
| mix_coeff | -0.065 | -0.064 | Predictive coding (identical mechanism to C) |
| ablation_gap | 0.182 | 0.193 | Load-bearing at inference |
| val_accuracy | 0.523 | 0.529 | vs A's 0.521 |

Evidence: [`report_j.json`](../../../experiments/wikitext_103/artifacts/closed_loop_prediction/report_j.json)

**The key finding is seed robustness, not raw improvement magnitude.**
- C was unreliable: seed 42 hit 1.660, seed 43 fell to 1.669 (range: 0.009)
- J is rock-solid: both seeds hit 1.660 (range: 0.0002)
- C's reported "-0.006 mean" was depressed by seed variance; J eliminates that variance

**Why this works:** Full-state prediction (C) optimizes for a target that block 0 already has — redundant. The older-window target is 2× more learnable (pred_loss 0.165 vs 0.29) because it asks for information that block 0's fast current processing may not preserve. The learning signal is cleaner and more consistent.

**What it does NOT yet tell us:**
- Whether farther-back windows (9-16, 17-32) would be better or worse (J_far_window running)
- Whether the older-window target composes with width (J + two helpers)
- Whether the benefit grows with context length (should, in principle)
- Whether an external/fixed target (J_fixed_embedding, L) would be even more robust

**Decision rule triggered:** J < C → target WAS the bottleneck. Scale the target semantics.

### J_far_window: DONE ✓ — Offset sensitivity is SHALLOW (gentle inverted-U)

Block 1 predicts `mean(h_{t-12}, ..., h_{t-9})` instead of J's `mean(h_{t-8}, ..., h_{t-5})`. Tests whether more temporal separation helps.

| Variant | Seed 42 | Seed 43 | Mean | Δ vs A | std |
|---------|---------|---------|------|--------|-----|
| A_single | 1.669 | 1.672 | 1.670 | — | 0.0015 |
| J_older_window (offset 8) | 1.660 | 1.660 | 1.660 | -0.010 | 0.00009 |
| **J_far_window (offset 12)** | **1.663** | **1.664** | **1.664** | **-0.007** | **0.0003** |

Full metrics at step 20000:

| Metric | J_far s42 | J_far s43 | J_older s42 | Interpretation |
|--------|-----------|-----------|-------------|----------------|
| pred_loss | 0.183 | 0.174 | 0.168 | Farther target harder to predict |
| mix_coeff | -0.067 | -0.070 | -0.065 | Slightly more aggressive predictive coding |
| ablation_gap | 0.184 | 0.174 | 0.182 | Load-bearing at inference |

Evidence: [`report_jfar.json`](../../../experiments/wikitext_103/artifacts/closed_loop_prediction/report_jfar.json)

**Pre-registered decision framework verdict: between "SAME" and "WORSE but still beats A."**
- J_far is slightly worse than J (-0.007 vs -0.010) — the inverted-U exists but is gentle
- Still well above C's old -0.006 ceiling
- Still extremely seed-robust (std 0.0003)
- The farther target is genuinely harder (pred_loss 0.179 vs 0.165) but still very learnable

**Implication for multi-helper:** Offsets 8 and 12 carry somewhat different information (different pred_loss values, different benefit magnitudes). If both can contribute simultaneously through separate gates, their bands should compose. The `J_dual_band` experiment (already implemented) tests this directly.

**Next steps per decision framework:**
1. J_fixed_embedding — separates "older content" from "older hidden-state codes" (still most discriminating)
2. J_dual_band (offsets 8,12) + J_dual_same_band (8,8) control — tests band composition

### Multi-block scaling implication

J implies that multi-block scaling should be built around **temporal role differentiation**, not duplicate prediction. In a deep local-learning stack, each block should own a different offset/timescale band matched to what the block below naturally forgets. The natural hierarchy: "block 2 remembers what block 1 drops after ~N steps, block 3 remembers what block 2 drops after a larger N." Falsified if identical targets/offsets scale equally well.

### Implementation notes (preserved)

**Implementation status:** `J_older_window` and `J_fixed_embedding` variants in `runs/closed_loop_prediction.py`. Same architecture as C, only target changes.

**Caveat:** "Block 0 can't compute this" is too strong information-theoretically. With ctx=128 and a recurrent-style rollout, block 0 could in principle encode past information. The real test is operational: can a dedicated helper objective make block 1 carry a **cleaner long-range signal** than block 0 bothers to preserve in its fast single state?

All variants below use the **same gated additive interface** (change one thing: the target). Same architecture, same d=256, same ctx=128, same rate-2 helper.

### Run order (updated post-J)

J won clearly. Remaining run order:
1. ✅ **J_older_window** — DONE. Target was the bottleneck.
2. 🔄 **J_far_window** (offset 12) — RUNNING. Tests offset sensitivity.
3. → **J_fixed_embedding** — next regardless of J_far outcome. Separates "older content" from "older hidden-state codes."
4. → **Width with J target** — multiple helpers at different offsets (only if J_fixed ≈ J or worse)
5. → **L** (future chunk code) — different hypothesis family, if memory hypothesis stalls
6. → **K** (nonlocal residue) — deprioritized. High collapse risk, less discriminating than J_fixed.

**Collapse risk ranking:** L (lowest, fixed external target) < J/J_far (moderate, self-generated but diverse) < K (highest, subtraction can produce near-zero targets).

**ctx=128 note:** offset 12 → valid positions = 116/128 = 91%. Offset 24 → valid = 104/128 = 81%. Still plenty of room.

### J: Older-window memory summary

**The direct test of Max's "information from a longer time ago" framing.**

Target: predict a pooled summary of block 0's older history, not its current state.

```
z_t = mean(h_{t-8}, h_{t-7}, h_{t-6}, h_{t-5})   # 4-step old window
pred_loss = 1 - cos(LN(p_t), LN(z_t))
```

Block 1's role: dedicated "older memory channel." No new inputs needed — block 1 already has a 1-step-stale view of block 0 and accumulates context over its rate-2 schedule. It just needs to retain older information rather than tracking current state.

**Why block 0 may not do this on its own:** Block 0 has no explicit pressure to preserve a clean 4–8-step-old summary while simultaneously doing immediate next-char CE. The helper gives a dedicated "remember this" role.

**Collapse risk:** Moderate. Self-generated target, but the 4-step-old window is diverse enough to avoid the degenerate constant-predictor trap that killed D.

**Fixed-target alternative (J'):** If J collapses or is ambiguous, a cleaner version uses raw token embeddings instead of hidden states:
```
z_t = mean(E(x_{t-8}), E(x_{t-7}), E(x_{t-6}), E(x_{t-5}))   # E = token embedding
```
This removes the self-generated target risk entirely (block 0 can't make this easier by co-adapting). Tests: "does block 0 benefit from being reminded which characters appeared 5-8 steps back?" — relevant because this architecture processes tokens ONE AT A TIME and block 0's single state vector must compress all history. Older character identity may be genuinely overwritten.

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

**Collapse/washout risk:** Highest of the three. If representations are temporally smooth, `m_old - m_recent` can become low-variance/near-zero, making the cosine objective noisy or uninformative. Run only after J establishes whether older memory helps at all.

### L: Future chunk code

**Different hypothesis family: helper as slow predictor, not memory store.**

Target: compressed representation of the next few tokens (not just next-1).

```
z_t = mean(E(x_{t+1}), E(x_{t+2}), E(x_{t+3}), E(x_{t+4}))   # E = token embedding
pred_loss = 1 - cos(LN(p_t), LN(z_t))
```

Block 1's role: forecast a 4-token chunk code. This connects to the hierarchical dynamic prediction direction from [dictation 2026-05-24-3](../../../dictations/2026-05-24-3.md).

**Why block 0 can't do this:** Block 0 is trained only on immediate next-token CE. It has no dedicated multi-step forecast objective. The helper provides a "what's coming next in aggregate" signal.

**Note:** Must mask last 4 positions of context window (no future available there). Valid positions: 124/128 = 97%.

**Collapse risk:** Lowest. Target is tied to token embeddings (external signal), not self-generated hidden states. Block 0 cannot make this target easier by co-adapting.

### What would NOT work (and why)

- **Predicting block 0's full state (current approach):** Block 0 already knows this. Useful for mechanism testing only.
- **Predicting next-token logits:** Recreates E_grounded (a second next-token model, not a distinct channel). Already tested — stable but doesn't help.
- **Predicting gradients/errors:** Available in training, not at inference. Block 1 learns something it can't use when generating.
- **Very long horizons (50+ steps):** Rate-4 was already too stale. With ctx=128 and rate-2 helper, start with 4-position windows and scale offset gradually.
- **Adding encoder/decoder before testing simple targets:** Too confounded. If it helps, you won't know if the win came from the target or the extra machinery.

### Decision framework

- **J helps, K doesn't:** Block 0 wants a coarse older-memory summary. Simple is best.
- **K helps more than J:** The right signal is specifically *nonlocal complement*, not raw memory.
- **L helps most:** Helper should be a slower predictive latent / chunk forecaster. Points toward hierarchical autoregressive direction.
- **None beat current C:** The binding issue is interface alignment / selection pressure (the CE-through-interface gradient), not target semantics alone. Current mechanism IS the right one. ~~But also: 5-8 char lag at ctx=32 may simply be too short.~~ (RESOLVED: J at ctx=128 confirmed the target works.)
- **All help but only marginally:** The helper channel at d=256 with ctx=128 may not have enough capacity to carry long-range information at its full potential. Would need to scale model size before concluding.

### Implementation notes

The change is **localized to the loss computation.** The rate/phase/buffer mechanism stays identical. After the time loop produces `state0_history` (shape: `[batch, ctx, d_model]`), derive a different `target_history` tensor and pass it to `prediction_loss_terms`. The prior valid mask must reflect where the new target is computable (e.g., invalid for first 8 positions for J/K).

For L, the target comes from `embeddings` (token embedding matrix), not from `state0_history`. This means `embeddings` must be accessible at loss computation time.

### Prerequisites — MET

1. ✓ I resolved: width saturates at full-state target (seed 42 complete, seed 43 consistent before interruption)
2. ✓ Results integrated and understood (Phase 4 I section above, daily report updated)
3. ✓ Prediction target change is clearly the next binding question

---

## What this does NOT cover

- Whether any local variant matches a transformer at matched compute. That's a separate question about absolute performance, not about whether locality works.
- Continuous-time or rate-coding variants of the update rule.
- Discrete communication channels between blocks — relevant only if the continuous interface bottleneck limits scaling.
- The hierarchical dynamic tokenization direction (from [dictation 2026-05-24-3](../../../dictations/2026-05-24-3.md)) — a separate thread.

Scope is: can we train hierarchical prediction blocks with local-ish gradients and still get task benefit? And does it scale with width?
