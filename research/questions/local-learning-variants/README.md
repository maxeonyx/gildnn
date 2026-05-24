# Local learning variants

## Core question

Serves [dictation 2026-05-23-5](../../../dictations/2026-05-23-5.md): "how can I get local learning, i.e. enabling parallelism?"

Max's framing: "It doesn't have to be totally local. We can be using backpropagation through a local neighborhood of blocks. Then if we can do that, we can parallelize training significantly more."

## Status: ANSWERED for N=2; scaling test in progress

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

**Now testing:** Does neighborhood-local scale to N=3? Running F_star_3block (two helpers, both predict block 0 via star topology).

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

### Phase 3: N=3 semi-local star — IN PROGRESS

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

**What this tests:** Does width (more parallel predictors) compose to give more benefit than a single predictor? If yes, the architecture scales toward Max's "many more parallel blocks."

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

Block coordinate descent: train block 0 for K steps with predictions frozen, then train block 1 for K steps with targets frozen. Genuinely parallel within each phase. Only interesting if semi-local instability emerges at larger N — not needed for N=2 or (so far) N=3.

### N-block topologies

**Star (current):** All helpers predict block 0 directly. Each gets direct CE gradient through its interface. Scales by adding spokes. This is the mainline topology.

**Chain:** Block N predicts block N-1, which predicts... eventually reaching block 0. Task relevance degrades with hop count (proven by E_grounded analysis: even one hop of indirection loses selection pressure). Only viable if each link has its own CE head — but that's just star with extra hops.

**Hybrid:** Some blocks in a star around block 0, others forming short chains. Might be needed at large N if the interface bottleneck limits how many signals block 0 can use simultaneously. Not explored yet.

## What this does NOT cover

- Whether any local variant matches a transformer at matched compute. That's a separate question about absolute performance, not about whether locality works.
- Continuous-time or rate-coding variants of the update rule.
- Discrete communication channels between blocks — relevant only if the continuous interface bottleneck limits scaling.
- The hierarchical dynamic tokenization direction (from [dictation 2026-05-24-3](../../../dictations/2026-05-24-3.md)) — a separate thread.

Scope is: can we train hierarchical prediction blocks with local-ish gradients and still get task benefit? And does it scale with width?
