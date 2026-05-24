# Local learning variants

## Core question

Serves [dictation 2026-05-23-5](../../../dictations/2026-05-23-5.md): "how can I get local learning, i.e. enabling parallelism?"

Max's framing: "It doesn't have to be totally local. We can be using backpropagation through a local neighborhood of blocks. Then if we can do that, we can parallelize training significantly more."

## Status: ANSWERED (neighborhood-local is the minimum viable locality)

The full experiment series is complete:

| Variant | Mean val_loss | vs A | Conclusion |
|---------|--------------|------|-----------|
| C_closed_loop (semi-local) | 1.665 | -0.006 | CE through interface HELPS |
| A_single | 1.670 | — | baseline |
| E_grounded (strict-local + local CE) | 1.696 | +0.026 | stable but hurts |
| D_strict_local | 3.047 | +1.377 | COLLAPSED |

**Three findings:**
1. D's collapse was caused by the ungrounded prediction target, not locality itself (E proves this — same locality, task-grounded, no collapse).
2. Task-grounded strict-local IS viable but doesn't add value — predictions slightly hurt.
3. **The CE-through-interface gradient teaches the predictor WHAT to predict.** That's specifically what makes semi-local C better than A. Local CE prevents collapse (finding a) but can't replace the feedback gradient's role in selecting useful prediction components (finding b).

**Conclusion:** Neighborhood-local (CE flows through block interfaces) is the minimum viable locality. The feedback gradient does two jobs: (a) prevent collapse, (b) select which prediction dimensions help the task. Local CE can replace (a) but not (b). Only (b) actually improves performance.

The remaining question: does neighborhood-local scale to N>2? Next: N=3 semi-local star topology.

## Key insight: "parallel" means three different things

Three properties get conflated under "local learning." They're independent:

1. **Forward parallelism** — can blocks compute their forward passes concurrently? Requires that block i's input doesn't depend on block i-1's output within the same timestep. Our architecture already has this: block 1 reads a *detached copy* of block 0's previous state, not its current one.

2. **Backward locality** — do parameter updates require gradients flowing through distant blocks? This is what "local learning rule" usually means. Semi-local (current v3) fails here: CE backprops through the feedback path into block 1. Strict-local passes: one extra detach and no gradient crosses the boundary.

3. **Update latency** — can block i's optimizer step happen before downstream blocks finish their backward pass? This requires backward locality plus no shared optimizer state. With strict-local + per-block optimizers, block 1 can step as soon as its own pred_loss backward completes — it doesn't wait for block 0's CE backward.

Most "local learning" papers address only (2). Genuine training parallelism — Max's goal of "parallelize training significantly more" — requires all three. The architecture already gives us (1). The open question is whether (2) costs us performance.

## Variants analyzed (from most to least local)

### Strict-local (recommended first test)

One additional `detach()` on the feedback path where predictions enter block 0. After this change:

- Block 0 trains on CE only. Its inputs include predictions, but gradients don't flow back through them into block 1.
- Block 1 trains on pred_loss only. It sees block 0's states (already detached) and tries to predict their future. No task gradient reaches it.
- No gradient crosses the block boundary in either direction. Fully parallel backward passes.

**Risk:** Without CE shaping block 1's representations, the predictor optimizes for easy-to-predict components of s0 rather than task-relevant ones. The gate might learn to ignore predictions because they're uninformative for the task. **Why it might still work:** block 0's states are CE-trained regardless — they encode token information because that's what block 0 is optimizing for. So prediction targets are indirectly task-shaped even without a direct gradient signal.

The test is whether the prediction ablation gap (val_loss with vs without predictions) stays positive under strict-local training. If the gap collapses to zero, block 1 is predicting something block 0 doesn't use.

### Semi-local (current v3 experiment)

What we're running now. `s0.detach()` prevents pred_loss from reaching block 0, but CE flows freely through the feedback path back into block 1's prediction head. Block 1 cannot update until block 0's full backward completes.

Better description: global task backprop through a bottlenecked interface. The interface is narrow (one additive vector, gated by a single scalar), but the gradient path is unbroken. This gives CE a way to shape what block 1 predicts — it's essentially supervised local learning, where the supervisor is the task loss flowing backward through the gate.

Expected: highest performance ceiling among the variants. But the locality claim is weak — you can't step block 1's optimizer independently of block 0's backward.

### Alternating optimization

Block coordinate descent: train block 0 for K steps with predictions frozen, then train block 1 for K steps with targets frozen. Within each phase, only one block's parameters move, so there's no cross-boundary gradient by construction.

Genuinely parallel within each phase (the idle block doesn't need gradients at all). Schedule-sensitive — K too large and the blocks oscillate; K too small and it's just slower standard training.

This is a contingency. Only interesting if strict-local looks promising but co-adaptation between the predictor and the gate causes instability. Not the mainline plan.

### N-block chains

Two variants, qualitatively different:

**Semi-local chain** (no detaches on feedback): CE from block 0 backprops through block 0's feedback input into block 1's predictions, then through block 1's feedback input into block 2's predictions, and so on. Global backprop in disguise — CE reaches block N through the chain.

**Strict-local chain** (all feedback detached): each block trains only on its own prediction loss, except block 0 which trains on CE. Genuinely local. But task information reaching block N must propagate *indirectly* — block N predicts block N-1's states, which predict block N-2's states, which... eventually ground out in block 0's CE-trained representations.

The main qualitative change beyond N=2: **grounding drift**. Task relevance degrades with hop count. Block 2's prediction targets are one step removed from CE; block N's are N-1 steps removed. Whether this degrades gracefully or catastrophically is an empirical question that N=2 cannot answer.

### REINFORCE / policy gradient

Not useful here. The interface between blocks is continuous and differentiable (additive gated vector). Strict-local already provides exact local gradients via the prediction loss. REINFORCE would give the same gradient direction with higher variance — strictly worse.

REINFORCE becomes relevant only if the interface becomes discrete (send/skip gates, codebook messages, routing decisions). That's not on the current roadmap.

## Experimental plan (decision tree)

### Phase 1: v3 semi-local — DONE ✓

Both seeds confirm C < A (mean -0.006). Mechanism works under semi-local conditions.

| Variant | Mean val_loss | vs A |
|---------|--------------|------|
| A_single | 1.670 | — |
| C_closed_loop | 1.665 | -0.006 |
| B_spectator | 1.677 | +0.006 |

Evidence: [`report_v3.json`](../../../experiments/wikitext_103/artifacts/closed_loop_prediction/report_v3.json)

### Phase 2: strict-local — DONE ✓ (COLLAPSE)

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

**Decision:** Strict-local with full-state cosine prediction is dead. The local objective doesn't select for task-relevance. Two paths remain: (a) accept neighborhood-local and scale to N>2, or (b) design a task-grounded local objective.

Evidence: [`report_strict_local.json`](../../../experiments/wikitext_103/artifacts/closed_loop_prediction/report_strict_local.json)

### Phase 3: N=3 neighborhood-local chain — NEXT

**Revised scope:** since strict-local fails, this is now the semi-local (neighborhood-local) variant. Three blocks: 0←1←2. Block 0 has CE. Block 1 predicts block 0 (CE flows through interface). Block 2 predicts block 1 (CE flows through interface).

Tests: does neighborhood-local scale beyond N=2? Can task grounding survive two hops via the chain 0←1←2? This is the question that determines whether the architecture has practical parallelism benefits — each adjacent pair is a "neighborhood" that can update partially independently.

Decision rules:
- **N=3 still helps:** Architecture scales. Depth = more parallelism.
- **Block 2 collapses but block 1 stays useful:** Grounding drift is real. Parallelism limited to adjacent pairs only.
- **Everything collapses at N=3:** Semi-local doesn't scale. Need stronger mechanisms (task-grounded targets, synthetic gradients).

## What this does NOT cover

- Whether any local variant matches a transformer at matched compute. That's a separate question about absolute performance, not about whether locality works.
- Continuous-time or rate-coding variants of the update rule.
- Discrete communication channels between blocks — relevant only as future work if strict-local fails and we need a different interface type.

Scope is tightly: can we train hierarchical prediction blocks with local gradients and still get task benefit?
