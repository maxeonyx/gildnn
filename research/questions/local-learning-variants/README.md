# Local learning variants

## Core question

<!-- TODO(author): Write a concise, dense opening for Max. Must explicitly say this question serves [dictation 2026-05-23-5](../../../dictations/2026-05-23-5.md) and preserve the exact quoted goal: "how can I get local learning, i.e. enabling parallelism?" Keep the framing as an open research question, not a settled conclusion. -->

## Key insight: "parallel" means three different things

<!-- TODO(author): Write a short section that distinguishes these three meanings of parallelism and makes the key claim that most proposed "local" schemes only address backward locality, while true parallelism requires all three:
1. Forward parallelism — can blocks compute concurrently?
2. Backward locality — do updates require gradients through distant blocks?
3. Update latency — can block i update before downstream backward finishes?
Keep it crisp and make the distinction load-bearing for the rest of the document. -->

## Variants analyzed (from most to least local)

### Strict-local (recommended first test)

<!-- TODO(author): Describe the strict-local variant. Must cover all of:
- Detach both directions: pred_loss doesn't reach block 0 (already done via s0.detach()), and CE doesn't reach block 1 (detach the feedback path too).
- Each block has only its own loss.
- Genuinely more parallel: yes; no cross-boundary gradient at all.
- Risk: block 1 predicts "easy/average" components of s0 rather than components that help CE, so the gate may learn to ignore predictions.
- Why it might still work: block 0's states are CE-trained, so prediction targets are indirectly task-shaped.
- Key test: whether the prediction ablation gap (val_loss with vs without predictions) stays positive.
State that this is the recommended first local variant to test, without overstating confidence. -->

### Semi-local (current V2 experiment)

<!-- TODO(author): Describe the semi-local variant. Must cover all of:
- s0.detach() blocks pred_loss from reaching block 0, but CE still flows through the feedback path to block 1.
- Genuinely more parallel: only weakly.
- Block 1 cannot update until block 0's backward completes.
- Better description: global task backprop through a bottlenecked interface.
- Expected: highest performance ceiling among these local variants, but weakest locality claim.
Keep the tone analytical rather than defensive. -->

### Alternating optimization

<!-- TODO(author): Describe alternating optimization as block coordinate descent. Must cover all of:
- Train block 0 with fixed predictions for K steps, then train block 1 with fixed targets for K steps.
- Genuinely more parallel: yes, within each phase.
- Tradeoff: schedule-sensitive and can oscillate if K is too large.
- Only interesting if strict-local looks promising but unstable because of co-adaptation.
Make clear that this is a contingency variant, not the mainline plan. -->

### N-block chains

<!-- TODO(author): Compare two chain variants. Must cover all of:
- Semi-local chain (no detaches on feedback): global backprop in disguise; CE reaches block N through the chain, so it is not genuinely local.
- Strict-local chain (all feedback detached): genuinely local, but task information reaching block N must propagate indirectly through prediction targets.
- Main risk: grounding drift grows with depth.
Keep the focus on what changes qualitatively when moving beyond N=2. -->

### REINFORCE / policy gradient

<!-- TODO(author): Explain why REINFORCE is not useful here. Must cover all of:
- For continuous differentiable predictions, strict-local already provides an exact local gradient.
- REINFORCE would be noisier for no gain.
- It only becomes relevant for discrete interfaces such as send/skip gates or codebook messages.
Keep this brief. -->

## Experimental plan (decision tree)

### Phase 1: current V2 (semi-local, running)

<!-- TODO(author): Write the Phase 1 decision branch using these exact outcome buckets and implications:
- If C < A: mechanism works. Proceed to Phase 2.
- If C ≈ A: feedback helps but no net benefit. Still proceed; it might be learnable with tuning.
- If C >> A: collapse is not fixed. Investigate further before locality variants.
Briefly remind the reader what the phase is testing, but do not redefine experiment notation unless needed for clarity. -->

### Phase 2: strict-local vs semi-local (N=2)

<!-- TODO(author): Write the Phase 2 comparison. Must cover all of:
- One extra detach() on the feedback path.
- Same everything else.
- Same three controls (A/B/C).
- Decision branches:
  - If strict-local < A: LOCAL LEARNING WORKS. This is the key result.
  - If strict-local ≈ A but semi-local < A: CE shaping matters. Not fully local yet.
  - If strict-local > A: predictor learns the wrong things without CE guidance.
Preserve the emphasis that this is the discriminating experiment. -->

### Phase 3: N=3 strict-local chain (only if Phase 2 positive)

<!-- TODO(author): Write the Phase 3 extension. Must cover all of:
- Three blocks: 0←1←2.
- Each predicts its neighbor below.
- Block 0 only has CE.
- Main test: whether task grounding survives two hops.
- Why N must be at least 3: N=2 cannot test multi-hop propagation.
Keep it short and conditional on Phase 2 being positive. -->

## What this does NOT cover

<!-- TODO(author): Write a brief non-goals section. Must explicitly exclude:
- Whether any of these variants match transformer performance at matched compute.
- Continuous-time / rate-coding variants.
- Discrete communication channels, except as future work if strict-local fails.
End with a short sentence that keeps the scope tightly on local-learning variants for parallel training. -->
