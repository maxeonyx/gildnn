# What local objective creates useful multi-timescale representations?

Serves [dictation 2026-05-31-01](../../../dictations/2026-05-31-01.md), [dictation 2026-05-30-03](../../../dictations/2026-05-30-03.md), and [dictation 2026-05-31-04](../../../dictations/2026-05-31-04.md) (explore many, don't settle).

## Status: exploration in progress

No single answer yet. Two candidates show life; neither is conclusive. The core question remains open: what local objective is both genuinely local AND creates representations useful for token prediction?

## The bracketing result

Three boundary points already narrow the design space:

- **InfoNCE on neighbor states**: almost orthogonal to token prediction; measured cosine with token gradient was 0.013.
- **Per-band CE**: rapidly creates specialization, but [dictation 2026-05-31-01](../../../dictations/2026-05-31-01.md) rejects it because it broadcasts the token objective globally.
- **Raw MSE on neighbor states**: "garbage" because a high-dimensional target invites copying without forcing compression.

So the problem is not "find any local loss." It is: find a loss that is local, compressive, and coupled to temporal structure that matters for prediction.

## Why InfoNCE failed

InfoNCE here asks: **which sample in the batch is my true neighbor?** That rewards features that distinguish one neighbor state from other contemporaneous states. Those distinctions can be arbitrary, batch-dependent, and unrelated to the temporal dynamics band 0 needs for token prediction.

So the loss is contrastive over **identity**, not over **dynamics**. A module can get very good at saying "this is my neighbor" without learning anything that helps model future tokens.

## Recommended approach: hierarchical future-state prediction through noise

- **Band 0** keeps the existing next-token CE loss.
- **Band k** predicts band `k-1`'s output at time `t + rate(k)`.
- **Laterals stay noisy**, so the channel is not an exact copy path.
- **Targets should be summaries or projections**, not the full raw state.

The intended logic is:

1. Noise on laterals creates an information bottleneck.
2. Future prediction adds compression pressure: model transition structure, not snapshot identity.
3. Alignment can then propagate upward: if band 0 learns token-relevant features, band 1 must model their dynamics; band 2 must model band 1's slower dynamics; and so on.

This differs from the rejected options in exactly the needed ways:

- **vs InfoNCE**: predict the future, not neighbor identity.
- **vs raw MSE**: add a noisy bottleneck so copying is harder.
- **vs per-band CE**: predict the neighbor's locally available representation, not the global token stream.

## Why the target should be a summary, not the full state

Jiang & Rao 2024, in *Dynamic Predictive Coding* ([PLOS Computational Biology](https://journals.plos.org/ploscompbiol/)), argue that higher levels can profitably predict **transition dynamics** rather than reconstructing raw high-dimensional activations. That matters here: if the target is the whole neighbor vector, the loss may mostly reward surface fidelity; if the target is a projection that preserves predictive structure, it is more likely to reward the causes of change.

Millidge et al. 2024, in *Temporal Predictive Coding* ([PLOS Computational Biology](https://journals.plos.org/ploscompbiol/)), decompose free energy into **sensory prediction errors** and **temporal prediction errors**. The key point for gildnn is simply that both signals are local.

So the best current target family is probably: **predict a learned or fixed projection of the lower band's future state**, not the full vector.

## Ranked alternatives

| Approach | Local? | Timescale diff? | Task-aligned? | Tested? | Result |
|----------|--------|-----------------|---------------|---------|--------|
| Predict-next-inputs (band0 token+neighbor) | ✓ | ~ | ✓ (band 0) | ✓ | CE 3.24 at 100 steps (detached readout) |
| Hierarchical targets (band k → mean of band k-1) | ✓ | ✓ | ✓ (via hierarchy) | ✓ | CE 2.67 at 100 steps (detached readout) |
| Hierarchical future prediction through noise | ✓ | ✓ | ✓ (via hierarchy) | ✗ | Untested — need future-state targeting |
| SFA / VICReg temporal | ✓ | ✓✓ | ~ (indirect) | ✗ | Good auxiliary candidate |
| Forward-Forward | ✓ | ? | ? | ✗ | Unclear (negative generation problem) |
| Equilibrium Propagation | ✓ | - | - | ✗ | Dead end (wrong dynamics) |
| Communication coherence | ✓ | - | ✗ | ✗ | Dead end (= InfoNCE problem) |

## Experimental results (day 10)

### Predict-next-inputs (`--band0-local-loss --attention-readout --detach-readout`)

Band 0 predicts l2_normalize(neighbor_sum + token_embedding). All other bands predict l2_normalize(neighbor_sum). Detached attention head for CE.

**Key finding: prediction improves dramatically but CE stagnates.**

```
Step     Band 0 pred   CE (attention)
  50      2.94          3.61
 100      2.40          3.24
 150      1.36          3.49
 200      1.18          3.27
 250      0.64          3.37
 300      0.59          3.33
```

Band 0 becomes excellent at predicting its inputs (loss → 0.59) but the detached attention head can't extract improving token predictions. Two hypotheses:
1. **Training dynamics**: the attention head is chasing rapidly-changing representations it can't influence
2. **Representation mismatch**: predicting inputs well doesn't create states linearly decodable as token logits

### Hierarchical targets (`--hierarchical-targets --attention-readout --detach-readout`)

Band k predicts the mean state of band k-1. CE 2.67 at 100 steps (best purely-local result so far). However, this may be partly "cheating": lower-band means ARE your neighbors in the 2D grid, so this is very similar to neighbor prediction but with spatial averaging.

## Why per-band CE works and these don't (fully)

Per-band CE gets CE 2.54 because it directly optimizes through the logit head — the representation IS optimized for token classification. The detached-readout experiments have a fundamental mismatch: modules optimize for local prediction, the readout must somehow exploit representations optimized for a different purpose.

This suggests the right answer might not be "purely local learning + detached readout" but rather "local learning + a small amount of task signal through the readout." The question becomes: how much task signal, and does it violate the local-learning principle?

## Experiment queue (next to try)

1. **Band0-local-loss WITHOUT detach** — band 0 gets CE + local prediction together. Tests: does local prediction help CE when combined? (~100 steps, 5 min)
2. **Combined: band0-local-loss + hierarchical-targets + attention-readout + detach** — all modules have local targets. Tests: does the combination beat either alone? (~100 steps, 5 min)
3. **Two-phase: local only → freeze → train readout** — train modules 300 steps, freeze, train only attention head 200 steps. Tests: is the problem dynamics or representation quality? (requires code change)
4. **Higher attention LR**: 10x LR on attention parameters only. Tests: can faster readout adaptation track changing representations? (requires code change)

## Why this still might not work

- The detached readout may be fundamentally unable to exploit representations optimized for a different objective.
- Even a projected neighbor target might still be too high-dimensional for MSE to be selective.
- The existing noise level might destroy too much information for the future state to be predictable at all.
- The alignment story assumes band 0 quickly learns something worth predicting; at initialization it does not.
- A warm-up where band 0 trains alone before higher-band prediction starts might be necessary.

## Open questions

- Is the CE stagnation a dynamics issue (readout chasing) or a representation issue (wrong features)?
- What loss should sit on the prediction head: MSE, cosine, or a temporal contrastive loss?
- Should the target be the full future state, a learned projection, or a fixed random projection?
- Does alignment really propagate upward? A direct check would be the gradient cosine between band 1's prediction loss and band 0's CE loss.
- How much lateral noise is enough to prevent copying without making the target effectively random?
- Is the "purely detached" constraint too strict? Maybe a small task signal to the readout is acceptable if modules still learn locally.

For now the design-space answer is: **predict something about your neighbors' future through a bottleneck**. The exact form (what target, what loss, how much global signal is acceptable) is the open experimental question with multiple candidates to test.
