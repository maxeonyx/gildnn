# What local objective creates useful multi-timescale representations?

Serves [dictation 2026-05-31-01](../../../dictations/2026-05-31-01.md) and [dictation 2026-05-30-03](../../../dictations/2026-05-30-03.md).

## Conclusion first

The best current hypothesis is **hierarchical future-state prediction through a noisy lateral bottleneck**: band _k_ predicts a summary of band _k-1_'s state at `t + rate(k)`, while only band 0 sees token cross-entropy. This is the only candidate here that is local, plausibly creates timescale separation, and stays aligned with the task without broadcasting the global objective to every band.

It is still only a hypothesis. The failure modes are obvious enough that it needs a 30-second sanity check before any larger run.

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

| Approach | Local? | Timescale diff? | Task-aligned? | Verdict |
|----------|--------|-----------------|---------------|---------|
| Hierarchical future prediction | ✓ | ✓ | ✓ (via hierarchy) | **Top pick** |
| SFA / VICReg temporal | ✓ | ✓✓ | ~ (indirect) | Good auxiliary |
| Forward-Forward | ✓ | ? | ? | Unclear (negative generation problem) |
| Equilibrium Propagation | ✓ | - | - | Dead end (wrong dynamics) |
| Communication coherence | ✓ | - | ✗ | Dead end (= InfoNCE problem) |

SFA-style slowness and VICReg-style variance/covariance penalties might help as auxiliaries, but by themselves they do not obviously couple the bands to token-relevant dynamics. Forward-Forward remains interesting only in principle; the negative-sample story in this architecture is unclear. Equilibrium Propagation solves the wrong problem because the model is not naturally an equilibrium-settling system.

## Simplest 30-second sanity check

```python
Band 0: CE loss on next token  # existing
Band 1: loss = MSE(
    linear_head(band1_output_t),
    detach(band0_output_{t + rate(1)})
)
```

Success criterion: **band 1's prediction loss decreases and band 0's CE improves relative to the no-band-1 control**. That would not prove the full theory, but it would show the mechanism can produce useful upward pressure through the noisy lateral channel.

## Why this still might not work

- Even a projected neighbor target might still be too high-dimensional for MSE to be selective.
- The existing noise level might destroy too much information for the future state to be predictable at all.
- The alignment story assumes band 0 quickly learns something worth predicting; at initialization it does not.
- A warm-up where band 0 trains alone before higher-band prediction starts might be necessary.

## Open questions

- What loss should sit on the prediction head: MSE, cosine, or a temporal contrastive loss?
- Should the target be the full future state, a learned projection, or a fixed random projection?
- Does alignment really propagate upward? A direct check would be the gradient cosine between band 1's prediction loss and band 0's CE loss.
- How much lateral noise is enough to prevent copying without making the target effectively random?

For now the design-space answer is fairly narrow: **predict the future through a bottleneck, and predict a summary of dynamics rather than a raw state**. Everything else tried so far either aligns poorly, cheats by reintroducing the global objective, or leaves too much room for trivial solutions.
