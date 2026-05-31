# What local objective creates useful multi-timescale representations?

Serves [dictation 2026-05-31-01](../../../dictations/2026-05-31-01.md), [dictation 2026-05-30-03](../../../dictations/2026-05-30-03.md), and [dictation 2026-05-31-04](../../../dictations/2026-05-31-04.md) (explore many, don't settle).

## Status: negative result (corrected 2026-05-31)

**No purely-local objective has produced representations competitive with CE training.** All tested purely-local objectives (with detached readout) produce CE 3.2-3.5, vs the control's 2.70. The 0.5 nat gap is consistent, large, and does not close over 500+ steps.

The earlier "CE 2.67 — best purely-local result" was mislabeled: it was a hybrid with CE on band 0. See correction section below.

## The bracketing result

Three boundary points already narrow the design space:

- **InfoNCE on neighbor states**: almost orthogonal to token prediction; measured cosine with token gradient was 0.013.
- **Per-band CE**: rapidly creates specialization, but [dictation 2026-05-31-01](../../../dictations/2026-05-31-01.md) rejects it because it broadcasts the token objective globally.
- **Raw MSE on neighbor states**: "garbage" because a high-dimensional target invites copying without forcing compression.

So the problem is not "find any local loss." It is: find a loss that is local, compressive, and coupled to temporal structure that matters for prediction.

## Why InfoNCE failed

InfoNCE here asks: **which sample in the batch is my true neighbor?** That rewards features that distinguish one neighbor state from other contemporaneous states. Those distinctions can be arbitrary, batch-dependent, and unrelated to the temporal dynamics band 0 needs for token prediction.

So the loss is contrastive over **identity**, not over **dynamics**. A module can get very good at saying "this is my neighbor" without learning anything that helps model future tokens.

## Previously recommended approach: hierarchical future-state prediction through noise (UNTESTED — likely insufficient)

This was the pre-correction recommendation. It may be better than the tested alternatives but the theoretical analysis (below) suggests it still fails the "sufficient statistic" test.

- **Band 0** keeps the existing next-token CE loss (violates purely-local constraint).
- **Band k** predicts band `k-1`'s output at time `t + rate(k)`.
- **Laterals stay noisy**, so the channel is not an exact copy path.
- **Targets should be summaries or projections**, not the full raw state.

The intended logic is:

1. Noise on laterals creates an information bottleneck.
2. Future prediction adds compression pressure: model transition structure, not snapshot identity.
3. Alignment can then propagate upward: if band 0 learns token-relevant features, band 1 must model their dynamics; band 2 must model band 1's slower dynamics; and so on.

**Why this likely still fails:** The theoretical conclusion identifies that this approach requires band 0 to already be trained by CE for the "upward propagation" story to work. In the purely-local case, band 0 has no token-prediction signal — it learns to predict its inputs, not to classify tokens. The chain "band k → band k-1 → ... → tokens" only exists if something at the bottom IS a token predictor. Without that anchor, future-state prediction is just a more sophisticated version of neighbor prediction, and the sufficient-statistic argument still applies.

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
| Predict-next-inputs (band0 token+neighbor) | ✓ | ~ | ✓ (band 0) | ✓ | CE 3.24→3.41 (detached, worsens) |
| Hierarchical targets (band k → mean of band k-1) | ✓ | ✓ | ✓ (via hierarchy) | ✓ | **Corrected:** hybrid=2.67 (≈control), purely-local=3.28→3.52 (worsens) |
| Hierarchical future prediction through noise | ✓ | ✓ | ✓ (via hierarchy) | ✗ | Untested — likely insufficient (see theoretical conclusion) |
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

### Hierarchical targets (`--hierarchical-targets`)

Band k predicts the mean state of band k-1.

**CORRECTION (discovered 2026-05-31 12:15):** The earlier "CE 2.67 at 100 steps (best purely-local result)" was mislabeled. That result used `--hierarchical-targets` with the STANDARD band-0 readout (CE gradient into band 0). It was a hybrid: CE trained band 0, hierarchical targets trained bands 1-7. The 2.67 came from the standard band-0 CE head — essentially the same as the control (2.70) within noise.

The truly purely-local configuration (`--hierarchical-targets --attention-readout --detach-readout`, where NO band gets CE gradient) gives **CE 3.28 at step 100** — significantly worse than the control. Band 0 is completely untrained in this config (no prediction target for the bottom band, no CE because detached).

So the honest results are:
- Hybrid (CE on band 0 + hierarchical targets on bands 1-7): CE ~2.67 (≈ control, within noise)
- Purely local (no CE anywhere, detached attention readout): CE 3.28 → **3.52 at 1000 steps** (actively worsens)
- Local prediction improves strongly (band 2: 4.17 → 1.97) while CE degrades — confirming anti-correlation.

## Why per-band CE works and local objectives don't

The gap is not about the readout. The standard model (CE on band 0, standard band-0 readout) achieves CE 2.70 — proving band 0's representations DO encode tokens when trained by CE. A detached attention head could attend to band 0 and achieve ~2.70 if band 0 had good representations. It doesn't (~3.2-3.3) because:

**Local objectives produce different representation geometries than CE.** CE directly optimizes "state → next-token logits." Local prediction optimizes "state → neighbor prediction accuracy." These create fundamentally different features:

- **CE-trained band 0**: encodes whatever combination of current context predicts the NEXT token. The representation IS a token classifier.
- **Locally-trained band 0**: encodes whatever helps predict the next INPUT (neighbor states + token embedding). This implicitly encodes the next token (it's part of the input), but the encoding is mixed with neighbor-prediction features and isn't in logit-extractable form.

Evidence: band0-local-loss prediction loss goes from 4.65 → 0.54 (excellent local prediction) while CE stays flat at 3.3 (terrible token classification). The representation IS learning — just not learning something the readout can exploit.

This is not a readout problem. It's a feature-geometry problem.

## The fundamental tension (theoretical conclusion)

**Formal argument:** For a local objective to produce token-useful representations, the locally-predicted quantity must be a sufficient statistic for the next token. If there exist contexts c1, c2 that produce the same optimal local behavior but require different token predictions, the local objective has no reason to distinguish them — and will collapse the distinction. Topology cannot create task information that the objective never rewards preserving.

In our experiments: neighbor states, lower-band means, and next-step inputs are all **invariant to distinctions that matter for next-token prediction**. You can predict your neighbors perfectly well regardless of whether the next character is 'a' or 'z' — the neighbor dynamics don't encode that distinction. So the local objective discards it.

The measured gradient cosine (0.013) is not bad luck — it's a structural consequence. The local objectives and CE are asking for different things, and optimizing one actively moves features away from the other.

**What's established:** The tested "predict neighbor state" family of local objectives does not produce token-useful representations on this architecture with detached laterals. The gradient is orthogonal to CE, and training actively degrades token-prediction performance.

**What remains open:** Whether a *different* class of local objective — one that targets bottlenecked predictive structure rather than raw state reconstruction, and possibly with a different communication architecture — could align local learning with token prediction. The necessary properties:
1. Target must be closer to a sufficient statistic for future tokens (not raw neighbor state)
2. Must be predictive (future-oriented), not reconstructive (present snapshot)
3. Must be bottlenecked (force selection, not copying)
4. Must create complementarity pressure (different modules learn different useful things)
5. Downstream usefulness must be locally visible (detached laterals remove this signal)

The tested family satisfies none of these except weakly (3) via L2 normalization. A working local learning rule would need a fundamentally different design — not a variant of "predict neighbors."

## Experiment queue (completed)

1. **Band0-local-loss WITHOUT detach** — band 0 gets CE + local prediction together. **Result: CE 3.04 at 100 steps.** Worse than control (2.70). Local prediction conflicts with CE even when combined.
2. **Combined: band0-local-loss + hierarchical-targets + attention-readout + detach** — all modules have local targets. **Result: CE 3.35 at 100 steps.** Worse than either alone. Multiple local objectives don't synergize.
3. ~~Two-phase: local only → freeze → train readout~~ — not tested (project concluded)
4. ~~Higher attention LR~~ — not tested (project concluded)

## Key insight: direction of coupling matters (but doesn't solve the problem)

**Horizontal coupling** (predict same-level neighbors): doesn't propagate token information upward. Band 3 predicting its band-3 neighbors learns about band-3 dynamics, not tokens. Token info stays trapped in band 0.

**Vertical coupling** (predict lower-band states): creates a chain from tokens (in band 0) through all bands. Band 1 predicting band 0 must model band 0's dynamics, which encode tokens. Band 2 predicting band 1 must model band 1's dynamics, which encode band 0's dynamics, which encode tokens.

**However:** the corrected purely-local results show vertical coupling ALSO fails at token prediction. Hybrid hierarchical targets (CE on band 0 + vertical on bands 1-7) matches the control at 2.67 — but that's because the CE on band 0 does all the work. When everything is purely local (CE 3.28 → 3.52), vertical coupling doesn't rescue it. The "chain from tokens through all bands" story only works if band 0 is already trained by CE — which defeats the purpose.

So vertical > horizontal for local learning, but neither is sufficient for token-useful representations without global task signal somewhere in the system.

## Why this still might not work

- The detached readout may be fundamentally unable to exploit representations optimized for a different objective.
- Even a projected neighbor target might still be too high-dimensional for MSE to be selective.
- The existing noise level might destroy too much information for the future state to be predictable at all.
- The alignment story assumes band 0 quickly learns something worth predicting; at initialization it does not.
- A warm-up where band 0 trains alone before higher-band prediction starts might be necessary.

## Literature context (quick survey, 2026-05-31)

**This problem appears to be open in the literature.** No published work demonstrates competitive language modeling with purely local learning rules in multi-module systems.

Closest results:
- **Predictive Coding Networks** (Millidge, Salvatori et al., 2022-2024): proven to *approximate* backprop gradients locally, but explicitly frames PC as converging to the same solution backprop would find. Benchmarks remain small-scale (MNIST-level). Not tested on competitive LM.
- **Forward-Forward** (Hinton, 2022): layer-local goodness objective. MNIST only. Explicitly "preliminary."
- **Blockwise/greedy training** (Löwe 2019, Belilovsky 2019): local contrastive objectives get within 2-5% of backprop on ImageNet classification. But: sequential (not parallel), classification only, no sequence prediction results.

The specific combination — parallel multi-module, local objectives, competitive token prediction — has no published positive result. Our negative result is consistent with the state of the field.

## Open questions

- Is the CE stagnation a dynamics issue (readout chasing) or a representation issue (wrong features)?
- What loss should sit on the prediction head: MSE, cosine, or a temporal contrastive loss?
- Should the target be the full future state, a learned projection, or a fixed random projection?
- Does alignment really propagate upward? A direct check would be the gradient cosine between band 1's prediction loss and band 0's CE loss.
- How much lateral noise is enough to prevent copying without making the target effectively random?
- Is the "purely detached" constraint too strict? Maybe a small task signal to the readout is acceptable if modules still learn locally.

## Evaluation of untested candidates (theoretical, 2026-05-31)

Scored against the 5 necessary properties. The fundamental blocker is **(5)**: with detached laterals, no module receives any signal about whether what it produces is downstream-useful for token prediction.

| Candidate | 1 Sufficiency | 2 Predictive | 3 Bottleneck | 4 Complementarity | 5 Downstream visible | Verdict |
|---|---|---|---|---|---|---|
| Delta prediction (state change) | ~ | ✓ | ~ | ✗ | ✗ | Marginal improvement — cheap probe at best |
| MI maximization (MINE) | ✗ | ✗ | ✗ | ✗ | ✗ | Anti-bottleneck, rewards copying shared info |
| VICReg temporal | ~ | ~ | ~ | ✗ | ✗ | Auxiliary shaping only, not a primary objective |
| Forward-Forward | ✗ | ✗ | ~ | ✗ | ✗ | "Looks real" detector ≠ token-useful |
| Contrastive temporal distinction | ✗ | ~ | ✓ | ✗ | ✗ | Same failure mode as InfoNCE with temporal negatives |
| Communication through coherence | ✗ | ✗ | ✗ | ~ | ✗ | Routing mechanism, not a learning rule |
| BYOL/SimSiam self-prediction | ✗ | ~ | ✗ | ✗ | ✗ | Self-consistency ≠ token sufficiency |
| Sparse random subset prediction | ✗ | ~ | ~ | ✗ | ✗ | Random bottleneck ≠ useful bottleneck |

**Conclusion:** No untested candidate satisfies even 3 of 5 properties. The fundamental issue is architectural (property 5), not objective-specific. With detached laterals, a module cannot learn what's useful downstream — so any objective that doesn't explicitly embed task information will produce features optimized for local prediction geometry, not global token utility. This suggests the path forward requires **architecture change** (non-detached communication or explicit utility signals), not just objective redesign.

For now the design-space answer is: **predict something about your neighbors' future through a bottleneck**. The exact form (what target, what loss, how much global signal is acceptable) is the open experimental question with multiple candidates to test.
