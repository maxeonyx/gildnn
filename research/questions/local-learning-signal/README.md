# Local learning signal for stale lateral blocks

Serves [dictation 2026-05-30-03](../../../dictations/2026-05-30-03.md) and [ROADMAP Pathway 3: Local Learning](../../../ROADMAP.md#pathway-3-local-learning).

## Status

<!-- TODO(author): Write a 2-4 sentence status block. Audience: Max. Must say this is a theory/research brief for the open question "what should the local learning signal be at each block?" in the current cellular-automaton-style architecture. Must explicitly state that the current MSE prediction loss is not accepted as adequate. Must preserve uncertainty: this document narrows candidates; it does not close the question. -->

## The question

<!-- TODO(author): Write one dense paragraph stating the exact question. Must describe: multiple independent feedforward blocks; stale lateral reads; per-block optimizers; current local loss = prediction of next lateral arrival; only block-local signals are guaranteed at each interior block. Must state the discriminating requirement: the signal must make blocks learn something useful and different from their neighbors under async/stale communication. -->

## Why the current MSE target fails

<!-- TODO(author): Write 2 short paragraphs plus the formula below. Must explain both failure modes precisely: (1) without noise, `L_MSE = ||p_i(h_i^t) - x_{i-1}^{t+1}||^2` is trivially minimized by copying or learning the identity statistics of the interface, so low loss does not imply useful representation learning; (2) with noise/corruption, the objective mostly measures denoising or noise-robustness, not whether the block extracted task-useful structure. Must connect this to the architecture's stale laterals: the target lives on the same marginal distribution as the input, so "predict the next arrival" can collapse into "reconstruct the stream". Must say explicitly that MSE is a bad objective here because it does not force selectivity. -->

```math
L_{\mathrm{MSE}}^{(i)} = \|p_i(h_i^t) - x_{i-1}^{t+1}\|_2^2
```

<!-- TODO(author): Add one compact example or thought experiment showing why copying solves the objective when the arrival distribution changes slowly. Keep it concrete and architecture-specific. -->

## Mechanism survey

<!-- TODO(author): Write a 1-2 sentence bridge paragraph. Must say the question is not "which papers are good" but "which mechanism creates specialization under stale, asynchronous communication." -->

### 1. InfoNCE / Contrastive Predictive Coding

<!-- TODO(author): Write this subsection with four explicit subheads: How it works; What makes blocks specialize; Async/stale compatibility; Prior evidence. Must include an InfoNCE-style objective such as `L = -log exp(s(q_i^t, z_{t+\Delta}))/[exp(s(q_i^t, z_{t+\Delta})) + Σ_k exp(s(q_i^t, z_k^-))]`. Must explain that the block predicts a future lateral representation contrastively, not by regression. Must explicitly say copying does not solve the task when negatives come from the same marginal distribution (other timesteps / other blocks), because the representation must identify the true future, not merely match distributional moments. Must explain that specialization pressure comes from negative choice: other timesteps push temporal specificity; other blocks push role specificity. Must mark it as fully compatible with async/stale reads because positives/negatives can be sampled from buffers without synchronous settling. Prior evidence to mention: CPC / wav2vec / modern contrastive predictive methods show strong sequential representation learning; exact transfer to this architecture remains untested. -->

### 2. VICReg with cross-block covariance penalty

<!-- TODO(author): Write this subsection with the same four subheads. Must explain the three standard VICReg terms: invariance, variance, covariance. Then add the project-specific extension: a cross-block covariance penalty between different blocks' outputs so `Cov(h_i, h_j) \approx 0` for `i != j` on matched inputs/windows. Must say this directly attacks redundant representations rather than hoping specialization emerges indirectly. Must include one compact formula block showing the total objective shape, e.g. `L = λ_inv L_inv + λ_var L_var + λ_cov Σ_i L_cov(h_i) + λ_xblk Σ_{i<j} ||Cov(h_i, h_j)||_F^2`. Must note complexity is cheap enough at current hidden sizes (`O(d^2)` per block pair or a sampled approximation). Must explicitly say this is fully async-compatible because the penalty only needs cached activations, not fresh gradients through other blocks. Prior evidence: VICReg / Barlow-style redundancy reduction is established for collapse prevention and feature diversity, but cross-block use in this exact stale-lateral setting is still a hypothesis. -->

### 3. Forward-Forward

<!-- TODO(author): Write this subsection with the same four subheads. Must explain Hinton's mechanism: local goodness such as `g(h)=Σ_j h_j^2`, positive passes on real data, negative passes on corrupted/fake data, each block trained to increase goodness on positive and reduce it on negative. Must say why it fits this architecture: each block can optimize independently with no backward graph through neighbors. Must also say clearly that Forward-Forward by itself does not strongly differentiate blocks; two blocks can still learn similar detectors unless paired with explicit cross-block decorrelation or role-specific negatives. Async/stale compatibility: fully compatible. Prior evidence: conceptually elegant and local, but evidence at scale is weaker than contrastive/self-supervised methods; mostly small-scale demonstrations. -->

### 4. Communication through coherence / phase-amplitude coupling

<!-- TODO(author): Write this subsection with the same four subheads. Must frame it as a communication mechanism and an inductive bias, not yet a complete loss by itself. Explain that blocks at different rates/frequencies can naturally occupy different temporal slots; useful information is exchanged when sender/receiver phases align. Must connect directly to the existing architecture: multi-rate blocks already exist, and stale reads can be interpreted as phase delay rather than merely a bug. Specialization mechanism: different frequencies/phase windows privilege different timescales, so blocks need not represent the same content. Async/stale compatibility: unusually strong, because staleness is part of the model rather than a violation of assumptions. Prior evidence: neuroscience motivation and oscillatory communication literature support the idea conceptually; evidence in deep learning systems is thinner and more architectural than loss-level. Must keep uncertainty explicit. -->

### 5. Anti-Hebbian / decorrelation rules

<!-- TODO(author): Write this subsection with the same four subheads. Must explain the core rule: discourage correlated activity across blocks or features, e.g. anti-Hebbian updates or Barlow-Twins-style cross-correlation penalties. Must distinguish within-block decorrelation from between-block decorrelation, and emphasize that the latter is the point here. Must say the appeal is simplicity and direct pressure against redundancy, but also that decorrelation alone does not guarantee task usefulness; it needs either a predictive/contrastive anchor or a positive objective. Async/stale compatibility: yes, because it operates on cached activations/statistics. Prior evidence: redundancy-reduction principles are strong; using them as the sole local learning signal for useful hierarchical computation remains uncertain. -->

### 6. Predictive coding (amortized residual prediction)

<!-- TODO(author): Write this subsection with the same four subheads. Must explain the relevant version, not the classical equilibrium-settling story: each block predicts residual information not already explained by neighboring blocks or slower-timescale context, in a single amortized pass. Must explicitly say why classical predictive coding is a poor fit here if it requires iterative synchronous settling, but amortized/single-pass variants may preserve the hierarchy-forcing idea without breaking async execution. Specialization mechanism: each block is rewarded for explaining what others have not already explained. Async/stale compatibility: classical form no; amortized form maybe yes. Prior evidence: strong theoretical relevance to hierarchical representation learning, but the implementation details here are fragile and underspecified relative to InfoNCE/VICReg. -->

### 7. Equilibrium propagation

<!-- TODO(author): Write a very short subsection. Must explicitly mark this as incompatible with the target architecture because it requires synchronous settling / two-phase dynamics. One sentence on mechanism, one sentence on incompatibility, then stop. -->

## What seems true so far

<!-- TODO(author): Write a compact synthesis section. Must separate knowledge from uncertainty. "We know" should include: plain MSE does not force selectivity; async/stale compatibility is a hard requirement, not a nice-to-have; the architecture already contains partial multi-rate / phase structure. "We do not know" should include: which objective actually yields load-bearing interior blocks; whether explicit anti-redundancy is enough or must be paired with predictive discrimination; whether the best signal is primarily loss-level, architecture-level, or both. -->

## Recommended candidates for this architecture

<!-- TODO(author): Write a ranked list with short but concrete justifications. Must preserve this ranking exactly unless the evidence in this file itself justifies a different order, which it currently does not. Each item should explain why it fits stale asynchronous communication and what key risk remains. -->

1. **InfoNCE + cross-block covariance**
   <!-- TODO(author): Justification requirements: say this is the current best fit because InfoNCE supplies discriminative predictive pressure and cross-block covariance directly suppresses redundancy. Must note that the combination addresses both failure modes of MSE: copying is not enough, and specialization is not left to chance. Risk: negative sampling design may dominate behavior; could become expensive or brittle if the queue design is wrong. -->

2. **Forward-Forward + cross-block decorrelation**
   <!-- TODO(author): Justification requirements: say this is simpler and more purely local than InfoNCE, with strong async compatibility, but weaker empirical backing and weaker built-in pressure toward task-relevant representations. Risk: choosing useful negatives/corruptions and goodness threshold. -->

3. **Explicit phase/frequency gating**
   <!-- TODO(author): Justification requirements: say this is already partially present via multi-rate blocks, so the cheapest path may be to make the temporal-slot specialization mechanism explicit rather than inventing a wholly new one. Must note that this is more an architectural bias than a complete learning signal, so it likely pairs with one of the losses above. Risk: may organize communication without guaranteeing useful content. -->

4. **Amortized predictive coding**
   <!-- TODO(author): Justification requirements: say this is conceptually strong because it directly targets hierarchical residual explanation, but it is the most design-sensitive and easiest to get wrong under async stale communication. Risk: implementation complexity and hidden synchrony assumptions. -->

## Proposed 30-second ablations

<!-- TODO(author): Write a short intro sentence saying these are discriminating probes, not long training runs. Then fill the table below. Each row must be concrete enough to run as a 30-second sanity check. -->

| Experiment | Change | Readout / metric | What would count as success | What it would teach us |
|---|---|---|---|---|
| `nce-vs-mse-two-block` | Replace MSE on one interface with InfoNCE using negatives from other timesteps in the same batch | Local loss, block ablation gap, short-run val loss | Non-collapse plus larger ablation gap than MSE | Whether discriminative prediction creates immediately more load-bearing helpers |
| `nce-negatives-block-vs-time` | Compare time negatives only vs time+other-block negatives | Same as above, plus cross-block correlation | Lower cross-block redundancy with mixed negatives | Whether negative design can explicitly force role separation |
| `vicreg-xblk` | Add cross-block covariance penalty to current local objective | Cross-block correlation matrix, local loss, ablation gap | Reduced redundancy without collapse | Whether anti-redundancy alone already buys specialization |
| `ff-decorrelation` | Forward-Forward local goodness with simple corruption + cross-block decorrelation | Goodness separation, ablation gap, short-run val loss | Stable positive/negative separation and nonzero helper contribution | Whether a simpler purely local rule is viable at all |
| `phase-gated-interface` | Gate lateral reads by explicit phase/rate alignment instead of unconditional stale read | Same as above, plus per-block utilization | Distinct utilization by rate/phase and improved helper usefulness | Whether temporal slotting is already enough to create different roles |
| `amortized-residual-target` | Predict residual of future lateral after subtracting lower-block baseline predictor | Residual norm explained, ablation gap | Better-than-MSE helper usefulness without synchrony | Whether "predict what others did not explain" is the right hierarchy pressure |

## Non-goals / what this does not settle

<!-- TODO(author): Write 3-6 bullets. Must include: this README does not prove any candidate works in this codebase; it does not settle the exact block topology; it does not answer whether local learning can match full backprop at scale; it does not decide between loss-only and architecture-plus-loss interventions. -->

## Bottom line

<!-- TODO(author): Write one final paragraph. Must say: MSE prediction of the next lateral arrival is a poor proxy for useful local learning because it rewards reconstruction/copying instead of discriminative, non-redundant representation building. The best current candidate is InfoNCE-style future prediction paired with an explicit cross-block anti-redundancy term. The cheapest backup is Forward-Forward plus decorrelation. Multi-rate phase structure should probably be exploited explicitly rather than treated as incidental. Keep uncertainty explicit. -->
