# Local learning signal for stale lateral blocks

Serves [dictation 2026-05-30-03](../../../dictations/2026-05-30-03.md) and [ROADMAP Pathway 3: Local Learning](../../../ROADMAP.md#pathway-3-local-learning).

## Status

Theory brief for the open question: what should the local learning signal be at each block in the cellular-automaton architecture? The current MSE prediction loss is not accepted as adequate — it rewards copying rather than useful representation learning. This document surveys mechanisms that could force blocks to learn useful, differentiated representations under stale asynchronous communication. It narrows candidates; it does not close the question.

## The question

The architecture consists of multiple independent feedforward blocks (MLPs), each with its own optimizer, communicating via stale (one-timestep-delayed, gradient-detached) lateral reads. Blocks fire at different rates (multi-rate: periods [1, 2, 4, 8, 16, 32, 64, 128]). Currently, each block's local loss is MSE prediction of the next lateral arrival from the block below: predict what will arrive next from your neighbor. Only block-local signals are available at each interior block — there is no end-to-end gradient. The discriminating requirement: the local signal must make each block learn representations that are (a) useful for downstream computation and (b) different from what neighboring blocks already represent, all under async/stale communication where you cannot assume synchronous settling or fresh gradients from other blocks.

## Why the current MSE target fails

Without noise on laterals, `L_MSE = ||p_i(h_i^t) - x_{i-1}^{t+1}||^2` is trivially minimized by learning the identity statistics of the interface — the block learns to copy or linearly reconstruct the lateral arrival. Loss reaches ~0.0008, but this reflects memorization of the stream's marginal distribution, not extraction of task-useful structure. The target lives on the same distribution as the input because stale laterals are just delayed copies of a neighbor's output. "Predict the next arrival" collapses into "reconstruct the stream" whenever the arrival distribution changes slowly relative to the prediction horizon.

With noise or corruption on laterals, the objective shifts to measuring noise-robustness: the block learns a denoising map, which is a real computation but not the one we want. It does not force selectivity — the block need not decide what matters, only smooth out perturbations. Neither mode forces blocks to learn differentiated representations. MSE is a bad objective here because it has no mechanism to penalize redundancy between blocks or reward discrimination between distinct inputs.

```math
L_{\mathrm{MSE}}^{(i)} = \|p_i(h_i^t) - x_{i-1}^{t+1}\|_2^2
```

Consider a two-block system where block 1 fires every step and block 2 fires every other step. Block 2's lateral target is block 1's output from the previous timestep. If block 1's output changes slowly (e.g., character-level language modeling where consecutive hidden states are highly correlated), block 2 can achieve near-zero MSE by outputting a running mean of recent arrivals — or even a fixed vector close to the stream's unconditional mean. The loss goes to zero; the block does nothing. This is not a pathological edge case; it is the default behavior whenever lateral distributions are autocorrelated, which they always are in sequential processing.

## Mechanism survey

The question is not "which papers are good" but which mechanism creates specialization under stale, asynchronous communication — i.e., forces blocks to learn different, useful things when they cannot see each other's gradients and receive only delayed snapshots.

### 1. InfoNCE / Contrastive Predictive Coding

**How it works.** Instead of regressing toward the next lateral value, the block produces a query representation and must identify the true future arrival among a set of negatives drawn from other timesteps or other blocks:

```math
L_{\mathrm{NCE}}^{(i)} = -\log \frac{\exp(s(q_i^t,\, z_{t+\Delta}))}{\exp(s(q_i^t,\, z_{t+\Delta})) + \sum_{k} \exp(s(q_i^t,\, z_k^-))}
```

where `s` is a learned similarity (e.g., bilinear or cosine), `z_{t+Δ}` is the positive (true future lateral), and `z_k^-` are negatives sampled from other timesteps in the same sequence or from other blocks' outputs.

**What makes blocks specialize.** Copying does not solve this objective. The block cannot merely match distributional moments — it must identify *which specific* future arrival corresponds to its current state among alternatives drawn from the same marginal distribution. Negative choice creates specialization pressure: negatives from other timesteps push the block toward temporal specificity (represent what distinguishes *now* from *then*); negatives from other blocks push toward role specificity (represent something the other block does not already capture).

**Async/stale compatibility.** Fully compatible. Positives and negatives are sampled from a buffer of past lateral arrivals. No synchronous settling is required. The block optimizes against cached data, which is exactly what stale laterals already provide.

**Prior evidence.** CPC (van den Oord et al., 2018), wav2vec 2.0, and subsequent contrastive predictive methods demonstrate strong sequential representation learning in audio and language. The core mechanism — learn representations that are predictive of future states in a discriminative sense — transfers cleanly in principle. Exact transfer to this stale-lateral architecture remains untested.

### 2. VICReg with cross-block covariance penalty

**How it works.** VICReg (Bardes et al., 2022) applies three terms per representation: *invariance* (embedding of augmented views should be similar), *variance* (each dimension should have non-trivial variance across the batch, preventing collapse), and *covariance* (off-diagonal covariance within a block's representation should be zero, preventing redundant features). The project-specific extension adds a fourth term: a cross-block covariance penalty that drives `Cov(h_i, h_j) ≈ 0` for `i ≠ j` on matched temporal windows, directly attacking redundant representations between blocks.

```math
L = \lambda_{\mathrm{inv}} L_{\mathrm{inv}} + \lambda_{\mathrm{var}} L_{\mathrm{var}} + \lambda_{\mathrm{cov}} \sum_i L_{\mathrm{cov}}(h_i) + \lambda_{\mathrm{xblk}} \sum_{i<j} \|C(h_i, h_j)\|_F^2
```

**What makes blocks specialize.** The cross-block term directly penalizes any linear relationship between different blocks' outputs. Unlike InfoNCE, which achieves specialization indirectly through negative design, VICReg+xblk makes redundancy reduction an explicit objective. Two blocks cannot represent the same thing without paying a loss penalty.

**Async/stale compatibility.** Fully compatible. The cross-block covariance penalty only needs cached activations from a recent batch — it does not require fresh gradients through other blocks. Each block's optimizer can compute its contribution to the penalty from its own outputs and a frozen snapshot of its neighbors' outputs.

**Prior evidence.** VICReg and Barlow Twins demonstrate effective collapse prevention and feature diversity in self-supervised learning. Using cross-block covariance in a stale-lateral cellular automaton is a hypothesis — it has not been tested in this exact setting. Computational cost is `O(d²)` per block pair, which is cheap at current hidden sizes (128–512d); a sampled approximation would scale further if needed.

### 3. Forward-Forward

**How it works.** Each block computes a scalar "goodness" measure — typically the sum of squared activations `g(h) = Σ_j h_j²` — and is trained to produce high goodness on positive (real data) passes and low goodness on negative (corrupted/synthetic) passes. Each block optimizes independently with no backward graph through neighbors: the loss is entirely local to the block's own activations.

**What makes blocks specialize.** Forward-Forward alone does not strongly differentiate blocks. Two blocks receiving similar inputs can learn similar goodness detectors without penalty. Specialization requires an additional mechanism — either cross-block decorrelation (same as the VICReg extension above) or role-specific negative generation where each block sees different corruptions targeted at its position in the hierarchy.

**Async/stale compatibility.** Fully compatible. Each block's optimization is entirely self-contained — it only needs its own activations on positive vs. negative passes. No inter-block gradient flow exists by construction.

**Prior evidence.** Hinton (2022) demonstrates the concept and shows it works on small-scale classification. Evidence at scale is weaker than contrastive/self-supervised methods. The algorithm is conceptually elegant and maximally local, but its capacity to produce genuinely load-bearing representations in a multi-block hierarchy — without additional decorrelation pressure — is uncertain.

### 4. Communication through coherence / phase-amplitude coupling

**How it works.** This is a communication mechanism and inductive bias, not yet a complete loss by itself. Blocks operating at different rates (as in this architecture's multi-rate design) naturally occupy different temporal slots. Useful information is exchanged preferentially when sender and receiver phases align — i.e., a fast block (period 1) might only meaningfully update its state from a slow block (period 128) at specific alignment points, and vice versa. The stale lateral read can be reinterpreted as a phase delay rather than merely a degradation.

**What makes blocks specialize.** Different frequencies and phase windows privilege different timescales of information. A block firing every 128 steps cannot represent fast-changing features; it is structurally forced to capture slow statistics. The multi-rate design already provides this inductive bias implicitly — the question is whether making it explicit (gating lateral reads by phase alignment, weighting arrivals by coherence) strengthens the specialization.

**Async/stale compatibility.** Unusually strong. Staleness is part of the model rather than a violation of assumptions — it *is* the phase delay. This mechanism does not fight the architecture; it embraces it.

**Prior evidence.** The neuroscience literature on communication through coherence (Fries, 2005; 2015) and phase-amplitude coupling provides strong conceptual motivation. Evidence in deep learning systems is thinner and more architectural than loss-level. Whether this can be formalized as a learning signal (rather than just an architectural choice) in a way that produces measurably better specialization remains open.

### 5. Anti-Hebbian / decorrelation rules

**How it works.** Discourage correlated activity across blocks: if two blocks produce correlated outputs on the same input window, apply an anti-Hebbian update or a Barlow-Twins-style cross-correlation penalty to push them apart. The core rule is: `Δw ∝ -h_i · h_j` across blocks, or equivalently penalize the off-diagonal entries of the cross-block correlation matrix.

**What makes blocks specialize.** *Within-block* decorrelation prevents feature collapse inside a single block (standard regularization). *Between-block* decorrelation is the point here — it directly prevents two blocks from representing the same thing. The appeal is simplicity: no contrastive sampling, no negative generation, just "be different from your neighbors."

**Async/stale compatibility.** Yes. The penalty operates on cached activations/statistics. Each block needs only its own current output and a frozen snapshot of its neighbor's recent output — exactly what stale laterals provide.

**Prior evidence.** Redundancy-reduction principles (Barlow, 1961; Barlow Twins, HSIC-based methods) are well-established for preventing collapse and encouraging diverse features. However, decorrelation alone does not guarantee task usefulness — it ensures blocks are *different* but not that the differences are *useful*. Without a predictive or contrastive anchor tying representations to the input structure, blocks could decorrelate into arbitrary orthogonal noise. It likely needs pairing with a positive objective.

### 6. Predictive coding (amortized residual prediction)

**How it works.** The relevant version is not the classical equilibrium-settling story (Rao & Ballard, 1999) but an amortized single-pass variant: each block predicts the residual information not already explained by neighboring blocks or slower-timescale context, in a single forward pass. Block `i` receives a prediction from block `i+1` (the slower, more abstract level) and is rewarded for explaining what that prediction missed — the residual `r_i = x_i - pred_{i+1→i}(h_{i+1})`.

**What makes blocks specialize.** Each block is explicitly rewarded for explaining what others have *not* already explained. Higher blocks capture slow/abstract regularities; lower blocks capture fast/specific residuals. This is a natural hierarchy-forcing signal.

**Async/stale compatibility.** Classical predictive coding requires iterative synchronous settling between levels — incompatible with this architecture. Amortized/single-pass variants avoid iterative settling but introduce a subtlety: the prediction from block `i+1` must be computed from a stale snapshot, not a fresh synchronized state. Whether this stale prediction is accurate enough to produce a meaningful residual target is implementation-sensitive. Async compatibility: classical form no; amortized form possibly yes, but fragile.

**Prior evidence.** Strong theoretical relevance to hierarchical representation learning (Friston's free-energy principle, hierarchical predictive processing). But the implementation details under stale asynchronous communication are underspecified relative to InfoNCE or VICReg. Getting the residual target right without synchronous information about what other blocks currently explain is the core difficulty.

### 7. Equilibrium propagation

Equilibrium propagation (Scellier & Bengio, 2017) requires synchronous settling to an equilibrium state followed by a perturbed phase — a two-phase dynamic where all units must reach steady state before learning can occur. This is fundamentally incompatible with the target architecture, which fires blocks asynchronously at different rates with no global settling phase. Not a candidate.

## What seems true so far

**We know:**
- Plain MSE prediction of the next lateral arrival does not force selectivity — it rewards copying/reconstruction and produces near-zero loss without useful representations.
- Async/stale compatibility is a hard constraint, not a nice-to-have. Any mechanism requiring synchronous settling or fresh inter-block gradients is ruled out by construction.
- The architecture already contains partial multi-rate/phase structure (block rates [1, 2, 4, 8, ...128]), which is an unexploited inductive bias for timescale separation.
- Specialization requires both a positive signal (learn something useful about the input) and a negative signal (learn something different from your neighbors). Neither alone is sufficient.

**We do not know:**
- Which objective actually yields load-bearing interior blocks — blocks whose removal measurably degrades system performance.
- Whether explicit anti-redundancy (cross-block decorrelation) is enough on its own, or must be paired with discriminative prediction (InfoNCE-style) to ensure the *content* of specialization is task-relevant.
- Whether the best intervention is primarily at the loss level, the architecture level (gating, phase structure), or necessarily both.
- Whether any of these candidates work in practice at the scales and sequence lengths this architecture targets. This document is theory; the experiments have not been run.

## Recommended candidates for this architecture

1. **InfoNCE + cross-block covariance**

   Current best fit. InfoNCE supplies discriminative predictive pressure — the block must identify the true future among plausible alternatives, which copying cannot solve. The cross-block covariance term directly suppresses redundancy between blocks. Together they address both failure modes of MSE: copying is insufficient (InfoNCE), and specialization is not left to chance (covariance penalty). Risk: negative sampling design may dominate behavior. If the negative queue is too easy (distant timesteps) or too hard (adjacent timesteps with near-identical content), the signal degrades. Could become expensive or brittle if queue/buffer management is wrong.

2. **Forward-Forward + cross-block decorrelation**

   Simpler and more purely local than InfoNCE — no contrastive queue, no bilinear scoring, just local goodness maximization plus a decorrelation penalty. Strong async compatibility by construction. But weaker empirical backing at scale and weaker built-in pressure toward task-relevant (as opposed to merely distinct) representations. Risk: choosing useful negatives/corruptions for the negative pass, and setting the goodness threshold correctly. Without careful negative design, blocks may learn to distinguish "real vs. garbage" without learning anything about the task structure.

3. **Explicit phase/frequency gating**

   Already partially present via multi-rate blocks — the cheapest path may be to make the temporal-slot specialization mechanism explicit (gate lateral reads by phase alignment, weight arrivals by coherence) rather than inventing a wholly new learning signal. This is more an architectural bias than a complete learning signal: it organizes *when* blocks talk but does not determine *what* they learn to say. Almost certainly pairs with one of the losses above rather than standing alone. Risk: may organize communication without guaranteeing useful content — clean temporal separation that carries no task-relevant information.

4. **Amortized predictive coding**

   Conceptually the strongest hierarchy-forcing signal: each block explains what others have not already explained, directly producing a division of labor. But it is the most design-sensitive candidate. The residual target depends on a stale prediction from the block above, which may be inaccurate enough to produce a noisy or misleading target. Hidden synchrony assumptions are easy to introduce accidentally. Risk: implementation complexity and the possibility that stale residual targets degrade the signal to the point where it offers no advantage over simpler methods.

## Proposed 30-second ablations

These are discriminating probes — 30-second sanity checks that test whether a mechanism shows any immediate signal, not training runs to convergence.

| Experiment | Change | Readout / metric | What would count as success | What it would teach us |
|---|---|---|---|---|
| `nce-vs-mse-two-block` | Replace MSE on one interface with InfoNCE using negatives from other timesteps in the same batch | Local loss, block ablation gap, short-run val loss | Non-collapse plus larger ablation gap than MSE | Whether discriminative prediction creates immediately more load-bearing helpers |
| `nce-negatives-block-vs-time` | Compare time negatives only vs time+other-block negatives | Same as above, plus cross-block correlation | Lower cross-block redundancy with mixed negatives | Whether negative design can explicitly force role separation |
| `vicreg-xblk` | Add cross-block covariance penalty to current local objective | Cross-block correlation matrix, local loss, ablation gap | Reduced redundancy without collapse | Whether anti-redundancy alone already buys specialization |
| `ff-decorrelation` | Forward-Forward local goodness with simple corruption + cross-block decorrelation | Goodness separation, ablation gap, short-run val loss | Stable positive/negative separation and nonzero helper contribution | Whether a simpler purely local rule is viable at all |
| `phase-gated-interface` | Gate lateral reads by explicit phase/rate alignment instead of unconditional stale read | Same as above, plus per-block utilization | Distinct utilization by rate/phase and improved helper usefulness | Whether temporal slotting is already enough to create different roles |
| `amortized-residual-target` | Predict residual of future lateral after subtracting lower-block baseline predictor | Residual norm explained, ablation gap | Better-than-MSE helper usefulness without synchrony | Whether "predict what others did not explain" is the right hierarchy pressure |

## Non-goals / what this does not settle

- This document does not prove any candidate works in this codebase. It is a theory brief, not an experimental report.
- It does not settle the exact block topology (number of blocks, hidden sizes, which blocks connect to which).
- It does not answer whether local learning can match full backprop at scale — that question is much larger and longer-term.
- It does not decide between loss-only interventions and architecture-plus-loss interventions; both may be necessary.
- It does not specify hyperparameters, learning rates, or loss weightings for any candidate.
- It does not address the readout/output loss — only the interior block local signals. How the final block's output connects to a task loss is a separate question.

## Bottom line

MSE prediction of the next lateral arrival is a poor proxy for useful local learning because it rewards reconstruction and copying rather than discriminative, non-redundant representation building. A block that achieves near-zero MSE may be doing nothing — memorizing the stream's unconditional statistics. The best current candidate is InfoNCE-style contrastive future prediction paired with an explicit cross-block anti-redundancy term: the first forces the block to identify specific futures (not just match moments), the second forces it to do so differently from its neighbors. The cheapest backup is Forward-Forward plus decorrelation — maximally local, no contrastive queue, but less proven at scale. The multi-rate phase structure already present in the architecture should probably be exploited explicitly (phase-gated communication) rather than treated as incidental, though it is an architectural bias rather than a complete learning signal. All of this remains theory until the ablations run.
