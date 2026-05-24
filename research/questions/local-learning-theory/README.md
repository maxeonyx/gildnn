# Local learning theory: first principles and literature

Serves [dictation 2026-05-24-5](../../../dictations/2026-05-24-5.md) and [dictation 2026-05-23-8](../../../dictations/2026-05-23-8.md).

## Status: RESEARCH DIRECTION — not yet experimentally tested

This document lays out the theoretical landscape of local learning rules and how they apply to Max's architecture. It is a research brief, not a results report.

---

## The architecture (corrected framing)

Per dictation 2026-05-24-5: **these are not RNNs with hidden state.** They are feedforward blocks:

- Each block takes an input at time t, produces an output at time t+1
- That output is **public** — available to other blocks, not a hidden state
- Block 0 receives token embeddings and has next-token prediction loss
- Other blocks receive laterally-propagated outputs from neighbors
- Block 1 should predict something **block 0 couldn't already know** — e.g. information dependent on input from a longer time ago

The key insight: upper blocks have access to *history* (via accumulated lateral signals over time). Their value is providing temporal context that the token-adjacent block can't compute from its current input alone.

---

## First principles: what must be true for local learning to work?

### 1. Alignment signal

A module without task access must receive *some* signal correlated with task usefulness. Without this, it's doing unsupervised learning — which can produce useful representations but with no guarantee they help the task.

Possible signals (ranked by directness):
1. **Targets** — "here's what your output should have been" (target propagation)
2. **Prediction errors** — "your prediction of my state was wrong by this much" (predictive coding)
3. **Gradient estimates** — "change your output in this direction" (synthetic gradients)
4. **Contrastive signal** — "this is what good output looks like, this is bad" (Forward-Forward)
5. **Scalar reward** — "that was good/bad overall" (RL-style, weakest)

### 2. Non-degeneracy

The local objective must prevent:
- Constant output (ignoring input)
- Identity function (passing input unchanged, no transformation)
- Dimensional collapse (using only a subspace)

This is what killed the strict-local experiment (D_strict_local collapsed because pred_loss alone didn't prevent the prediction head from converging on trivially-predicted but task-irrelevant features).

### 3. Information preservation

The module must not discard information that the task-adjacent block needs. An aggressive compression objective can be catastrophic. The module's output should preserve task-relevant information even if the local objective doesn't explicitly require it.

### 4. Compatibility

Independently-trained modules must produce outputs in a form others can use. This requires either shared normalization conventions, fixed-dimension interfaces, or gradual co-adaptation.

### 5. The fundamental trade-off

**A local learning rule can match global backprop if and only if the local signal contains information equivalent to the projection of the global gradient onto the local parameter space.** All known methods are different approximations of this projection.

---

## Literature: how has this been solved?

### Predictive coding (most relevant)

**Core idea (Rao & Ballard 1999, Friston 2005-2010):** The brain maintains a hierarchical generative model. Each level predicts the level below via top-down connections. Bottom-up connections carry only *prediction errors*. Learning = adjust weights to minimize local prediction error.

**Operationalization in neural nets:** Each layer l has representation neurons μ_l and error neurons ε_l = x_l − f(μ_l). Weight update: Δθ ∝ ε_l · ∂f/∂θ — purely local. Whittington & Bogacz 2017 proved this converges to exact backprop gradients at equilibrium.

**Why it's most relevant here:** Blocks predict each other's outputs. Prediction errors flow laterally. The block with task access generates top-level errors that propagate through inter-block predictions. No global backward pass needed.

**Limitation:** Requires iterating to equilibrium (multiple forward passes before weight update). For Max's architecture, this could mean multiple lateral communication rounds per timestep, or we accept a biased approximation (one round, like the current setup).

### Target propagation

**Core idea (Lee et al. 2015, Meulemans et al. 2020):** Propagate *targets* backward instead of gradients. Each layer has a learned approximate inverse. Given a target for the output, compute a target for the input using the inverse. Each layer minimizes distance to its target.

**Why it's relevant:** The task-adjacent block can compute targets for its input (what lateral signal would have been most useful). Those targets become teaching signals for upstream blocks.

**Limitation:** Requires learning inter-block inverse mappings. Non-invertible transformations (ReLU, dimensionality reduction) are problematic. Recently scaled to ImageNet (Meulemans et al. 2022).

### Local greedy training (Belilovsky et al. 2019, Löwe et al. 2019)

**Core idea:** Each block gets its own auxiliary task head. Train blocks greedily.

**When it works:** When individual blocks naturally extract useful features even without coordination. Performance gap is 1-3% on ImageNet for CNN blocks.

**Why it's relevant:** Simple and proven to scale. Each block could have its own next-token prediction head. The question is whether this produces representations that *help the task-adjacent block*, not just solve the task independently.

**Limitation:** Greedy features may be locally optimal but globally suboptimal. Early blocks may discard info that later blocks need.

### Forward-Forward (Hinton 2022)

**Core idea:** Two forward passes (positive on real data, negative on corrupted data). Each layer maximizes "goodness" (sum of squared activations) for positive data, minimizes for negative.

**Limitation:** Only tested on MNIST. Performance well below backprop. Negative data generation is crucial and not well understood.

**Relevance:** Interesting because fully local and forward-only. Each block could have its own goodness function. The challenge is what constitutes "negative" data in a lateral-block setting.

### Synthetic gradients (Jaderberg et al. 2017)

**Core idea:** Learn a small network that predicts what the gradient will be. Update immediately using predicted gradient. Train the predictor when actual gradient arrives.

**Why it mostly failed:** Bootstrapping problem (early predictions are garbage → layers learn from garbage), circular dependency, instability at scale. Worked only on small problems.

### Equilibrium propagation (Scellier & Bengio 2017)

**Core idea:** For energy-based nets. Two phases: free-running (settle to equilibrium), then nudge output toward target and re-settle. Weight update = difference in Hebbian correlations between phases. Mathematically exact as nudge → 0.

**Limitation:** Requires energy-based architecture, symmetric weights, two full settling phases. Hard to scale. Not directly applicable to feedforward blocks unless you frame lateral communication as iterative settling.

### Contrastive Hebbian / HSIC bottleneck

Less directly applicable. HSIC (Ma et al. 2020) maximizes kernel independence between representations and labels — but requires label access at every layer. Could work if the task-adjacent block broadcasts a compressed signal.

---

## What neuroscience says

**Cortical columns:** ~150,000 columns in the cortex, each ~0.5mm diameter. Connected laterally within areas and hierarchically between areas. Learning uses local synaptic plasticity (Hebbian, STDP).

**Layers carry different signals:** Superficial layers (2/3) carry prediction errors *forward/upward*. Deep layers (5/6) carry predictions *backward/downward*. This is the neuroanatomical basis for predictive coding.

**No evidence of backprop-like mechanism:** No symmetric forward/backward weights, no separate phases, no continuous error signal propagation. But there IS evidence for feedback alignment (random feedback weights work — Lillicrap et al. 2016) and dendritic error computation (apical dendrites may carry top-down error signals).

**Neuromodulatory broadcast:** Dopamine, acetylcholine, norepinephrine modulate plasticity globally — acting as broadcast reward/surprise/uncertainty signals. Combined with local eligibility traces, this could solve credit assignment without backprop.

**Current consensus:** The brain probably implements something that *approximates* credit assignment across layers — likely predictive coding or dendritic computation — but not exact backprop.

---

## How this maps to Max's architecture

Max's blocks are parallel, communicate laterally, and only block 0 has task access. The key question: **what should upper blocks' local objective be?**

### The natural fit: predictive coding with temporal context

Block 1 sees block 0's output history (from past timesteps). It can:
1. **Predict block 0's future output** — using temporal patterns that block 0 can't compute from its current input alone
2. Feed that prediction to block 0 as a "prior" or conditional input
3. Train on prediction error (how wrong was the prediction?)

This is predictive coding applied laterally. The prediction error is the local signal. Block 0's actual output (shaped by CE loss) provides the grounding.

**Why this is different from what the agent tested:** The agent's "closed-loop" experiment had block 1 predict block 0's *state* (full hidden vector). Max's framing is different: block 1 should predict something block 0 *doesn't already know*. This means the prediction target should NOT be block 0's current state (which block 0 already has), but rather:
- Block 0's *future* state (which block 0 hasn't computed yet)
- A compression of block 0's *history* (which block 0 can't store in one vector)
- A higher-level pattern that explains multiple timesteps of block 0's output

### Immediate next step: Phase 5 prediction targets

Concrete experiments are designed in [`research/questions/local-learning-variants/README.md`](../local-learning-variants/README.md) — see Phase 5 (J/K/L variants). These test Max's framing directly:
- J: predict a 4-step-old window summary of block 0's history
- K: predict the nonlocal residue (older context minus recent context)
- L: predict a future 4-token chunk code

All three keep the semi-local architecture (CE through interface) while changing the prediction target. This tests whether the -0.006 benefit ceiling is a target problem (full-state is redundant info block 0 already has) or an interface problem (the gate can only carry ~0.006 nats regardless).

### Longer-term investigations

1. **Literature deep-dive on predictive coding implementations** — specifically Whittington & Bogacz 2017, Millidge et al. 2021, and any 2023-2025 work on scaling predictive coding to language models or sequential tasks.

2. **Target propagation feasibility** — can the task-adjacent block compute useful targets for the lateral signal it receives? What does the inverse mapping look like for this architecture?

3. **The connection to dynamic hierarchical prediction** — if each block operates at a different timescale (per the autoregressive autoencoder vision in dictation 2026-05-24-3), then each level's "local learning" is just next-chunk prediction at its own level. The hierarchy provides the alignment signal naturally: level K's prediction errors are level K+1's input.

4. **Good target + strict-local:** If Phase 5 shows that a better target significantly helps under semi-local, follow-up: does the same target also work under strict-local (prediction error only, no CE through interface)? This would test whether a *well-chosen target* removes the need for task-gradient shaping entirely. The earlier E_grounded failure was with a bad target (full-state); a good target might change the outcome.

---

## What this does NOT cover

- The async/parallelism execution question (separate from learning rule)
- Graph topology beyond chain/star (separate question)
- CUDA implementation concerns
- Whether this is competitive with transformers (separate question — the point is understanding the mechanisms)
