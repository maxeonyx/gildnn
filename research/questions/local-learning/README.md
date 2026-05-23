# Local learning for parallel multi-rate blocks

Serves [dictation 2026-05-23-5](../../../dictations/2026-05-23-5.md): "How can I get local learning, enabling parallelism — some mechanism to learn well without requiring global backpropagation?"

## Status

**INVALIDATED — tested on wrong architecture.** The experiment below was run with `token_injection="all"` (every block sees tokens directly). Per [dictation 2026-05-23-7](../../../dictations/2026-05-23-7.md): only block 0 should receive the token embedding. Higher blocks depend on lateral propagation for their input.

Max's exact words: "of course they don't matter if every block already has direct access to the tokens. In my intended architecture, deeper blocks *depend* on lateral propagation for their input."

The results below are factually correct for the old (wrong) architecture but do not answer the local learning question for Max's intended design. Must re-test after architecture correction (`token_injection="block0"`, committed in `93147c1`).

### Old result (wrong architecture, kept for reference)

## Results

| Seed | Full backprop | Lateral detached | Delta |
|------|---------------|------------------|-------|
| 42 | 1.718 | 1.720 | +0.002 |
| 43 | 1.749 | 1.707 | **-0.042** |
| 44 | 1.737 | 1.752 | +0.015 |
| **Avg** | **1.735** | **1.726** | **-0.009** |

Artifact: [`experiments/fixed_multi_rate/artifacts/local_learning_gradient/`](../../../experiments/fixed_multi_rate/artifacts/local_learning_gradient/)

The "1-hop truncated" condition produced identical results to "fully detached" — because with only neighbor-lateral connections in a parallel architecture, there's at most 1 hop per timestep. The distinction is architecturally meaningless here.

**Interpretation:** The shared additive stream provides ALL the useful gradient signal. The lateral connections carry useful *forward* information (for quality — we know removing them in the forward pass hurts), but their *gradient* contribution is negligible. Blocks can be trained as if they were independent modules all writing to the same output, because mathematically that's what the gradient tells us they are.

## The mathematical result

Consider the current architecture: parallel blocks writing additively to a shared residual stream.

```
S = x + h_1 + h_2 + ... + h_N
L = loss(S)
```

The gradient of L with respect to any block's output h_i is:

```
dL/dh_i = dL/dS    for all i
```

This is exact. Every block receives the *same* gradient signal — the shared adjoint `g = dL/dS`. Each block then only needs its own local Jacobian to compute parameter gradients. No block needs to see through any other block.

**Training parallelism for the readout path is literally free.** There is no approximation. The full gradient decomposes into a single shared broadcast plus N independent local backwards.

### Multi-rate extension

When block i fires at interval k_i (output cached between firings):

```
G_i = Σ_{t in caching interval} g_t
```

The block's effective gradient is the time-integrated shared adjoint over its caching window. Slower blocks train on the aggregate usefulness of their cached representation across multiple steps. This is also exact for the additive readout path.

### What breaks exactness

The current architecture has **lateral propagation**: block j reads block i's previous output. This introduces extra gradient terms beyond the shared adjoint:

```
true gradient = g + [lateral terms from downstream blocks reading this block's output]
```

These lateral terms are the only thing preventing fully parallel training. The design space is a spectrum:

| Truncation level | What you compute | Parallelism | Quality cost |
|---|---|---|---|
| Full backprop | All terms | Serial | Zero (baseline) |
| k-hop truncation | Lateral terms within radius k | Partial | Unknown |
| Shared-adjoint only | g broadcast, lateral terms dropped | Fully parallel | **Unknown — this is what we measure** |

## Hypotheses

1. **H1: Lateral terms are small.** For the additive multi-rate architecture, most gradient signal comes from the shared adjoint; lateral terms contribute little. If true, parallel training is nearly free.

2. **H2: Lateral terms matter but are approximable.** Full truncation hurts noticeably, but 1-hop truncation recovers most of the quality — allowing partial parallelism with small cost.

3. **H3: Lateral terms are critical.** The blocks learn cooperative representations that require global backprop. Local learning approaches in this architecture need synthetic gradient models or distillation heads.

## Mechanisms (ranked by expected quality–parallelism tradeoff)

1. **Shared-adjoint exact training** — Compute g = dL/dS once, broadcast to all blocks, each does local backward. Exact for readout path. Zero quality cost for that component. Lateral terms truncated.

2. **Per-block LM distillation heads** — Each block has its own token prediction head, trained against teacher logits from a detached global head. Fully parallel, safe LM objective. Expected cost ~0.03–0.15 nats.

3. **Neighborhood-truncated backprop** — Backprop within radius-r neighborhood only. Blocks with non-overlapping neighborhoods train concurrently. Expected cost ~0.05–0.20 nats.

4. **Synthetic gradient interfaces (DNI)** — Learn a model that predicts the downstream gradient. Maximum parallelism but historically unstable. Expected cost ~0.10–0.30 nats.

5. **Rate-matched hierarchical CPC** — Slow blocks predict future fast-block features. Matches Max's hierarchical prediction vision but may diverge from the LM objective.

## Prior art

- Decoupled Neural Interfaces (Jaderberg et al. 2017) — synthetic gradient predictors between modules
- Deeply-Supervised Nets (Lee et al. 2015) — auxiliary heads at intermediate layers
- Contrastive Predictive Coding (van den Oord et al. 2018) — predicting future latents
- Difference Target Propagation (Lee et al. 2015) — local targets without backprop

## Existing evidence in this repo

| Experiment | Result | Relevance |
|---|---|---|
| Stop-gradient local residual prediction ([local-learning-residual](../local-learning-residual/)) | val_loss 2.08 vs 1.64 global — catastrophic | Different architecture (recurrent stacks, not shared-adjoint). Negative for that family, not directly applicable here. |
| Self-prediction auxiliary loss (this session) | Zero task effect | Blocks specialize naturally but local prediction signal adds nothing on top of global backprop. |

## Discriminating experiment

The cheapest test that produces real evidence about which hypothesis holds:

**Setup:**
- Current best model: parallel 4-block [1,2,4,8] rates, d=256
- Same dataset, same 20K steps, 3 seeds each

**Conditions:**
- **(a) Full backprop** — baseline, all gradient terms flow
- **(b) Lateral terms detached** — shared-adjoint only (g broadcast, all inter-block gradient paths cut via stop-grad)
- **(c) 1-hop truncated** — each block receives gradient from its immediate neighbors but not further

**Decision rule:**
- If (b) ≈ (a): H1 confirmed, parallel training is nearly free. Pursue shared-adjoint training directly.
- If (b) ≪ (a) but (c) ≈ (a): H2 confirmed. Pursue neighborhood-truncated backprop.
- If (b) ≪ (a) and (c) ≪ (a): H3 confirmed. Need synthetic gradients or distillation heads.

## Non-goals

This question does NOT address:

- Whether the multi-rate architecture itself is good (that's measured elsewhere)
- Inter-GPU parallelism (future work contingent on intra-GPU results)
- Optimal block count or rate schedule
- The graph topology question from the dictation (separate from gradient routing)
- Normalizing flows / norm preservation (acknowledged as separate direction in the dictation)

## Next steps

1. **Re-run on corrected architecture** (`token_injection="block0"`). In this architecture, detaching lateral gradients removes the ONLY information pathway to higher blocks. H1 likely fails — but how badly?
2. **Test whether all-block readout provides sufficient local signal.** Even without lateral gradients, each block gets a direct task gradient from its readout contribution. This may partially compensate.
3. **If H3 holds:** Explore predictive coding / target propagation style local objectives for interior blocks. Max mentions "something where the local objective is grounded in real prediction error, not just social selection." The attention-based "usefulness" signal is disfavored.
