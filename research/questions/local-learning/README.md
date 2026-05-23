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

## Architecture problem: upper blocks have no exclusive information

**Early empirical result (seed 42, all-rate-1, 20K steps):**

| Variant | val_loss |
|---------|----------|
| Single block | 1.719 |
| 4-block token_injection="all" (old wrong) | 1.689 |
| 4-block token_injection="block0" (corrected) | **1.719** |

The corrected architecture matches a single block exactly. Upper blocks are spectators.

**Root cause analysis (multiple think iterations):** In the corrected architecture, upper blocks are **strictly staler decoders of block 0's lossy summary**. They never see raw tokens, never see current position, and only receive whatever block 0 exposes in its compressed state. On TinyShakespeare (where recent suffix is most predictive), upper blocks are exactly the modules that LACK the recent suffix. The static weighted-sum readout gives the optimizer permission to ignore them.

**This is NOT just a training problem.** It's architectural: upper blocks have no exclusive information that block 0 doesn't already provide through its own readout contribution.

**Cheapest diagnostic:** Fixed equal readout weights during training. If corrected arch + equal weights beats single block → problem is training collapse. If it ties → problem is forward information flow.

**Likely architectural fix:** Give upper blocks a WINDOW of past lower-block states (not just the latest). This provides exclusive temporal information (patterns in block 0's state evolution over multiple steps) that block 0 doesn't trivially expose in its current single state.

**Connection to predictive coding:** If upper blocks had exclusive temporal information (a window), then predicting block 0's FUTURE state becomes a meaningful task — they can extrapolate from the trajectory. Without that window, "predict the future" is just "predict a slightly staler version of what you already have."

## Theoretical analysis (corrected architecture)

With `token_injection="block0"`, the local learning question changes fundamentally:

### Why detached lateral likely fails differently than expected

With detached lateral gradients in the corrected architecture:
- Block 0 still gets full task gradient (sees tokens + contributes to output)
- Block 1+ get task gradient from their readout contribution (dL/dh_i = dL/dS)
- Block 1+ CAN learn a function of their lateral input (like a frozen-encoder + trainable-decoder)
- Block 0 CANNOT learn to emit features useful for block 1 (no gradient signal from block 1's needs)

The all-block readout creates an **escape hatch**: the network doesn't NEED to make interior blocks useful. It can let block 0 dominate and downweight upper blocks. This produces "shallow collapse" — the system trains but upper blocks become vestigial.

### Predicted outcome spectrum

**UPDATE:** Seed 42 of the running experiment shows even full-backprop corrected = single block (1.719). The "shallow collapse" happens EVEN WITH full gradients. This suggests the problem is not gradient flow but INFORMATION FLOW — upper blocks simply have nothing useful to add when they only get stale compressed state.

| Condition | Expected result | Actual (seed 42) |
|---|---|---|
| Full backprop, corrected | Works (slower blocks learn from delayed info) | **= single block (1.719)** |
| Detached lateral, corrected | Trains, but upper blocks get near-zero readout weight | Not yet tested |
| Single-block baseline | Similar to detached (if collapse is complete) | 1.719 |

The decisive signal is NOT val loss — it's **per-block ablation magnitude** and **readout weight distribution**.

### Candidate local signals (ranked)

Per [dictation 2026-05-23-7](../../../dictations/2026-05-23-7.md): "More promising directions are closer to predictive coding or target propagation — something where the local objective is grounded in real prediction error."

1. **Predictive coding on inter-block interfaces** — Each block predicts the future incoming signal at its input boundary. Trains the communication protocol, not just decoding. Horizon matched to rate (fast blocks predict 1 step, slow blocks predict their update interval).
   - Concrete: `L_local_i = ||stopgrad(target_from_below_{t+Δ_i}) - pred_i(h_i_t)||²`
   - Trains both "send useful things" (via main loss on block 0) AND "expect useful things" (via local prediction)
   
2. **Per-block LM heads** (auxiliary) — Each block has its own next-token prediction head. Grounds learning in real prediction error. BUT only trains "use what you get", not "form good protocol." Useful as diagnostic and weak anchor, not as the main signal.

3. **1-hop truncated backprop** — Allow gradient to flow from block i to block i-1 but no further. Gives immediate neighbors gradient signal about what's useful. Allows O(2) parallelism instead of O(N). Practical hybrid baseline.

4. **Target propagation** — Compute per-block targets from above. More complex machinery (learned inverses), historically brittle with recurrence/stale dynamics. Lower priority.

The attention-based "usefulness" signal is explicitly **disfavored** by Max (salience learning, grounding decay with depth, chicken-and-egg problems).

## Next steps

1. **[RUNNING] Confirm spectator result** across seeds 43-44 (PID 14456)
2. **Fixed equal readout weights** — cheapest diagnostic. Does forced participation change anything? If yes → training collapse. If no → forward info flow is broken.
3. **Windowed lateral input** — give upper blocks a FIFO of recent lower-block states (e.g. last 4 steps), not just the latest. This provides exclusive temporal information. Simplest version: concatenate last K states, project down.
4. **Re-evaluate local learning AFTER upper blocks have value** — detached-lateral only matters once upper blocks contribute. Don't test local learning on a broken architecture.
5. **Predictive coding** — once upper blocks have exclusive temporal info (windowed input), predict lower-block future states. This becomes meaningful because they can extrapolate from the trajectory.
