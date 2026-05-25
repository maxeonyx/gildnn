# Local learning for parallel multi-rate blocks

Serves [dictation 2026-05-23-5](../../../dictations/2026-05-23-5.md): "How can I get local learning, enabling parallelism — some mechanism to learn well without requiring global backpropagation?"

## Status

**Bridge experiment PRE-REGISTERED (surrogate architecture).** Script ready: `runs/bridge_detach.py` (commit 93854b6). Launches after temporal_window + 4-block follow-up.

The original local learning experiment (TinyShakespeare, token_injection="all") is factually correct but tested the wrong architecture per [dictation 2026-05-23-7](../../../dictations/2026-05-23-7.md). Results kept for reference below.

**C_old lateral ablation (2026-05-25): CLEARLY LATERAL HELPS** — prerequisite for bridge experiment is met. See detailed results below.

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

## C_old lateral ablation — pre-registered interpretation (2026-05-25)

**Experiment:** `runs/c_old_ablation.py`. C_lateral (topology="upward") vs C_isolated (topology="isolated"), 4 blocks, d=256, ff=512, token_injection="all", 2 seeds (42, 43), 20K steps, WikiText-103 ctx=128.

**Scope:** Tests whether lateral connections are load-bearing in the only current multi-block regime that helps (token_injection="all"). Does NOT directly validate Pathway 3 in its intended form — every block still sees tokens directly.

**Quantity:** Paired difference per seed: d_s = val_loss(C_isolated, s) − val_loss(C_lateral, s). Positive means lateral is better.

**Calibration:** distinct_matched std across seeds = 0.001. Stale-read cost = +0.005 ± 0.005 (treated as negligible). A_single→distinct_matched gap = 0.021 (known meaningful architecture effect).

| Outcome | Criterion | Pathway 3 implication | Next step |
|---|---|---|---|
| **Clearly lateral helps** | Δ ≥ 0.015 AND both seeds > 0.005 | Prerequisite met: cross-block communication is genuinely load-bearing in this regime. Does NOT prove local learning works. | Bridge experiment: full-backprop vs detached-lateral training in same regime. |
| **Modest lateral use** | 0.005 ≤ Δ < 0.015 AND both seeds positive | Weak increase. Lateral probably matters some. | Add 3rd seed, or test corrected-architecture version. |
| **Clearly an ensemble** | |Δ| < 0.005 AND both seeds individually < 0.005 | Decrease. Gain from multi-block is compatible with independent token-fed blocks + shared readout. | Do NOT test local learning on C_old. Redirect to architectures where upper blocks depend on laterals. |
| **Ambiguous / unstable** | Discordant seed signs or one seed large and one near zero | No update from this experiment alone. | Add more seeds or run cleaner experiment. |

**Statistical note:** With 2 seeds, require concordance (both seeds agree directionally) for any "clear" claim. The mean alone is insufficient when one seed carries the effect.

### Results (2026-05-25, 22:27 NZST)

| Variant | Seed 42 | Seed 43 | Mean ± Std |
|---------|---------|---------|------------|
| C_lateral (upward) | 1.765 | 1.756 | 1.760 ± 0.005 |
| C_isolated (isolated) | 1.794 | 1.785 | 1.790 ± 0.004 |
| **Δ (isolated − lateral)** | **+0.029** | **+0.029** | **+0.030** |

**Outcome: CLEARLY LATERAL HELPS.** Both seeds Δ = +0.029, well above 0.015 threshold. Seeds are concordant. Lateral communication is genuinely load-bearing in this regime.

**Readout ablation (C_lateral models only):**

| Blocks included | Seed 42 val_loss | Seed 43 val_loss | Ablation cost |
|----------------|-----------------|-----------------|---------------|
| All 4 (full) | 1.765 | 1.756 | — |
| Block 0 only | 3.326 | 3.228 | +1.5 nats (catastrophic) |
| Blocks 0+1 | 2.303 | 2.203 | +0.5 nats |
| Blocks 0+1+2 | 2.177 | 2.107 | +0.4 nats |

ALL four blocks are load-bearing. The model distributes useful computation across all blocks — this is NOT a spectator architecture.

**Artifacts:** `experiments/wikitext_103/artifacts/c_old_ablation/report.json`, `run.jsonl`

### Interpretation

The prerequisite for local learning experiments is **met**: lateral connections carry information that measurably improves prediction. The next step per pre-registration is the **bridge experiment** (full-backprop vs detached-lateral in the same regime).

**What this proves:**
- ✅ Lateral communication is load-bearing in the token_injection="all" regime
- ✅ ALL four blocks contribute meaningfully to prediction (not just block 0)
- ✅ The +0.030 nat gap is 6× the stale-read noise floor (0.005)

**What this does NOT prove:**
- ❌ Does NOT prove local learning works (still needs the bridge experiment)
- ❌ Does NOT prove laterals matter in the INTENDED architecture (token_injection="block0") — the temporal_window experiment tests that separately
- ❌ Does NOT tell us WHERE the lateral info is used — could be purely for readout enrichment, or could be for within-block computation


---

## Bridge experiment design (if C_old positive) — pre-registered 2026-05-25

**What it is:** Same C_old regime (`topology="upward"`, `token_injection="all"`, 4 blocks, d=256), training with `detach_lateral=True` vs `detach_lateral=False`. The flag already exists in `core/model.py` (`ParallelDiagonalModel(..., detach_lateral=True)`). No new code required — only a config change from C_lateral.

**What detach_lateral does:** In `topology="upward"`, block i>0 receives `0.5 * (own_state + neighbor_state)` where neighbor is the lower block's previous-timestep state. With `detach_lateral=True`, `neighbor_state = lateral_source.detach()` — the forward signal flows but no gradient passes backward through the lateral edge. **Importantly, `detach_lateral` also detaches the temporal_window path** (line 680 in model.py: `lower_history = self._maybe_detach_lateral(temporal_history[block_index - 1])`). So an intended-architecture bridge_detach (with temporal_window > 0) requires NO code changes — just config.

**What this tests:** Can the model learn to use lateral communication without gradient credit assignment through those edges? Full-backprop lets block i+1 train block i to emit useful features. Detached lets block i+1 learn to *use* block i's output but cannot train block i to make it better.

**What this is NOT:**
- Not true local learning (still uses one global CE loss, full backprop inside each block)
- Not the corrected architecture (still token_injection="all")
- Not testing temporal locality (still backprops through each block's own temporal state)
- Not predictive coding or target propagation

**⚠ Critical measurement requirement: val_loss alone is AMBIGUOUS.**

If `detached ≈ full`, two explanations exist:
1. (Exciting) The model learned to use laterals without lateral gradients
2. (Boring) The model learned to NOT use laterals — collapsed to independent token-fed blocks

**Required second measure:** Post-training forward ablation on BOTH trained models. Eval normally, then eval with laterals zeroed/shuffled. If the detached-trained model still shows a large ablation cost → laterals are genuinely learned without gradient. If ablation cost is small → model collapsed to ensemble.

**Pre-registered interpretation:**

| Outcome | Criterion | Meaning |
|---|---|---|
| **Detached learns laterals** | val_loss(detached) ≈ val_loss(full) AND lateral-ablation cost ≈ same for both | Cross-block gradient not needed in this regime. Strong positive for Pathway 3. |
| **Detached collapses to ensemble** | val_loss(detached) ≈ val_loss(full) BUT detached shows no lateral-ablation cost | Model found independent solution. Null result for Pathway 3. |
| **Detached clearly worse** | val_loss(detached) > val_loss(full) by ≥ 0.015 | Cross-block gradient terms matter. Negative for Pathway 3 in this regime. |

---

## Next steps

1. **Confirm spectator result** across seeds — done (3-seed result: corrected = single block on TinyShakespeare at d=256). See `experiments/fixed_multi_rate/artifacts/token_injection_sanity/`.
2. **WikiText-103 baseline** — test whether multi-block helps on a dataset where single-block hasn't saturated. Running or queued.
3. **Re-evaluate local learning AFTER upper blocks have value** — detached-lateral only matters once upper blocks contribute. Don't test local learning on a broken architecture.
4. **Predictive coding** — once upper blocks have exclusive temporal info (e.g. windowed input), predict lower-block future states. This becomes meaningful because they can extrapolate from the trajectory.
