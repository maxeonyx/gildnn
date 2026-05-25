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

### Quantitative prediction (theory-informed, 2026-05-26)

Based on first-principles analysis of what detach preserves vs removes:

**Known endpoints:**
- Full backprop lateral benefit: Δ = +0.030 (C_lateral=1.760 vs C_isolated=1.790)
- Detach preserves: forward channel, receiver learning, task-trained sender features
- Detach removes: sender shaping by downstream gradient, joint optimization of communication protocol

**Predicted survival fraction:** ~65% of the 0.030 lateral benefit survives under detach (plausible range: 45%–80%).

**Concrete prediction:**
- `full_backprop`: ~1.760 (should match C_old baseline)
- `detached`: ~1.770 (band: 1.766–1.777)
- `isolated` (reference): ~1.790

**Interpretation thresholds (sharper than the ±0.015 coarse criterion):**
- ≤ 1.768: strong evidence detach works (≥73% retained)
- 1.768–1.773: moderate positive (57%–73% retained)
- 1.773–1.782: ambiguous (some value survives but meaningful degradation)
- ≥ 1.783: strong evidence detach fails (most of the 0.030 is gone)

**Low-probability wild card:** Detach could *beat* full backprop (by ~0.002–0.010 nats) if lateral gradients create harmful co-adaptation. This would be theoretically very interesting — implying that in the surrogate regime, joint optimization HURTS because it creates fragile communication protocols that simple feature interpretation would not.

**Why ensemble collapse is unlikely (but still must be tested via ablation):** The upward topology breaks block symmetry (block 0 gets no lateral, block 3 gets 3 levels of processed info). Independent parameters + readout competition create differentiation pressure even without co-adaptation. But upper blocks (esp. block 3) are at highest risk of marginal contribution.

**Dynamics prediction (when curves diverge):**
- Steps 0–4K: effectively identical (laterals are noise; receiver can't benefit from sender shaping yet)
- Steps 5K–8K: first tiny opening possible
- Steps 8K–10K: first clearly visible gap (full_backprop ahead)
- Steps 10K–20K: gap widens moderately, then plateaus

Empirical anchor: C_old lateral-vs-isolated gap only emerged at ~7K–8K steps. Same architecture.

If curves diverge EARLY (step 2K–4K): sender shaping matters immediately — bad for local learning.
If curves diverge LATE and gap closes: detached finds alternative path — good for local learning.
If gap snowballs without plateau: co-adaptation compounds — bad, suggests fragile joint protocol.

**Note:** bridge_detach uses flat AdamW lr=3e-4 (no cosine-annealing, no LR warmup). Divergence timing comes from representation bootstrapping, not LR schedule.

### Pre-registered ablation pattern predictions (2026-05-26)

**Full_backprop** should reproduce C_old: lateral-zeroing cost ~+0.030, steep cumulative ablation, all 4 blocks load-bearing. If it doesn't, that's a reproduction issue before interpreting detach.

**Detached** patterns determine the outcome class:

| Measurement | Learned laterals | Ensemble collapse | Clearly worse |
|---|---|---|---|
| Lateral-zeroing cost | Large: +0.020 to +0.030 | Near zero: 0 to +0.005 | Small: 0 to +0.015 |
| Per-block readout | All 4 non-spectator (≥+0.02) | Modest/flat (independent contributors) | Bottom-heavy (upper blocks spectator-ish) |
| Cumulative ablation | Steep (block0_only catastrophic) | Flat/ensemble (each block adds little) | Flat (upper blocks add nothing) |

**Practical reading order (most to least discriminating):**
1. Detached final val_loss vs full (separates "clearly worse" from not)
2. **Detached lateral-zeroing cost** (THE key discriminator — uniquely identifies learned laterals vs collapse)
3. Detached cumulative ablation (steep vs flat distinguishes real distribution from spectator/ensemble)
4. Detached per-block readout (confirms specific block contributions but ambiguous alone)
5. Full-backprop diagnostics (sanity check only)

**Edge cases not to force into 3 buckets:** Intermediate lateral-zeroing (+0.010–0.020) means partial lateral use. Detached clearly worse BUT still substantial lateral-zeroing means "learned somewhat but sender shaping still matters." Detached beating full with large lateral-zeroing would be the anti-co-adaptation wild card.

### Post-result action plan (surrogate bridge_detach)

**Note on "locality sweep":** In this architecture (nearest-neighbor-only laterals, upward topology), gradient radius sweep is degenerate — 1-hop truncation = full detach because there's at most 1 hop per timestep. The original plan's "locality sweep" is replaced by a bootstrapping-vs-grounding fork.

| Outcome | Next step | Rationale |
|---|---|---|
| **Detached learns laterals** | If 4-block intended arch is validated → intended-architecture bridge_detach. Otherwise → report strong positive, move to other pathways. | The cheap question is answered for the surrogate. The real discriminant is the intended architecture where blocks DEPEND on communication. |
| **Detached collapses to ensemble** | If intended arch validated → intended bridge_detach (no ensemble escape hatch there). If not → predictive coding at interfaces (forces communication to matter locally). | Ensemble bypass means the surrogate is too permissive. The intended architecture removes the bypass, making it the better substrate. |
| **Detached clearly worse** | **Warmup full→detach diagnostic** (train 5-10K steps full, then switch to detach for remaining 10-15K). | Discriminates bootstrapping (protocol just needs to form before detach) from fundamental signal insufficiency (shared adjoint is too weak). |

**If warmup→detach recovers most of the gap:** Problem was bootstrapping. Detach works once a communication protocol exists. Strong positive with a caveat (needs warmup phase). Next: test whether intended architecture shows the same warmup→recovery pattern.

**If warmup→detach does NOT recover:** Shared adjoint is genuinely insufficient. Escalation path:
1. **Predictive coding at interfaces (innovation-predicting variant)** — each block predicts the INNOVATION of its neighbor's next state. See theory analysis below.
2. **Multi-timestep Wasserstein local loss** — each block trains on its own prediction error of the neighbor's next distribution (from dictation 2026-05-25-1). Bigger conceptual jump but the most principled long-term solution.

### Predictive coding failure-mode analysis (theory, 2026-05-26)

**Naive "predict neighbor's full state" is likely to degenerate in this architecture.** Reasons:

1. **Target triviality:** With `token_injection="all"`, all blocks receive the same token embedding. The neighbor's next state is dominated by the shared next-token embedding, not block-specific computation. An auxiliary head optimizes for "predict next token dynamics" (trivial shared knowledge) rather than "predict unique block contribution" (the useful signal).

2. **State imitation ≠ task relevance:** Being good at predicting a neighbor's future state does not imply producing useful representations for next-char prediction. The auxiliary gradient drives blocks toward emulation/imitation, not toward task-aligned specialization.

3. **Implementation is clean:** 1-step delayed target buffer works. Not a conceptual blocker.

**Fix: predict the INNOVATION (residual component).** Subtract the shared token-driven baseline from the target:
- Target = `neighbor_next_state - shared_component` (where shared_component ≈ token embedding + stream)
- This focuses the auxiliary loss on what the neighbor's block SPECIFICALLY contributed, not what all blocks share.
- Concretely: target is `neighbor.block_output` (the feedforward residual before it's added to the stream), not the full stream state.

**Assessment:** Innovation-predicting variant is viable as an escalation. Key design choices: (1) predict block residual output, not full state; (2) keep auxiliary loss coefficient small (0.01-0.1× main CE); (3) use stop-gradient on target (block i cannot train block i-1 through this path — preserves locality).

**Per-block auxiliary LM heads** are a useful diagnostic at any outcome (quick to implement, shows whether task grounding alone helps) but are not the primary next step for any outcome — they don't specifically improve communication.

### Pre-interpretation: divergence timing (seed 42 mid-run observation, 08:27 NZST 2026-05-26)

Both variants track identically for 12K steps. Gap opens at step 15K (not the pre-registered 8K-10K), stabilizes at ~0.018 by step 16-17K. This observation narrows the interpretation space:

**What it tells us about bootstrapping:** The lateral communication protocol bootstraps fine WITHOUT cross-block gradient. 12K steps of identical learning curves is strong evidence that detached blocks can learn to consume lateral signals effectively during early/mid training. This weakens "bootstrapping failure" as an explanation if detached ends clearly worse.

**What the late divergence likely represents:** A refinement/specialization phase where full_backprop blocks start co-adapting across the lateral interface — training neighbors to emit better features. This coincides with a plateau→jump in the full_backprop curve (14-15K plateau, then rapid improvement to 1.765 by 20K). Detached stops at a weaker coordination point because it cannot drive sender improvement.

**Sharpened warmup→detach interpretation:** If detached lands "clearly worse," the trajectory data changes the warmup→detach question from "can the protocol bootstrap at all?" to "is lateral gradient needed to reach the good refinement regime, or needed continuously?" The weaker bootstrap hypothesis (detach can start but not sustain late-stage improvement) is not eliminated by this observation.

**Recommendation:** Still run warmup→detach if clearly worse, but with lower priority — the trajectory data already provides partial discriminant power. If warmup→detach recovers the gap, it specifically means "co-adaptation is needed briefly to reach the refinement regime." If it doesn't recover, shared adjoint is insufficient for continuous improvement.

**Caveats:** Single seed. Optimizer effects (Adam running averages accumulate). Could be capacity/symmetry-breaking coincidence rather than lateral-specific.

### Pre-registered ablation prediction (seed 42 gap = C_old lateral cost, 2026-05-26)

**Observation:** Bridge_detach gap (0.029) ≈ C_old lateral ablation cost (0.030, exact on seed 42). Both on same architecture (4-block, upward, token_injection=all).

**Hypothesis:** Detached laterals provide zero net benefit. Without gradient to shape the sender, the receiver cannot extract useful information. Detached model is functionally equivalent to a model with NO laterals.

**Testable prediction for ablation results:**

| Measurement | Predicted (if dead-lateral hypothesis correct) | Alternative (partial use) |
|---|---|---|
| full_backprop lateral-zeroing cost | ~0.030 (same as C_old) | ~0.030 |
| detached lateral-zeroing cost | ~0 to +0.005 (laterals not used) | +0.010 to +0.020 |
| detached final vs detached-zeroed | negligible difference | meaningful difference |

**What decides:** If detached zeroing cost is ~0 → hypothesis confirmed (laterals functionally dead without sender gradient). If +0.010-0.020 → laterals partially used (receiver CAN extract some value from unoptimized sender output, but not as much as with co-adaptation).

**Supporting evidence:** Trajectory data shows both variants identical for 12K steps (neither uses laterals early), divergence at 15K (co-adaptation begins in full only). Consistent with "laterals are irrelevant until sender is optimized for communication."

---

## Bridge_detach results — COMPLETE (2026-05-26, 3 seeds)

### Final val_loss

| Seed | full_backprop | detached | Gap |
|---|---|---|---|
| 42 | 1.765 | 1.794 | **+0.029** |
| 43 | 1.756 | 1.794 | **+0.038** |
| 44 | 1.796 | 1.811 | **+0.016** |
| **Mean** | **1.772 ± 0.017** | **1.799 ± 0.008** | **+0.027** |

**Outcome: "Clearly worse"** — all 3 seeds above the 0.015 pre-registered threshold. Mean gap +0.027 nats.

Seed 44 is shifted up for both variants (~0.03 worse than seeds 42-43) but the gap pattern holds. Seeds 42-43 share an eerily consistent detached ceiling (1.794 both seeds); seed 44 converges higher (1.811). The ceiling may be initialization-dependent rather than universal.

### Readout ablation (per-block zeroing cost at eval)

| Block | Full s42 | Full s43 | Full s44 | Det s42 | Det s43 | Det s44 |
|---|---|---|---|---|---|---|
| 0 | +1.12 | +0.93 | +0.96 | +1.11 | +0.84 | +1.05 |
| 1 | +0.10 | +0.20 | +0.10 | **+0.44** | **+0.68** | **+0.41** |
| 2 | +0.05 | +0.03 | +0.05 | +0.16 | +0.11 | +0.16 |
| 3 | **+0.42** | **+0.35** | **+0.42** | +0.07 | +0.07 | +0.07 |

**Pattern (perfectly consistent across all 3 seeds):**
- Full_backprop is "U-shaped": blocks 0 and 3 are load-bearing. The last block (furthest from input tokens) develops substantial representation (+0.35-0.42) when lateral gradient shapes what earlier blocks send it.
- Detached is "front-loaded": blocks 0 and 1 dominate. Block 3 is nearly useless — **exactly +0.07 across all 3 seeds** (at noise floor, remarkably stable). Without lateral gradient, the model settles for a shallower solution using blocks closest to input.
- The gap (~0.027 nats mean) is explained by: block 3 loses ~0.35 nats of contribution, block 1 gains ~0.35-0.48 nats compensating → net loss is partial because the model partially self-heals.

**Causal hypothesis (plausible, not uniquely proven):** Without gradient through lateral connections, senders (blocks 0-2) don't learn what to send to block 3. Block 3 receives uninformative lateral input → cannot develop useful specialization. Alternative: detached simply settles for a shallower independent-block solution because it requires less coordination. The readout competition under `readout_mode="all"` makes these hard to distinguish without further experiment.

### ⚠️ Methodological learning: lateral-zeroing ablation is uninformative

The pre-registered "lateral-zeroing cost as THE key discriminator" (line 344 above) **does not work** in this experimental setup.

**What happened:** Both variants produce catastrophic ablation values (10^11-10^15) when `model.topology` is switched to `"isolated"` at eval time.

**Why:** Both models were TRAINED with forward lateral connections (detach only cut the backward pass). Their learned representations deeply assume lateral input exists. Removing laterals at eval is architectural mutilation — it tests "can this model survive losing an entire input pathway?" not "how useful are the laterals?"

**Contrast with C_old:** The C_old experiment trained TWO SEPARATE MODELS (one with laterals, one without). Its Δ=0.030 compared models each trained for their respective regime. That's a valid comparison. This bridge_detach ablation asks a trained-with-laterals model to work without them — fundamentally different question, catastrophic answer.

**Lesson:** Post-training lateral removal is not comparable to training without laterals. The actual discriminator is the **readout pattern** (which blocks develop useful representations), not the lateral-zeroing cost.

**Impact on decision tree:** The "lateral-zeroing cost" row in the interpretation table (above) is now marked as methodologically uninformative for this regime. The "clearly worse" determination is made from the loss gap + readout pattern instead.

### Assessment against pre-registered prediction

The "dead lateral hypothesis" (detached zeroing cost ≈ 0) cannot be directly tested because zeroing produces catastrophic values. However, the READOUT pattern is consistent with a weaker version of the hypothesis: block 3 is effectively unused (contributes only +0.07 in detached vs +0.35-0.42 in full), suggesting the later blocks fail to develop useful computation without lateral gradient — even though the forward lateral pathway exists.

This is "partially dead" — not that laterals carry zero information, but that without gradient to shape the sender, the receiver can't extract enough value from unoptimized lateral output to develop useful specialization.

### Decision: proceed with warmup→detach

Per pre-registered action plan: clearly worse → warmup→detach diagnostic. The question sharpens from "is lateral gradient needed?" to "when can lateral gradient be safely removed — during bootstrapping, during refinement, or never?"

---

## Warmup→detach diagnostic — pre-registered 2026-05-26

**Conditional on:** bridge_detach "detached clearly worse" outcome (final val_loss ≥ 1.783 or gap consistently > 0.015 across seeds). This is the first diagnostic in the escalation path.

**Question:** Is lateral gradient needed only to REACH the late refinement regime, or needed CONTINUOUSLY to keep refining?

### Conditions

| Condition | Construction | Purpose |
|---|---|---|
| `full_20k` | existing bridge_detach baseline | reference best trajectory |
| `detached_20k` | existing bridge_detach baseline | reference no-gradient endpoint |
| `warm12_detach` | train full_backprop, switch `model.detach_lateral=True` at step 12K, continue to 20K | pre-divergence warmup (before gap opens) |
| `warm15_detach` | train full_backprop, switch `model.detach_lateral=True` at step 15K, continue to 20K | post-divergence warmup (after entering refinement regime) |

**Implementation:** Fresh 20K-step runs with mid-training switch. At step N, flip `model.detach_lateral = True` and continue. No checkpoint branching needed — `detach_lateral` is a runtime attribute checked during forward pass (`_maybe_detach_lateral()`), not an architectural change. Optimizer state, model weights, and dataloader RNG all continue uninterrupted. This is the cleanest implementation: no resume artifacts, no checkpoint format issues.

**⚠️ CUDA Graph caveat:** `GraphTrainer` captures the entire forward+backward pass as a static CUDA graph. Changing `model.detach_lateral` at runtime has NO EFFECT on an already-captured graph — the original computation is replayed exactly. The warmup→detach switch must: (1) delete the old `GraphTrainer`, (2) set `model.detach_lateral = True`, (3) create a new `GraphTrainer` and re-capture with the updated model. This maintains CUDA graph speed (~116K tok/s) for both phases. Without graphs, eager mode is ~10× slower — unacceptable for 6 runs × 20K steps.

### Hypotheses and predictions

| Outcome | What it means |
|---|---|
| Both warmups drift back toward detached | Lateral gradient needed CONTINUOUSLY for refinement |
| `warm15_detach` stays near full; `warm12_detach` partial | Only need to REACH the refinement regime |
| Even `warm12_detach` mostly closes the gap | Early protocol formation is the bottleneck (unlikely given trajectory data) |

### Interpretation thresholds

**Retained-gap fraction:** `R = (L_detached - L_warm→detach) / (L_detached - L_full)` using 20K final val_loss. Higher R = more recovery.

| Interpretation | Criterion |
|---|---|
| Recovered / sustained | R ≥ 0.75 AND final loss within 0.005 of full |
| No meaningful recovery | R ≤ 0.25 OR final loss within 0.005 of detached |
| Partial | everything between |

**Gap re-opening check:** if warmup run's gap vs full grows by ≥ 0.010 nats after the switch point, it's "temporary head start only," not sustained recovery.

### Seeds and budget

3 seeds × 2 warmup conditions = **6 full 20K-step runs** ≈ **~120 min** GPU time (each 20K steps at ~1.7 min/1K = ~34 min; or ~20 min at actual 1000 steps/min observed speed). No resume controls needed — mid-training switch has no artifacts.

### Confounds

- **Optimizer state:** Adam moments from the full phase carry through. The warmup condition has "richer" moments at the switch point. This is inherent to the question being asked (it IS a training-procedure test).
- **What this does NOT show:** true local learning. Only shows whether detach can inherit/sustain a regime created by full backprop.
- **Optional if recovery is surprising:** post-training lateral ablation on recovered models to confirm they still USE the lateral channel rather than coasting.

### Sharpened by trajectory data

The divergence-timing observation (identical curves for 12K, gap at 15K) ALREADY tells us bootstrapping is not the primary problem. This diagnostic is now mainly asking: "once in the good regime, can detach SUSTAIN it?" If `warm15_detach` drifts back, the answer is clearly no — gradient is needed continuously. If it stays, the 5K of full training during the regime transition was enough.

### Structural persistence despite loss regression (seeds 42-43)

Readout ablation on warm→detach final models reveals a dissociation: val_loss regresses to detached (R≈0), but the **internal structure** partially retains the full_backprop pattern — and the retention is proportional to warmup length.

**Decay rate of block 3 contribution (seed 42):**

| Condition | Detached steps | Block 3 cost | % full-BP retained | R (val_loss) |
|---|---|---|---|---|
| Full_backprop | 0 | +0.419 | 100% | 1.00 |
| Warm15→detach | 5K | +0.373 | 87% | 0.26 |
| Warm12→detach | 8K | +0.295 | 64% | -0.14 |
| Pure detached | 20K | +0.072 | 0% | 0.00 |

Block 1 compensation (the "front-loaded" pattern of pure detached, where block 1 swells to +0.44-0.68) does NOT develop at 5K or 8K detached steps — it requires the full 20K.

**Key dissociation:** Val_loss regresses faster than structure. At 8K detached steps (warm12), R≈0 (loss fully regressed) but block 3 still retains 64% of its structural contribution. This means block 3 is still USED by readout but produces lower-quality output — like a degraded but still-wired module.

**Implication for predictive-residual:** A successful local loss should recover BOTH val_loss (R≥0.5) AND the U-shaped readout structure (block 3 contribution at +0.35+ rather than +0.07). The local loss doesn't need to prevent structural collapse (that's slow) — it needs to maintain output QUALITY in block 3 during the period when the block would otherwise degrade.

---

## Next escalation: predictive-residual local loss — pre-registered 2026-05-26

**Conditional on:** warmup→detach confirming "gradient needed continuously" (both warm12 and warm15 drift back toward detached; R ≤ 0.5 for both). If warmup shows bootstrapping is enough, skip this and characterize minimum warmup instead.

**Question:** Can a grounded local prediction signal replace cross-block gradient?

### Design

Same surrogate substrate as bridge_detach. Two new conditions (plus existing baselines):

| Condition | Config | Notes |
|---|---|---|
| `full_20k` | existing bridge_detach baseline | ceiling |
| `detached_20k` | existing bridge_detach baseline | floor |
| `detached_predres_20k` | `detach_lateral=True` + aux prediction loss (real targets) | **mechanism test** |
| `detached_shuffled_20k` | `detach_lateral=True` + aux prediction loss (shuffled targets) | **regularization control** |

The `shuffled` control uses the same aux heads and same loss weight, but predicts a **time-shuffled** version of the lower block's residual. This controls for the regularization/multitask effect of having extra gradient flow through the block — if shuffled matches predres, then the benefit is from the extra objective generically, not from the predictive content.

The aux head on each block `i > 0` predicts block `(i-1)`'s next output/residual. Target is **stop-grad** (no gradient flows back through the target — otherwise we've smuggled cross-block credit assignment back in).

Architecture: identical to bridge_detach (4 blocks, d=256, ff=512, token_injection=all, readout_mode=all, topology=upward). Same 20K steps, batch_size=64, lr=3e-4, ctx=128, seeds 42/43/44.

### Why predict the block residual, not the full stream state?

With `token_injection=all`, the full stream state at each position is token-dominated. Predicting it is too easy and not communication-specific — the aux head would mostly learn token dynamics. The block-specific residual/output (what the lower block actually ADDS to the stream) is the signal that captures bridge usefulness.

### Aux loss weight

Pre-register one coefficient, calibrated once from a single batch so aux gradients are same-order-of-magnitude as CE gradients on bridge activations. No hyperparameter sweep (budget constraint).

### Known confounds (from adversarial review 2026-05-26)

1. **Moving-target problem:** Lower block's output is nonstationary (changes as CE trains it). The predictor chases a moving target. A null result could mean the idea is bad OR the target is too unstable early in training. This is inherent but interpretation must acknowledge it.
2. **Readout competition:** `readout_mode="all"` uses learned softmax weights. A block can appear more important because readout shifted toward it, not because its representation genuinely improved. Use readout ablation as **supporting** evidence only, not sole criterion.
3. **This is a surrogate, not the intended architecture.** Results apply to `token_injection=all` regime only. Do not generalize to the intended `block0` architecture without further testing.

### Interpretation

Primary metric: val_loss gap closure (R). Supporting: block 3 readout ablation pattern.

| Outcome | Criterion | Meaning | Next step |
|---|---|---|---|
| **Strong positive** | R ≥ 0.5 (predres) AND R(predres) > R(shuffled) + 0.15 AND block 3 ablation rises | Local prediction CONTENT specifically helps (not just regularization). | Wasserstein/distributional version |
| **Partial** | R(predres) > R(shuffled) but modest (0.1-0.3 gap) | Prediction has traction beyond regularization but isn't enough alone. | Upgrade to distributional or try combined with 1-hop truncation |
| **Regularization only** | R(predres) ≈ R(shuffled) (both improve or both don't) | The predictive content doesn't matter — any multitask signal helps (or doesn't). | Redirect: the problem isn't the local loss content, it's something else |
| **Null** | R < 0.2 for both predres and shuffled | Neither the content nor the extra objective helps. Point-vector prediction alone not enough. | Wasserstein remains possible (geometry matters), but confidence drops |

Key metric anchors from bridge_detach:
- Block 3 full_backprop: +0.35 to +0.42
- Block 3 detached: +0.07 (noise floor)
- Mean val_loss gap: +0.027

### What this does NOT settle

- Whether DISTRIBUTIONAL prediction (Wasserstein) would work where point-vector doesn't — that's a separate experiment
- Whether the local signal works on the intended architecture (this is still surrogate)
- Whether the loss coefficient matters (one-shot calibration only)
- Whether the prediction target should be "future" vs "current" (this predicts next-step; could also try current-step prediction of arriving lateral input)
- Whether the moving-target problem would resolve with a warmup phase (predict only after block 0 stabilizes)

### Connection to multi-timestep theory

This is the **point-vector approximation** of the same interface-prediction idea from `research/questions/multi-timestep-architecture/README.md`. It tests whether local prediction of arriving information helps at all, without importing the full distributional stream semantics. A positive result supports the predictive-processing family; a negative result does NOT falsify the Wasserstein direction (geometry matters). The shuffled control separates "prediction is the right idea" from "any aux objective helps."

---

## Intended-architecture bridge_detach design — pre-registered 2026-05-26

**Conditional on:** 4-block follow-up STRONGLY POSITIVE (W8_all > A_all ≥ 0.015, blocks load-bearing). Only makes sense if the intended architecture has been validated.

### What makes this different from the surrogate bridge_detach above

The surrogate version tests an easier question: blocks that already have independent token input (can function alone) — can they ALSO exploit lateral info without lateral gradient? That's "optional enrichment."

The intended version tests a harder and more fundamental question:

> **Can upper blocks learn to use lower-block communication when that communication is their ONLY source of task-relevant information, and when they cannot train the sender to emit better features?**

In the intended architecture (`token_injection="block0"`), upper blocks get NO direct tokens. Their information comes ONLY through:
1. Lateral connections (block i-1's previous state)
2. Temporal window (last N states from block i-1)

Under `detach_lateral=True`, both paths are stop-gradiented. The forward pass is identical — upper blocks still SEE everything. But gradient cannot flow backward through the bridge. Lower blocks cannot be trained to emit features that are useful for upper blocks.

This distinguishes two learning regimes:
- **Co-adaptive:** sender and receiver jointly negotiate a communication protocol (full backprop)
- **Opportunistic:** receiver learns to exploit whatever the sender naturally produces (detached)

### Why this is the project's core question

Per [dictation 2026-05-23-5](../../../dictations/2026-05-23-5.md): "How can I get local learning, enabling parallelism?"

If detached works on the intended architecture, it proves that blocks can learn useful roles from a purely local signal (shared adjoint + own parameters + detached inputs). Lower blocks emit whatever is good for their own contribution; upper blocks independently learn to interpret it. No global coordination needed. This is the strongest possible evidence for Pathway 3.

If detached fails, it proves the intended architecture requires end-to-end gradient — the communication channel needs active shaping from both ends. The fallback is semi-local methods (1-hop truncation, distillation heads, predictive coding at interfaces).

### Experiment design

**Config:** Identical to 4-block W8_all variant (the validated intended architecture):
- 4 blocks, d=256, ff=512
- `token_injection="block0"`, `temporal_window=8`
- `readout_mode="all"`, `topology="upward"`, rates=(1,1,1,1)
- 20K steps, batch_size=64, lr=3e-4, context_size=128, WikiText-103

**Conditions:**
- `W8_full`: `detach_lateral=False` (full backprop across bridges)
- `W8_detached`: `detach_lateral=True` (stop-gradient on lateral + temporal_window inputs)

**Seeds:** 42, 43, 44 (same as temporal_window and 4-block)

**No new code required:** `detach_lateral` already covers both lateral and temporal_window paths (model.py line 680).

### What gradient signal remains for upper blocks under detach?

Upper blocks still receive:
- Gradient from the task loss through their own readout contribution (shared adjoint: dL/dh_i = dL/dS)
- Full backprop through their own parameters and internal state
- Gradient through `window_proj` weights (learning how to project the detached temporal input)

What is cut:
- Gradient flowing backward INTO lower blocks through the lateral/temporal path
- Lower blocks cannot be trained to emit upper-block-friendly features

So this is NOT "upper blocks get no gradient." It's "upper blocks can learn to USE inputs, but cannot SHAPE what those inputs contain."

### Cold-start / bootstrapping concern

More severe than surrogate version:
- Early in training, block 0 emits immature/noisy features
- Upper blocks try to learn from a moving, noisy feature stream they can't influence
- `readout_mode="all"` may shift weight toward block 0 if it learns faster (readout competition)
- Risk: upper blocks fall behind early and never recover

This means **learning curves matter**, not just final val_loss. If detached shows delayed convergence but catches up, that's meaningful (bootstrapping latency, not impossibility). If it never catches up, that's more concerning.

### Pre-registered interpretation

| Outcome | Criterion | Meaning |
|---|---|---|
| **Detached learns bridges** | val_loss(W8_detached) ≈ val_loss(W8_full) (|Δ| < 0.015) AND per-block readout ablation shows all blocks load-bearing in BOTH conditions | Local learning is viable on the intended architecture. Upper blocks can independently learn useful roles from natural lower-block emissions. Strongest Pathway 3 evidence. |
| **Partial degradation** | val_loss(W8_detached) 0.015–0.050 worse than W8_full, OR some blocks lose their contribution under detach | Cross-block gradient helps alignment/bootstrapping but isn't strictly necessary. Suggests semi-local approaches (1-hop truncation, warmup with full backprop then detach) could recover most of the gap. |
| **Full collapse** | val_loss(W8_detached) > 0.050 worse, OR upper blocks become spectators under detach | Intended architecture REQUIRES end-to-end gradient for useful delayed hierarchy. Pivot to predictive coding, target propagation, or per-block auxiliary objectives to provide local grounding. |

### Required measurements (beyond val_loss)

1. **Per-block readout ablation** (same as 4-block follow-up) — do blocks 1/2/3 actually contribute under detached training?
2. **Learning curve comparison** — plot val_loss vs step for both conditions. Delayed convergence vs permanent gap.
3. **Cumulative block ablation** — ablate blocks top-down: [0+1+2+3], [0+1+2], [0+1], [0]. Compare patterns between full and detached.
4. **Lateral-zeroing ablation** — at eval time, zero the lateral/temporal inputs. Does the detached model still use them? (Rules out "collapsed to spectator" masquerading as "detached ≈ full")

### What this experiment will NOT settle

- Whether the intended architecture is better than the surrogate at scale (different question)
- Whether semi-local methods (1-hop, distillation) can recover quality if full-detach fails
- Whether multi-rate interacts with detach (this experiment uses uniform rates)
- Whether longer training would close any gap (fixed budget comparison)
- Whether predictive coding is the right local objective (this tests the cheapest version: no local objective at all, just shared-adjoint)

---

1. **Confirm spectator result** across seeds — done (3-seed result: corrected = single block on TinyShakespeare at d=256). See `experiments/fixed_multi_rate/artifacts/token_injection_sanity/`.
2. **WikiText-103 baseline** — test whether multi-block helps on a dataset where single-block hasn't saturated. Running or queued.
3. **Re-evaluate local learning AFTER upper blocks have value** — detached-lateral only matters once upper blocks contribute. Don't test local learning on a broken architecture.
4. **Predictive coding** — once upper blocks have exclusive temporal info (e.g. windowed input), predict lower-block future states. This becomes meaningful because they can extrapolate from the trajectory.

---

## Theory note (2026-05-26): why A_all block 1 survives, and what that predicts for bridge_detach

This section is a pre-analysis written **without reading the running `bridge_detach` results**.

### First: one framing correction

The simplified math `S = x + h_0 + h_1 + h_2 + h_3` is not the exact repo implementation. In code, `readout_mode="all"` uses a learned softmax-weighted sum:

```
S = Σ_i r_i h_i,    r = softmax(readout_logits)
```

So the shared adjoint is not literally identical in magnitude across blocks. The exact direct readout term is:

```
∂L/∂h_i = r_i g,    g = ∂L/∂S
```

The direction is shared; the scale is block-specific. Everything below uses the exact weighted form when that distinction matters.

### Setup and notation

- `Y` = next-token target
- `H_i^T` = final output of block `i`
- `H_i^{1:T-1}` = trajectory of block `i` outputs over the preceding timesteps

Per-block readout ablation is **not** literally a mutual information measurement, but it is a usable proxy for whether block `i` carries predictive information that the remaining readout cannot cheaply replace.

Define the trajectory residue available at depth `i+1` as:

```
R_1 = I(Y; H_0^{1:T-1} | H_0^T)
R_2 = I(Y; H_1^{1:T-1} | H_0^T, H_1^T)
R_3 = I(Y; H_2^{1:T-1} | H_0^T, H_1^T, H_2^T)
```

These are the "still-predictive bits that are in the sender's trajectory but not already present in the lower blocks' final snapshots."

### 1) Why block 1 is load-bearing in A_all

`A_all` (`token_injection="block0"`, no temporal window) is **not** a pure snapshot architecture. Block 1 does not get an explicit window, but over the sequence it repeatedly receives `H_0^{t-1}` and has its own state. That means block 1 can compute a recurrent summary of the **trajectory** of block 0, not just its final state.

So block 1 has a specific information advantage over blocks 2/3:

1. **It is the first receiver of the strongly task-grounded source.** Block 0 is the only block that sees tokens, and it is also massively load-bearing itself. Its emitted states therefore contain substantial task-correlated structure even before any special shaping for upper blocks.
2. **It can store first-order trajectory residue.** If two recent contexts produce similar `H_0^T` but different trajectories `H_0^{1:T-1}`, block 1 can in principle preserve the difference. The observed `+0.07` ablation is evidence that `R_1 > 0` at this scale.
3. **It only has to beat block 0 on the residual.** Because readout is additive, block 1 is useful only to the extent that it carries information about `Y` not already recoverable from block 0's contribution. `+0.07` means there is a small but real first-order residue.

That same argument weakens sharply for blocks 2 and 3.

### 2) Why the advantage decays with depth

The decay is **not** just generic "information gets noisier." There is a stronger structural reason.

Block 2 does **not** get direct access to block 0's trajectory residue. It only gets block 1's outputs, which are already a lossy transform of that residue and are trained under weaker grounding. Formally, by data processing:

```
I(Y; H_1^{1:T-1} | H_0^T) ≤ I(Y; H_0^{1:T-1} | H_0^T)
```

But the relevant quantity for block 2 is smaller still:

```
R_2 = I(Y; H_1^{1:T-1} | H_0^T, H_1^T)
```

This is a **second-order residue**: information in block 1's trajectory that is not already captured by the final snapshots of blocks 0 and 1. Block 3 chases a third-order residue. So each extra hop does two bad things at once:

1. **Compression:** the sender trajectory is itself already a lossy summary of the level below.
2. **Conditioning penalty:** the new block must carry information unique relative to all lower blocks already in the additive readout.
3. **Weaker grounding:** block 1 is directly downstream of the token-grounded block; block 2 is downstream of a block whose own useful residue is already small; block 3 is worse again.

That predicts a hierarchy `R_1 >> R_2 >= R_3`, which matches the empirical pattern much better than the old "all upper blocks should be spectators" hypothesis.

This also explains why `W8_all` amplifies block 1 much more than blocks 2/3. The temporal window gives block 1 direct access to a larger chunk of the **best possible sender trajectory** (block 0's). It does not solve the deeper problem that block 2 and block 3 still depend on already-weak upper-block trajectories.

### 3) What detach removes and what it leaves intact

For block 1 parameters, the exact weighted-readout gradient under full backprop has the form:

```
∇_{θ1} L = r_1 J_1^T g + downstream cross-block terms
```

With `detach_lateral=True`, the cross-block terms disappear but the direct readout term remains:

```
∇_{θ1}^{detach} L = r_1 J_1^T g
```

So detach does **not** remove block 1's task signal. It removes block 1's ability to train block 0 to emit block-1-friendly features.

For block 0, the contrast is:

```
∇_{θ0}^{full} L = r_0 J_0^T g + "make your output useful to upper blocks" terms
∇_{θ0}^{detach} L = r_0 J_0^T g
```

Therefore the detached regime still allows:

- **receiver learning** (`block 1 learns to use whatever arrives`)
- **shared task grounding** (`block 1 still sees CE-aligned gradient through its own readout contribution`)

But it no longer allows:

- **sender shaping** (`block 0` being optimized specifically for block 1's later use)
- **joint protocol formation** across the bridge

### 4) Prediction: should block 1 remain load-bearing under detach?

**Yes, probably for block 1; probably not much for blocks 2/3.**

Reason:

- The `A_all` result already says the one-hop bridge carries genuinely useful first-order residue.
- That residue originates in the trajectory of a strongly task-grounded sender (block 0), so a substantial fraction of it should exist even without sender shaping.
- Block 1 still has enough gradient to become a better **decoder** of that natural trajectory.
- Blocks 2/3 are different: their senders are much less grounded, and their useful residue is already tiny. Those are exactly the blocks most likely to collapse when co-adaptation is removed.

So the most likely detached pattern is:

- block 0 still dominant
- block 1 still clearly non-spectator, but weaker than full backprop
- blocks 2/3 near-spectator or fully spectator

### 5) Registered falsifiable predictions

#### Hard prediction for an A_all-style detached run

If we trained the **same architecture as `A_all`** with detached bridges, the block-1 per-block readout ablation should remain **positive and clearly above noise**.

Registered range:

```
block 1 ablation cost: +0.025 to +0.055 nats
block 2 ablation cost: 0 to +0.010 nats
block 3 ablation cost: 0 to +0.010 nats
```

Interpretation:

- `< +0.010` for block 1 would falsify the claim that shared-adjoint receiver learning is sufficient even for the first bridge.
- `+0.025 to +0.055` would support the view that most of block 1's benefit is opportunistic decoding of naturally useful block-0 emissions, with some loss from missing co-adaptation.
- `>= +0.070` would imply lateral gradient terms were unnecessary or even harmful for block 1 in this regime.

#### Lower-confidence prediction for the currently running surrogate `bridge_detach`

The running experiment is **not** `A_all`; it is the easier surrogate with `token_injection="all"`. That architecture gives block 1 more independent grounding than `A_all`, so block 1 should be at least as robust to detach.

I therefore predict that detached block 1 in the surrogate run will remain **clearly load-bearing**, with per-block readout ablation most plausibly in:

```
+0.10 to +0.35 nats
```

This number is lower-confidence than the `A_all` prediction because the repo does not yet contain a matching per-block full-backprop ablation baseline for the surrogate run; the cleaner claim is directional: **detach should shrink block 1, not zero it.**

### 6) What would falsify this analysis cleanly?

The strongest falsifiers are:

1. **Detached block 1 collapses to noise** (`< +0.010`) while full-backprop block 1 is clearly positive. That would mean sender shaping is not a small correction; it is the main thing making the bridge useful.
2. **Detached block 2 stays clearly positive** (`> +0.020`). That would mean the rapid residue-decay argument is too pessimistic, and higher-order residues survive better than expected.
3. **Detached block 1 matches or beats full-backprop block 1.** That would imply cross-block gradients are causing harmful co-adaptation rather than helpful protocol formation.

The cleanest conceptual summary is:

> Block 1 is special because it is the first module that can cache predictive residue from the trajectory of the only strongly grounded sender. Detach should hurt its magnitude, but not its existence. The deeper blocks are not just farther away; they are chasing higher-order residues that are structurally much smaller.
