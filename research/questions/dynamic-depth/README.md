# Dynamic Depth / Early Exit (Pathway 5)

> **Roadmap connection:** Pathway 5 — "Can a loss predictor decide when recurrent iterations are done?"

---

## Question

Do different tokens benefit differently from additional tied-depth iterations? If so, dynamic depth (early exit) could improve both speed and quality.

## Exploratory probe (2026-05-25)

Trained depth=8 tied-depth transformer (d_model=72, TinyShakespeare ctx=32, seed=42). Measured per-token cross-entropy at all 8 depths across all validation window positions.

### What was measured

- Per-token marginal improvement (Δloss from depth d to d+1) across 638K position instances
- Distribution of "earliest depth within ε of depth-8 loss"
- Fraction of tokens with negative total improvement (depth-8 worse than depth-1)

### Raw numbers

Mean total improvement d1→d8: 0.112 nats (std 1.413)
Earliest-done depth mean: 3.1 (std 2.57) at ε=0.1

Marginal improvements (mean / std):
- d1→d2: 0.062 / 0.584
- d2→d3: 0.009 / 0.466
- d3→d4: -0.020 / 0.340
- d4→d5: 0.005 / 0.273
- d5→d6: 0.015 / 0.167
- d6→d7: 0.023 / 0.127
- d7→d8: 0.019 / 0.133

45.5% of token instances have negative total improvement (depth-8 worse than depth-1).

### Honest interpretation

**Supported:** Substantial token-level heterogeneity exists. Marginal gains are dispersed (std >> mean at every depth transition).

**Not supported:**
- "Most tokens are done early" — ε=0.1 is nearly the size of the mean total improvement (0.112), so the "done" threshold is too loose to be meaningful
- "Dynamic depth would improve quality" — haven't shown a halting mechanism would help, and self-attention means tokens can't be independently halted
- "45.5% are harmed by more depth" — more likely normal probability redistribution (model improves average loss by helping some tokens more than it hurts others)

### Methodological concerns

1. **Non-standard evaluation frame** — all positions × overlapping windows ≠ standard next-token val (mean depth-8 loss here is 3.37 vs standard val 1.65)
2. **Single seed** — percentages could shift substantially across seeds
3. **Self-attention coupling** — can't independently halt tokens in this architecture
4. **Train/val comparison not done** — haven't ruled out overfitting as explanation for late-depth degradation

### Status

**Interesting hint, not evidence.** Pathway 5 remains plausible but unvalidated. Worth revisiting with cleaner methodology when dynamic depth becomes the active question.

### What would constitute real evidence

1. Clean evaluation (one prediction per raw validation position with maximal context)
2. Oracle best-depth measurement (not "within ε")
3. Train vs val comparison to rule out overfitting
4. Multi-seed replication
5. Test whether shallow state can predict sign/magnitude of future marginal improvement

---

## Clean measurement design (pre-registered 2026-05-26)

**Substrate:** Existing trained tied-depth transformer (d_model=72, 4 heads, TinyShakespeare ctx=32, depth=8, 3 seeds). Eval-only — no new training required. GPU time: minutes.

### Methodology

**Standard next-token eval only.** For each corpus position t, build exactly one example with maximum available left context (up to 32 chars). Predict next token ONCE. Record logits at the FINAL position after every tied iteration d=1..8. Compute per-token cross-entropy L_t(d). Do this on val AND a matched train subset of equal size, for all 3 seeds.

**Self-attention coupling addressed:** Depth is defined per PREDICTION EVENT (whole forward pass uses same depth), not per-token within a single pass. This is valid — you can't halt different positions independently, but you CAN choose different depths for different examples.

### Metrics to compute

1. **Mean loss by depth** E[L(d)] — recover normal depth-vs-quality curve
2. **Oracle-best loss** E[min_d L_t(d)] — if this beats depth-8, some tokens are HARMED by extra depth
3. **Oracle depth histogram** — distribution of d*_t = argmin_d L_t(d)
4. **No-regret shallowest depth** d_t^δ = min{d: L_t(d) ≤ L_t(8) + δ} with δ=0.01 (NOT 0.1 which was too loose)
5. **Oracle speedup** 8 / E[d_t^δ]
6. **Predictability** — can shallow features (depth-1 or depth-2 state) predict d*_t or marginal improvement?

### Decision criteria

| Result | Interpretation |
|---|---|
| Oracle speedup > 1.3× AND oracle-best improves over depth-8 | **Worth pursuing** — real headroom exists |
| d*=8 almost always, speedup 1.0-1.1× | **Not worth it** — depth-8 is near-optimal for all tokens |
| Oracle-best substantially better but not predictable from shallow state | Headroom exists but halting is hard to learn (still interesting for Pathway 5) |
| Train speedup >> val speedup | Overfitting artifact — depth isn't helping on new data |

### Implementation notes

- The existing `per_token_depth_analysis.py` script in `runs/` does something similar but with flawed methodology (overlapping windows). Could be adapted.
- Key change: use the standard eval function, not sliding windows. One prediction per position.
- Save raw L_t(d) matrix for all positions — enables later analysis without re-running.

---

## Clean measurement results (2026-05-26)

**Script:** `runs/clean_depth_eval.py` | **Artifacts:** `experiments/wide_recurrent_vs_transformer/artifacts/clean_depth_eval/`

Ran on seed 42 only (seeds 43/44 would require retraining tied-depth models — not done yet). CPU-only eval of existing trained model.

### Methodology validation

Depth-8 val loss measured = **1.6505** vs standard eval = **1.6505** (difference < 0.0001). The methodology fix works — this is the real next-token prediction loss, not the inflated 3.37 from overlapping windows.

### Mean loss by depth (val)

| Depth | Mean loss | Δ from depth-8 |
|---|---|---|
| 1 | 2.371 | +0.720 |
| 2 | 1.948 | +0.298 |
| 3 | 1.757 | +0.106 |
| 4 | 1.702 | +0.051 |
| 5 | 1.675 | +0.024 |
| 6 | 1.666 | +0.015 |
| 7 | 1.658 | +0.007 |
| 8 | 1.650 | — |

Diminishing returns: 78% of total improvement happens by depth 3, 93% by depth 5.

### Oracle-best vs depth-8

| Split | Depth-8 loss | Oracle-best | Improvement | Tokens harmed by depth-8 |
|---|---|---|---|---|
| Val (19968 pos) | 1.650 | 1.365 | **0.285 nats (17.3%)** | **71.5%** |
| Train (19968 pos) | 1.369 | 1.173 | 0.196 nats (14.3%) | 67.7% |

Val improvement proportionally larger than train → **NOT an overfitting artifact.**

### Oracle depth histogram (val)

```
d1: ████████████████  15.2%  (3034)
d2: ███████████       11.0%  (2192)
d3: █████████          9.2%  (1838)
d4: ██████████         9.4%  (1867)
d5: ████████           8.3%  (1665)
d6: ██████████         9.7%  (1937)
d7: █████████          8.8%  (1750)
d8: ████████████████████████████  28.5%  (5685)
```

Surprisingly flat across depths 1-7, with a spike at d8 meaning ~28.5% of tokens genuinely benefit from full depth. The remaining ~71.5% would be better at some d < 8.

### No-regret shallowest depth (δ=0.01)

Mean optimal depth: **4.09** (val) / 4.34 (train)

**Oracle speedup: 1.96×** (val) / 1.84× (train)

At δ=0.01 (nearly lossless threshold), 25.3% of val tokens can safely stop at depth 1, and only 19.0% truly need depth 8.

### Decision per pre-registered criteria

**Oracle speedup (1.96×) > 1.3× ✓ AND oracle-best improves over depth-8 ✓**

→ **Worth pursuing.** Real headroom exists for dynamic depth.

### What this does NOT show

1. ~~**Predictability**~~ — NOW MEASURED (see below). Shallow state is weakly predictive.
2. **Multi-seed replication** — single seed only. The magnitude could shift.
3. **Causal mechanism** — we don't know WHY some tokens are harmed by more depth. Possible explanations: attention pattern interference at higher depths, or the model learns depth-8-optimal representations that sacrifice shallower accuracy.
4. **Architecture-specific** — this is a TinyShakespeare ctx=32 tied-depth transformer. Larger/different models may behave differently.

---

## Predictability probe (2026-05-26)

**Script:** `runs/depth_predictability_probe.py` | **Artifacts:** `experiments/wide_recurrent_vs_transformer/artifacts/depth_predictability_probe/`

Can shallow hidden state (after 1 or 2 tied iterations) predict which tokens benefit from more depth?

### Method

Extract hidden state at depth 1 and depth 2 for all 19968 val positions. Train linear probe and 1-hidden-layer MLP (72→32→output) on 80% of positions, evaluate on 20%. Three prediction targets:
- Binary: "harmed by depth-8" (oracle d* < 8)
- Binary: "done by depth 3" (no-regret depth ≤ 3 at δ=0.01)
- Multiclass: predict exact oracle depth d*

### Results

| Feature | Target | Baseline | Linear | MLP |
|---|---|---|---|---|
| Depth-1 hidden | Harmed by d8 | 71.6% | 73.5% | 75.0% |
| Depth-2 hidden | Harmed by d8 | 71.6% | 73.6% | 74.8% |
| Depth-1 hidden | Done by d3 | 52.5% | 59.7% | — |
| Depth-2 hidden | Done by d3 | 52.5% | 60.3% | — |
| Depth-1 hidden | Oracle d* | 28.4% | 33.0% | 34.4% |
| Depth-2 hidden | Oracle d* | 28.4% | 32.8% | 35.6% |

### Interpretation

**Shallow state is weakly predictive — not enough for easy halting.**

- "Harmed by depth-8" (majority class 71.6%): best probe 75.0% — only 3.4% above baseline
- "Done by depth 3" (more balanced, 52.5% baseline): best probe 60.3% — 7.8% above baseline, moderate
- Exact oracle depth (chance 28.4% = always predict d8): best probe 35.6% — 7.2% above baseline

Depth-2 is slightly better than depth-1 (especially for multiclass with MLP), but the difference is small.

### What this means for Pathway 5

Per pre-registered decision criteria: **"Oracle-best substantially better but not predictable from shallow state → Headroom exists but halting is hard to learn."**

The headroom is real (1.96× speedup if you could halt optimally). But a post-hoc probe on a model trained without halting objectives can barely identify which tokens need more depth. This means:

1. A simple "add a halt head to an existing model" approach likely won't work
2. The model would need to be **trained with** a halting/depth-allocation objective to develop depth-predictive features
3. Or a more sophisticated halting mechanism is needed (e.g., predict marginal improvement at each depth, not binary stop/go from shallow state alone)

This doesn't kill Pathway 5 — it calibrates expectations. "Dynamic depth" is an architectural training objective, not a cheap inference trick on existing models.

### Next steps (Pathway 5, revised)

1. ~~**Predictability test**~~ — DONE. Result: weakly predictive.
2. **Multi-seed** — retrain 2 more tied-depth models (seeds 43/44) and confirm the 1.96× speedup is reproducible.
3. **Halting-aware training** — train a model WITH a small confidence head from the start (predicting "how much will the next iteration help?"). The model may learn to encode depth-useful features in its hidden state when incentivized to do so.
4. **Per-depth marginal prediction** — instead of predicting from depth-1 state only, predict at EACH depth whether the next iteration will help. This matches the architecture better (you make the stop decision incrementally, not from shallow state alone).

---

## Oracle measurement at d=256, 4 iterations (2026-05-27)

**Script:** `runs/oracle_depth_analysis.py` | **Artifacts:** `experiments/tinyshakespeare/artifacts/oracle_depth_analysis/`

Tests whether the oracle opportunity persists at the larger scale used in Pathway 1 experiments (d=256, 4 iterations, 900K chars, 20K steps). This is a different model from the d=72/depth-8 measurement above.

### Mean loss by depth (val)

| Depth | Mean loss | Δ from depth-4 |
|---|---|---|
| 1 | 2.375 | +0.816 |
| 2 | 1.729 | +0.170 |
| 3 | 1.584 | +0.026 |
| 4 | 1.558 | — |

Diminishing returns: 94.9% of total improvement happens by depth 3 (0.816→0.026 marginal left).

### Oracle-best vs depth-4

| Split | Depth-4 loss | Oracle-best | Improvement | Tokens harmed by depth-4 |
|---|---|---|---|---|
| Val (19872 pos) | 1.558 | 1.396 | **0.163 nats (10.4%)** | **44.9%** |

### Oracle depth histogram (val)

```
d1: ██████████████        12.8%  (2548)
d2: █████████████████     16.5%  (3278)
d3: ████████████████      15.5%  (3089)
d4: ████████████████████████████████████████████████████████  55.1%  (10957)
```

55% of tokens genuinely benefit from full depth-4. Much higher than the d=72/depth-8 model (28.5% needed depth-8) — which makes sense: fewer iterations means each one carries more weight.

### No-regret shallowest depth (δ=0.01)

Mean optimal depth: **2.93**

**Oracle speedup: 1.37×**

At δ=0.01, 19.7% of tokens can safely stop at depth 1. 47.9% truly need depth 4.

### Decision per pre-registered criteria

**Oracle speedup (1.37×) > 1.3× ✓ AND oracle-best improves over depth-4 ✓**

→ **Worth pursuing.** Real headroom exists, though less than the d=72/depth-8 case (1.96×).

### Comparison: d=72/depth-8 vs d=256/depth-4

| Metric | d=72, depth-8 | d=256, depth-4 |
|---|---|---|
| Oracle speedup | 1.96× | 1.37× |
| Oracle-best improvement | 0.285 nats (17.3%) | 0.163 nats (10.4%) |
| Tokens harmed by full depth | 71.5% | 44.9% |
| Tokens needing full depth | 28.5% | 55.1% |

The deeper model (8 iterations) has far more headroom for early exit. This suggests: **dynamic depth is most valuable with MORE iterations, not fewer.** The d=256/4-iter model is already fairly well-utilized at each depth.

### Implication

If pursuing dynamic depth, prefer deeper models (8+ iterations) where the opportunity is larger. At 4 iterations, the headroom is real but modest (1.37×). The 1.96× result at 8 iterations is far more attractive — and our stability results showed training is stable through 8 iterations (and likely more).

### Next step

**Halting-aware training at depth-8** — train a d=256, 8-iteration model with a per-depth confidence head from the start. Use the oracle measurement to set expectations. The d=72 result suggests ~2× speedup is achievable if the halt head can learn the pattern.

---

## Oracle measurement at d=256, 8 iterations (2026-05-27)

**Script:** `runs/oracle_depth_analysis.py --recurrent-iterations 8` | **Artifacts:** `experiments/tinyshakespeare/artifacts/oracle_depth_analysis/`

Tests whether the larger oracle opportunity from d=72/depth-8 (1.96×) persists at d=256 with 8 iterations.

### Mean loss by depth (val)

| Depth | Mean loss | Δ from depth-8 |
|---|---|---|
| 1 | 2.747 | +1.169 |
| 2 | 2.043 | +0.465 |
| 3 | 1.798 | +0.220 |
| 4 | 1.683 | +0.105 |
| 5 | 1.628 | +0.050 |
| 6 | 1.597 | +0.019 |
| 7 | 1.582 | +0.004 |
| 8 | 1.578 | — |

### Oracle-best vs depth-8

| Split | Depth-8 loss | Oracle-best | Improvement | Tokens harmed by depth-8 |
|---|---|---|---|---|
| Val (19872 pos) | 1.578 | 1.395 | **0.183 nats (11.6%)** | **53.4%** |

### Oracle depth histogram (val)

```
d1: ████████       7.3%   (1441)
d2: █████████      8.6%   (1715)
d3: ██████████     9.5%   (1893)
d4: ███████        6.9%   (1366)
d5: ██████         6.5%   (1284)
d6: ███████        7.0%   (1384)
d7: ████████       7.7%   (1521)
d8: ████████████████████████████████████████████████  46.6%  (9268)
```

Flatter distribution than depth-4 (which had 55% at max depth). At depth-8, 46.6% truly need the full depth, while 53.4% are optimally served by some d < 8.

### No-regret shallowest depth (δ=0.01)

Mean optimal depth: **5.07**

**Oracle speedup: 1.578×**

At δ=0.01: 15.7% can safely stop at depth 1, 35.2% truly need depth 8.

### Cross-scale comparison

| Metric | d=72, depth-8 | d=256, depth-4 | d=256, depth-8 |
|---|---|---|---|
| Oracle speedup | 1.96× | 1.37× | **1.58×** |
| Oracle-best improvement | 0.285 nats | 0.163 nats | 0.183 nats |
| Tokens harmed by full depth | 71.5% | 44.9% | 53.4% |
| Tokens needing full depth | 28.5% | 55.1% | 46.6% |

The d=256/depth-8 model sits between the other two. More iterations always gives more headroom, but larger models (d=256 vs d=72) seem to use their depth more efficiently (less waste). The 1.58× result confirms dynamic depth is worth pursuing at practical model sizes.

### Implication for halting experiment

With 1.58× oracle ceiling, a learned halt head achieving even 50% oracle efficiency would give ~1.29× speedup at ≤0.02 nats loss. Getting to 70% efficiency would give ~1.4× — meaningful compute savings for no quality loss.

---

## Halting-aware training: pilot results (2026-05-27)

**Script:** `runs/halting_aware_training.py` | **Artifacts:** `experiments/tinyshakespeare/artifacts/halting_aware_training/`

Trained d=128, 8-iteration model with jointly-trained halt head (10K steps, ~7 min). The halt head predicts "safe to stop now?" at each depth using oracle labels.

### Key result: PRACTICAL FAILURE

The learned halt head **does worse than a trivial fixed-depth baseline**.

| Policy | Avg depth | Val loss hit | Speedup |
|---|---|---|---|
| Always depth 8 (baseline) | 8.00 | 0.000 | 1.00× |
| **Fixed depth 6 (trivial)** | **6.00** | **0.015** | **1.33×** |
| Learned halt (τ=0.80) | 7.79 | 0.015 | 1.03× |
| Oracle | 5.08 | 0.000 (by definition) | 1.58× |

At the same loss budget (0.015 nats), fixed depth-6 gives 1.33× speedup while the learned halt gives only 1.03×. The adaptive policy adds complexity for no benefit.

### What the halt head DID learn

- **AUROC per depth:** [0.693, 0.662, 0.669, 0.665, 0.654, 0.642, 0.605]
- Best AUROC (depth 1): 0.693 — above random (0.5) but below the 0.70 threshold
- The head learned real token-specific signal (not just the depth prior)
- But discrimination is too weak for useful halting

### Against pre-registered criteria

| Criterion | Required | Actual | Verdict |
|---|---|---|---|
| AUROC > 0.70 at ≥1 depth | > 0.70 | 0.693 | ❌ Marginal miss |
| Avg depth ≤ 6.5 | ≤ 6.5 | 7.79 | ❌ Clear miss |
| Val loss hit ≤ 0.02 | ≤ 0.02 | 0.015 | ✓ (but only because barely halting) |
| Better than post-hoc | Yes | Probably yes | ✓ (AUROC metric vs accuracy not directly comparable) |
| **Does NOT meet failure bar** | AUROC ≤ 0.60 everywhere | 0.693 at d1 | Not formal failure |

### Diagnosis

1. **The objective formulation is likely wrong** — binary BCE "safe-to-stop-now?" doesn't encode the cost asymmetry of halting. A shallow false positive is catastrophic (loss explodes); a deep false negative just wastes one iteration.

2. **Training was still improving** — AUROC climbed from 0.60→0.66 during training, eval AUROC 0.693 slightly higher than final training metric. Undertraining is possible but unlikely to close the gap to beating fixed depth-6.

3. **Model capacity is not the bottleneck** — the signal exists (AUROC > 0.5), it's just not being turned into a useful policy.

### Next step: falsification run

Running 30K steps (same setup) to definitively rule out undertraining. If AUROC at evaluation still doesn't yield a policy that beats fixed depth-6, abandon binary BCE and switch to:

**Marginal improvement regression** — predict `remaining_gain = ℓ(d) - ℓ(N)` as a continuous value. Halt when predicted remaining gain < ε. This preserves severity information and matches the actual deployment decision.

> **Purpose:** Determine whether a jointly-trained halt head can learn to predict oracle depth, given that post-hoc probing failed (3.4% above baseline).

### Why joint training is necessary

The predictability probe (above) showed that a frozen model's features are barely predictive of halting decisions. This means the model doesn't naturally develop depth-indicative features. Joint training creates gradient pressure for the model to encode "am I done?" information in its hidden state.

### Architecture

**Base model:** SharedRecurrentCore (d_model=128, 4 heads, ff_dim=512, 8 iterations, ctx=128, TinyShakespeare)

Smaller than the d=256 measurement model — this is the cheapest honest test of the mechanism.

**Halt head** (shared across all depths):
```
input:  h_d[:, -1, :]          [batch, 128]     (last-position hidden state at depth d)
        LayerNorm(128)
        concat d/N scalar       [batch, 129]     (normalized depth index)
        Linear(129, 64) + GELU
        Linear(64, 1)           [batch, 1]       (halt logit)
output: sigmoid → halt probability p_d
```

The head is shared across depths and receives the depth index explicitly. This is simpler than separate heads and allows generalization across depths.

### Training objective

**Supervised oracle labels** (not REINFORCE — too many confounds for a first test):

```
y_{i,d} = 1[ℓ_{i,d} ≤ ℓ_{i,N} + δ]    (δ = 0.01 nats)
```

where ℓ_{i,d} is the per-example CE loss using hidden state at depth d.

**Total loss:**
```
L = L_LM + λ · L_halt

L_LM   = mean CE at full depth (depth N)
L_halt = (1/(N-1)) Σ_{d=1}^{N-1} BCEWithLogits(s_{i,d}, y_{i,d})
```

- Depth N excluded from halt loss (trivially "stop here")
- λ warmup: 0 → 0.1 linearly over first 10% of training
- Per-depth pos_weight from running average of class balance (prevents collapse to "always continue")
- Labels are **detached** — no gradient flows through oracle label construction

**Gradient flow:** Halt loss gradients DO flow into the shared model (through h_d). This is the key difference from post-hoc probing — the model is incentivized to make its states halt-predictive.

### Training protocol

- Steps: 10,000
- Batch size: 64
- Always run all 8 iterations during training (oracle labels require full-depth losses)
- Log per-depth: halt positive rate, BCE loss, running AUROC
- No actual halting during training

### Evaluation protocol

After training, on validation set:

1. **Threshold sweep:** For τ ∈ {0.05, 0.10, ..., 0.95}, halt at first depth where p_d ≥ τ
2. **Per threshold, report:**
   - Average depth used
   - Val loss achieved (using the halted depth's hidden state)
   - Learned speedup = N / avg_depth
3. **Oracle efficiency** = (learned_speedup - 1) / (oracle_speedup - 1)
4. **Threshold-free metric:** AUROC per depth for "safe to stop now?" classification

### Success bar (minimum to continue)

| Metric | Threshold |
|---|---|
| AUROC at ≥1 early/mid depth | > 0.70 |
| Best threshold: avg depth | ≤ 6.5 (of 8) |
| Best threshold: val loss hit | ≤ 0.02 nats vs full depth |
| Clearly better than post-hoc probe | Yes (by visual inspection of AUROC) |

### Failure bar (stop condition)

- AUROC ≤ 0.60 at all depths, AND
- No useful frontier point in threshold sweep (either no depth reduction, or obvious loss damage > 0.05 nats)

If failure: conclude that simple supervised joint halt training doesn't make halt-predictive features emerge. Consider alternatives: marginal improvement regression, or auxiliary halting loss formulations.

### Expected runtime

~20-45 min on RTX 3090 (d=128, 8 iterations, 10K steps). Affordable as a single pilot run.
