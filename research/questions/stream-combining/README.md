# Stream Combining: Does folding predictions into the stream produce useful enrichment?

> Serves: [Pathway 3 (Local Learning)](../../../ROADMAP.md), specifically the combining function mechanism.
> Source: [dictation 2026-05-27-11](../../../dictations/2026-05-27-11.md)

---

## The question

When Block 1 predicts Block 0's next output, and we fold that prediction into the stream using tempered PoE, does Block 2 benefit from seeing the enriched combined stream vs just seeing raw Block 0 output?

This tests whether the combining mechanism adds value — whether a hierarchy of predictors creates progressively richer representations.

---

## Architecture

```
Block 0 (rate 1): reads tokens, produces distributions A0_t
Block 1 (rate 1): predicts A0_t from stale context, produces P1_t
Combine:          C1_t = TemperedPoE(A0_t, P1_t) + process noise
Block 2 (rate 1): predicts C1_t from stale context, produces P2_t

Losses:
  L1 = W₂²(P1_t, A0_t)           -- Block 1's prediction error
  L2 = W₂²(P2_t, C1_t)           -- Block 2's prediction error
```

Key: **Block 2 reads the COMBINED stream, not raw Block 0 output.** This is what dictation 11 means by "the stream accumulates."

All blocks at rate 1 for the first test (isolate combining from multi-rate).

---

## Combining function (validated in piece test 06)

Tempered diagonal-Gaussian Product of Experts:

```
τ_stream = 1/σ_stream², τ_pred = 1/σ_pred²
τ_new = τ_stream + λ·τ_pred
μ_new = (τ_stream·μ_stream + λ·τ_pred·μ_pred) / τ_new
σ_new = 1/√τ_new
σ_propagated = √(σ_new² + q²)
```

Where:
- λ ∈ (0, 1] controls how much the prediction influences the stream
- q > 0 is process noise preventing precision collapse

---

## Hypotheses

**H1 (combining helps):** Block 2's prediction error is lower when reading C1 (combined) than when reading raw A0.

**H2 (real prediction matters):** The benefit disappears if Block 1's predictions are random/shuffled. So combining helps BECAUSE the predictions are informative, not just because of smoothing.

**H3 (hierarchy develops):** With synthetic multi-timescale data, higher blocks' representations correlate more with slower latents.

---

## Planned controls

1. **Pass-through actual** (no combine): C1_t := A0_t. Block 2 sees raw Block 0.
2. **Random predictions** before combine: P1 is random noise instead of Block 1's learned prediction.
3. **Oracle prediction**: P1_t = A0_t exactly (perfect prediction). Tests combiner capacity in isolation from learning quality.

---

## Key risks

- **Poisoned bootstrap:** Early bad predictions from Block 1 corrupt C1, Block 2 trains on garbage → mitigate by annealing λ from 0 to target value
- **Overconfidence collapse:** Wrong low-σ predictions dominate PoE → σ floor + process noise
- **Upper-block starvation:** Combine becomes too sharp → track precision growth

---

## Design decisions (frozen for this experiment)

- Upward carrier = combined stream (not surprise)
- Surprise = derived side-channel (for loss, not injected into stream)
- Stale = one global tick (not block-rate-relative)
- Rates = all equal (rate 1) for this test
- Block 2 predicts next combined arrival

---

## Status

**Complete.** Report: [`report.json`](../../../experiments/tinyshakespeare/artifacts/stream_combining/report.json).

---

## GPU results (seed 42, d_model=96, 800 steps per condition)

| Condition | Final eval MSE | Copy baseline | Gain (G = 1 - eval/copy) |
|---|---|---|---|
| Combined (real B1 predictions) | 0.001613 | 0.002782 | **42.0%** |
| Passthrough (raw Block 0) | 0.003064 | 0.004852 | **36.9%** |
| Random (noise combined) | 0.005208 | 0.008040 | **35.2%** |

**Ordering: combined > passthrough > random.** Matches pre-registered Case A (combined better AND random worse than passthrough).

### Interpretation

The result is **provisional Case A** — the ordering is correct but the margins are modest:
- Combined vs passthrough: **+5.1pp** (likely real, but single-seed)
- Passthrough vs random: **+1.7pp** (marginal)

Real Block 1 predictions provide modest benefit over raw passthrough. Combining doesn't poison the stream. But at equal-rate H=1, the effect isn't dramatic. This is consistent with the recurrence-null at H=1 — if B1's predictions are near-Markov (basically "what A0 just said, slightly transformed"), then combining them adds incremental value, not transformative information.

**Why combined's copy baseline is lower:** The combined stream incorporates B1's predictions, which are a smoothed/averaged version of A0. This reduces timestep-to-timestep variance, making copy a better baseline (lower). Despite this tighter baseline, Block 2 still achieves a proportionally larger gain — suggesting real information content, not just smoothing.

**Why random barely differs from passthrough:** Combining random noise makes the stream noisier (higher copy baseline: 0.008040 vs 0.004852), but Block 2 can still learn temporal structure in a noisy stream. The gain metric normalizes this: Block 2 beats its own (worse) copy baseline by nearly as much as passthrough beats its (better) copy baseline. Block 2 is robust.

### What this settles

- ✅ H1 (combining helps): Yes, modestly. 5pp gain over passthrough.
- ⚠️ H2 (real prediction matters): Barely — random is only 1.7pp worse than passthrough.
- ❌ H3 (hierarchy develops): Not tested here (requires multi-timescale data or multi-rate).

### What this doesn't settle

- Whether combining helps MORE at longer horizons (→ horizon sweep)
- Whether multi-rate makes the combined predictions carry genuinely novel information
- Whether the 5pp gap is seed-robust (single seed = provisional)

### Decision

Per pre-registered softened conditional: **multi-rate proceeds.** Combining is positive (even if modest), so the condition is satisfied. Horizon sweep will determine whether longer predictions carry novel info that makes combining transformative rather than incremental.

---

## CPU sanity check results (60 training steps, seq_len=128)

| Condition | Block 2 MSE | Copy baseline | vs Copy | vs Random |
|---|---|---|---|---|
| Combined (Block 1 predictions) | 0.002073 | 0.001431 | — | 64% better |
| Passthrough (raw Block 0) | 0.003774 | 0.002879 | — | 35% better |
| Random (noise combined) | 0.005770 | 0.006411 | — | — |

**Ordering: combined < passthrough < random.** This is the H2 discriminative test — combining with random noise HURTS prediction (worse than passthrough), while combining with informative predictions HELPS. The combining function needs real information, not just smoothing.

**Important nuance:** Copy baselines differ across conditions because the combined stream is intrinsically smoother (incorporating predictions reduces timestep-to-timestep variance). The meaningful comparison for the full GPU run is whether each Block 2 beats ITS OWN copy baseline by a larger margin in the combined condition. At 60 steps, Block 2 hasn't yet beaten copy for combined/passthrough (short training), but the relative ordering between conditions is already clear.

---

## Next steps

1. ~~Design synthetic multi-timescale data source~~ — using TinyShakespeare (real text), same as 2-block
2. ~~Implement 3-block equal-rate experiment~~ — done (`runs/stream_combining.py`)
3. ~~Launch on GPU~~ — ✅ complete, provisional Case A
4. **Horizon sweep** (running now) — tests whether H=2/H=4 predictions carry more novel info
5. If horizon sweep positive + combining positive: **multi-rate experiment** (rates 1/2/4 with horizon-matched control)

---

## Multi-rate extension design (for after equal-rate confirms)

Key design decisions (from theory analysis):

**Prediction target:** Each block predicts the stream value AT ITS NEXT FIRING TIME. Block 1 (rate 2) predicts A0 at t+2. Block 2 (rate 4) predicts C1 at t+4. This couples rate and horizon — which is the point (higher blocks forced onto longer timescales).

**Combining semantics:** Combine ONLY WHEN DUE. A prediction from Block 1 at time t is stored until t+2, then combined with the arriving actual. Between firings, C1_t = A0_t (no stale prediction contaminating intermediate steps).

**Required control:** All-rate-1 with horizon-matched targets [+1, +2, +4]. This isolates the effect of multi-rate firing from the effect of predicting further ahead. Without this control, can't distinguish timescale separation from horizon difficulty.

**What to measure:** Per-block loss vs copy baseline at its own horizon. Whether higher-block representations vary more slowly over time (temporal autocorrelation). Whether B2 benefits more from combined stream when multi-rate (predictions more valuable when blocks are further apart).
