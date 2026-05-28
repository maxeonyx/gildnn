# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-28, 11:21 NZST)

**3-block stream combining experiment running** (PID 6728). Tests whether folding Block 1's predictions into the stream helps Block 2.

**2-block prediction experiment COMPLETE.** Key finding: prediction works (37% better than copy), but recurrence doesn't help — static transform matches/beats recurrence.

**~2 days remain in timebox. GPU: BUSY (stream_combining, PID 6728).**

---

## ⚠️ DICTATIONS 11-14 — implications

### Dictation 11: Stream accumulation changes the combining function story

The stream at a node ACCUMULATES predictions over time. It's not "passthrough actual." The stream at t=2 is combine(actual_t1, prediction_based_on_t0). This reframes piece test 04's conclusion — "passthrough is the only stable option" was testing the wrong question.

**Real question:** "What does the stream become when you fold in a prediction?" → Bayesian combining (prior × likelihood → posterior) is back in play.

**Impact on current work:** The 2-block assembly doesn't exercise the combining function (there's no Block 2 reading the combined stream). So the assembly is still valid as a prediction-mechanism test. But the NEXT piece test should be: Bayesian combining in distribution-parameter space.

### Dictation 12: Model goal is high-level representations, not token prediction

"The model isn't supposed to be good at predicting tokens. It's supposed to be good at building high-level representations."

**Impact:** Block 0's CE loss is just GROUNDING (gives it a reason to produce meaningful representations). The point of the experiment is Block 1's ability to predict — that's what tests the predictive processing mechanism. Don't optimize Block 0's CE.

### Dictation 13: Multi-rate ≠ halting (redaction)

Multi-rate = how often each block fires across timesteps (temporal extent). Halting = how many recurrent iterations within a single timestep. Independent dimensions.

### Dictation 14: Future direction — learned distribution family

Not current work. But shows where this is heading: learned density evaluator, sampler, combiner, distance. For now: diagonal Gaussians with explicit params.

---

## ⚠️ DICTATION 10 — what it changes

The previous plan (noise on laterals, more A/B tests) was refining a model that ISN'T the target. The target is:

1. Block 0 produces a **distribution** (not logits, not point vectors)
2. Block 1 **predicts block 0's distribution** from one timestep ahead
3. The loss is **Wasserstein distance** between predicted and actual distributions
4. **Up** = surprisal (what wasn't predicted). **Down** = what was predicted.
5. Test ALL pieces in isolation — sample distributions, put them through operations, verify properties.

Max explicitly says: "Don't train a full model. Don't even train. Just do the test with some distribution of activations."

---

## What's next: piece tests for predictive processing

All pieces validated ✅:

### ✅ Piece 1: Wasserstein loss on diagonal Gaussians
W₂² = ||μ₁-μ₂||² + ||σ₁-σ₂||². All metric properties pass. Autograd correct. Script: `experiments/predictive_processing/01_wasserstein_diag_gaussian.py`

### ✅ Piece 2: Combining function candidates
W₂ residual decomposition is the clear winner for analysis. PoE is fusion-only, not a decomposition. Script: `experiments/predictive_processing/02_combiner_candidate_sweep.py`

### ✅ Piece 3: Up/down decomposition
W₂ residual: exact reconstruction, zero-on-identity, strong loss correlation, monotonic sweeps. Script: `experiments/predictive_processing/03_up_down_decomposition.py`

### ✅ Piece 4: Stream chain dynamics
Key finding: **surprise cannot BE the stream** (W₂ residual produces invalid Gaussians under chaining). The stream carries ACTUAL distributions. Surprise is a side-channel metric. Script: `experiments/predictive_processing/04_stream_chain_dynamics.py`

### ✅ Piece 5: W₂² learning convergence
A small MLP learns to predict distributions via W₂² loss. Converges cleanly. Gradients well-behaved. W₂² = MSE on (μ,σ) numerically identical. Script: `experiments/predictive_processing/05_w2_learning_convergence.py`

---

## ✅ DONE: 2-block prediction experiment

**Complete.** Report: `experiments/tinyshakespeare/artifacts/predictive_processing/report.json`

Results:
- Phase A: Block 0 CE 5.38 → 2.51 ✅
- Phase B: Block 1 eval MSE = **0.003038** vs copy baseline 0.004852 = **37% better than copy** ✅
- No-recurrence ablation: eval MSE = **0.003027** — **marginally BETTER than recurrence** (0.4% difference = noise)

**Key finding:** Prediction works (both versions beat copy by 37-38%). But recurrence provides ZERO benefit for 1-step-ahead prediction at this scale. The model learns a static transform of the previous activation, not temporal prediction.

**Implications:**
- Combining experiment is unaffected — static predictions still enrich the stream
- Multi-rate is where recurrence SHOULD matter (predicting 2+ steps ahead requires temporal context)
- For the combined architecture: recurrence may only become load-bearing at longer prediction horizons or larger scale

## ✅ DONE: 3-block stream combining experiment

**Complete.** Report: `experiments/tinyshakespeare/artifacts/stream_combining/report.json`

Results (Phase C — Block 2 prediction under three conditions):

| Condition | Eval MSE | Copy Baseline | Gain |
|---|---|---|---|
| Combined (real B1 predictions) | 0.001613 | 0.002782 | **42.0%** |
| Passthrough (raw Block 0) | 0.003064 | 0.004852 | **36.9%** |
| Random (noise combined) | 0.005208 | 0.008040 | **35.2%** |

**Decision tree: provisional Case A** (combined > passthrough AND random < passthrough). Ordering is correct but gaps are modest (5pp combined-vs-passthrough, 1.7pp passthrough-vs-random) on one seed. Interpretation: real predictions provide modest benefit beyond raw passthrough; combining doesn't poison; but the effect at equal-rate H=1 is not dramatic. Consistent with recurrence-null — H=1 predictions are near-redundant, so folding them in doesn't transform much.

**Implication:** Multi-rate proceeds (softened conditional satisfied). Horizon sweep is the real discriminator for whether longer-horizon predictions carry genuinely novel info.

## IN PROGRESS: Horizon sweep H=2 (GPU active)

**Running now** (PID 15096, `runs/active.lock` present). Log: `experiments/tinyshakespeare/artifacts/horizon_sweep/run_h2.jsonl`. Config: 200 Block 0 steps, 2×800 Phase B steps (with/without recurrence). Started 2:00pm NZST. Expected ~79 min → completion ~3:20pm.

Tests whether recurrence becomes load-bearing at horizon 2 (predict A0_{t+2} from context up to t). Design: `research/questions/horizon-sweep/README.md`.

After H=2 completes: launch `--horizon 4` (another ~79 min). Both must finish before multi-rate design.

## Remaining timebox sequence (~2 days)

1. ~~**Finish combining**~~ ✅ DONE (provisional Case A)
2. **Horizon sweep** — H=2 running now. Then H=4. Total ~2.6 hours remaining.
3. **Reports** (daily + weekly, due after 4pm Thursday) — can write while horizon sweep runs
4. ~~**Noise sanity check**~~ ✅ DONE (piece test 07)
5. **Multi-rate** (proceeds regardless — combining positive, consistent with plan) → rates 1/2/4 with horizon-matched control
6. **One confirmation seed** if anything is clearly positive

Strategy: breadth first (answer more questions), then one depth step. Code dedup deferred until after key experiments.

### Decision tree outcome (was pre-registered, now resolved)

Pre-registered gain metric (G = 1 - MSE/own_copy_baseline) applied. Result: **Case A** (combined 42% > passthrough 37%, random 35% < passthrough 37%). Modest margins on one seed → "provisional" qualifier. Multi-rate proceeds.

Note: "Noise on laterals for information hierarchy" (dictation 2026-05-27-2) is validated by piece test 07. Adding noise makes prediction MORE valuable relative to copy (gain increases from 32%→94%). However, the recurrence ablation shows noise does NOT make recurrence more valuable — advantage DECREASES with noise (23%→0.15%). Mechanism: at high noise, both GRU and FF converge to the same "predict the mean" strategy; past noisy observations don't add independent information about a quickly-changing state. The regime where noise WOULD make recurrence load-bearing: slow latent state + independent observation noise + partial observability — exactly what MULTI-RATE provides (slow block fires infrequently, needs to filter across multiple noisy observations). Conclusion: noise alone is not the forcing mechanism for recurrence; multi-rate + noise together likely are.

## ✅ DONE: Bayesian combining piece test

Piece test 06 validates tempered PoE + process noise. All 26 checks pass. Script: `experiments/predictive_processing/06_bayesian_combining.py`

---

## What's been validated (keep, don't re-do)

| Finding | Evidence | Still relevant? |
|---|---|---|
| Local learning works (detach laterals) | A/B test: block 0 0.215 nats better | ✅ — predictive processing requires local learning |
| Separate optimizers per block | Architecture correctness | ✅ — blocks must be independent |
| Fixed random embeddings work | Training runs | ❓ — may change with distributional stream |
| Multi-rate processing works | Assembly sanity check | ✅ — multi-timescale is core |
| Per-timestep normalization needed | Stability fix | ❓ — may not apply to distributional stream |
| Prediction beats copy by 37% | 2-block experiment report.json | ✅ — mechanism validated |
| Recurrence doesn't help 1-step prediction | No-recurrence ablation matches/beats | ✅ — static transform sufficient at H=1 |
| Stream combining modestly helps (Case A) | 3-block report.json: G=42% vs 37% | ✅ — combining doesn't poison, provides modest benefit |
| Noise doesn't break prediction | Piece test 07: all σ levels beat copy | ✅ — noise on laterals is compatible |
| Noise doesn't force recurrence alone | Piece test 07 ablation: advantage drops | ✅ — need multi-rate + noise together |

---

## Terminology (per dictation 10)

- ❌ "states" — implies RNN hidden states, which this isn't
- ✅ "residual streams" — at certain times, at certain lateral distances
- ✅ "distributions" — what the stream carries and what blocks produce

---

## Key references

- `dictations/2026-05-27-10.md` — the course correction
- `research/questions/multi-timestep-architecture/README.md` — full theoretical design
- `ROADMAP.md` Pathway 3 — local learning via distributional predictive coding
- `core/model.py` — existing architecture (still useful as scaffolding)
