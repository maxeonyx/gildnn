# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-28, 10:30 NZST)

**2-block prediction experiment running.** Block 1 consistently beating copy baseline (26% better at step 350/800). Trend still improving. No-recurrence ablation will follow.

**3-block combining script ready.** `runs/stream_combining.py` committed. CPU sanity check shows 45% advantage for combined stream. Waiting for GPU.

**Piece test 06 done.** Bayesian combining (tempered PoE + process noise) validated.

**~2 days remain in timebox. GPU: BUSY (predictive_processing).**

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

## IN PROGRESS: 2-block prediction experiment (GPU active)

**Running now** (PID 14980, `runs/active.lock` present). Phase B step 350/800, then no-recurrence ablation (800 steps).

Results so far:
- Phase A: Block 0 CE 5.35 → 2.51 ✅
- Phase B: Block 1 MSE steadily beating copy baseline (0.00357 vs 0.00485 = **26% better** at step 350)
- Still improving — trend is clear

**Key question remaining:** Does the no-recurrence ablation perform worse? That proves recurrence enables temporal prediction (not just learning a static transform).

## READY TO LAUNCH: 3-block stream combining experiment

Script: `runs/stream_combining.py` (committed, sanity-checked on CPU)
Config: 200 Block 0 steps, 800 Block 1 steps, 800 Block 2 steps, lambda=0.5

Tests whether folding Block 1's predictions into the stream helps Block 2. Design: `research/questions/stream-combining/README.md`

CPU sanity check showed combined (0.002073) 45% better than passthrough (0.003774) — very promising signal, but need full GPU run to confirm.

**Launch immediately after 2-block experiment finishes** (once `runs/active.lock` is cleared).

## ✅ DONE: Bayesian combining piece test

Piece test 06 validates tempered PoE + process noise. All 26 checks pass. Script: `experiments/predictive_processing/06_bayesian_combining.py`

## TODO: Code deduplication

`runs/stream_combining.py` duplicates ~300 lines from `runs/predictive_processing.py` (blocks, data utils, training helpers). Should extract shared pieces to `core/` once both experiments are stable. Not urgent — both run independently.

---

## What's been validated (keep, don't re-do)

| Finding | Evidence | Still relevant? |
|---|---|---|
| Local learning works (detach laterals) | A/B test: block 0 0.215 nats better | ✅ — predictive processing requires local learning |
| Separate optimizers per block | Architecture correctness | ✅ — blocks must be independent |
| Fixed random embeddings work | Training runs | ❓ — may change with distributional stream |
| Multi-rate processing works | Assembly sanity check | ✅ — multi-timescale is core |
| Per-timestep normalization needed | Stability fix | ❓ — may not apply to distributional stream |

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
