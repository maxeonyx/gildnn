# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-27, 19:30 NZST)

**COURSE CORRECTION per dictation 2026-05-27-10.** Stop refining "blocks predict token CE." Start implementing predictive processing: blocks predict EACH OTHER's distributions. Test pieces in isolation — don't train.

**Piece tests: ALL DONE.** Every piece of the predictive processing architecture validated independently.
**Daily report: NEEDS UPDATE** with piece test results (append to existing).

**~3 days remain in timebox. GPU: FREE.**

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

## NEXT: Assemble minimal predictive processing model

The pieces work independently. The next step is to wire them into the simplest possible predictive processing model:

**Minimal design (2 blocks, no multi-rate yet):**
1. Block 0: reads token input, produces a distribution (μ, σ) — its representation of the current token
2. Block 1: receives Block 0's distribution from the PREVIOUS timestep (stale lateral)
3. Block 1's loss: W₂²(Block 1's prediction of what Block 0 will produce, Block 0's actual output)
4. Block 0's loss: could be token-level CE (it's the interface to the world) OR something else

**Open design questions (need resolution before implementing):**
- What is Block 0's loss? It needs SOME grounding to the world (tokens). CE on token logits is simplest.
- Does Block 1 also predict tokens, or ONLY predict Block 0's distribution?
- How does the combining function work at each node? (Not needed for this 2-block test — just pass the distributions directly via stale laterals)
- Should this be tested with synthetic data first, or go straight to TinyShakespeare?

**Suggested approach:** Start with the simplest version. Block 0 predicts tokens (CE loss). Block 1 predicts Block 0's output distribution (W₂² loss). Run on TinyShakespeare. See if Block 1 learns anything useful.

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
