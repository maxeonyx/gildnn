# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-27, 19:00 NZST)

**COURSE CORRECTION per dictation 2026-05-27-10.** Stop refining "blocks predict token CE." Start implementing predictive processing: blocks predict EACH OTHER's distributions. Test pieces in isolation — don't train.

**Architecture prerequisites: DONE.** Local learning validated, separate optimizers, fixed embeddings.
**Daily report: NEEDS UPDATE** with course correction note.

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

Per the theoretical design in `research/questions/multi-timestep-architecture/README.md`, the pieces are:

### Piece 1: Wasserstein loss on diagonal Gaussians

The formula: W₂² = ||μ₁ - μ₂||² + ||σ₁ - σ₂||²

Test: sample random diagonal Gaussians, compute W₂², verify metric properties (symmetric, triangle inequality, zero iff same, differentiable).

This is trivial but worth writing down as executable code.

### Piece 2: The combining function (THE open question)

Given two distributions (block prediction + lateral arrival), produce:
- The stream state at this node
- Surprisal signal (goes rightward/up)
- Prior/context signal (goes leftward/down)

Candidates:
- Product of experts (precisions add)
- Bayesian update (one is prior, one is likelihood)
- Learned decomposition with conservation constraint

Test: sample many random pairs, apply operation, verify:
- Information conservation (up + down = original information)
- Surprisal is genuinely "what wasn't predicted"
- The operation is symmetric / has right asymmetry

### Piece 3: Up/down signal decomposition

Given prediction P and actual A, decompose into what was predicted vs what was surprising.

Test: can you reconstruct A from (what-was-predicted, what-wasn't-predicted)? Is there a clean decomposition?

### Piece 4: Block producing a distribution

A block reads a stream state (which is ALSO a distribution) and outputs a distribution. What's the minimal block that does this?

Test: random input distributions → block → output distribution. Does the output have sensible properties?

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
