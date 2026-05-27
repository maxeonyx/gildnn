# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-27, 15:00 NZST)

**Architecture correctness: DONE.** Per dictations 8-9:
- Fixed embeddings (random unit vectors, not learned)
- Separate optimizer per block (truly independent learners)
- Sanity check passed with new setup (commit `6250ccd`)

**Stability fix: DONE.** Per-timestep L2 normalization prevents hidden state explosion.

**~3 days remain in timebox. GPU: FREE.**

---

## ✅ DONE: Separate optimizers + fixed embeddings (dictations 2026-05-27-8, 2026-05-27-9)

Implemented and verified in commit `6250ccd`. Sanity check passed — all 3 block losses decrease.

- Fixed embedding = random unit vectors, `requires_grad=False`
- One AdamW per block (FFN + mix params)
- No shared trainable parameters between blocks

---

## Architecture checklist

| Piece | Status | Evidence |
|---|---|---|
| Multi-block grid | ✅ | `core/model.py` ParallelDiagonalModel |
| Stale laterals (one-timestep delay) | ✅ | `detach_lateral=True`, topology="upward" |
| Multi-rate blocks (1,2,4) | ✅ | Assembly sanity check |
| Per-block local CE loss (predict rate ahead) | ✅ | Assembly sanity check — all blocks learn |
| Weight-tied normalized readout | ✅ | `tied_logits` with normalize=True |
| TBPTT on long sequences | ✅ | `runs/assembled_architecture.py` (seq=2048, chunk=128) |
| CUDA graph execution | ❌ | Exists in `core/training.py` GraphTrainer, but not wired for TBPTT |
| Dynamic depth / halting | ❌ | Validated separately, not yet in assembly |
| Noise on laterals | ❌ | Not implemented |

---

## What's next

### 1. Run assembly for real (~10-15 min) — IN PROGRESS

Now that the architecture is correct, run at full scale and answer: **does the higher block learn something different from the lower block?**

Pathway connection: Pathway 3 (Local Learning) + Pathway 8 (Multi-Rate Processing)
- Evidence for: blocks converge to different loss levels, representation probing shows distinct features
- Evidence against: all blocks converge to same representation despite different objectives

Metrics: per-block loss curves (script already produces these). For stronger evidence, would need probing/similarity analysis as a follow-up.

### 2. Fix temperature → re-run (quick)

Temperature 0.07 is wrong for fixed embeddings (see note above). Change to 0.5, sanity-check, re-run. Block 0 should start near uniform and learn faster.

### 3. Local-only vs full-backprop A/B (Pathway 3: Local Learning)

**The most discriminating next test.** Same architecture, one change: `detach_lateral=True` (current) vs `detach_lateral=False` (full gradient through laterals).

- Hypothesis: most learning signal is already captured by local objectives; full backprop helps little
- Success for local learning: block losses within ~0.05-0.1 nats of full-backprop control
- Success for architecture: upper blocks are load-bearing (ablation hurts by >0.05 nats)
- Failure: full-backprop dramatically outperforms local-only → local learning thesis weakens

Explicitly listed in dictation 2026-05-27-6 as a valid one-change comparison.

### 4. Further valid comparisons (per dictation 2026-05-27-6)

- Noise on laterals → timescale separation?
- CUDA graph integration → expected speedup?

---

## Architecture (validated hyperparams)

- **Fixed** normalized token embeddings (random unit vectors, not learned)
- Weight-tied readout (dot product against fixed embedding vectors)
- Normalized block outputs (L2-norm before readout)
- Addition-based lateral combination (lateral_scale=0.2)
- Per-timestep L2-norm on hidden states (stability fix)
- CE local loss for all blocks (cosine/L2 worse — tested earlier)
- Temperature = 0.07 (**⚠️ see note below**)
- Separate optimizer per block (AdamW, no shared params)

### ⚠️ Temperature note (discovered 2026-05-27)

Temperature 0.07 was validated with LEARNED embeddings. With FIXED RANDOM embeddings in 96 dimensions:
- Dot products between random unit vectors have std ≈ 1/√96 ≈ 0.102
- Dividing by 0.07 gives logit std ≈ 1.46 → randomly peaked softmax
- Expected initial CE ≈ 5.19 (vs uniform 4.13) — exactly what we observe

**However: τ=0.5 makes learning WORSE, not better.** The near-uniform init (good) comes with diffuse gradient geometry (bad). AdamW normalizes gradient scale, so the issue is gradient SHAPE not magnitude. τ=0.07's "random overconfidence" creates strong contrastive signal that accelerates learning. The model corrects from 5.19 → 3.85 in 200 steps — the high init is cosmetic, not harmful.

**Conclusion: keep τ=0.07.** Revert from 0.5 for the next run.

---

## Key references

- `dictations/2026-05-27-2.md` through `2026-05-27-7.md` — the assembly directive and principles
- `core/model.py` — `ParallelDiagonalModel` + `ParallelDiagonalCarryState`
- `core/tied_readout.py` — weight-tied readout + local loss functions
- `runs/assembled_architecture.py` — the assembled training script
- `PROCESS.md` — experiment discipline (updated with valid-experiment criteria)

---

## Deferred: codebase cleanup

`runs/` has ~30 dead experiment scripts, all superseded by `assembled_architecture.py`. Git preserves history. Delete them at the next natural stopping point (between experiment threads). Only `assembled_architecture.py` and `generate_text.py` (template for generation) are potentially relevant.
