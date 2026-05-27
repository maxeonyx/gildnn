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

### 2. Valid comparisons (per dictation 2026-05-27-6)

Once training is confirmed stable and blocks diverge:
- Noise on laterals → timescale separation?
- Global backprop vs local-only → blocks still learn differently?
- CUDA graph integration → expected speedup?

---

## Architecture (validated hyperparams)

- **Fixed** normalized token embeddings (random unit vectors, not learned)
- Weight-tied readout (dot product against fixed embedding vectors)
- Normalized block outputs (L2-norm before readout)
- Addition-based lateral combination (lateral_scale=0.2)
- Per-timestep L2-norm on hidden states (stability fix)
- CE local loss for all blocks (cosine/L2 worse — tested earlier)
- Temperature = 0.07
- Separate optimizer per block (AdamW, no shared params)

---

## Key references

- `dictations/2026-05-27-2.md` through `2026-05-27-7.md` — the assembly directive and principles
- `core/model.py` — `ParallelDiagonalModel` + `ParallelDiagonalCarryState`
- `core/tied_readout.py` — weight-tied readout + local loss functions
- `runs/assembled_architecture.py` — the assembled training script
- `PROCESS.md` — experiment discipline (updated with valid-experiment criteria)
