# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-27, 14:15 NZST)

**Assembly sanity check: PASSED.** All 3 blocks' local losses decrease independently.
**Stability fix: DONE.** Per-timestep L2 normalization prevents hidden state explosion.
**Training stability: CONFIRMED.** Ran 120+ steps without crash (killed by timeout, not error).

Per-block losses at step 120 (stable run):
- Block 0 (rate 1, next-token): 8.59 → 6.91
- Block 1 (rate 2, 2-ahead): 5.05 → 3.34
- Block 2 (rate 4, 4-ahead): 4.57 → 3.27

Script: `runs/assembled_architecture.py` (commit `7aee143`)

**~3 days remain in timebox. GPU: FREE.**

---

## ⚠️ PRIORITY: Separate optimizers + fixed embeddings (dictations 2026-05-27-8, 2026-05-27-9)

**Blocks are separate networks with separate optimizers.** Each block gets its own loss and its own optimizer. They are independent learners that communicate only via stale (detached) laterals.

**Fixed embeddings eliminate gradient coupling entirely.** The embedding table is initialized as random unit vectors and NOT learned. This makes blocks truly independent — no shared parameters at all. Learned embeddings are an optimization for later, once the architecture is proven.

### What needs to change

1. **Fixed embedding** — initialize as random unit vectors, `requires_grad=False`
2. **Separate optimizer per block** — each block's FFN + mix params get their own AdamW
3. **Separate backward per block** — one `block_loss.backward()` per block (no retain_graph needed since no shared params)
4. **No embedding optimizer** — embedding is fixed

### Why this simplifies everything

- No shared parameters between blocks → no gradient coupling → no design question about who trains what
- Each block is a fully independent learner that sees stale lateral info from neighbors
- The tied readout (dot product against fixed embeddings) still works — it just uses fixed reference vectors instead of learned ones
- Block independence is now architecturally enforced, not just hoped for via detach_lateral

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

### 1. Implement separate optimizers per block (dictation 2026-05-27-8) — HIGHEST PRIORITY

See details above. This is an architectural correction, not a new feature.

### 2. Re-run training with separate optimizers and check per-block learning

After fixing the optimizer coupling, run the assembly again. The key question remains: **does the higher block learn something different?**

### 3. Valid comparisons (per dictation 2026-05-27-6)

Once training is stable and architecturally correct:
- Noise on laterals → timescale separation?
- Global backprop vs local-only → blocks still learn differently?
- CUDA graph integration → expected speedup?

---

## Architecture (validated hyperparams)

- Shared normalized token embeddings (weight-tied readout)
- Normalized block outputs (L2-norm before readout)
- Addition-based lateral combination (lateral_scale=0.2)
- CE local loss for all blocks (cosine/L2 worse — tested earlier)
- Temperature = 0.07

---

## Key references

- `dictations/2026-05-27-2.md` through `2026-05-27-7.md` — the assembly directive and principles
- `core/model.py` — `ParallelDiagonalModel` + `ParallelDiagonalCarryState`
- `core/tied_readout.py` — weight-tied readout + local loss functions
- `runs/assembled_architecture.py` — the assembled training script
- `PROCESS.md` — experiment discipline (updated with valid-experiment criteria)
