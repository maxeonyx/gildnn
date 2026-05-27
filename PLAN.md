# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-27, 16:40 NZST)

**Architecture correctness: DONE.** Fixed embeddings + separate optimizers (commit `6250ccd`).
**Stability fix: DONE.** Per-timestep L2 normalization (commit `7aee143`).
**Temperature investigation: DONE.** τ=0.07 is correct — τ=0.5 worsens learning.
**Local-vs-global A/B: DONE.** Local is strictly better (block 0: 0.215 nats better, blocks 1&2 identical).
**Codebase cleanup: DONE.** 47K lines of dead code removed.
**Daily report: UPDATED.** `research/daily/2026-05-27.md`

**~3 days remain in timebox. GPU: FREE.**

---

## ✅ DONE: Local-only vs global-backprop A/B

| | Block 0 | Block 1 | Block 2 |
|---|---|---|---|
| Local (detach=True) | **3.851** | 3.286 | 3.269 |
| Global (detach=False) | 4.066 | 3.290 | 3.269 |

Local learning is strictly superior. Gradient through laterals interferes with block 0's own objective. Pathway 3 (Local Learning) validated.

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
| Local learning (detach laterals) | ✅ | A/B test — strictly better than global |
| CUDA graph execution | ❌ | Exists in `core/training.py` GraphTrainer, but not wired for TBPTT |
| Dynamic depth / halting | ❌ | Validated separately, not yet in assembly |
| Noise on laterals | ❌ | Not implemented |

---

## What's next

### 1. Noise on laterals (next experiment)

**Pathway connection:** Pathway 5 (Information Hierarchy) — noise creates pressure for blocks to develop genuinely different representations (timescale separation).

**Setup:** Same architecture, one change: add Gaussian noise to the lateral pathway. The noise forces higher blocks to not simply copy lower-block representations — they must extract signal that's robust to corruption.

**Question:** Does noise promote specialization (visible in cross-horizon matrix or representation similarity analysis)?

**Exit condition:** Cross-horizon matrix shows non-flat rows (blocks genuinely specialize for different horizons), OR clear evidence that noise at any tested level hurts convergence without benefit.

### 2. Further valid comparisons (per dictation 2026-05-27-6)

- CUDA graph integration → expected speedup?
- Dynamic depth / halting integration

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
- Detached laterals (proven superior to global backprop)

---

## Key references

- `dictations/2026-05-27-2.md` through `2026-05-27-9.md` — the assembly directive and principles
- `core/model.py` — `ParallelDiagonalModel` + `ParallelDiagonalCarryState`
- `core/tied_readout.py` — weight-tied readout + local loss functions
- `runs/assembled_architecture.py` — the assembled training script
- `PROCESS.md` — experiment discipline (updated with valid-experiment criteria)
