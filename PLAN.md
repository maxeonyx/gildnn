# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-27, 13:30 NZST)

**Assembly sanity check: PASSED.** All 3 blocks' local losses decrease independently in 23 seconds. The architecture is not broken.

- Block 0 (rate 1, next-token): 8.54 → 8.20
- Block 1 (rate 2, 2-ahead): 5.19 → 3.65
- Block 2 (rate 4, 4-ahead): 4.80 → 3.57

Script: `runs/assembled_architecture.py` (commit `65f4118`)
Config: `ParallelDiagonalModel(num_blocks=3, rates=(1,2,4), topology="upward", detach_lateral=True)`, temperature=0.07, normalize=True, lateral_scale=0.2

**~3 days remain in timebox. GPU: FREE.**

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

### 1. Run the assembly for real (10-15 min)

The sanity check proved it doesn't explode. Now train longer and answer the KEY scientific question: **does the higher block learn something different?**

Metrics to watch:
- Per-block loss curves — do they diverge or converge?
- Representation similarity between blocks (cosine of hidden states)
- Does block 2 (rate 4) develop more abstract/smoother representations?

### 2. Valid comparisons (per dictation 2026-05-27-6)

Once the assembly is running well, these are the faithful A/B tests:
- Noise on laterals → timescale separation?
- Global backprop vs local-only → blocks still learn differently?
- CUDA graph integration → expected speedup?
- Single block + far-lower-rate observer → observer learns?

### 3. Remaining assembly pieces (if time permits)

- CUDA graphs for TBPTT (complex — GraphTrainer assumes fixed-window, TBPTT has variable control flow)
- Dynamic depth integration
- Noise injection

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
