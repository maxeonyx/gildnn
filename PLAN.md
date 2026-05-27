# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-27, 15:57 NZST)

**Architecture correctness: DONE.** Fixed embeddings + separate optimizers (commit `6250ccd`).
**Stability fix: DONE.** Per-timestep L2 normalization (commit `7aee143`).
**Temperature investigation: DONE.** τ=0.07 is correct — τ=0.5 worsens learning. See note below.
**Daily report: WRITTEN.** `research/daily/2026-05-27.md`

**~3 days remain in timebox. GPU: BUSY (global-backprop run, PID 924).**

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

### 1. ⏳ Local-vs-global A/B test — IN PROGRESS (PID 924, started 15:56 NZST)

Global-backprop condition (B) running. Same as local-only but with `--global-backprop` (detach_lateral=False).

**Local-only results (A, already have, 200 steps τ=0.07):**
- Block 0: 5.19 → 3.85
- Block 1: 5.14 → 3.29
- Block 2: 5.09 → 3.27

**When B finishes:** Compare per-block losses. If gap < 0.1 nats → local learning works. If gap > 0.3 → locality costs something real. Cross-horizon matrix will show specialization.

Log file: `experiments/tinyshakespeare/artifacts/assembled_architecture/run.jsonl` (look for the latest `run_restarted` marker, then the `global_backprop: true` entry)

### 2. Temperature comparison data (completed)

τ=0.5 run (200 steps): Block 0→3.87, Block 1→3.75, Block 2→3.74. All worse than τ=0.07. Cross-horizon showed NO specialization (all rows flat). Confirms τ=0.07 is better — sharper gradients drive faster/deeper learning.

### 3. After A/B: noise on laterals

Once local-vs-global is settled: add noise injection to laterals and check if it promotes timescale separation. Single change from the working assembly.

### 4. Further valid comparisons (per dictation 2026-05-27-6)

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
