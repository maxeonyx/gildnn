# Project Synthesis

> **⚠️ STALE — DO NOT TRUST (written ~2026-05-21, project has since moved to a different architecture and scale)**
>
> This document is from the GRU-based recurrent phase. The project now uses `ParallelDiagonalModel` (feedforward blocks, not GRU), WikiText-103 at ctx=128 with ~3M params, and has new results that contradict several claims below:
> - "Local learning is catastrophic" → was tested on wrong architecture (no propagation delay); Pathway 3 is still alive
> - "Single-GPU async cannot produce speedup" → CUDA Graphs produced 28% speedup (not streams)
> - "Drop local learning" → Pathway 3 blocked pending C_old ablation, not abandoned
> - "Backend: JAX" → settled on PyTorch + torch.compile
>
> For current state, read **PLAN.md**. For current results, read **research/daily/** and **research/weekly/**. This file will be rewritten or deleted when the next weekly synthesis is written.

---

A two-week exploration of recurrent-over-depth architectures for language modeling — specifically whether GRU blocks iterated across time with various enhancements (local learning, attention, async execution, dynamic gating) could match transformer baselines. The core recurrent-over-depth architecture underperforms transformers by +0.071 nats at matched parameters, and CUDA-stream async execution cannot produce wall-clock speedup on single GPU.

**However:** a subsequent experiment found that **fixed multi-rate execution** (blocks on predetermined schedules, skipping computation entirely on inactive steps) produces 12-15% wall-clock speedup with BETTER quality than all-blocks-every-step. This changes the picture — the speedup path is not parallelism but computation skipping.

## Decision Matrix

| Idea | Result | Confidence | Implication |
|------|--------|------------|-------------|
| Local learning at block boundaries | NEGATIVE | High | val 2.081 vs 1.644 e2e — gradient isolation destroys quality catastrophically |
| Attention-residual (depth-only) | MARGINAL | High | transient 0.013 nat edge regresses; 33% slower — not worth pursuing |
| Causal triangle attention (depth+seq) | NEGATIVE | High | +0.028 nats, 1.8× slower — strictly dominated |
| Selective/dynamic computation | NEGATIVE | High | 26-29% slower wall-clock despite fewer block executions — learned gating overhead exceeds savings |
| **Fixed multi-rate execution** | **POSITIVE** | **High** | **12-15% speedup, BETTER quality — fixed schedule eliminates gating overhead entirely** |
| Async stale-read quality | POSITIVE | Medium | +0.005 nats, not significant across 5 seeds — stale reads don't hurt |
| Async wall-clock (training) | NEGATIVE | High | always ≤1.00×, sequential fastest across all configs |
| Async wall-clock (inference) | NEGATIVE | High | 0.50-0.86× (worse); stream overhead dominates |
| GRU speed | POSITIVE | High | 1.8-2.8× faster training, 2.5-3.7× faster decode vs transformer blocks |
| Broadcast channel | NEGATIVE | High | +0.005 worse than plain async — adds noise, not signal |
| Self-prediction / compute compression | NEGATIVE | High | +0.011 to +0.018 nats at every depth tested |
| Residual-stream-across-time (overall) | QUALIFIED NEGATIVE | High | +0.071 nats at 900k params, +0.011 MSE images, 18× slower |
| Muon / orthogonal stability | POSITIVE | High | enables k=8 window where AdamW diverges; 27% gap reduction |
| Partial detach | NEUTRAL | High | zero quality cost, zero speed benefit at 900k scale |
| Orthogonal parameterization (matrix exp) | MIXED | Medium | stability confirmed, quality -0.366 nats, 30-50× runtime cost |
| Image patches (MNIST) | NEGATIVE | High | MSE 0.061 vs 0.050 baseline, 18× slower |

## What the Project Established

- **Stale reads are quality-neutral.** Blocks reading one-timestep-old states from other blocks costs +0.005 nats, indistinguishable from noise at 5 seeds. The async *semantics* are viable; the async *execution* is not.
- **GRU blocks are genuinely fast.** 1.8-2.8× training speedup, 2.5-3.7× decode speedup over transformer blocks. Raw arithmetic is cheap.
- **Muon optimizer enables long temporal windows.** k=8 truncation window works stably where AdamW diverges. 27% gap reduction vs short windows.
- **Local learning is catastrophic for quality.** Not marginal — the 0.44 nat gap (2.081 vs 1.644) rules out gradient-isolated block training.
- **Single-GPU async parallelism cannot produce speedup.** Not "doesn't yet" — fundamentally cannot, given CUDA stream scheduling and kernel launch overhead relative to GRU arithmetic intensity.

## Key Mechanisms Understood

**The orthogonality story.** Three independent lines of evidence converge: Muon optimizer (implicitly orthogonal updates) enables long windows; explicit matrix-exp parameterization confirms stability but at 30-50× cost; partial detach shows the gradient magnitude problem is already solved at this scale. The recurrent stability problem is real but solvable — it's just not the binding constraint.

**The hardware story.** GRU blocks have low arithmetic intensity — they're memory-bound, not compute-bound. CUDA streams parallelize compute, but when kernels are memory-bound, streams compete for the same bandwidth. Tensor Cores sit idle. This is why "run blocks in parallel" produces overhead (stream synchronization, kernel launch) without offsetting gains. The selective-gating result confirms it from another angle: skipping cheap blocks saves almost no time while the gating logic adds real cost.

**The stale-read story.** Quality tolerance for stale information is high (+0.005 nats). The architecture *could* tolerate async execution if the hardware supported it. But "the hardware" here means multiple independent memory systems — i.e., multiple GPUs or custom silicon, not CUDA streams on one device.

## What Didn't Work and Why

- **Attention mechanisms** add parameters and FLOPs but the recurrent state already carries sufficient inter-block information. The attention is solving a problem that doesn't exist at this scale.
- **Dynamic/selective computation** fails because the savings from skipping a GRU block (~microseconds) are smaller than the cost of deciding whether to skip it. *(Note: fixed-schedule multi-rate avoids this by eliminating the decision entirely.)*
- **Self-prediction** adds a training signal that competes with the language modeling objective rather than complementing it. The auxiliary loss pushes representations toward predictability rather than expressiveness.
- **The overall architecture** pays a 0.071 nat tax for temporal recurrence. At 900k parameters this gap is structural — the recurrent path doesn't carry enough information to justify the sequential dependency it introduces.

## The Async Question — Partially Resolved

The CUDA-stream parallelism hypothesis is fully resolved (negative). But a different path to speedup — **fixed multi-rate execution** — is positive. See `research/questions/fixed-multi-rate/README.md`.

**CUDA-stream parallelism:** Confirmed impossible on single GPU. GRU kernels are memory-bandwidth-bound; streams share bandwidth.

**Fixed multi-rate (new):** Blocks on predetermined schedules (rate 1, 2, 4, 8) reuse cached output on inactive steps. No gating, no decisions at runtime. Results:
- Rates [1,1,2,4]: 12-15% wall-clock speedup, quality BETTER (acts as regularizer)
- Rates [1,2,4,8]: testing in progress (17.8% speedup at step 0, expected >20% once trained)

**Why multi-rate works where selective execution failed:** Selective execution uses learned gating — the gating logic itself costs more than the block it skips. Multi-rate has ZERO decision overhead (the schedule is a compile-time constant). The full compute reduction translates directly to speed.

## What Remains Genuinely Open

- **Multi-rate scaling.** Does speedup increase with more aggressive rates? How far can you push it before quality degrades? Active exploration.
- **Diagonal + multi-rate.** Adding explicit time-offset connections between blocks may further improve quality. Question doc at `research/questions/diagonal-multi-rate/README.md`.
- **Larger scale.** The 0.071 nat gap at 900k params might narrow or widen at 10M+. Multi-rate's regularization benefit might also scale differently.
- **Different tasks.** Temporal recurrence might matter more for tasks with longer-range dependencies.
- **Muon + longer windows at scale.** The 27% gap reduction from k=4→k=8 suggests further gains from k=16+.
- **Backend/compilation.** JAX or torch.compile for long-lived reusable code.

## Recommendations

The CUDA-stream async path is dead. The multi-rate path is alive and promising.

If continuing:
1. **Scale multi-rate aggressively** — push rates, add blocks, find the quality/speed frontier. This is the active direction.
2. **Combine diagonal connections with multi-rate** — test whether explicit time-offset signals between blocks improve quality.
3. **Backend migration** — move to JAX or compiled torch for clean, fast, reusable code.
4. **Drop:** local learning, attention augmentation, dynamic gating, self-prediction, broadcast — all definitively negative.
