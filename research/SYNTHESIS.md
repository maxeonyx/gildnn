# Project Synthesis

A two-week exploration of recurrent-over-depth architectures for language modeling — specifically whether GRU blocks iterated across time with various enhancements (local learning, attention, async execution, dynamic gating) could match transformer baselines. The bottom line: the core recurrent-over-depth architecture underperforms transformers by +0.071 nats at matched parameters, none of the proposed enhancements close that gap, and the async execution strategy that motivated much of the design cannot produce wall-clock speedup on a single GPU. The project's value is in definitively resolving these questions with clean evidence rather than leaving them as open speculation.

## Decision Matrix

| Idea | Result | Confidence | Implication |
|------|--------|------------|-------------|
| Local learning at block boundaries | NEGATIVE | High | val 2.081 vs 1.644 e2e — gradient isolation destroys quality catastrophically |
| Attention-residual (depth-only) | MARGINAL | High | transient 0.013 nat edge regresses; 33% slower — not worth pursuing |
| Causal triangle attention (depth+seq) | NEGATIVE | High | +0.028 nats, 1.8× slower — strictly dominated |
| Selective/dynamic computation | NEGATIVE | High | 26-29% slower wall-clock despite fewer block executions — overhead exceeds savings |
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
- **Dynamic/selective computation** fails because the savings from skipping a GRU block (~microseconds) are smaller than the cost of deciding whether to skip it.
- **Self-prediction** adds a training signal that competes with the language modeling objective rather than complementing it. The auxiliary loss pushes representations toward predictability rather than expressiveness.
- **The overall architecture** pays a 0.071 nat tax for temporal recurrence. At 900k parameters this gap is structural — the recurrent path doesn't carry enough information to justify the sequential dependency it introduces.

## The Async Question — Fully Resolved

This was the motivating hypothesis: if blocks can read stale state, they can execute in parallel across timesteps, hiding latency.

**Quality:** Confirmed viable. +0.005 nats is noise. (5 seeds, `experiments/async_gru_corpus/artifacts/multiseed_corpus_report.json`)

**Training wall-clock:** Negative across all configurations tested.
- d_model=64, 4 blocks: sequential 24.78ms, parallel 30.35ms, async 36.11ms
- d_model=512, 8 blocks: sequential 1453ms, parallel 1509ms, async 1505ms
- 9/12 configurations tested showed ≤1.00× speedup. Sequential always fastest.

**Inference wall-clock:** Worse still.
- 8 blocks, d_model=128: sequential 341ms, parallel 508ms, async 1022ms (0.50×)
- Stream overhead dominates when individual kernels are tiny (single-token, single-batch).

**Why it can't work on single GPU:** GRU kernels are memory-bandwidth-bound. Parallel streams on one GPU share the same memory bus. You don't get parallelism — you get contention plus scheduling overhead. This is architectural, not implementational.

## What Remains Genuinely Open

- **Multi-GPU async.** Physically separate memory systems would actually enable the parallel execution that single-GPU cannot. Untested — requires different hardware.
- **Larger scale.** The 0.071 nat gap at 900k params might narrow or widen at 10M+. No evidence either way.
- **Different tasks.** Temporal recurrence might matter more for tasks with longer-range dependencies than character-level LM on small data.
- **Muon + longer windows at scale.** The 27% gap reduction from k=4→k=8 suggests further gains from k=16+, but only Muon could stabilize it. Untested beyond k=8.

These aren't "promising leads." They're conditions under which the negative results *might* not generalize. The default expectation should be that they do.

## Recommendations

The core ideas are tested. The architecture underperforms transformers, and the execution model that would justify the performance cost (async parallelism) doesn't work on available hardware.

If continuing:
1. **Multi-GPU experiment** — the one condition where async might actually pay off. Requires at least 2 GPUs with independent memory. This is the only experiment that could change the fundamental picture.
2. **Scale the GRU speed advantage differently** — GRUs are 2-3× faster per block. Instead of recurrence-across-time, use them as cheap replaceable layers in a standard architecture. Less interesting theoretically, but the speed result is real.
3. **Drop everything else.** Local learning, attention augmentation, dynamic gating, self-prediction, broadcast — all definitively negative. No amount of tuning changes the mechanistic reasons they fail.

The honest answer: the project produced clear findings. The ideas were tested fairly and most didn't work. That's a complete result, not an incomplete one.
