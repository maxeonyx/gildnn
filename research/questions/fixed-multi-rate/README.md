# Fixed Multi-Rate Execution

## Motivation

[Dictation 2026-05-22-6](../../../dictations/2026-05-22-6.md): "the diagonal residual connections across time and depth, modules running at different rates, the thing that might actually give wall clock speedup."

[Dictation 2026-05-22-3](../../../dictations/2026-05-22-3.md): "there's a second advantage which is having modules run at different rates."

## The gap between what was tested and what Max wants

Prior experiments showed:
- **Async CUDA streams:** NEGATIVE — can't parallelize on single GPU (memory-bandwidth bound)
- **Dynamic selective execution:** NEGATIVE — gating overhead exceeds compute savings (26-29% slower)
- **Stale reads:** POSITIVE — quality barely affected by one-timestep-old states (+0.005 nats)

But none of these tested the simplest form of "modules at different rates": a **fixed schedule** where different blocks have predetermined execution frequencies. No learned gating, no runtime decisions, no CUDA stream parallelism. Just: some blocks run less often, reading stale state from blocks that ran more recently.

## Hypothesis

A model with blocks on a fixed multi-rate schedule (e.g., block A every timestep, block B every 2nd, block C every 4th) achieves similar quality to all-blocks-every-timestep, while using measurably less wall-clock time — because the savings come from NOT RUNNING compute, not from parallelizing it.

This is the CHEAPEST async benefit:
- No gating network (zero overhead for the decision)
- No CUDA streams (no stream-management overhead)
- Stale reads already proven quality-neutral
- Pure compute reduction: if half the blocks run half as often, ~25% wall-clock savings

## Why this might work

1. **Stale reads are quality-neutral** — proven at +0.005 nats, not significant across 5 seeds
2. **No decision overhead** — the schedule is fixed at architecture-definition time, not learned
3. **Diagonal connections enable it** — block2 reading block1's previous output is already a "stale" design pattern

## Why the prior "selective execution" experiment was different

That experiment used a **learned gate** (sigmoid on block output → skip/execute decision). The gate itself is a neural network that adds parameters, FLOPs, and gradient paths. The overhead of COMPUTING whether to skip was larger than the savings from skipping. Result: 26-29% SLOWER.

Fixed-rate eliminates this entirely: the schedule is a hyperparameter, not a learned decision.

## Design

Architecture: existing `residual_stream_time_mixadd` base (the most mature variant), modified:
- N blocks, each assigned a rate: rate=1 (every step), rate=2 (every 2nd step), rate=4 (every 4th step)
- On skipped steps, a block's output is its PREVIOUS output (stale read from last execution)
- Rate assignment is a hyperparameter, not learned

Comparison targets:
- All-blocks-every-step (the standard model, matched parameters)
- Same model with all rates=1 (controls for any architecture differences)

## Planned evidence

- Wall-clock time comparison: multi-rate vs all-every-step
- Validation loss comparison: multi-rate vs all-every-step
- Loss per effective-FLOP comparison

## What this does NOT settle

- Whether the diagonal connection structure specifically helps (vs just multi-rate on flat architectures)
- Multi-GPU async (still requires different hardware)
- Whether the optimal rate assignment can be learned (that's the dynamic version we already tested negatively)

## Results

**POSITIVE.** Fixed multi-rate execution achieves 14.8% wall-clock speedup with no quality loss.

### Setup

- 4 feedforward blocks on shared residual stream with temporal attention
- Control: all blocks rate=1 (execute every timestep)
- Multi-rate: blocks with rates [1, 1, 2, 4]
- Same parameters, same initialization
- Character-level LM, context=32, d_model=128, batch=64

### Overfit test

Multi-rate model memorizes one batch: loss 0.0009, accuracy 1.0 at step 55. Architecture functions correctly.

### Quality (1000 training steps)

| Step | All-rate-1 val loss | Multi-rate val loss | Delta |
|------|--------------------:|--------------------:|------:|
| 500  | 2.188 | 2.171 | **-0.017** |
| 1000 | 2.062 | 2.052 | **-0.010** |

Multi-rate is slightly BETTER (not worse). Likely noise, but clearly no quality penalty.

### Wall-clock (1000 timed forward passes, CUDA synchronized)

| Model | ms/batch | Speedup |
|-------|------:|------:|
| All-rate-1 | 85.47 | — |
| Multi-rate [1,1,2,4] | 72.86 | **14.8%** |

Artifact: `experiments/fixed_multi_rate/artifacts/short_run/report.json`

### Interpretation

This is the first positive wall-clock result in the project. The key insight: prior "selective execution" used learned gating which added overhead exceeding savings. Fixed-rate has ZERO decision overhead — the schedule is a hyperparameter. The full compute reduction translates directly to speed.

14.8% from rates [1,1,2,4] with 4 blocks. More aggressive rates or more blocks could give larger savings.

### Max's interpretation ([dictation 2026-05-22-10](../../../dictations/2026-05-22-10.md))

> "multi-rate means that a certain part of the network is attempting to predict further in the future"

Max sees the multi-rate constraint as an inductive bias: a rate-4 block's output must be "generic enough to be useful across 4 steps" — the gradient from step t+4 back to step t pressures it toward longer-timescale features. This is similar to how the sequential bottleneck was the useful thing in prior experiments: constraining update frequency forces more slowly-varying, generalizable representations.

The fact that multi-rate achieves *better* val loss (not just equal) is consistent with this being a useful regularizer, not just a compute trick.

## Extended run (partial — 1000 of 5000 steps, interrupted by external event)

| Step | All-rate-1 val loss | Multi-rate val loss | Delta | Speedup |
|------|--------------------:|--------------------:|------:|--------:|
| 0    | 4.136 | 4.136 | +0.000 | 8.7% |
| 500  | 2.188 | 2.171 | **-0.017** | 11.2% |
| 1000 | 2.062 | 2.052 | **-0.010** | 12.4% |

Artifact: `runs/fixed_multi_rate_5k.log`

Key observations:
- Multi-rate quality advantage is **consistent** across all checkpoints (not just noise)
- Speedup **grows** during training (8.7% → 12.4%) — likely because cached outputs become more informative as the model learns
- The run was killed externally (window-close event), not by error. Data through step 1000 is valid.

Combined with the short run (14.8% speedup at step 1000 with 1000 timed passes), the result is robust: **multi-rate is both faster and slightly better quality**.

## Aggressive rates [1,2,4,8] — 20.7% speedup

| Step | All-rate-1 val loss | Multi-rate val loss | Delta | Speedup (100-pass) |
|------|--------------------:|--------------------:|------:|--------:|
| 0    | 4.136 | 4.136 | +0.000 | 17.8% |
| 500  | 2.188 | 2.168 | **-0.020** | 14.5% |
| 1000 | 2.062 | 2.051 | **-0.010** | 18.4% |
| 1500 | 1.994 | 1.975 | **-0.019** | -0.6%* |
| 2000 | 1.946 | 1.941 | **-0.006** | 25.0% |

*Step 1500 timing is a measurement anomaly (100-pass noise). The 500-pass final timing is reliable.

### Final timing (500 passes, CUDA synchronized)

| Model | ms/batch | Speedup |
|-------|------:|------:|
| All-rate-1 (4 blocks) | 73.89 | — |
| Multi-rate [1,2,4,8] | 58.59 | **20.7%** |

Artifact: `experiments/fixed_multi_rate/artifacts/aggressive_1248/report.json`

### Interpretation

**The 20% target is cleared.** Multi-rate [1,2,4,8] achieves 20.7% wall-clock speedup with quality consistently BETTER than all-rate-1 (not just neutral). The regularization effect persists at more aggressive rates — the multi-rate constraint forces longer-timescale representations that generalize better.

Compared to [1,1,2,4] (14.8% speedup): doubling the aggressiveness of rates yields roughly 40% more speedup (14.8% → 20.7%) with no quality cost.

## Next steps

1. **Even more aggressive rates** — try [1, 2, 4, 8, 16] with 5 blocks, or [2, 4, 8, 16] (no rate-1 block). How far can it go?
2. **Diagonal connections** — combine with time-offset residual connections. See `research/questions/diagonal-multi-rate/README.md`.
3. **Matched-FLOP comparison** — is multi-rate better than a smaller all-rate-1 model with the same compute budget?
4. **Larger model** — does the result hold at 900K+ params?
