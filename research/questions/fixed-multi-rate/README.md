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

## Next steps

1. **Longer training run** (5000+ steps) to confirm the quality gap stays neutral at convergence
2. **More aggressive rates** — [1, 2, 4, 8] or more blocks with varied rates
3. **Scale up** — larger model, more data, check if the result holds
4. **Compare effective quality/compute frontier** — how does multi-rate compare to a smaller all-rate-1 model with matched wall-clock time?
