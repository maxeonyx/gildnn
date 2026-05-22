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

_Pending._
