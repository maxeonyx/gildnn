# Diagonal Connections + Multi-Rate Execution

## Motivation

[Dictation 2026-05-22-6](../../../dictations/2026-05-22-6.md): "the diagonal residual connections across time and depth, modules running at different rates"

[VISION.md](../../../VISION.md): "The interesting connections are **diagonal** — from residual block A at time t to the next block at time t+1."

## The insight

The diagonal connection (block A's output at time t feeds block B at time t+1) and multi-rate execution (block B runs every Nth step, reusing cached output between executions) are the **same mechanism viewed from different angles**:

- **Diagonal:** information crosses time AND depth simultaneously. Block B always receives delayed information from block A.
- **Multi-rate:** block B only computes on some steps. On other steps, its cached output is reused — which is inherently "delayed" relative to the current state.

When block B runs at rate 2, the diagonal connection IS the stale-read: block B at time t=2 sees block A's output from t=1 (one step stale).

## Prior results

- **Multi-rate [1,1,2,4]:** 12-15% speedup, quality BETTER than baseline. The rate constraint acts as a regularizer.
- **Diagonal 2-block:** Inconclusive at small scale — val loss 1.788 at 777K params, identical to single-block at 481K. Doesn't hurt, doesn't help.

## Hypothesis

Adding explicit diagonal connections to a multi-rate architecture may improve quality — block B's execution on its active steps is better informed by an explicit fresh signal from the block above, rather than just its own cached state + stream.

The question: does the explicit diagonal signal give block B something it wouldn't get from the residual stream alone?

## What to build

Modify the `fixed_multi_rate` model to add diagonal connections:

- When block B **executes** at time t: it receives `block_A_output[t-1]` (the cached delta from block A at the previous timestep) as an additional input
- When block B **doesn't execute** (stale-read): its cached output is used as before

This is a small architectural change: each block stores its last delta, and the next block receives it when executing.

## Experiment plan

1. Test rates [1,2,4,8] WITH diagonal vs WITHOUT diagonal (current multi-rate baseline)
2. Compare quality and speed (diagonal adds no significant compute — it's just adding a cached tensor)
3. If quality improves: the diagonal is providing useful signal. If neutral: the stream already conveys the information.

## Status

_Pending — waiting for aggressive multi-rate [1,2,4,8] results first._

## What this doesn't settle

- Whether diagonal helps at larger scale (small models may not benefit)
- Whether the diagonal should cross MORE than one timestep (block A at t-2, t-3, etc.)
- How diagonal interacts with temporal attention (the attention mechanism already provides cross-time information)
