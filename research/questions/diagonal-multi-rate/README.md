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

Single-seed [1,2,4,8] comparison completed: multi-rate **with** diagonal vs the same multi-rate schedule **without** diagonal.

Artifact: `experiments/fixed_multi_rate/artifacts/diagonal_1248_vs_baseline/report.json`

### 2000-step run

| Step | Multi-rate no diagonal | Multi-rate + diagonal | Delta |
|------|-----------------------:|----------------------:|------:|
| 500  | 2.192 | 2.169 | **-0.022** |
| 1000 | 2.066 | 2.064 | **-0.002** |
| 1500 | 1.983 | 1.992 | **+0.009** |
| 2000 | 1.988 | 1.935 | **-0.052** |

This is **suggestive but not settled**. The final checkpoint is the strongest diagonal result seen so far (`-0.052` nats), and it came from a divergence in late training: the no-diagonal control worsened slightly from step 1500 to 2000 (`1.983 -> 1.988`) while the diagonal model continued improving (`1.992 -> 1.935`). That is consistent with the diagonal acting as a useful regularizer or stabilizer.

But the trajectory is non-monotonic: diagonal was slightly better at 500 and 1000 steps, worse at 1500, then much better at 2000. With one seed and one short run, that means we should treat the result as a promising lead rather than a confirmed effect.

## Multi-seed confirmation

Artifact: `experiments/fixed_multi_rate/artifacts/diagonal_1248_multi_seed/report.json`

| Seed | Control val_loss | Diagonal val_loss | Delta |
|------|-----------------:|------------------:|------:|
| 42 | 1.988 | 1.935 | -0.052 |
| 43 | 1.934 | 1.934 | +0.000 |
| 44 | 1.959 | 18.386 | +16.4 |

The single-seed `-0.052` result did not replicate. Across three seeds, the apparent gain was seed-specific rather than a reliable improvement.

More importantly, the mechanism is unstable as currently implemented. On seed 44, the diagonal model trained normally through step 1500 (`val_loss 2.044`) and then catastrophically diverged by step 2000 (`18.386`), while the no-diagonal control remained normal (`1.959`). That makes the naive diagonal path a clear negative overall, even though one seed looked good.

The most likely current interpretation is that the diagonal signal can be too strong or improperly scaled, so in some initializations it injects enough extra activation or gradient energy to destabilize training. That does not prove the idea is bad in principle, but it does mean this direct implementation is not reliable enough to treat as a win.

### Speed

The final 500-pass timing showed an 8.7% diagonal "speedup" (69.8ms -> 63.7ms), but both models use the same execution schedule `[1,2,4,8]`. The diagonal path only adds a cached tensor input on execute steps, so a real speedup of that size is unlikely. Most likely this timing gap is measurement noise or run-state noise, not a meaningful throughput gain.

### Current read

- **Quality:** **NEGATIVE as currently implemented. Unstable and unreliable.** The single-seed gain did not hold up across seeds.
- **Speed:** effectively neutral; the measured gap should not be trusted as a real diagonal benefit
- **Next:** if this idea is revisited, try a stabilized diagonal path rather than this naive additive version

## What this doesn't settle

- Whether diagonal helps at larger scale (small models may not benefit)
- Whether a scaled/gated diagonal (e.g. learnable mixing coefficient starting near zero) would be stable
- Whether the diagonal should cross MORE than one timestep (block A at t-2, t-3, etc.)
- How diagonal interacts with temporal attention (the attention mechanism already provides cross-time information)
