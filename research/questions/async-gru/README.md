# Async GRU on shared residual memory

Serves Thread 1 boundary-mechanism work and the GPU-utilization question in [VISION.md](../../../VISION.md) and the residual-block clarification in [dictations/2026-05-20-10.md](../../../dictations/2026-05-20-10.md).

## Question

Can the GRU block function from the earlier throughput work be dropped into the existing async shared-memory architecture without breaking convergence, and does the GRU speed advantage still matter once reads can be stale?

## Simplification

- Shared memory is the residual stream at width `d_model`
- Each module is one residual block implemented as `LayerNorm -> GRU -> Linear -> residual delta`
- All modules run every tick; only committed-memory visibility changes between variants
- Task is TinyShakespeare char-level next-token prediction with `context_size=32`

## Matched comparison rules

- Primary comparison is `synchronous_control` vs `async_stale_reads`
- Same initial weights
- Same `d_model`, module count, tick count, optimizer, seed, dataset split, and batch order
- Same parameter count by construction; only `read_lags` changes
- Zero-lag async must collapse exactly to sync before trusting stale-read runs

## Current experiment config

- `d_model = 152`
- `num_modules = 6`
- `num_ticks = 4`
- Target scale: about 1M parameters
- Sync read lags: all zero
- Async stale read lags: `0, 1, 2, 3, 4, 5`

## Hypotheses

- H1: zero-lag async is exactly equivalent to sync
- H2: stale committed reads do not break one-batch memorization
- H3: the async variant remains trainable enough to justify timing and full-training runs
- H4: the GRU-based block still looks materially faster than a transformer-based block in the same shared-memory frame

## Non-goals

- Not a claim about the final architecture for Thread 1
- Not a comparison against a plain transformer LM for quality
- Not a claim that any observed throughput difference will survive larger-scale tasks
- Not a test of selective updates, stop-gradient boundaries, or local predictive losses

## Planned evidence

- `experiments/async_gru/artifacts/mechanics_report.json`
- `experiments/async_gru/artifacts/overfit_report.json`
- `experiments/async_gru/artifacts/timing_report.json`
- `experiments/async_gru/artifacts/training_report.json`

## Results

### Mechanics

Source: `experiments/async_gru/artifacts/mechanics_report.json`

```text
parameter_count = 1,002,349
d_model = 152
num_modules = 6
num_ticks = 4
zero-lag equivalence: max_history_abs_diff = 0.0, max_logits_abs_diff = 0.0
stale-read witness: tick 1, read_version_ids = [1, 0, 0, 0, 0, 0], max_gap = 0.283326
```

### One-batch memorization

Source: `experiments/async_gru/artifacts/overfit_report.json`

```text
sync  : memorized = true, hit_step = 12, final_loss = 1.2e-05, final_accuracy = 1.0
async : memorized = true, hit_step =  9, final_loss = 1.3e-05, final_accuracy = 1.0
```

### Timing

Source: `experiments/async_gru/artifacts/timing_report.json`

```text
sync  : 78.466 ms/step, 104,401.6 tok/s
async : 75.845 ms/step, 108,009.2 tok/s
delta : -2.621 ms/step (-3.34%) for async in this rung
```

This rung does not show measurable async overhead. If anything, async came out slightly faster on this measurement.

### Full matched training

Source: `experiments/async_gru/artifacts/training_report.json`

```text
sync  : best val loss = 1.597662 (epoch 3), final val loss = 1.736813
async : best val loss = 1.603266 (epoch 4), final val loss = 1.787625
best-val delta  (async - sync) = 0.005604
final-val delta (async - sync) = 0.050812
```

Per-epoch val losses:

```text
epoch | sync_val | async_val
    1 | 1.762488 | 1.741779
    2 | 1.622636 | 1.619856
    3 | 1.597662 | 1.615498
    4 | 1.597924 | 1.603266
    5 | 1.603321 | 1.627696
    6 | 1.620984 | 1.616795
    7 | 1.639658 | 1.662912
    8 | 1.667086 | 1.682383
    9 | 1.660366 | 1.669019
   10 | 1.705833 | 1.726963
   11 | 1.708953 | 1.763734
   12 | 1.722970 | 1.764821
   13 | 1.736813 | 1.787625
```

Sample excerpts from the matched prompt `First Citizen:\nBefore we proceed`:

```text
sync:
First Citizen:
Before we proceed the city flattery the bears of the wars of the market-place,
And welcome to the people are the belly be loved to beg
The blood to curse the bellow the bellow the bellow...

async:
First Citizen:
Before we proceed to the matter? The begging the senate, whose great stand not the consul his belly,
Let the people and the controver to the people,
And the consine that the can on the people...
```

## Next steps

- Compare these numbers against the earlier plain GRU throughput anchor so the speed claim is tied back to the original GPU-utilization result
- If this architecture family remains interesting, repeat the matched run at one smaller and one larger scale to see whether the small best-val gap is stable or noise
- If quality matters more than raw speed here, investigate regularization or a milder stale-read schedule before changing the block function again

## Remaining open question

Does stale-read GRU training converge cleanly enough that the async mechanism still looks like a useful composition rather than a throughput-only curiosity?

Current evidence: yes for convergence and mechanics; likely yes for throughput on this rung; quality is slightly worse than sync but close at best epoch and somewhat worse by the final epoch.
