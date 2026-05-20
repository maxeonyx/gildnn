# Async GRU scale-up

Serves the wider-module async-execution question in [VISION.md](../../../VISION.md), continuing the earlier [async GRU rung](../async-gru/README.md) and the residual-block clarification in [dictations/2026-05-20-10.md](../../../dictations/2026-05-20-10.md).

## Question

Does the async quality gap shrink when modules are wider?

## Answer

Yes. At 11.12M parameters (`d_model=512`), the best-validation gap is indistinguishable from seed noise on this TinyShakespeare rung: async-minus-sync best-val gap was `mean = -0.003401`, `std = 0.008603` across 5 seeds, versus the earlier 1M rung where async was worse by `+0.005604` on the matched best-val comparison.

This does not show a robust async win. It shows that the small-scale quality penalty no longer appears cleanly at this width.

## Setup

- `d_model = 512`
- `num_modules = 6`
- `num_ticks = 4`
- sync read lags: `0, 0, 0, 0, 0, 0`
- async read lags: `0, 1, 2, 3, 4, 5`
- batch size: `256`
- learning rate: `0.0015`
- dataset: TinyShakespeare fixed slice, `100k` train / `20k` val
- training schedule: max `2000` steps, eval every `250`, early stop after `3` non-improving evals
- matching rules: same seed, same initial weights, same batch ordering, only `read_lags` changed

Primary artifacts:

- [mechanics](../../../experiments/async_gru_scaleup/artifacts/mechanics_report.json)
- [overfit](../../../experiments/async_gru_scaleup/artifacts/overfit_report.json)
- [timing](../../../experiments/async_gru_scaleup/artifacts/timing_report.json)
- [single-seed screen](../../../experiments/async_gru_scaleup/artifacts/screen_training_report.json)
- [5-seed screen](../../../experiments/async_gru_scaleup/artifacts/multiseed_report.json)

## Stage-2 correctness checks

Source: [mechanics](../../../experiments/async_gru_scaleup/artifacts/mechanics_report.json), [overfit](../../../experiments/async_gru_scaleup/artifacts/overfit_report.json)

```text
parameter_count = 11,117,629
zero-lag equivalence: max_history_abs_diff = 0.0, max_logits_abs_diff = 0.0
stale-read witness: tick 1, read_version_ids = [1, 0, 0, 0, 0, 0], max_gap = 0.25927
one-batch memorization:
  sync  : memorized = true, hit_step = 36, final_loss = 1e-06, final_accuracy = 1.0
  async : memorized = true, hit_step = 17, final_loss = 4e-06, final_accuracy = 1.0
```

So the 11M rung still satisfies the same mechanical checks as the 1M rung before trusting the quality comparison.

## Results

### Per-seed best-validation gaps

Source: [multiseed report](../../../experiments/async_gru_scaleup/artifacts/multiseed_report.json)

| seed | sync best val | async best val | gap (async - sync) |
| --- | ---: | ---: | ---: |
| 42 | 1.679603 | 1.673056 | -0.006547 |
| 123 | 1.714973 | 1.722035 | +0.007062 |
| 456 | 1.690610 | 1.682577 | -0.008033 |
| 789 | 1.694887 | 1.698850 | +0.003963 |
| 1337 | 1.711372 | 1.697921 | -0.013451 |
| mean |  |  | -0.003401 |
| std |  |  | 0.008603 |

Three seeds favored async, two favored sync, and the spread across seeds was larger than the mean effect.

### Single-seed training curves used for the initial screen

Source: [single-seed screen](../../../experiments/async_gru_scaleup/artifacts/screen_training_report.json)

```text
seed 42

step | sync_val | async_val
 250 | 1.829619 | 1.812072
 500 | 1.746760 | 1.743276
 750 | 1.678454 | 1.675736
1000 | 1.676129 | 1.684057
1250 | 1.704146 | 1.739824
1500 | 1.755496 | 1.750164
1750 | 1.806962 |    n/a

sync best  = 1.676129 @ step 1000
async best = 1.675736 @ step 750
gap        = -0.000393
```

That first screen suggested near parity. The 5-seed follow-up kept the same basic conclusion, but made clear that the effect size lives inside seed-sized noise.

### Throughput

Source: [11M timing](../../../experiments/async_gru_scaleup/artifacts/timing_report.json), [1M timing](../../../experiments/async_gru/artifacts/timing_report.json)

```text
11M rung:
  sync  : 237.962 ms/step, 34,425.7 tok/s
  async : 241.696 ms/step, 33,893.8 tok/s
  delta : +3.734 ms/step (+1.569%) for async

1M rung:
  sync  : 78.466 ms/step, 104,401.6 tok/s
  async : 75.845 ms/step, 108,009.2 tok/s
  delta : -2.621 ms/step (-3.34%) for async
```

On wall-clock alone, the wider rung did not show clear utilization improvement. Async was slightly slower at 11M, which is a small reversal from the 1M timing rung.

## Caveats

- Dataset saturation: `100k` training characters is tiny for an 11M-parameter model, and both variants typically hit best val around step `750-1000`.
- No profiler evidence: the utilization claim here is only wall-clock timing, not kernel-level evidence.
- Near-zero gap does not prove async is free. It only says the quality cost is small enough to be hard to separate from seed noise on this rung.

## Implications

The async mechanism looks viable at this scale. The earlier small-scale quality cost does not survive cleanly once the modules are wide enough, at least on this matched TinyShakespeare screen.

What this settles: the 1M async penalty is not a stable property of this mechanism family.

What this does not settle: whether larger-scale or less-saturated tasks would reveal a consistent async cost again, and whether wider GEMMs really improve GPU fit in a profiler-visible way.
