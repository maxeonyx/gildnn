# Async volatile memory

## Question

Can a tiny dense residual-width module bank demonstrate volatile shared-memory semantics honestly if the only changed variable between control and variant is memory visibility: latest committed reads vs stale committed reads?

Relevant dictations: [`2026-05-20-6`](../../../dictations/2026-05-20-6.md), [`2026-05-20-9`](../../../dictations/2026-05-20-9.md), [`2026-05-20-10`](../../../dictations/2026-05-20-10.md), [`2026-05-20-11`](../../../dictations/2026-05-20-11.md).

## Locked simplification

- Shared memory width is `d_model`
- Fixed dense module bank; every module executes every tick
- Fixed tick count; writes commit only at tick boundaries
- Control and async variant share weights, `d_model`, module count, tick count, initial memory, and write rule
- Only read visibility changes: control reads latest committed memory; async reads an older committed version by fixed module-level lag
- No per-token branching, no ragged batches, no training claim, no hardware-level async claim yet

## Stage 1 semantics

| Item | Choice |
|---|---|
| Shared state | committed memory versions `memory[v]` with shape `[batch, slots, d_model]` |
| Prototype size | `batch=1`, `slots=2`, `d_model=4`, `modules=3`, `ticks=4` |
| Module execution | all 3 modules execute on all 2 slots every tick |
| Module input | one visible committed memory snapshot per module |
| Module output | residual delta with shape `[batch, slots, d_model]` |
| Commit rule | `memory[t+1] = memory[t] + sum(module_deltas[t])` |
| Sync control reads | all modules read `memory[t]` |
| Async reads | module `m` reads `memory[max(0, t - lag[m])]` |
| Async lag schedule | `[0, 1, 2]` |
| Gradients | not part of this Stage 2 prototype |

## Planned evidence

- zero-staleness equivalence check
- stale-read witness trace
- write-rule check
- inline trace excerpt plus full JSON artifact in [`experiments/async_volatile_memory/artifacts/stage2/mechanical_trace.json`](../../../experiments/async_volatile_memory/artifacts/stage2/mechanical_trace.json)

## Non-goals

- training quality
- literal independent CUDA scheduling
- stop-gradient claims
- any conclusion about transformer-vs-RNN quality

## Results

### Mechanical trust

Artifact: [`mechanical_trace.json`](../../../experiments/async_volatile_memory/artifacts/stage2/mechanical_trace.json)

The locked comparison stayed narrow: 3 modules, `d_model=4`, 4 ticks, identical weights and write rule, with the only changed variable being committed-memory visibility.

| Check | Result | Inline evidence |
|---|---:|---|
| Zero-staleness equivalence | pass | max final diff `0.0`, max all-version diff `0.0` |
| Write rule | pass | max tensor diff `0.0` in both control and async variants |
| Stale-read witness | pass | tick `1` read versions `[1, 0, 0]`; stale modules saw max read-vs-latest diff `0.417959` |

Inline excerpt from `mechanical_trace.json`:

```json
{
  "zero_staleness_equivalence": {
    "max_final_memory_abs_diff": 0.0,
    "max_all_versions_abs_diff": 0.0
  },
  "stale_read_witness": {
    "witness_tick": 1,
    "latest_version_id": 1,
    "read_version_ids": [1, 0, 0],
    "unique_read_version_ids": [0, 1],
    "max_read_vs_latest_abs_diff": 0.417959
  }
}
```

The traced async run shows the intended semantics directly:

- tick 0: all modules read version 0
- tick 1: module 0 reads version 1, modules 1 and 2 still read version 0
- tick 2: modules read versions `[2, 1, 0]`
- tick 3: modules read versions `[3, 2, 1]`

So this is not ordinary synchronous depth. Within the same logical tick, different modules consume different committed memory versions while still executing densely over the same tensor shapes.

### What this does and does not show

Settled at Stage 2:

- dense stale-read shared-memory semantics can be implemented cleanly in a tiny PyTorch prototype
- the sync control and async variant can be matched except for memory visibility
- a concrete stale-read witness exists

Still open:

- whether this prototype path is a fair GPU vehicle rather than mostly software overhead
- whether stop-grad boundaries help or hurt
- whether any trained version is viable
- whether literal GPU-level desynchronisation is achievable

### Stage 3 GPU-friendliness check

Artifact: [`timing_report.json`](../../../experiments/async_volatile_memory/artifacts/stage3/timing_report.json)

Matched char-level probe on the standard `100K/20K`, `ctx=32` TinyShakespeare frame:

| Variant | Params | Mean train step (ms) | Delta vs sync |
|---|---:|---:|---:|
| Synchronous control | 158,965 | `46.851` | `0%` |
| Async stale reads | 158,965 | `44.736` | `-4.5%` |

Forward-time breakdown from `timing_report.json`:

```json
{
  "synchronous_control": {
    "module_compute_fraction_of_accounted": 0.92941,
    "untracked_fraction_of_wall": 0.183456
  },
  "async_stale_reads": {
    "module_compute_fraction_of_accounted": 0.938291,
    "untracked_fraction_of_wall": 0.09814
  }
}
```

So at this rung the async implementation looks like a fair vehicle for the idea: dense module compute dominates, bookkeeping stays small, and the variant did not reintroduce the previously-failed sparse execution pattern.

### Stage 4 overfit one batch

Artifact: [`overfit_report.json`](../../../experiments/async_volatile_memory/artifacts/stage4/overfit_report.json)

One-batch memorization on the same TinyShakespeare frame, same optimizer, same LR, same seed, same `d_model`, same module count, same ticks:

| Variant | Memorized? | Hit step | Final loss | Final accuracy |
|---|---:|---:|---:|---:|
| Synchronous control | yes | `18` | `0.000052` | `1.0` |
| Async stale reads | yes | `11` | `0.000036` | `1.0` |

Inline excerpt from `overfit_report.json`:

```json
{
  "synchronous_control": {
    "memorized": true,
    "hit_step": 18,
    "final_loss": 0.000052,
    "final_accuracy": 1.0
  },
  "async_stale_reads": {
    "memorized": true,
    "hit_step": 11,
    "final_loss": 0.000036,
    "final_accuracy": 1.0
  }
}
```

So the stale-read mechanism is not obviously breaking optimization at the first rung. On this fixed batch, both matched variants memorize cleanly.

### Stage 5 tiny matched training-and-timing rung

Artifact: [`tiny_training_report.json`](../../../experiments/async_volatile_memory/artifacts/stage5/tiny_training_report.json)

Tiny matched rung on the standard `100K/20K`, `ctx=32` TinyShakespeare frame, 12 epochs, same optimizer/LR/seed, same `d_model=72`, same `num_modules=3`, same `num_ticks=4`, same params `158,965`.

Best-vs-final comparison:

| Variant | Best epoch | Best val loss | Best val acc | Final train loss | Final val loss | Final val acc |
|---|---:|---:|---:|---:|---:|---:|
| Synchronous control | `9` | `1.656889` | `0.513021` | `1.348393` | `1.669227` | `0.509465` |
| Async stale reads | `6` | `1.658197` | `0.504758` | `1.362514` | `1.668071` | `0.506761` |

Epoch-time comparison:

| Variant | Typical epoch wall-clock |
|---|---:|
| Synchronous control | about `18.0–19.8 s` |
| Async stale reads | about `18.4–19.7 s` |

The training curves are close. The synchronous control reached the slightly better best validation loss and best validation accuracy, but the gap stayed small: about `0.0013` loss and `0.0083` accuracy at best epoch.

Final samples:

```text
synchronous_control
First Citizen:
Before we proceed his he had have
which his he had the had have him the company, the had have him
The capting the had have the have him the stal
A call his hing the people him.

MENENIUS:
What the common the consul, w

async_stale_reads
First Citizen:
Before we proceed to the people and the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people the people the p
```

At this tiny rung the async mechanism remains viable in the narrow sense that it trains, matches wall-clock closely, and stays near the synchronous control on validation metrics. But it is not yet a positive modeling result: the sync control is still slightly better, and both samples are still clearly weak.
