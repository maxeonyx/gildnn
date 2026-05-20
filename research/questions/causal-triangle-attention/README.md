# Causal Triangle Attention Residual

## Question

Does extending the depth-only attention-residual path to a 2D causal memory over layer depth and prior sequence positions help a small TinyShakespeare character LM, compared with both the matched baseline transformer and the earlier depth-only attention-residual variant?

Relevant dictations: [`dictations/2026-05-20-9.md`](../../../dictations/2026-05-20-9.md), [`dictations/2026-05-20-5.md`](../../../dictations/2026-05-20-5.md), [`dictations/2026-05-20-10.md`](../../../dictations/2026-05-20-10.md).

## Locked simplification

- Residual blocks, not recurrent mini-stacks
- Shared `d_model` residual stream
- No weight sharing, stop-gradients, local losses, async execution, or graph routing
- Same baseline frame as the earlier attention-residual experiment: TinyShakespeare char LM, `ctx=32`, `d_model=72`, 3 blocks, matched parameter budget
- New mechanism: at layer `L`, position `T`, the readout can attend to any `(L', T')` where `T' < T`, or `T' == T` and `L' < L`
- Controls retained: external control (plain transformer) and depth-only attention-residual

## Comparison matrix

| Model | Blocks | `d_model` | Params | FFN width | New path |
|---|---:|---:|---:|---:|---|
| External control | 3 | 72 | 186,805 | 256 | none |
| Depth-only attention-residual | 3 | 72 | 186,946 | 159 | earlier layers, same token only |
| Causal-triangle attention-residual | 3 | 72 | 186,799 | 110 | earlier layers + prior timesteps under 2D causal mask |

## Hypotheses

1. The 2D residual-memory path is mechanically distinct from both the baseline transformer and the depth-only variant.
2. If access to prior timesteps' intermediate states is useful, the causal-triangle variant should beat the depth-only variant and baseline on the matched training frame.
3. If the extra history access mostly adds cost/noise at this scale, the causal-triangle variant should be slower and not improve validation loss.

## Planned evidence

- Stage 1 mask probe
- One-batch overfit traces
- Tiny-run histories and samples
- Full standardized comparison if time permits

## Non-goals

- Looped/shared blocks
- Any claim about stop-gradients, async execution, or local losses
- Any claim that this settles Max's broader looped-RNN direction

## Results

### Architecture

The 2D readout is inserted in the same place as the depth-only path: before ordinary causal sequence attention and before the FFN. The difference is the memory bank and its mask.

```python
allowed = (memory_positions < query_positions) | (
    (memory_positions == query_positions) & (memory_layer_indices < current_layer_index)
)
```

Source: `experiments/causal_triangle_attention/model.py`.

### Stage 1 — mechanical trust

Parameter counts from [`experiments/causal_triangle_attention/artifacts/stage1_mask_probe/mechanical_trust.json`](../../../experiments/causal_triangle_attention/artifacts/stage1_mask_probe/mechanical_trust.json):

| Model | Params | Delta vs baseline | Rough compute vs baseline |
|---|---:|---:|---:|
| External control | 186,805 | 0 | 1.0000 |
| Depth-only | 186,946 | +141 | 1.0000 |
| Causal triangle | 186,799 | -6 | 1.1435 |

Positive/negative probe for the 2D mask on block 2, query position 3:

```json
{
  "allowed_same_layer_past_timestep": 0.001070261001586914,
  "allowed_earlier_layer_same_timestep": 0.001018136739730835,
  "forbidden_same_layer_same_timestep": 0.0,
  "forbidden_same_layer_future_timestep": 0.0,
  "forbidden_earlier_layer_future_timestep": 0.0
}
```

The causal-triangle probe shows both required positives and negatives: an allowed past-timestep edit changes the query output, an allowed earlier-layer same-timestep edit changes the query output, and the forbidden same/future cells do not.

All three models also passed one-batch memorization:

| Model | Steps | Final loss |
|---|---:|---:|
| External control | 15 | 0.016186 |
| Depth-only | 14 | 0.014183 |
| Causal triangle | 14 | 0.014787 |

### Stage 3 — tiny rung

Tiny rung (`4096/1024`, 2 epochs):

| Model | Best val loss | Runtime (s) |
|---|---:|---:|
| External control | 2.586105 | 2.06 |
| Depth-only | 2.581598 | 2.73 |
| Causal triangle | **2.568915** | 3.98 |

Tiny fixed-prompt samples:

```text
external_control
First Citizen:
Before we proceed the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the

depth_only
First Citizen:
Before we proceed t t t t t t t the an:
[...]
Fin: t t t t t t t t t ane ane ane ane an:

causal_triangle
First Citizen:
Before we proceed the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
```

### Stage 4 — full standardized comparison

Full rung (`100K/20K`, 13 epochs):

| Model | Best val loss | Final val loss | Runtime (s) | Best epoch |
|---|---:|---:|---:|---:|
| External control | 1.638869 | 1.640817 | 83.96 | 11 |
| Depth-only | **1.618050** | **1.640417** | 116.41 | 12 |
| Causal triangle | 1.666854 | 1.696673 | 152.25 | 8 |

Margins vs external control best val loss:

```text
depth_only      -0.020819
causal_triangle +0.027985
```

Selected training-curve rows from `training_history.json`:

| Epoch | External control | Depth-only | Causal triangle |
|---|---:|---:|---:|
| 0 | 4.264182 | 4.253161 | 4.270424 |
| 1 | 1.945497 | 1.957869 | 1.926124 |
| 5 | 1.673320 | 1.698953 | 1.694844 |
| 8 | 1.644734 | 1.651312 | **1.666854** |
| 11 | **1.638869** | 1.636701 | 1.722525 |
| 12 | 1.642318 | **1.618050** | 1.707038 |
| 13 | 1.640817 | 1.640417 | 1.696673 |

Full fixed-prompt samples:

```text
external_control
First Citizen:
Before we proceed the people the people,
And the people the people.

CORIOLANUS:
What the people!

depth_only
First Citizen:
Before we proceed the people, and the people,
And the people the people.

CORIOLANUS:
The people, and the people, and the people,

causal_triangle
First Citizen:
Before we proceed the consul.

COMINIUS:
I will do you are the conster their be the can they and
The charge the consul, and the cannot the cannot
```

## Conclusion

Negative at this scale.

- The 2D causal-triangle mechanism is mechanically correct.
- It passed the required positive/negative mask probe.
- It memorized one batch and trained stably through the tiny and full rungs.
- It did not improve the matched full comparison. Its best validation loss (`1.666854`) is worse than both the baseline (`1.638869`) and the earlier depth-only variant (`1.618050`).
- It is also materially slower: about `1.81x` baseline runtime and about `1.31x` depth-only runtime.

This experiment supports: the 2D history readout is implementable and testable in the simplified transformer frame.

This experiment does not support: the claim that this specific 2D residual-history access is useful on the matched TinyShakespeare comparison frame.

## Artifacts

- Mechanical trust: `experiments/causal_triangle_attention/artifacts/stage1_mask_probe/mechanical_trust.json`
- Comparison summary: `experiments/causal_triangle_attention/artifacts/comparison_summary.json`
- Overfit rung: `experiments/causal_triangle_attention/artifacts/overfit/`
- Tiny rung: `experiments/causal_triangle_attention/artifacts/tiny/`
- Full rung: `experiments/causal_triangle_attention/artifacts/full/`
