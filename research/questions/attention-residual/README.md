# Attention-Residual Transformer

## Question

Does adding a content-based read over earlier residual-stream boundary states for the same token help a small character-level transformer on the fixed TinyShakespeare frame, compared with both a standard transformer and a same-memory token-independent mix control?

Relevant dictations: [`dictations/2026-05-20-10.md`](../../../dictations/2026-05-20-10.md), [`dictations/2026-05-20-11.md`](../../../dictations/2026-05-20-11.md).

## Locked simplification

- Residual blocks, not recurrent mini-stacks
- Shared `d_model` residual stream
- No weight sharing, stop-gradients, local losses, async execution, or graph routing
- Primary tested mechanism: depth-only attention over earlier boundary states `h_t^0 .. h_t^{l-2}` for the same token, inserted before ordinary sequence attention and FFN
- Internal control: same memory bank, but token-independent learned depth mixing instead of content-based attention
- External control: ordinary transformer
- Parameter matching by shrinking FFN width in the variants

## Comparison matrix

| Model | Blocks | `d_model` | Param target | New path |
|---|---:|---:|---:|---|
| External control | 3 | 72 | ~186K | none |
| Depth-only attention-residual | 3 | 72 | ~186K | content-based depth attention |
| Internal control | 3 | 72 | ~186K | token-independent depth mix |

## Hypotheses

1. The depth-only model is mechanically distinct from a standard transformer because block `l` can read an explicit bank of earlier residual boundary states instead of only `h^{l-1}`.
2. If content-based depth selection matters, the depth-only model should beat the internal control at matched budget.
3. If residual-history exposure alone is enough, the internal control should match the depth-only model.
4. If the new path is not worth the extra complexity, both new variants should be similar to or worse than the external control.

## Planned evidence

- Exact parameter counts
- Shape / gradient / causal-mask / depth-memory trust checks
- One-batch overfit traces
- Tiny-run histories and samples
- Full standardized training histories, runtimes, best/final val loss, and fixed-prompt samples

## Non-goals

- Variant B (depth+sequence / 2D memory)
- Looped/shared blocks
- Any claim about stop-gradients, local losses, async execution, or broader column-style routing

## Results

### Mechanical trust

Parameter counts on the standardized `100K/20K`, `ctx=32`, `d_model=72`, 3-block frame:

| Model | FFN width | Params | Delta vs baseline |
|---|---:|---:|---:|
| External control | 256 | 186,805 | 0 |
| Depth-only attention-residual | 159 | 186,946 | +141 |
| Internal control | 231 | 186,733 | -72 |

`experiments/attention_residual/artifacts/stage3_checks/mechanical_trust.json`:

```json
{
  "rough_compute": [
    {"label": "external_control", "relative_to_external_control": 1.0},
    {"label": "depth_only", "relative_to_external_control": 1.0},
    {"label": "internal_control", "relative_to_external_control": 0.9988425925925926}
  ]
}
```

The locked access pattern was actually implemented:

- external control: depth path stays exactly zero at all blocks
- depth-only: block 2 reads one earlier boundary state, block 3 reads two; perturbing the same token's stored boundary state changes only that token's depth update
- internal control: same memory bank access pattern, but with token-independent learned mixing weights

Inline evidence from `mechanical_trust.json`:

```json
{
  "depth_only_block_3": {
    "memory_length": 2,
    "attention_preview": [
      [0.5676719546318054, 0.4323280453681946],
      [0.5480096936225891, 0.4519903063774109],
      [0.44423386454582214, 0.5557661652565002],
      [0.47653186321258545, 0.5234681367874146]
    ],
    "same_token_position_change": [0.005744993686676025, 0.005025189369916916],
    "unaffected_position_change": 0.0,
    "other_token_position_change": 0.0
  },
  "internal_control_block_3": {
    "memory_length": 2,
    "mix_weights": [0.5, 0.5],
    "same_token_position_change": [0.0052030086517333984, 0.0046569108963012695],
    "unaffected_position_change": 0.0,
    "other_token_position_change": 0.0
  },
  "causal_mask": {
    "earlier_positions_max_abs_diff": 0.0
  }
}
```

All three models also passed one-batch memorization:

| Model | Steps to hit `loss < 0.02`, `acc = 1.0` |
|---|---:|
| External control | 15 |
| Depth-only attention-residual | 14 |
| Internal control | 14 |

### Ladder

Tiny-run (`4096/1024`, 2 epochs) comparison:

| Model | Best val loss | Runtime (s) |
|---|---:|---:|
| External control | 2.586129 | 2.30 |
| Depth-only attention-residual | 2.581604 | 3.01 |
| Internal control | 2.578392 | 2.27 |

All three learned something on the tiny rung, but all three samples were still highly repetitive. Example fixed-prompt outputs:

```text
external_control
First Citizen:
Before we proceed the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the

depth_only
First Citizen:
Before we proceed t t t t t t t the an:
[...]
Fin: t t t t t t t t t ane ane ane ane an:

internal_control
First Citizen:
Before we proceed t t t t t t t t t t t t t t t the anoure t t t t t t t t t t t t t t t anoure [...]
```

### Standardized comparison

Full run (`100K/20K`, 13 epochs, `ctx=32`, `d_model=72`, 3 blocks, fixed prompt `First Citizen:\nBefore we proceed`):

| Model | Params | Best val loss | Final val loss | Runtime (s) |
|---|---:|---:|---:|---:|
| External control | 186,805 | 1.643539 | 1.654424 | 84.60 |
| Depth-only attention-residual | 186,946 | **1.631051** | 1.657420 | 112.42 |
| Internal control | 186,733 | 1.645569 | **1.645569** | 88.46 |

Best-loss margin vs external control:

```text
depth_only      -0.012488
internal_control +0.002030
```

Abbreviated training curves from `experiments/attention_residual/artifacts/full/*/training_history.json`:

| Epoch | External control | Depth-only | Internal control |
|---|---:|---:|---:|
| 0 | 4.264182 | 4.253161 | 4.251589 |
| 1 | 1.947030 | 1.960112 | 1.946136 |
| 5 | 1.671214 | 1.691712 | 1.701630 |
| 9 | 1.648534 | 1.659250 | 1.649552 |
| 11 | 1.662453 | **1.631051** | 1.647083 |
| 13 | 1.654424 | 1.657420 | **1.645569** |

Fixed-prompt samples:

```text
external_control
First Citizen:
Before we proceed thee!

Citizens:
The consul the people, and the people,
And their be the people.

CORIOLANUS:
When they have been the people and
The people the people.

depth_only
First Citizen:
Before we proceed the people and the people,
And the people and the people, and the people,
And the people and the people, and the people,
And the people and the people, and the people,

internal_control
First Citizen:
Before we proceed the people, and they are they do the people, and the people, and the common the people, and the people the people, and the common the people, and the people [...]
```

## Conclusion

### Result

Marginal / inconclusive. The depth-only attention-residual variant is mechanically distinct from a standard transformer, but at this budget and frame it shows no meaningful language-modeling improvement.

### Key evidence

- The best validation-loss edge is only `0.012488` nats (`1.643539 -> 1.631051`), which is small enough to treat as single-seed noise until shown otherwise.
- The advantage is transient: the depth-only model peaks at epoch 11, then regresses to a final validation loss of `1.657420`, worse than the baseline's `1.654424`.
- The internal control, which exposes the same residual-history bank without content-based selection, stays essentially baseline-like (`best 1.645569`, `final 1.645569`), so content-based depth selection adds no clear value here.
- The depth-only model costs about `33%` more runtime (`112.42s` vs `84.60s`) for no stable quality gain.

### What this settles

At roughly `186K` parameters on TinyShakespeare character-level LM, content-based attention over earlier residual-stream boundary states does not meaningfully help quality. The mechanism works mechanically, but it does not earn its compute cost on this comparison frame.

### What this does not settle

- Whether the 2D depth+sequence variant would behave differently
- Whether weight sharing / looped blocks would change the picture
- Whether a larger-scale regime would make the mechanism more worthwhile

Those remain open, but this result does not justify pursuing them from within the current depth-only line.

## Artifacts

- Mechanical trust: `experiments/attention_residual/artifacts/stage3_checks/mechanical_trust.json`
- Ladder summary: `experiments/attention_residual/artifacts/comparison_summary.json`
- Overfit rung: `experiments/attention_residual/artifacts/overfit/`
- Tiny rung: `experiments/attention_residual/artifacts/tiny/`
- Standardized comparison: `experiments/attention_residual/artifacts/full/`
