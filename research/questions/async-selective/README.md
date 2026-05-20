# Async selective block execution

## Question

On TinyShakespeare char-level, can a transformer where upper blocks selectively skip some token positions preserve language-model quality while reducing effective computation in a way that actually runs faster on this GPU?

## Conclusion

Cut. The core premise is falsified for this setup.

The learned gates did work as a mechanism: they learned genuine token/block differentiation, did not collapse during the overfit rung, and reduced logical executed-block ratio by 15-32% on the tiny rung. But wall-clock got worse, not better. The tiny learned runs were slower than the synchronous control even though they executed fewer logical block updates.

So this settles the question I actually asked here: fine-grained per-token/per-block skipping is not a viable path to compute savings on the current GPU execution path. The gate overhead is real, and the implementation still executes dense batched kernels, so logical sparsity did not become hardware savings.

This does not settle all conditional compute. Coarser routing, expert-style batching, hardware-friendly sparse kernels, or genuinely different execution models could still work.

## Setup

- 3-layer causal transformer, `d_model=72`, context `32`, TinyShakespeare char-level
- Block 1 always executes
- Blocks 2 and 3 can skip by outputting zero residual delta on individual token positions
- Gate: `hard_sigmoid(Linear(RMSNorm(h)))` with straight-through estimator
- Learned targets tested on tiny rung: `r=0.75` and `r=0.50`
- Controls: synchronous transformer, forced-open learned-gate model, random-skip model (implemented but not promoted because the process cut at tiny)

Key artifacts:

- Mechanical trust: [`mechanical_trust.json`](../../../experiments/async_selective/artifacts/stage1_checks/mechanical_trust.json)
- Tiny control: [`final_metrics.json`](../../../experiments/async_selective/artifacts/tiny/synchronous_control/final_metrics.json)
- Tiny learned `r=0.75`: [`final_metrics.json`](../../../experiments/async_selective/artifacts/tiny/learned_gate_r75/final_metrics.json)
- Tiny learned `r=0.50`: [`final_metrics.json`](../../../experiments/async_selective/artifacts/tiny/learned_gate_r50/final_metrics.json)

## Mechanical trust

All preregistered mechanical checks passed. The implementation is doing the thing I meant to test, which matters here because this is a negative result.

| check | result | inline evidence |
|---|---:|---|
| Forced-open equivalence | pass | max logits diff `0.0`, max grad diff `0.0` |
| Skip semantics | pass | skipped tokens changed by `0.0`; executed tokens changed by up to `1.60` / `1.95` |
| Causality | pass | earlier-position max diff `0.0`; edited-position diff `2.11` |
| Gradient routing | pass | gate grad norms block 2 `0.1605`, block 3 `0.1633` |
| Budget control | pass | exact-target penalty `0.0`; all-open `0.25`; all-closed `0.25` |

The forced-open check is especially important: the learned-gate path exactly matches the synchronous control when gates are clamped open, so the mechanism did not quietly change the underlying model.

## Overfit

Both learned-budget settings memorized a single batch quickly, with no gate-collapse signal.

Artifacts:

- [`overfit_trace.json`](../../../experiments/async_selective/artifacts/overfit/learned_gate_r75/overfit_trace.json)
- [`final_metrics.json`](../../../experiments/async_selective/artifacts/overfit/learned_gate_r75/final_metrics.json)
- [`overfit_trace.json`](../../../experiments/async_selective/artifacts/overfit/learned_gate_r50/overfit_trace.json)
- [`final_metrics.json`](../../../experiments/async_selective/artifacts/overfit/learned_gate_r50/final_metrics.json)

| variant | steps | final LM loss | accuracy | block 2 open | block 3 open | effective ratio | collapse |
|---|---:|---:|---:|---:|---:|---:|---|
| learned `r=0.75` | `15` | `0.0170` | `1.0` | `0.695` | `0.792` | `0.829` | none |
| learned `r=0.50` | `15` | `0.0180` | `1.0` | `0.270` | `0.680` | `0.650` | none |

So the mechanism can fit data while using different skip regimes. This was not just a dead gate that happened to pass shape checks.

## Tiny rung

This is the decisive table.

| variant | best val LM loss | Δ vs control | final val acc | effective executed-block ratio | logical block reduction | runtime (s) | Δ runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| synchronous control | `2.583280` | `0.000000` | `0.2863` | `1.000` | `0.0%` | `2.53` | `0%` |
| learned `r=0.75` | `2.601659` | `+0.018379` | `0.2994` | `0.848` | `15.2%` | `3.27` | `+29%` |
| learned `r=0.50` | `2.606893` | `+0.023613` | `0.3024` | `0.681` | `31.9%` | `3.20` | `+26%` |

The quality hit is modest, not catastrophic: about `+0.018` to `+0.024` nats. If runtime had improved, this would have been worth taking seriously.

But runtime moved the wrong way. Both learned-gate variants are slower than the control, and not by a rounding-error amount.

That is enough to cut here under the preregistered process. The experiment asked for real compute savings, not just lower logical execution counts.

## Learned selectivity was real

The gates did learn differentiation rather than collapsing to a trivial mode.

Tiny-rung validation gate usage:

| variant | block 2 mean open | block 3 mean open | note |
|---|---:|---:|---|
| learned `r=0.75` | `0.550` | `0.993` | block 3 is basically always on; most skipping came from block 2 |
| learned `r=0.50` | `0.280` | `0.764` | stronger skipping, again mostly concentrated in block 2 |

So the learned policy was not uniform noise. It found that block 2 is the easier place to skip, while block 3 stayed more valuable. That is a real modeling signal, even though it did not become a systems win.

## Interpretation

I think the operative reason is simple: this is fine-grained logical skipping layered onto a dense GPU execution path.

- The model still runs dense attention and feedforward kernels over batched tensors
- The per-token gates add their own overhead
- There is no hardware-visible sparse execution here to cash in the skipped logical updates

So on this hardware, at this scale, the mechanism reduces logical block usage but increases wall-clock. That breaks the premise I cared about.

## What this settles

- Fine-grained per-token/per-block skipping can be implemented cleanly and verified mechanically
- It can learn nontrivial selective behavior without immediate collapse
- On the current GPU execution path, that behavior does not translate into runtime savings
- So this approach is not a viable compute-saving direction for this project in its current form

## What remains open

- Coarser routing that preserves batching better
- Expert-style or group-wise conditional compute that can actually change which dense kernels run
- Hardware-friendly sparse kernels or compiler/runtime support that can turn logical sparsity into real savings
- Execution models that are genuinely asynchronous in a hardware sense rather than just logically selective inside dense PyTorch ops

## Cut decision

Stopped at the tiny rung, as preregistered.

The reason is not that the model failed to learn. It did learn something interesting. The reason is that the core premise of the experiment was compute savings, and the first rung with real runtime evidence already showed the wrong sign.
