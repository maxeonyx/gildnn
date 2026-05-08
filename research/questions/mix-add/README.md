# Question: Mix-Add

## What we're asking

This remains a broad open question: does `mix(a, b, m) = a*sqrt(σ(m)) + b*sqrt(1-σ(m))` seem useful as a residual update operation, and if so, when, why, and under what conditions?

The bounded units recorded below do not close that broader question unless they explicitly say they do.

## Direct source framing from the dictations

- Max described Mix-Add as "interesting" and said that even just the mix function has been "quite useful" in his experience. See [`dictations/2025-05-08-1.md`](../../../dictations/2025-05-08-1.md) when checking the original wording.
- The claimed upside is hedged and conditional: "It helps make training very stable, even if it makes it less performant, I guess." See [`dictations/2025-05-08-1.md`](../../../dictations/2025-05-08-1.md).
- Max explicitly agreed that the claim needs caveat: "If you have normally distributed embeddings, in my experience it works quite well. If you have raw data getting put into your network, it doesn't necessarily work so well." See [`dictations/2025-05-08-2.md`](../../../dictations/2025-05-08-2.md).
- The noise gate is not the main thing here: it is "kind of a side quest." See [`dictations/2025-05-08-1.md`](../../../dictations/2025-05-08-1.md).

## The operation

From [modularity-loss](https://github.com/maxeonyx/modularity-loss):

```
mix(a, b, m) = a * sqrt(σ(m)) + b * sqrt(1 - σ(m))
```

where σ is sigmoid and m is a learnable scalar (or per-channel parameter).

As a residual update: `residual(x, f) = mix(x, f(x), m)`

Noise gate extension: `noise_gate(x, f) = mix(x, mix(noise, f(x), m), m)`.

## Current state of knowledge

- This repo now has one completed bounded Mix-Add unit on the tiny fixed-window character stack; see the bounded investigation log below.
- The broad Mix-Add question is still open. This repo does not yet have a general norm-preservation study, a general training-stability study, or a settled answer on whether Mix-Add is generally worth using.
- The strongest source-backed prior remains conditional and experiential: approximately normally distributed embeddings seem like the friendlier case; raw-input surfaces may be less predictable.
- For bounded investigations on residual surfaces, the most useful saved observables so far are branch RMS before/after combination, branch correlation, effective gate values, and gate saturation frequency.

## Bounded investigation log

| Unit | Scope | Status |
|---|---|---|
| Unit 01 | Tiny transformer host surface; plain residual vs scalar Mix-Add on the tiny fixed-window character next-token task | Complete |

## Unit 01 — tiny transformer host, plain residual vs scalar Mix-Add

### Scope

This unit compared one plain residual branch against one scalar-`m` Mix-Add branch on the existing tiny transformer host surface. The saved scope artifact is [`comparison_scope.json`](artifacts/comparison_scope.json).

```json
{
  "host_surface": "tiny_transformer_baseline",
  "variant_under_test": "scalar_m_mix_add",
  "primary_comparator": "plain_residual",
  "bounded_task_surface": "tiny_fixed_window_character_next_token"
}
```

### What this bounded unit shows

Both branches cleared the same cheap ladder. Overfit reached `final_accuracy = 1.0` for both [`plain`](artifacts/plain_residual/overfit_metrics.json) and [`mix`](artifacts/scalar_mix_add/overfit_metrics.json).

```json
{
  "plain_overfit_final_accuracy": 1.0,
  "scalar_mix_add_overfit_final_accuracy": 1.0,
  "both_reached_memorization_bar": true
}
```

On the saved tiny task surface, the two branches stayed tied on the main saved task metrics: [`plain`](artifacts/plain_residual/tiny_run_metrics.json), [`mix`](artifacts/scalar_mix_add/tiny_run_metrics.json).

```json
{
  "plain_final_loss": 0.03202458471059799,
  "plain_final_accuracy": 0.9793282151222229,
  "scalar_mix_add_final_loss": 0.03205796703696251,
  "scalar_mix_add_final_accuracy": 0.9793282151222229
}
```

The saved prompt continuations also match on this bounded surface: [`plain`](artifacts/plain_residual/tiny_samples.json), [`mix`](artifacts/scalar_mix_add/tiny_samples.json).

```json
{
  "hello": "hello world.\nsmall text.\nhello world.\nsmall text.\nhello world.\nsmall text.\nhello worl"
}
```

Where the bounded result does separate is on the saved residual-behaviour surface. Mix-Add saved lower `combined_output_rms` at both recorded residual sites, while the saved gates stayed interior and non-saturated: [`plain branch observables`](artifacts/plain_residual/plain_branch_observables.json), [`mix branch observables`](artifacts/scalar_mix_add/mix_branch_observables.json).

```json
{
  "tiny_surface_attention_combined_output_rms": {
    "plain_residual": 1.946786642074585,
    "scalar_mix_add": 1.373215913772583
  },
  "tiny_surface_feedforward_combined_output_rms": {
    "plain_residual": 2.3190948963165283,
    "scalar_mix_add": 1.2020200490951538
  },
  "tiny_surface_mix_effective_gate": {
    "attention": 0.4354681670665741,
    "feedforward": 0.4655552804470062
  },
  "tiny_surface_mix_gate_saturation_frequency": {
    "attention": 0.0,
    "feedforward": 0.0
  }
}
```

The narrowest honest reading is: on this one tiny transformer host surface, scalar Mix-Add preserved the same saved task-surface outcome while changing saved residual behaviour in one consistent direction — lower recorded `combined_output_rms` with interior non-saturated gates.

### What this bounded unit does not show

- It does not show that Mix-Add generally preserves norms in practice.
- It does not show that Mix-Add generally improves training stability.
- It does not show that Mix-Add is a better residual operator overall.
- It does not settle scalar `m` versus per-channel `m`.
- It does not say anything decisive about the noise gate, cortical-column work, async execution, or broader architecture direction.

## Open sub-questions

- Under what conditions does the approximate norm-preservation intuition actually hold well enough to matter?
- When, if ever, does Mix-Add help training stability enough to matter beyond a bounded residual-behaviour difference?
- What changes if `m` is per-channel rather than scalar?
- What happens on less toy or less approximately normalised host surfaces?
- Is the noise gate useful, or is it just a side quest?

## Status

One bounded unit is complete. The broader Mix-Add question remains open.
