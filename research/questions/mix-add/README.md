# Question: Mix-Add

## What we're asking

Does `mix(a, b, m) = a*sqrt(σ(m)) + b*sqrt(1-σ(m))` actually preserve norms in practice? When does it help training stability vs hurt performance? Is it worth using as the residual update operation?

## The operation

From [modularity-loss](https://github.com/maxeonyx/modularity-loss):

```
mix(a, b, m) = a * sqrt(σ(m)) + b * sqrt(1 - σ(m))
```

where σ is sigmoid and m is a learnable scalar (or per-channel parameter).

As a residual update: `residual(x, f) = mix(x, f(x), m)`

Noise gate extension: `noise_gate(x, f) = mix(x, mix(noise, f(x), m), m)` — this is a side quest, not core.

## Current state of knowledge

- Max has used it and found it stabilises training when embeddings are approximately normally distributed
- Behaviour with raw (non-normalised) inputs is less predictable
- The norm-preservation claim is approximate — it holds when a and b are uncorrelated and similarly scaled. Positive correlation inflates norm; anti-correlation shrinks it; gate saturation can still cause pathologies.

Recommended instrumentation if investigating: track RMS before/after mix, correlation between branches, gate values and saturation frequency, state norm drift across long horizons.

## Open sub-questions

- Under what conditions does norm-preservation actually hold?
- How does it compare against plain residual + RMSNorm and GRU-like interpolation?
- Does gate saturation (sigmoid(m) → 0 or 1) cause gradient problems?
- Per-channel m vs scalar m — what difference does it make?
- The noise gate extension: is it useful as a modularity tool, or just interesting?

## Status

Not yet investigated experimentally in this repo.

