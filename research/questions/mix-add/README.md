# Question: Mix-Add

## What we're asking

Does `mix(a, b, m) = a*sqrt(σ(m)) + b*sqrt(1-σ(m))` actually preserve norms in practice? When does it help training stability vs hurt performance? Is it worth using as the residual update operation?

## Current state of knowledge

- Max has used it and found it stabilises training when embeddings are approximately normally distributed
- Behaviour with raw (non-normalised) inputs is less predictable
- The norm-preservation claim is approximate — it holds when a and b are uncorrelated and similarly scaled. Correlation between branches, gate saturation, and covariance structure all affect it

## Open sub-questions

- Under what conditions does norm-preservation actually hold?
- How does it compare against plain residual + RMSNorm?
- Does gate saturation (sigmoid(m) → 0 or 1) cause gradient problems?
- Per-channel m vs scalar m — what difference does it make?
- The noise gate extension: is it useful as a modularity tool, or just interesting?

## Status

Not yet investigated experimentally in this repo.
