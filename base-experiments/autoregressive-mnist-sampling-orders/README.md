# Autoregressive MNIST – Sampling Orders

This branch experiment keeps the pixel transformer intact but explores order-agnostic training and sampling. We shuffle raster positions during training, supply 2D positional embeddings, and evaluate how different decoding schedules affect sample quality.

## Planned Scope
- Extend the baseline dataset loader to emit random but reproducible pixel orderings per sample.
- Inject 2D positional encodings so the transformer can reason about spatial neighbours even when the prefix is shuffled.
- Train with order dropout (e.g., mix of raster, column-major, random permutations) while logging metrics per ordering.
- Evaluate sampling strategies: classic raster rollouts, greedy sorted by model confidence, and order-conditioned infilling.

## Guardrails
- Do not introduce new architectures; reuse the pixel baseline model and hyperparameters.
- Document the exact permutation set and RNG seeds in `config.json` so runs stay deterministic.
- Focus on sampling behaviour and qualitative panels—if it drifts into patch tokens, move those changes to `base-experiments/autoregressive-patch-mnist`.
