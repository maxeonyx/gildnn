# Autoregressive MNIST (Pixel Transformer)

We reset this experiment to its simplest useful form: a tiny decoder-only transformer that models downsampled MNIST pixels in raster order. Earlier iterations tried to juggle multiple model families, patch pipelines, and sampling permutations at once; this pass narrows the scope so we can validate one clean baseline before layering anything else on top.

## Current Focus
- Downsample MNIST digits to $14\\times14$ (or smaller) grids so sequence length stays tiny.
- Tokenise pixels as discrete values in $[0, 255]$ with a single start/end token.
- Train a causal transformer with just enough capacity (e.g., 4 heads, 4 layers, width 128) to overfit a mini-batch before scaling up.
- Teacher-force during training; sample autoregressively with temperature $1.0$ to start.

## Implementation Checklist
1. Build a deterministic loader that returns `(input_tokens, target_tokens)` given the downsampled image and raster order.
2. Implement the lightweight transformer block (embedding, rotary/learned positions, masked self-attention, MLP).
3. Hook the model into the shared experiment harness with a single configuration entry and no architecture switching.
4. Produce report artefacts that walk through:
   - Dataset slices and visual examples.
   - Token construction for one digit (prefix/next-token pairs).
   - Training curves for both train/test metrics.
   - Sampled completions at early, mid, and late training checkpoints.
5. After the baseline trains end-to-end, snapshot `config.json`, `benchmark.json`, and regenerate `report.md` directly from the harness.

## Guardrails
- If new ideas pop up (patches, sampling tricks, etc.), capture them in follow-up experiments (`autoregressive-patch-mnist`, `autoregressive-mnist-sampling-orders`) instead of expanding this crate.
- Keep the implementation runnable on CPU with batch sizes ≤32 so iterative development stays quick.
- Regenerate the deterministic report after every meaningful change to keep the artefact canonical.
