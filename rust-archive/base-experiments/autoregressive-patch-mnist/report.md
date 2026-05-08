# Autoregressive Patch MNIST

This report will document the discrete-VAE patch pipeline layered on top of the autoregressive transformer baseline.

## Walkthrough Outline

1. **Patch Tokeniser Preview** – visualise reconstructed patches vs. originals for both train/test digits.
   <!-- OUTPUTSLOT:patch-tokeniser start -->
   _Placeholder for VAE codebook inspection and reconstruction grids._
   <!-- OUTPUTSLOT:patch-tokeniser end -->
2. **Sequence Construction** – show how patch indices map to the rasterised transformer sequence.
   <!-- OUTPUTSLOT:patch-sequence start -->
   _Placeholder for token tables with patch coordinates, code indices, and positional encodings._
   <!-- OUTPUTSLOT:patch-sequence end -->
3. **Training Metrics** – compare train/test losses alongside pixel-baseline curves.
   <!-- OUTPUTSLOT:patch-training start -->
   _Placeholder for loss/bits-per-dim charts._
   <!-- OUTPUTSLOT:patch-training end -->
4. **Sampling Panels** – include both sampled digits and their decoded patch reconstructions.
   <!-- OUTPUTSLOT:patch-samples start -->
   _Placeholder for early/mid/late sampling panels._
   <!-- OUTPUTSLOT:patch-samples end -->
5. **Final Evaluation** – run a larger batch for test metrics and sample diversity.
   <!-- OUTPUTSLOT:patch-final start -->
   _Placeholder for final evaluation summary._
   <!-- OUTPUTSLOT:patch-final end -->

## Status

- ⬜ Train or import the discrete VAE and document reconstruction quality.
- ⬜ Integrate patch tokenisation into the autoregressive harness.
- ⬜ Regenerate this report plus `benchmark.json` deterministically.

Remember to keep this crate focused on patch tokens only—sampling-order experiments live in `base-experiments/autoregressive-mnist-sampling-orders`.
