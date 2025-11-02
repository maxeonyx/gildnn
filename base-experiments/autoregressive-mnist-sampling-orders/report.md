# Autoregressive MNIST – Sampling Orders

This report captures experiments where we shuffle pixel orderings during training and explore flexible decoding schedules at inference time.

## Walkthrough Outline

1. **Permutation Gallery** – visualise representative orderings (raster, column-major, snake, k-random, random).
   <!-- OUTPUTSLOT:order-gallery start -->
   _Placeholder for ordering diagrams._
   <!-- OUTPUTSLOT:order-gallery end -->
2. **Training Dynamics** – plot loss/bits-per-dim for each ordering bucket.
   <!-- OUTPUTSLOT:order-training start -->
   _Placeholder for per-order metric charts._
   <!-- OUTPUTSLOT:order-training end -->
3. **Prefix Conditioning Examples** – demonstrate completing digits given sparse, shuffled context.
   <!-- OUTPUTSLOT:order-prefix start -->
   _Placeholder for infilling panels._
   <!-- OUTPUTSLOT:order-prefix end -->
4. **Decoding Strategies** – compare raster rollout vs. entropy-based ordering.
   <!-- OUTPUTSLOT:order-decoding start -->
   _Placeholder for sampling comparisons._
   <!-- OUTPUTSLOT:order-decoding end -->
5. **Final Evaluation** – summarise metrics and show a larger test batch sampled with multiple schedules.
   <!-- OUTPUTSLOT:order-final start -->
   _Placeholder for final evaluation summary._
   <!-- OUTPUTSLOT:order-final end -->

## Status

- ⬜ Extend the data pipeline to emit deterministic shuffled orders.
- ⬜ Update the transformer to accept order-specific positional embeddings.
- ⬜ Regenerate this report after running the flexible sampling experiments.

If this scope grows beyond sampling strategies (e.g., introduces patch tokens), pause and migrate that work to the appropriate experiment crate.
