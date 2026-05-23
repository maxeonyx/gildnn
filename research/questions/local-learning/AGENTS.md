# research/questions/local-learning/

This directory covers the theory and experimental plan for local learning on the current architecture: parallel multi-rate blocks on a shared residual stream.

The core question: how much quality does truncating gradient flow between blocks cost, and where on the full-backprop ↔ fully-local spectrum is the sweet spot?

Prior negative results from the old recurrent-stack framing are in `research/questions/local-learning-residual/`. Those used a fundamentally different architecture (detached 3-layer recurrent stacks) and are not directly comparable.
