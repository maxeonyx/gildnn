# Variable-Length Byte Prediction Heads

## Objective
Develop output layers that can emit predictions for multiple byte counts per step, enabling dynamic chunking strategies during inference.

## Key Ideas
- Provide specialized heads for 1, 2, 4, 6, 8, and 16-byte continuations with shared underlying representations.
- Train the model to estimate confidence or expected loss for each head alongside the token predictions.
- Create selection policies that choose the longest head meeting a loss threshold to accelerate decoding.

## Open Questions
- How should training batches balance different head lengths to avoid bias toward short outputs?
- What loss formulations couple byte-level accuracy with head-selection reliability?
- Can beam search or speculative decoding incorporate variable-length head decisions effectively?
