# Arbitrary-Order Sampling with Cross-Attention

## Motivation

From [dictation 2026-05-21-10](../../../dictations/2026-05-21-10.md) and [VISION.md](../../../VISION.md) (arbitrary-order image patches as a PRIMARY dataset).

Max's master's work demonstrated this on MNIST: train a transformer to predict patches in arbitrary order using cross-attention to query positions. At inference, choose the prediction order strategically. A single forward pass gives conditional marginals for ALL unobserved positions simultaneously — like a Gaussian process.

## Architecture

Encoder-decoder with cross-attention:

```python
# Encoder: bidirectional self-attention on observed tokens
# (no causal mask — can see all observed context)
observed = token_emb(obs_tokens) + pos_emb(obs_positions)
context = encoder(observed)  # full self-attention

# Decoder: position queries cross-attend to encoded context
queries = query_pos_emb(target_positions)
predictions = cross_attention(queries, context)
logits = head(predictions)  # [n_targets, vocab_size]
```

This is simpler than XLNet's two-stream mechanism. The encoder processes whatever subset is observed; position queries ask "what should be at position X?" The key insight: target positions enter ONLY as queries, never seeing their own content.

## What this serves

From VISION.md: "images split into patches, presented in many orderings during training [...] Model learns to predict any patch given any subset in any order."

Key properties Max described:
- Single forward pass → all conditional marginals (Gaussian process-like)
- Can choose sampling order at inference (predict easy parts first)
- Works on MNIST pixels — "whatever orders I trained on worked well"
- "Training on a single order worked better than training on many orders — specialization beats generalization, at least at that scale"
- Can plug in one pixel and get conditional marginals of all others in one pass

## Hypotheses

1. **Works at all:** A small encoder-decoder transformer can learn to predict MNIST patches given arbitrary subsets, trained with random subset selection.
2. **Order-agnostic:** Training on random orderings produces a model that handles any query order at inference.
3. **Single-order specialization:** Training on one fixed order outperforms training on many random orders (Max's prior finding — replicable?).
4. **Conditional marginals:** Given partial observation, the model produces sensible conditional distributions for all unobserved positions simultaneously.

## Planned evidence

- Loss curves: overfit-one-batch → tiny model → full MNIST
- Sample outputs: given partial image, show model's predictions for remaining patches
- Comparison: random-order training vs fixed-order training
- Conditional marginal visualization: observe one patch, show predicted distributions for all others

## What this does NOT settle

- Whether this composes with the residual-stream-across-time architecture (Max asked this as an open question)
- Text applications (VISION.md explicitly says "arbitrary-order text is not a goal")
- Optimal sampling strategies at inference (heuristic ordering)
- Scale beyond MNIST

## Implementation plan

Start with the simplest version:
- MNIST as 7×7 grid of 4×4 patches (49 patches, 16 pixels each)
- Or: MNIST as 784 individual pixels (simpler but longer sequence)
- Vocabulary: quantized pixel values (e.g., 16 bins or 256 bins)
- Tiny model first: 2-4 encoder layers, 1-2 decoder layers, d_model=64-128

Training:
- Each sample: randomly choose how many patches to observe (uniform 0 to N-1)
- Randomly select which patches are observed
- Predict all unobserved patches simultaneously
- Loss: cross-entropy on quantized pixel values

## Results

_Pending._

## Next steps

_Pending._
