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
- Each sample: randomly choose how many patches to observe (uniform 1 to N-1)
- Randomly select which patches are observed
- Predict all unobserved patches simultaneously
- Loss: MSE on continuous pixel values (0-1 range)

## Results

**The mechanism works.** Encoder-decoder transformer successfully learns to predict arbitrary MNIST patches from arbitrary subsets.

### Training (50 epochs, full MNIST, random subset selection)

| Epoch | Train MSE (per-pixel) | Val MSE (per-pixel) |
|-------|-----:|-----:|
| 1 | 0.0574 | 0.0515 |
| 10 | 0.0253 | 0.0248 |
| 25 | 0.0208 | 0.0207 |
| 50 | 0.0196 | 0.0195 |

Final per-pixel MSE: **0.0195** (RMSE ≈ 0.14 per pixel on 0-1 scale). This is well below the trivial baseline (~0.05) — the model genuinely predicts missing patches using observed context.

### Architecture

- d_model=128, 4 encoder layers, 2 decoder layers, 4 heads
- Total training time: ~68 minutes on RTX 3090 (50 epochs × ~80s/epoch)
- Cosine LR schedule with 2-epoch warmup

### Interpretation

The architecture is viable. Given any subset of observed MNIST patches, the model predicts unobserved patches with meaningful accuracy. The conditional-independence assumption (all targets predicted simultaneously) appears not to be a major limitation on this task.

Note: metric was initially misreported as per-patch (16× too high). Corrected by dividing by PATCH_DIM in the normalization.

## Status

**Deprioritized** per [dictation 2026-05-22-6](../../../dictations/2026-05-22-6.md): Max explicitly says this is not the current focus. "I don't want to be doing the arbitrary order sampling work now." Reference material for later: `maxeonyx/msc` and `maxeonyx/thesis` on GitHub.

## Next steps (deferred)

- Conditional marginal visualization (observe 1 patch, show predictions for all others)
- Comparison: random-order training vs fixed-order training
- Autoregressive sampling (predict one, add to observed, repeat)
- Compose with the residual-stream-across-time architecture (Max's open question)
