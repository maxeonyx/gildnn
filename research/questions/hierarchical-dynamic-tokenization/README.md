# Hierarchical dynamic autoregressive prediction

## Core question

Serves [dictation 2026-05-24-3](../../../dictations/2026-05-24-3.md): Can we build a hierarchical architecture where each level dynamically tokenizes the level below via autoregressive autoencoders, with loss prediction determining chunk boundaries?

## Status: QUEUED (not started)

This is a substantially different architecture direction from the current "block 1 predicts block 0's state" experiments. It should be explored separately once the current local-learning-variants line is resolved.

## Max's description (from dictation)

Multi-level autoregressive prediction with dynamic tokenization:

1. **Level 0:** Character/byte-level autoregression (the raw input)
2. **Autoregressive autoencoder:** Takes N tokens, encodes to a bottleneck, decodes by unrolling autoregressively from the bottleneck. The bottleneck is a representation of those N tokens.
3. **Dynamic chunk boundaries:** A loss-prediction head estimates how many tokens this chunk should contain. Where the loss predictor says "prediction quality drops off" → chunk boundary.
4. **Level 1:** Sequence of bottleneck vectors (each representing a variable-length chunk). Autoregressive prediction over these.
5. **Recurse:** Level 1's bottlenecks get chunked and compressed to Level 2, and so on up.

Two training objectives at each level:
- **Reconstruction:** autoencoder loss (decode the chunk from its bottleneck)
- **Next-chunk prediction:** predict the next bottleneck in the sequence

The bottleneck is useful both as context for the level above AND for the following chunk at the same level ([dictation 2026-05-24-4](../../../dictations/2026-05-24-4.md)).

## Relationship to current work

| Current experiments | This direction |
|---|---|
| Block 1 predicts block 0's full state | Each level predicts the next *chunk bottleneck* |
| Fixed 2-block hierarchy | Recursive hierarchy with variable depth |
| Prediction target is full hidden state | Prediction target is a learned compressed representation |
| No dynamic computation | Dynamic chunk size (variable tokens per bottleneck) |
| Local learning = no cross-boundary gradient | Local learning = each level trains its own autoencoder + predictor |

The loss-prediction mechanism (from [dictation 2026-05-24-1](../../../dictations/2026-05-24-1.md)) connects to the current work: it's the mechanism for deciding chunk boundaries. The current strict-local failure (target isn't task-grounded) is potentially addressed here by construction — the autoencoder's bottleneck IS task-grounded (it must reconstruct the input).

## What would be needed to start

1. Resolve whether the simpler "local CE + prediction" approach works (current experiment)
2. Design the autoregressive autoencoder architecture (encoder → bottleneck → autoregressive decoder)
3. Design the loss-prediction head for dynamic chunk boundaries
4. Start at 2 levels only (character + chunk)

## What this does NOT cover

- The attention-based routing mechanism from the original vision note
- The broadcast mechanism
- Async hardware execution
- Multi-rate blocks (separate concept)
