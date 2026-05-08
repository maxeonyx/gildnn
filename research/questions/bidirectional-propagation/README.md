# Question: Bidirectional Propagation

## What we're asking

What does bidirectional propagation look like in this architecture, and is it useful or necessary?

## Current state of knowledge

- Mentioned as an open question in the original vision — not designed at all
- Several possible interpretations: bidirectional over graph edges, backward-in-time smoothing, iterative relaxation across the column graph
- Standard transformers are unidirectional (causal). Standard RNNs are unidirectional. Bidirectional models (BERT-style) require seeing the full sequence before producing outputs.
- Whether bidirectionality is desirable depends heavily on what tasks we're running and what the I/O contract looks like.

## Open sub-questions

- Is this about graph topology (edges going both ways) or about temporal direction?
- Does bidirectional propagation require any changes to the training objective?
- Is this compatible with the arbitrary-order image patch task (which might already be inherently bidirectional)?
- Does it interact with the surprisal-trigger framing — does backward signal look like surprisal?
- Is this a priority, or a stretch goal after basic forward propagation works?

## Status

Not yet investigated experimentally in this repo.
