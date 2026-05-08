# Question: Graph vs Global Channel

## What we're asking

Does the graph structure matter, or does the global communication channel do most of the work — making graph locality cosmetic?

## Current state of knowledge

- The architecture has two communication paths: local graph edges (each column connects to its spatial neighbours) and a global channel (broadcast router or all-to-all attention)
- If the global channel is expressive enough, columns might effectively have access to everything anyway, making local graph edges redundant
- This is a key empirical question — not something to assume either way

## Open sub-questions

- Can we design an experiment that separates the contribution of graph edges vs global channel?
- Does the graph topology (ring, grid, random, GPU-topology-matched) affect outcomes measurably?
- Is there a regime (e.g. very large column count, constrained global channel bandwidth) where graph locality matters more?
- Does the answer differ for the language task vs the image patch task?

## Status

Not yet investigated experimentally in this repo.
