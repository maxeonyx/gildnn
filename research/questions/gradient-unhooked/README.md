# Question: Gradient Unhooked

## What we're asking

What should be unhooked from gradient propagation between columns, and what effect does it have? Does less gradient coupling produce useful modular specialisation, or does it cause communication breakdown?

## Current state of knowledge

- Intent: minimal gradient coupling between columns — each learns somewhat independently
- "No global gradient propagation" is the direction, but the exact boundary is not settled
- Embeddings may need partial global gradient signal to form a shared representation basis
- No gradients through time across columns is a hypothesis — plausible but not confirmed
- Unhooked gradients create what might be a coordination game: columns learn against a moving target induced by other columns, with no direct credit assignment for sending useful messages — whether this leads to useful specialisation or protocol breakdown is unknown

## Open sub-questions

- What exactly is stopped — gradients through messages? through attention weights? through time?
- Do columns co-adapt into useful shared protocols, or diverge into private dialects?
- Does any shared convention (normalisation space, codebook, quantisation) help stabilise communication under unhooked gradients?
- Can you retrain one column region while freezing others?

## Status

Not yet investigated experimentally in this repo.

