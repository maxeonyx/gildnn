# Question: I/O Contract

## What we're asking

What is the input/output contract for the cortical column architecture? How does input enter the system and how does output leave?

## Current state of knowledge

- Completely open. The architectural ideas have been sketched but the I/O boundary hasn't been designed.
- For the character-level language task: input is a sequence of characters, output is a probability distribution over next character. How does that map onto a graph of recurrent columns?
- For the arbitrary-order image patch task: input is a subset of patches in some order, output is a patch prediction. The RNN statefulness matters here — re-presenting patches in a different order requires replaying from scratch.
- "Inputs initially handled by adjacent columns" is one intuition but not designed.

## Open sub-questions

- Does each token/patch get mapped to one input column, or is input broadcast to many?
- Does output come from one output column, from all columns averaged, or from a separate read-out head?
- How does the positional encoding interact with the graph structure?
- Is the I/O contract the same for both tasks, or does it differ?
- Does input injection happen once (at the start of a macrostep) or continuously?

---

## Bounded-unit status

There is currently **no active bounded I/O-contract unit**.

An attempted Unit 01 tried to narrow the first prototype's full I/O boundary by comparing a designated-column-style contract against an adapter-mediated shared-latent-style contract. That framing did **not** survive the process gates.

What failed in the current repo record:

- the full I/O two-option frame did not freeze honestly enough for the exceptional bounded-unit category
- a hidden third option was already visible, especially on the output side
- the shared-latent framing still leaned on unresolved interface / residual-stream semantics

After that, a narrower ingress-only reframe was tested, but it also failed adversarial review. The local-input side did not stay honestly binary either: the source set supported adjacent columns more clearly than one designated column, which left a local multi-column / adjacent-neighborhood option still doing real work.

So this question remains broad and open. Do **not** resume I/O-contract narrowing from either failed frame by drift. A future selection pass would need to choose a genuinely different narrower unit before any new bounded memo work starts.

## Status

Not yet investigated experimentally in this repo. No bounded I/O-contract unit is currently active.
