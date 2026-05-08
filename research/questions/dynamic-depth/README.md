# Question: Dynamic Depth

## What we're asking

Does adaptive computation depth work on a standard recurrent model or transformer — using the same weights multiple times per output token, with a learned halting criterion based on loss prediction?

This is a completely separate research direction from the cortical column work. It starts on a standard model and only potentially connects to Thread 1 much later, if both make sense independently.

## Current state of knowledge

- The mechanism: run multiple forward passes per token (1, 2, 4, ... iterations — exponential schedule). Record loss at each depth. Train a loss-prediction head to predict those losses. At inference, use the head to decide when to stop.
- Dynamic hierarchical encodings might be relevant — the representation at each depth could encode different levels of abstraction, and the halting criterion might interact with that.
- This is well-studied territory. The goal here is not novelty but understanding whether it works on our tasks and hardware.
- Must be validated on a standard model before being combined with cortical columns.

## Open sub-questions

- What training schedule for depth rollouts works best? (exponential? uniform? curriculum?)
- How well does the loss-prediction head calibrate?
- Is there a meaningful quality/compute tradeoff, or does the model just learn to always use max depth?
- How does this interact with the RNN statefulness in the arbitrary-order patch task?
- If it works independently, does it compose with the cortical column architecture?

## Relevant prior work

There is prior work in this space (Adaptive Computation Time, PonderNet, Universal Transformers). Read only if a paper has results directly relevant to our setup — not for coverage.

## First bounded unit

The broad question above remains the research direction. The first bounded unit is deliberately narrower.

- **Host surface:** one standard model only, currently planned as the existing tiny transformer.
- **Task surface:** the existing tiny fixed-window character next-token task.
- **Comparator:** the ordinary one-pass version on the same host and task surface.
- **Bounded question:** does repeated within-token reuse of the same model weights produce a readable depth-related outcome surface on this bounded task, relative to the one-pass comparator?
- **What this unit is not:** it is not a claim that dynamic depth works in general, not a full learned-halting or loss-prediction study, not a compute/quality benchmark, not a backend decision, and not a step toward cortical-column composition.
- **Loss-prediction / learned halting:** not assumed to be in scope for this first unit. That must be explicitly decided before coding rather than absorbed from the broader question framing.

The broader sub-questions above remain open even if this first bounded unit succeeds, fails, or closes ambiguously.

## Status

Not yet investigated experimentally in this repo. The first bounded unit is now planned, but not yet implemented or run.
