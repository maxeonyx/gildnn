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

## Active bounded unit — Unit 01: I/O contract narrowing

**Status:** framing in progress.

**Type:** exceptional pre-prototype architecture-boundary narrowing unit. This is not an attempt to resolve the broad I/O-contract question. It is one tightly scoped comparison intended to narrow one downstream prototype choice.

**Decision surface:** one architecture boundary only — the first cortical-column prototype's input/output contract.

**Comparator:** exactly two candidate options:

1. **Direct designated-column contract** — task inputs are injected into designated input columns and outputs are read from designated output columns or a readout attached directly to that designated boundary.
2. **Adapter-mediated shared-latent contract** — task inputs first map into a shared latent/message interface and task outputs are decoded from that shared interface via adapters, rather than by special-purpose designated I/O columns.

These names are provisional labels for the bounded unit. The unit itself still has to define them precisely enough to compare.

**Concrete boundary examples used:** current character next-token prediction and arbitrary-order image-patch prediction. These are boundary-pressure examples, not a commitment to build the patch experiment inside this unit.

**Expected artifact:** one short comparison memo with side-by-side contract comparison, worked boundary examples for both task surfaces, explicit rejection criteria, and an explicit falsifier for any provisional front-runner.

**Explicit non-goals for this unit:**

- not a runnable cortical-column prototype
- not a residual-stream decision except where the interface strictly forces one
- not a broadcast-router decision
- not an async-execution decision
- not a gradient-unhooking decision
- not an update-trigger decision
- not a bidirectional-propagation decision
- not implementation planning for the full architecture
- not a claim that the patch task should be implemented immediately after this memo

**What remains open after this unit:** all five sub-questions above remain open. This unit only tries to narrow whether the first prototype's I/O boundary should start from a designated-column contract or from an adapter-mediated shared-latent contract, and only provisionally.

## Status

Not yet investigated experimentally in this repo. One bounded narrowing unit is now framed, but not yet completed.
