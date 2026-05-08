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

## Status

Not yet investigated experimentally in this repo.
