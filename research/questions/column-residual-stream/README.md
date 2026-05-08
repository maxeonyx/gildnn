# Question: Column Residual Stream

## What we're asking

What is the right definition of a cortical column's internal state — the thing it maintains, predicts, and communicates?

## Current state of knowledge

- Options include: a persistent hidden state vector; a token-conditioned activation at the current macrostep; a stream of incoming messages combined with local memory; something else entirely
- The answer shapes the local prediction objective, the communication protocol, and what "surprisal" means
- Not designed yet — this is a core open question

## Open sub-questions

- Is the state primarily a memory (carries information across time) or a representation (encodes current input)?
- Should public communication state be separate from private recurrent state?
- Does the column predict its own next state, or the next incoming message, or something else?
- What initialisation works — learned default, data-dependent, warmup steps, persistent carryover?

## Status

Not yet investigated experimentally in this repo.
