# Question: Column Residual Stream

## What we're asking

What is the right definition of a cortical column's internal state — the thing it maintains, predicts, and communicates?

## Current state of knowledge

- Options include: a persistent hidden state vector; a token-conditioned activation at the current macrostep; a stream of incoming messages combined with local memory; something else entirely
- The answer shapes the local prediction objective, the communication protocol, and what "surprisal" means
- Not designed yet — this is a core open question
- A column likely needs to separate at least two kinds of latent: **private recurrent state** (internal memory, not exposed) and **public message state** (what other columns can read). If one vector does both jobs, it may be hard to train and hard to interpret.

## The local objective trap

"Predict your own next residual state" may be too easy to satisfy — a column could learn smooth self-persistence (copy + damp) without learning meaningful cross-column communication. Ways to make the local objective harder to satisfy trivially:
- predict *innovation* (unexpected change), not raw next state
- predict incoming messages
- train under corruption or ablation so local self-copy is insufficient
- add explicit pressure for communication to be useful

This is a design decision with real consequences. Track whether the local loss improves while global task performance does not — that's the failure signal.

## Open sub-questions

- Is the state primarily a memory (carries information across time) or a representation (encodes current input)?
- Should public communication state be separate from private recurrent state?
- Does the column predict its own next state, the next incoming message, or something else?
- What initialisation works — learned default, data-dependent, warmup steps, persistent carryover?
- What makes the local objective nontrivial enough to force real communication?

## Status

Not yet investigated experimentally in this repo.

