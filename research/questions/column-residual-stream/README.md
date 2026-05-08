# Question: Column Residual Stream

## What we're asking

What is the right definition of a cortical column's internal state — the thing it maintains, predicts, and communicates?

## Current state of knowledge

- Options include: a persistent hidden state vector; a token-conditioned activation at the current macrostep; a stream of incoming messages combined with local memory; something else entirely
- The answer shapes the local prediction objective, the communication protocol, and what "surprisal" means
- Not designed yet — this is a core open question
- A column likely needs to separate at least two kinds of latent: **private recurrent state** (internal memory, not exposed) and **public message state** (what other columns can read). If one vector does both jobs, it may be hard to train and hard to interpret.

## The local prediction objective

Each column predicts its own *next input* — the latent it will receive from surrounding nodes and the global broadcast at the next timestep. Not its own output. This is a local next-timestep prediction of a latent, not a self-copy objective.

Whether this collapses (e.g. to smooth persistence of the input) still depends on the surrounding design — the input changing enough, the column having to integrate across neighbours, etc. That's an empirical question. Track whether the local loss improves while global task performance does not — that's the failure-mode signal.

## Open sub-questions

- Is the state primarily a memory (carries information across time) or a representation (encodes current input)?
- Should public communication state be separate from private recurrent state?
- What initialisation works — learned default, data-dependent, warmup steps, persistent carryover?
- Under what conditions does the local prediction objective collapse vs stay informative?

## Status

Not yet investigated experimentally in this repo.

