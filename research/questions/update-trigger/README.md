# Question: Update Trigger

## What we're asking

What triggers a column to update? The current framing is surprisal — a column updates when its incoming residual stream changes enough. But the details are completely open.

## Current state of knowledge

- The surprisal framing is intuitive but not designed
- Surprise relative to what baseline? EMA of previous state? Prediction of expected next state?
- Who measures surprisal — the column itself, the sender, or something on the edge?
- What happens to a column during dormancy — does it freeze, decay, drift?
- Whether surprisal-triggered updates produce a useful sparsity pattern, or whether it collapses to always-on or always-off, is unknown

## Open sub-questions

- What is the right surprisal metric? (residual delta norm? prediction error? EMA of change?)
- Who measures it — sender, receiver, or edge monitor?
- What happens to column state during dormancy?
- Can cascading updates happen within a macrostep, or do we need a refractory period?
- Does surprisal-gating interact with the async approximation in any useful way?
- Is this even the right framing, or does something else make more sense as a trigger?

## Status

Not yet investigated experimentally in this repo.
