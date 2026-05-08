# Question: Broadcast Router

## What we're asking

What should the global communication channel between cortical columns look like? Is a single stateful broadcast component (reads all columns, outputs one mixed message to all) the right design? Or all-to-all attention? Or something else?

## Current state of knowledge

- Max's intuition: a central stateful router that aggregates all column states and broadcasts back a single mixed signal. Possibly many broadcast channels.
- All-to-all attention is an alternative worth trying
- If all-to-all is available, local graph structure may become irrelevant — this is a key risk to test for
- The roles of graph edges vs global channel are not settled

## Open sub-questions

- Does a broadcast router actually do useful aggregation, or does it just average?
- How many broadcast channels are needed?
- If we compare broadcast vs all-to-all vs graph-only, which performs better and why?
- Does the global channel make the graph structure redundant?

## Status

Not yet investigated experimentally in this repo.
