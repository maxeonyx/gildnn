# Question: Broadcast Router

## What we're asking

What should the global communication channel between cortical columns look like? Is a single stateful broadcast component (reads all columns, outputs one mixed message to all) the right design? Or all-to-all attention? Or something else?

## Current state of knowledge

- Max's intuition: a central stateful router that aggregates all column states and broadcasts back a single mixed signal. Possibly many broadcast channels.
- All-to-all attention is an alternative worth trying
- **Key architectural tension:** if all-to-all attention is expressive enough and cheap enough, local graph edges may become decorative — every column effectively has access to everything and graph locality no longer matters. Whether this is what happens needs to be tested explicitly, not assumed either way.
- A workspace/slot design (small bank of latent slots; columns write summaries, columns read summaries) would preserve a communication bottleneck and may better match the original vision of meaningful graph locality.

## Open sub-questions

- Does a broadcast router actually do useful aggregation, or does it just average?
- How many broadcast channels are needed?
- If we compare broadcast vs all-to-all vs graph-only vs workspace slots, which performs better and why?
- Does the global channel make the graph structure redundant?
- What bandwidth/compression is right for the global path? (full vectors? projected? top-k?)

## Status

Not yet investigated experimentally in this repo.

