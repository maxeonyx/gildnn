# Depth-Staggered Temporal Processing

## Objective
Investigate recurrent architectures where successive layers operate on different temporal offsets, allowing many time steps to propagate simultaneously through depth.

## Key Ideas
- Maintain per-layer queues of hidden states so that layer k receives input from time t-k.
- Use continuous-time or clockwork-style gating to regulate when layers advance their internal state.
- Explore convolutional temporal filters or learnable delays to blend information across offsets.

## Open Questions
- How should gradients flow when time steps overlap across depth without strict synchronization?
- What initialization or normalization strategies keep activations stable under staggered updates?
- Can hardware-friendly scheduling policies map this structure efficiently onto accelerators?
