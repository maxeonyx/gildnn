# Graph topology for the multi-rate architecture

Serves [dictation 2026-05-23-5](../../../dictations/2026-05-23-5.md): "Originally my vision is a graph. I don't know if that graph helps, but I kind of think it might."

## Status

**Theory only.** No experiments run. Three concrete alternatives designed, ready to implement.

## The question

Does graph topology matter for the parallel multi-rate architecture? The current design is a linear chain — block i reads block i-1's previous output. Max's original vision was a graph, "especially dense around the lower level features," with "multiple different modules that have different, that have overlapping rates."

Two sub-questions:

1. Does denser fast-module connectivity improve feature quality?
2. Do overlapping same-rate modules (multiple modules at the same rate with different neighborhoods) capture different feature families?

## Current architecture

Four blocks in a chain. Each reads only its left neighbor's previous-timestep output via a scaled diagonal connection:

```
Block 0 ──→ Block 1 ──→ Block 2 ──→ Block 3
rate=1       rate=2       rate=4       rate=8
```

Neighbor table:

| Block | Rate | Reads from |
|-------|------|------------|
| 0     | 1    | (none)     |
| 1     | 2    | Block 0    |
| 2     | 4    | Block 1    |
| 3     | 8    | Block 2    |

All blocks write additively to a shared residual stream. The diagonal connection is the only inter-block forward path.

Total params at d=256: ~4 block-equivalents.

## Why topology is primarily a forward question

The [local learning result](../local-learning/) showed that lateral gradient terms are negligible — detaching all inter-block gradients costs nothing (Δ = -0.009 nats, favoring detach). The shared adjoint `dL/dS` carries all useful training signal regardless of graph shape.

This means: changing the graph changes what information each block *sees* during forward, but doesn't meaningfully change how it *learns*. We can evaluate topologies purely on forward-pass quality without worrying about training dynamics.

## Alternative topologies

### 1. Dense Lower Pyramid Lattice

More modules at faster rates, dense connectivity at the bottom:

```
Layer:   ┌─ F0 ─┐  ┌─ F1 ─┐  ┌─ F2 ─┐  ┌─ F3 ─┐
rate=1   │      │  │      │  │      │  │      │
         └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘
             ╲    ╱╲    ╱╲    ╱╲    ╱
              ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱
Layer:        M0       M1
rate=2        │        │
              └───┬────┘
                  │
Layer:           S0
rate=4           │
                 │
Layer:           H0
rate=8
```

8 modules: 4×rate=1, 2×rate=2, 1×rate=4, 1×rate=8.

Neighbor table:

| Module | Rate | Reads from          |
|--------|------|---------------------|
| F0     | 1    | (none)              |
| F1     | 1    | F0                  |
| F2     | 1    | F1                  |
| F3     | 1    | F2                  |
| M0     | 2    | F0, F1              |
| M1     | 2    | F2, F3              |
| S0     | 4    | M0, M1              |
| H0     | 8    | S0                  |

Param budget: each module gets 0.5× the per-block params of the current architecture (total = 4 block-equivalents). Fast modules are small but numerous.

**Tests:** Does dense fast-feature lateral exchange improve quality? Does having many small fast modules beat few large ones?

### 2. Overlapping-Rate Diamond Graph

Multiple modules at the same rate, each with different neighborhoods:

```
        ┌── A1 ──┐      ┌── B1 ──┐
rate=1  │        │      │        │
        │    ┌── A2 ────┤── B2 ──┘
rate=2  │    │          │
        └─── A4 ────────┘
rate=4       │
             H
rate=8
```

7 modules with overlapping rates:

| Module | Rate | Reads from   |
|--------|------|--------------|
| A1     | 1    | (none)       |
| B1     | 1    | A1           |
| A2     | 2    | A1, B1       |
| B2     | 2    | B1, A2       |
| A4     | 4    | A1, A2, B2   |
| B4     | 4    | B1, A2, B2   |
| H      | 8    | A4, B4       |

Param budget: each module gets ~0.57× the current per-block params (total ≈ 4 block-equivalents).

**Tests:** Do overlapping same-rate modules with different neighborhoods specialize into different feature families? This directly tests Max's "overlapping rates" concept — same fire-rate, different input neighborhoods, potentially different specializations.

**Recommended first test.** Smallest conceptual jump from the current chain. Directly addresses the dictation's core idea.

### 3. Slow-Hub Prior Graph

Star topology — fast modules all read from slow hubs, no fast-fast lateral connections:

```
         F0    F1    F2    F3
rate=1    \    |    |    /
           \   |    |   /
            ╲  │    │  ╱
              M0
rate=2        │
              S0
rate=4        │
              H0
rate=8
```

7 modules: 4×rate=1, 1×rate=2, 1×rate=4, 1×rate=8.

| Module | Rate | Reads from |
|--------|------|------------|
| F0     | 1    | M0, S0, H0 |
| F1     | 1    | M0, S0, H0 |
| F2     | 1    | M0, S0, H0 |
| F3     | 1    | M0, S0, H0 |
| M0     | 2    | F0, F1, F2, F3 |
| S0     | 4    | M0         |
| H0     | 8    | S0         |

**Tests:** Is the useful inter-block information just "slow context broadcast to fast modules"? If this matches or beats the chain, the value of hierarchy is purely top-down priors, not lateral feature exchange.

## Discriminating experiment

**Setup:** Any alternative vs 4-block chain baseline, matched total params (d scaled per module to hold param count constant), ctx=32, TinyShakespeare, 20K steps, 3 seeds.

**Recommended first comparison:** Overlapping-Rate Diamond (alt 2) vs current chain.

**Decision rules:**

- If diamond ≈ chain: topology doesn't matter at this scale. Graph complexity not worth pursuing until we're past TinyShakespeare.
- If diamond > chain: overlapping rates help. Follow up with pyramid and hub to isolate whether it's the overlap or the density.
- If diamond < chain: the chain's simplicity has value (fewer lateral reads = less noise?). Try the hub graph as an even simpler topology.

**Implementation notes:** The main change is generalizing `diagonal_input = previous_timestep_deltas[block_index - 1]` to read from a configurable neighbor list. Each module's forward gets `[previous_timestep_deltas[j] for j in neighbors[i]]`, aggregated (sum or mean — test both).

## Non-goals

- Doesn't test scalability. TinyShakespeare has a low ceiling; topology benefits may only show at larger scale.
- Doesn't test context > 32. Rate spacing matters more with longer sequences.
- Doesn't address hierarchical prediction (separate question — how higher blocks predict lower block features).
- Doesn't settle the "how many blocks" question.

## Epistemic note

Max explicitly said "I don't know if that graph helps." This is uncertain. The alternatives are worth testing, but should be compared fairly against the chain — not assumed better. The chain is simple, and simplicity has real value in a system that's already doing unusual things (multi-rate, shared stream, diagonal propagation).
