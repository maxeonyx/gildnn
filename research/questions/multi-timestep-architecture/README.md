# Multi-Timestep Architecture: Block-Timestep Grid with Local Learning

**Pathways:** 1 (Wide Recurrent), 3 (Local Learning)
**Status:** Design exploration (from conversation 2026-05-25)

---

## Core idea

The architecture is a 2D grid. The horizontal axis is lateral distance from the input. The vertical axis is timesteps. The stream flows diagonally through this grid — laterally with no processing. Blocks sit on the temporal (vertical) edges, adding computation with 1-tick delay.

A block IS a temporal edge. Information goes in, the block processes it, it comes back to the same lateral position 1 timestep later. There are no blocks on the lateral path — the fastest route for a token's information to reach all positions is the unprocessed diagonal.

Multiple timesteps per token gives higher-level blocks the chance to send information via right-to-left connections (processed info flowing back toward input).

---

## Grid diagram

4 lateral positions, stride 2 (2 timesteps per token). All connections shown.

```
        L0         L1         L2         L3

  T0    ○══════╲
        │       ╲
       [B0]      ╲
        │         ╲
  T1    ○══════╲   ○══════╲
        │       ╲  │       ╲
       [B0]      ╲[B1]      ╲
        │         ╲│         ╲
  T2    ○══════╲   ○══════╲   ○══════╲                <-- x1 enters at L0
        │       ╲  │       ╲  │       ╲
       [B0]      ╲[B1]      ╲[B2]      ╲
        │         ╲│         ╲│         ╲
  T3    ○══════╲   ○══════╲   ○══════╲   ○
        │       ╲  │       ╲  │       ╲  │
       [B0]      ╲[B1]      ╲[B2]      ╲[B3]
        │         ╲│         ╲│         ╲ │
  T4    ○══════╲   ○══════╲   ○══════╲   ○           <-- x2 enters at L0
        │       ╲  │       ╲  │       ╲  │
       [B0]      ╲[B1]      ╲[B2]      ╲[B3]
        │         ╲│         ╲│         ╲ │
  T5    ○          ○          ○          ○
```

```
○          = stream state (node — where info sits between processing)
══════╲    = lateral flow (the residual stream, NO processing, fastest path)
│[Bk]│     = block k (temporal edge — processing happens HERE, adds 1 tick delay)
```

Each ○ receives two inputs:
1. The lateral flow from upper-left (═══╲ arriving from the adjacent position, unprocessed)
2. The block output from directly above (│[Bk]│, the processed contribution from the same position)

These are merged by a combining function (same everywhere — see principles below).

The fastest path from x0 entering at (L0, T0) to reaching L3 is 3 ticks along the diagonal — pure stream, no blocks. Block contributions trail behind: B0's processed output of x0 arrives at L1 at T2 (1 tick behind the raw stream), at L2 at T3 (2 ticks behind), etc.

**The stream is only purely unprocessed on the very first diagonal wavefront.** After that, each node holds a temporal superposition — differently-aged information at different levels of processing coexisting in the stream:
- Raw token info (just arrived via lateral, zero processing)
- Block output from 1 tick ago (1 level of processing)
- Contributions derived from 2 ticks ago (2 levels of processing)
- etc.

"Depth" of processing is encoded as temporal age in the stream. The combining function determines how these differently-aged contributions coexist — it could differ by dimension or direction. Don't over-specify; the goal is getting information flow right so that parallel processing falls out.

---

## What's on each edge

**Lateral (══╲):** The residual stream, flowing without processing. This is not a block's output — it's the stream itself moving to the next position. The diagonal is the natural rate of information propagation.

**Temporal (│[Bk]│):** A block. The block reads the stream at this position, computes on it (1 tick), and writes its contribution back to the same position. The combining function merges the block's output with the incoming lateral flow.

**Key insight:** there are no blocks on the lateral path. The fastest information route is pure stream. Blocks add richness (processed representations) but at the cost of delay. This means raw tokens arrive at distant positions BEFORE any processed representations do.

---

## Right-to-left connections (noted, not yet designed)

Block outputs could also propagate leftward (back toward the input). This would let processed representations from later positions feed back to earlier ones. We're leaving this out of the initial design but bearing it in mind — the grid structure naturally supports bidirectional lateral flow.

---

## Architecture principles

From Max's description (2026-05-25). Preserved at his level of specificity.

### What's clear

1. **Grid structure.** Lateral positions on the horizontal axis, timesteps on the vertical axis. The grid is the computation.

2. **Timesteps >= tokens.** There must not be fewer timesteps than tokens. There could be equal, or more.

3. **Blocks are temporal edges.** A block reads from the stream, processes, and writes back to the same position. This takes 1 timestep. That's all a block is — a temporal edge in the grid.

4. **Lateral = unprocessed stream.** The diagonal flow is the residual stream moving without any processing. It's the fastest path. No blocks on this path.

5. **Token injection at L0.** Raw tokens enter at lateral position 0 (closest to input). Interior positions get the token's information only via the lateral stream — it takes N timesteps for a token to reach position N.

6. **Extra timesteps are for right-to-left flow.** More timesteps per token isn't about giving the left-to-right stream time to arrive (that's already the fastest path). It's about giving the processed, right-to-left connections time to send information back toward the input.

7. **Two separate concerns:**
   - **Block internals:** learned LOCALLY. Each block trains via a local signal (predict what the left neighbor will send next). This is the "lifetime learning."
   - **Communication scheme (combining function):** how block output merges with the incoming lateral stream at each node. This is SHARED (tied weights across all positions and timesteps). Could be learned globally (backpropped) or fixed. Same scheme everywhere — "evolved, not learned per-block." Could differ by dimension or direction.

8. **The prediction is the loss, not the output.** The local prediction (of next input from left) provides the training signal. The block's actual output is its representation, shaped by that training pressure. What goes back into the stream is determined by the combining function.

9. **Information flow drives the design.** The combining function, the stride, the right-to-left connections — these are all about getting information flow right. If information flow is correct, parallel processing falls out naturally. The biological inspiration is a consequence of this (brains also do parallel processing), not the starting point.

### What's uncertain

- **Should interior positions also get the raw token?** There's an argument for raw input plus derived representations. This would ground each position independently. Not decided.

- **What exactly does a block produce?** It processes its inputs and produces output. The combining function handles how that merges with the stream. The specifics are TBD.

- **The combining function.** "Combined in some fashion." Don't over-specify. Could differ by dimension or direction. Could be learned (tied, backpropped) or fixed. The point is getting information flow right — the specifics are empirical.

- **Right-to-left flow.** Block outputs going back toward the input. Conceptually present but not yet designed. This is what the extra timesteps are FOR.

---

## Why this design

The motivation chain:

1. Want truly parallel blocks (Pathway 2: async execution)
2. Want blocks to learn independently (Pathway 3: local learning, no global backprop between blocks)
3. Need lateral propagation for information to reach all positions
4. The lateral flow is the RESIDUAL STREAM — unprocessed, no blocks on it, fastest path
5. Blocks sit on temporal edges — they process in-place, adding delay but also adding richness
6. Multiple timesteps per token so lateral info has time to reach all positions before the next token arrives
7. The "predict left neighbor" local loss grounds each block without requiring cross-block gradients
8. The shared combining function is the minimal global structure — same everywhere, like evolved wiring vs learned synapses

---

## Local objective: distributional predictive processing

The stream doesn't carry point vectors — it carries **distributions**. Each block also outputs a distribution. The combining function operates on two distributions, and "surprisal" is the geometrically meaningful distance between them.

### Both sides are distributions

- **Lateral input:** a distribution (the stream's state of knowledge arriving from the left)
- **Block output:** a distribution (what the block predicts/expects based on its state)
- **Combining function:** operates on two distributions to produce the stream state at this node

### What "surprisal" means

With two distributions, "how much new information arrived" is well-defined as the transport cost between them. Earth mover's distance (Wasserstein) is the correct metric here — it measures the actual geometric cost to reconcile the block's belief with what arrived.

For diagonal Gaussians (each side parameterized by μ, σ per dimension):

```
W₂² = ||μ_lateral - μ_block||² + ||σ_lateral - σ_block||²
```

Cheap, closed-form, differentiable, symmetric, always finite. No log, no division, no infinite values when supports don't overlap.

Why Wasserstein over KL:
- KL is asymmetric
- KL doesn't respect the geometry of the underlying space
- KL is infinite when supports don't overlap
- Wasserstein measures actual transport cost — "how much work to move from one belief to the other"

### What flows where

- **Rightward (the surprise):** The information gain — what the block's distribution DIDN'T already account for. The component of the lateral distribution that required updating the block's beliefs.
- **Leftward (the prior/context):** The block's distribution itself — what it expected. This is the processed, higher-level model flowing back toward the input.

### Local loss

Each block minimizes the Wasserstein distance between its predicted distribution and the actual lateral distribution. That's the local training signal — "become a better predictor of what arrives from the left." No cross-block gradients needed.

### Information bottleneck

No external bottleneck constraint (MMD, KL penalty) needed — the distributional framework IS the information theory. The variances encode confidence per dimension. Dimensions the block models well → tight distribution → low surprisal → little flows rightward. Dimensions it can't model → wide distribution → admitting ignorance → more flows through.

### Self-prediction connection (speculative)

If blocks maintain state across timesteps (mix previous output into themselves), a well-trained block that predicts accurately produces low surprisal → less flows rightward → the stream downstream goes quiet. This IS computation compression emerging from the architecture — no explicit self-prediction loss needed. The better the block's model, the less downstream processing is required.

### Open: the combining function

How exactly do two distributions get combined into the stream state + surprise + prior? Options:
- Product of experts (multiply, precisions add) for Gaussians
- Bayesian update (one is prior, one is likelihood)
- Learned decomposition with conservation constraint
- Something else entirely

The combining function is shared/tied across all positions and timesteps. It could be fixed (analytical, like product of Gaussians) or learned (with tied weights). Not decided — this is the key remaining design question.

### Experimentally tested (2026-05-27): stream closure

Piece tests in [`experiments/predictive_processing/`](../../experiments/predictive_processing/) tested three candidate "rightward operators" for how the stream evolves after a prediction is compared to the actual:

| Operator | Valid stream? | Silences on perfect prediction? | Stable under bad prediction? |
|---|---|---|---|
| Raw W₂ residual (μ_A-μ_P, σ_A-σ_P) | ❌ σ goes negative | ✅ exact zero | ❌ degenerate |
| Pass-through actual (stream unchanged) | ✅ always | ❌ no silencing | ✅ always stable |
| Precision-weighted residual | ✅ always | ✅ exact zero | ❌ explodes (mean energy →∞) |

**Key finding:** The W₂ residual decomposition is the correct ANALYSIS TOOL (for computing loss and understanding what was predicted vs surprising). But it CANNOT be the stream itself — it produces invalid distributions under chaining.

**Architectural implication:** The stream carries ACTUAL distributions (always valid). The surprise is a side-channel quantity computed for:
1. The local loss (W₂² between prediction and actual)
2. Routing decisions (what goes up to the next block)

The combining function at a node takes (lateral arrival, block output) and produces a new stream state. The surprise is DERIVED from the comparison, not injected back INTO the stream.

This resolves the "what flows where" question differently than originally framed: the stream itself is NOT the surprise. The stream is the lateral residual flow (always valid, always a proper distribution). Blocks READ from the stream, PREDICT the stream, and the LOSS is the mismatch — but the stream continues regardless.

### Open: covariance structure

In theory we want rotated ellipsoids (full covariance), not just axis-aligned. But full covariance Wasserstein requires O(d³) matrix square root.

Practical option: **shared learned basis.** The combining function includes a learned rotation R (tied everywhere). All blocks output diagonal (μ, σ) in this shared rotated basis. Wasserstein stays cheap (diagonal formula applies in that basis). The rotation R is part of the "evolved wiring" — defines what directions mean in the stream. Blocks adapt to it locally.

However: neural networks often learn their own internal basis regardless. The explicit rotation might be unnecessary overhead — a diagonal parameterization with sufficient capacity might achieve the same thing implicitly. This is an empirical question: compare shared-rotation vs plain diagonal and see if it matters.

---

## Connection to existing work

- **C_old ablation (COMPLETE — positive):** Lateral connections ARE load-bearing in the surrogate architecture (token_injection=all). Δ=+0.030, all 4 blocks contribute. This validates that blocks CAN communicate usefully through laterals. The multi-timestep design provides a theory for WHY they'd communicate (propagation delay creates information asymmetry) and WHAT they'd send (predictions of left neighbor's future output).

- **Predictive coding literature:** "Predict your left neighbor's next output" is essentially predictive coding. Each level predicts the activity of the level below/beside. Prediction errors drive learning.

- **Residual streams (transformers):** The lateral flow IS a residual stream — blocks read from it and write to it without gating the flow. Similar to how transformer layers read/write to the residual stream, but here the stream also flows laterally (across positions), not just vertically (through layers).

---

## Open questions (for future experiments)

1. Does "predict left neighbor" actually produce useful representations, or does it degenerate?
2. What's the minimum stride (timesteps per token) needed for full propagation given N positions?
3. Should the combining function be learned (tied, backpropped) or fixed?
4. At what scale does the local-only learning signal become insufficient vs global backprop?
5. Can this compose with multi-rate (different positions process at different speeds)?
6. What does right-to-left flow look like? Does it use the same combining function?
7. With stride > 1, does the extra processing time (more block applications before next token) substitute for deeper blocks?
