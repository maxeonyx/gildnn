# Arbitrary-Order Image Patches

**Question:** Can a recurrent residual-stream architecture handle arbitrary-order patch conditioning as well as (or better than) a set-transformer baseline? Is the ordering dimension where this architecture's advantages show up?

**Status:** Experiment designed. Implementation in progress.

**Grounded in:** [VISION.md](../../../VISION.md) (primary dataset specification), [dictations/2026-05-20-15.md](../../../dictations/2026-05-20-15.md) (architecture clarification)

---

## Why this experiment

The residual-stream-across-time architecture is [conclusively less efficient than transformers for text](../residual-stream-across-time/README.md) at matched parameters (+0.071 nats). But text is a task where full-sequence attention dominates — a recurrent model with limited temporal context is inherently disadvantaged.

Image patches in arbitrary order are a fundamentally different setting. The model must build up a "scene state" from evidence arriving in unpredictable order. This is closer to the streaming/accumulation setting where a residual stream across time might have a natural advantage — or at least not have the same disadvantage.

---

## Task formulation: query-conditioned patch prediction

Given an image split into patches:
1. Sample a **target patch** q
2. Sample a **context subset** C (some fraction of the remaining patches)
3. Present context in a **random or structured order**
4. Ask the model to predict the target patch's pixels

**Input sequence format:**
- For each context patch i: `(type=context, coord=(row, col), pixels=x_i)`
- Final token: `(type=query, coord=(row_q, col_q), pixels=MASK)`

**Output:** predicted pixel values of the target patch (from the query token position)

**Loss:** MSE on normalized [0,1] pixel values. Report PSNR for readability.

This directly tests "fill in missing patches given revealed context" — Max's stated inference use case.

---

## Dataset and patch scheme

**MNIST, zero-padded to 32×32.**

- Patch grid: 8×8
- Patch size: 4×4 = 16 scalars per patch
- Total patches per image: 64
- Patch coordinates: 2D normalized (row/7, col/7)

Why MNIST: fastest honest first test. Tiny models can learn something in minutes. 64 positions is enough for ordering to matter.

---

## Training protocol

Per example:
- Target: uniform random from 64 patches
- Context fraction: uniform from {12.5%, 25%, 50%, 75%} (8, 16, 32, 48 patches)
- Context patches: uniform random subset of the remaining 63
- Order: 70% random permutation, 10% raster, 10% reverse raster, 10% column-major

---

## Baseline: set-transformer

A small transformer with **no causal mask** — full bidirectional attention over all tokens:
- Patch-content embedding + 2D coord embedding + token-type embedding (context vs query)
- Reads from the query position for prediction
- No 1D positional embedding — ordering should be invisible to this model
- ~250K params

This is the honest comparison: a model that treats context as a **set** (order-invariant). If this wins, the task is mostly set-conditioning, not order-processing.

## Probe: residual-stream-across-time

Same token format, processed sequentially:
- Each context token updates the residual stream (mix-add, temporal attention over past states)
- Final query token reads the accumulated state and predicts

**Hypothesized advantage:** efficient streaming, builds scene state incrementally, compute cost scales with context size not total patches.

**Known disadvantage:** order-sensitive by construction. Different query/order requires replay from scratch.

---

## Evaluation

**Primary metrics:**
- Patch MSE / PSNR on held-out images
- Broken out by context fraction (8, 16, 32, 48 patches)
- Broken out by evaluation order family

**Discriminating tests:**
1. **Order sensitivity:** same context set, 5 different random orders → prediction variance
2. **Cross-order generalization:** train mixed, eval each family separately
3. **Wall-clock efficiency:** best error reached within fixed time budget
4. **Iterative reconstruction:** repeatedly query all missing patches → whole-image MSE

**Positive result for residual-stream:** at matched params and time budget, lower error on random-order evaluation, OR reaches same error faster, OR shows lower order-sensitivity variance.

---

## Deliberate simplifications

**Included:** single-patch query, MNIST, fixed 32×32, 4 order families, matched-param comparison.

**Excluded on purpose:** full autoregressive permutation likelihood, CIFAR-10, knight's-move order, variable image sizes, outward extension as primary metric. These are future work if the basic concept shows signal.

---

## Results

*(Pending — experiment in progress)*

---

*Last updated: 2026-05-22. Design phase.*
