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

### Set-transformer baseline (228K params, 41s training)

| Context patches | MSE | PSNR |
|-----------------|-----|------|
| 8 | 0.0521 | 12.83 |
| 16 | 0.0512 | 12.91 |
| 32 | 0.0516 | 12.87 |
| 48 | 0.0513 | 12.90 |

Reference baselines: zero-prediction MSE ~0.088, mean-patch MSE ~0.078. The set-transformer is meaningfully better than trivial.

**Notable:** context fraction barely matters (8 patches → 48 patches gives only -0.001 MSE improvement). This suggests either (a) MNIST is too simple — positional priors dominate, or (b) the model isn't fully utilizing context.

### Residual-stream-across-time probe (229K params, 720s training)

| Context patches | MSE | PSNR |
|-----------------|-----|------|
| 8 | 0.0616 | 12.10 |
| 16 | 0.0607 | 12.17 |
| 32 | 0.0631 | 12.00 |
| 48 | 0.0624 | 12.05 |

**Negative.** The recurrent model is:
- **Worse on quality:** +0.010 MSE gap across all context fractions
- **Much slower:** 18x slower training (720s vs 41s)
- **Same flat context curve** — doesn't benefit more from additional patches

Artifacts: [`experiments/image_patches_baseline/artifacts/`](../../../experiments/image_patches_baseline/artifacts/), [`experiments/image_patches_probe/artifacts/`](../../../experiments/image_patches_probe/artifacts/).

---

## Interpretation

The hypothesized advantage of the recurrent model — "builds up scene state incrementally, better for streaming/ordering" — does **not** appear on this task. The set-transformer's bidirectional attention is simply the right inductive bias for "what goes at position (r,c) given these other patches?"

The flat context curve in BOTH models is the most interesting finding: going from 8 to 48 context patches barely helps either model. This means:
1. MNIST positional priors dominate — the model learns "what's typical at (row, col)" regardless of context
2. The task is too easy at this dataset / patch size to discriminate architectures on context utilization

The recurrent model's slowness is inherent: processing 49 tokens sequentially (worst case) vs the set-transformer's single parallel forward pass. This disadvantage would only grow with more patches.

## What this tells us about the architecture

Combined with the [text results](../residual-stream-across-time/README.md) (+0.071 gap on 900k text):

The residual-stream-across-time architecture is **strictly dominated** by transformers on both tasks tested:
- Text: worse quality at matched params
- Images: worse quality AND 18x slower

The architecture's theoretical advantages (pipelining, streaming, local learning) have not translated into measurable improvements on ANY discriminating metric in ANY experiment.

---

## Open questions

- **Would a harder dataset (CIFAR-10) show different relative performance?** Possibly, but unlikely to reverse the 18x speed gap.
- **Would the recurrent model benefit from a truly sequential reveal task** (where later patches depend on which earlier patches were seen)? Possibly — but that's a different task formulation than Max specified.
- **Is there a task where streaming processing genuinely helps?** Unknown. Neither text nor image patches have shown one.

---

*Last updated: 2026-05-22. Baseline and probe results.*
