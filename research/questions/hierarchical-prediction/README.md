# Hierarchical prediction via bottleneck latents

Serves [dictation 2026-05-23-5](../../../dictations/2026-05-23-5.md): "Block one tries to predict the next word, but block two tries to predict the features that predict the distribution over the next word."

## Status

**NULL (on wrong architecture).** Tested with `token_injection="all"` — every block saw tokens directly. Per [dictation 2026-05-23-7](../../../dictations/2026-05-23-7.md), only block 0 should receive tokens; higher blocks depend on lateral propagation. When every block already has the answer, *of course* predicting features between blocks adds nothing.

This result may reverse on the corrected architecture (`token_injection="block0"`), where higher blocks genuinely depend on lower blocks for information and predictive objectives could help form the inter-block protocol. See `research/questions/local-learning/README.md` for the re-test plan.

## Results

| Seed | A: baseline | B: probe-only | C: hierarchical | C − A |
|------|-------------|---------------|-----------------|-------|
| 42 | 1.738 | 1.734 | 1.734 | -0.005 |
| 43 | 1.761 | 1.755 | 1.755 | -0.006 |
| 44 | 1.725 | 1.743 | 1.744 | +0.019 |
| **Avg** | **1.741** | **1.744** | **1.744** | **+0.003** |

Artifact: [`experiments/fixed_multi_rate/artifacts/hierarchical_prediction/`](../../../experiments/fixed_multi_rate/artifacts/hierarchical_prediction/)

The hierarchical loss does learn — dropping from ~0.94 (random cosine distance) to ~0.41 — proving that slow blocks CAN predict fast blocks' future features. But this predictive ability doesn't translate into better task loss. The probes remain healthy (no collapse; per-dimension variance ∈ [0.12, 0.27]).

## Architecture

Standard `ParallelDiagonalModel` (4 blocks, rates [1,2,4,8], d=256, readout_mode="all") with auxiliary-only additions:

- **Anchored bottleneck latents:** Per-block `z_i(t) = normalize(P_i(LN(h_i(t))))`, `z_i ∈ R^32`
- **Local next-token probes:** Per-block `logits_i = U_i(z_i)` → CE against next token. Anchors the latents.
- **Hierarchical cosine prediction:** For adjacent pairs (1→0, 2→1, 3→2), predictor head outputs predicted future latents. Loss = `1 - cos(predicted, stopgrad(target))`.

Three conditions:
- **A:** Task loss only (baseline)
- **B:** Task + 0.1 × probe loss (tests whether latent probes alone help)
- **C:** Task + 0.1 × probe + 0.1 × hier (tests whether hierarchical prediction helps)

## Interpretation

Two prior experiments produced the same pattern: **auxiliary prediction objectives learn their target but don't improve the main task.**

| Experiment | Aux learns? | Task benefit? |
|---|---|---|
| Self-prediction (cosine alignment) | ✓ aux loss drops 3× | ✗ zero task effect |
| Hierarchical future-latent prediction | ✓ hier loss drops 0.94→0.41 | ✗ zero task effect |

The blocks already specialize naturally from the shared adjoint gradient (dL/dS) alone. Adding explicit predictive objectives doesn't help because:

1. The task loss already provides sufficient gradient signal for multi-rate specialization
2. At this scale (TinyShakespeare, 250K tokens, ctx=32), the model memorizes the data regardless — auxiliary regularization doesn't help generalization
3. The bottleneck latents may not capture the "right" features for the main task

## What this doesn't settle

- Whether hierarchical prediction would help on a **larger dataset** where generalization matters more
- Whether a **top-down injection** (not just aux loss, but actually feeding predictions back into lower blocks) would change the dynamics
- Whether the concept is sound but the **operationalization** (32-d bottleneck, cosine loss, linear probes) is too weak
- Whether predictive processing requires **closed-loop** interaction (predict → correct → predict), not just one-way prediction

## Non-goals

- Does not test the "graph architecture" question (separate direction)
- Does not test local learning (already confirmed separately)
- Does not test scaling to larger data/context

## Next steps (deferred)

If revisiting this direction:
1. Try on a **larger dataset** where overfitting isn't the ceiling
2. Try **top-down injection** — feed slow-block predictions into fast-block inputs (closed-loop predictive processing, not just auxiliary loss)
3. Try **contrastive/InfoNCE** instead of cosine distance (stronger anti-collapse, might surface different features)
