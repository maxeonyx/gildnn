# Chain + Dynamic Depth — older architecture interpretation

> **⚠ Superseded architecture — historical record**
>
> This experiment was run before the architecture clarification in [dictation 2026-05-20-10](../../../dictations/2026-05-20-10.md). The "chain" here uses a chain of blocks with their own hidden structures — not single residual blocks on a shared `d_model` residual stream, which is the clarified current design. The dynamic-depth mechanism is evaluated on top of that older chain family.
>
> The results are valid evidence about dynamic depth on **this specific architecture**. They are not direct evidence about dynamic depth on the clarified single-residual-block design.

## Question

Does adaptive depth add anything useful on top of the current predictive-chain family when everything is re-run on the standardized `~186K params / ctx=32 / 100K train / 20K val / AdamW / best-val` frame?

## Phase boundary

This phase is about the current predictive-chain family, not a claim that the full cortical-column architecture has been validated. The existing chain implementation still uses a simplified global readout over chain states. That simplification stays fixed here so the only intended architectural delta in Stage 2 is adaptive depth.

## Hypotheses

1. **Additive benefit:** chain + dynamic depth reaches lower best validation cross-entropy than the plain standardized chain at similar parameter count and similar wall-clock cost.
2. **Compute allocation benefit:** chain + dynamic depth matches the plain standardized chain's quality while using less effective depth / less wall-clock compute.
3. **No benefit:** chain + dynamic depth is equal or worse on quality and slower, so adaptive depth is redundant or harmful inside this chain family.
4. **Inconclusive benefit:** any gain is too threshold-sensitive or unstable to claim as a reliable improvement.

## Comparison targets

| Model | Params | Best Val Loss | Final Val Loss | Training Time | Status |
|---|---:|---:|---:|---:|---|
| Transformer | 186,805 | 1.632 | 1.648 | 73s | fixed trust anchor |
| Vanilla RNN | 186,125 | 1.706 | 1.720 | 78s | fixed trust anchor |
| Plain predictive chain | 186,237 | 1.676 | 1.676 | 528s | Stage 1 complete |
| Predictive chain + dynamic depth | pending | pending | pending | pending | Stage 2 |

## Stage 1 anchor

The standardized plain-chain re-run is now in place. On the fixed frame it reached best validation loss `1.675908` at epoch `13`, with final validation loss also `1.675908`, at `186,237` parameters and about `528s` wall-clock training time.

## Interpretation rule for this question

- better quality at similar compute -> additive benefit
- similar quality at lower compute -> useful adaptive allocation
- same or worse quality at higher compute -> redundant or harmful
- unstable threshold-sensitive result -> mechanism may be real but operationally weak

## Scope kept open on purpose

- whether the current chain matches Max's actual architecture vision
- whether first-node-only readout should replace the current simplified readout
- whether a standalone standardized dynamic-depth rerun is needed to resolve ambiguity
- whether adaptive depth matters more for later async/selective-update work than for immediate validation loss
