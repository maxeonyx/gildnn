# Can shared-weight recurrence match distinct-layer depth?

**Pathway:** 1 (Wide Recurrent vs Deep Transformer) — THE fundamental thesis
**Dictation context:** The entire project premise; see VISION.md
**Connection:** If recurrence works, everything else follows — dynamic depth, parallelism, local learning composition

## The question

A standard N-layer transformer has N sets of unique weights, each applied once. A recurrent transformer has 1 set of weights applied N times. At matched compute (same number of block applications per token), can recurrence match distinct-layer quality?

This is the simplest possible test of the project's core thesis.

## Why now

Pathway 3 (Local Learning) proved that locally-trained blocks work. Pathway 8 (Multi-Rate) is fully closed. The project needs to test its fundamental premise directly — does recurrence substitute for depth?

## Simplifications

- Char-level LM on TinyShakespeare (fast feedback, known baselines)
- d_model=128, ctx=128, matching our established baseline scale
- N=4 (enough iterations to be non-trivial, not so many that stability dominates)
- No laterals, no local learning, no multi-rate — pure recurrence test (isolation before composition)
- Separate LayerNorms per iteration (so normalization isn't the confound)

## Hypotheses

**H1 (optimistic):** Recurrent-4 matches distinct-4 within ~0.02-0.03 nats despite having 1/4 the parameters. Weight sharing is free or nearly free.

**H2 (pessimistic):** Recurrent-4 is significantly worse (>0.05 nats). Same weights can't learn diverse layer-specific features.

**H3 (instability):** Recurrent-4 training is unstable (gradient explosion through 4 iterations). Would need optimizer changes (Muon, orthogonal init).

## What this does NOT settle

- Whether recurrence works at larger scale (d=512+, many more iterations)
- Whether width can compensate for the gap (if any)
- Whether Muon/orthogonal parameterization improves stability at higher iteration counts
- How local learning (Pathway 3) composes with recurrence
- Whether the gap shrinks or grows with more training

## Planned evidence

| Measurement | Purpose |
|---|---|
| val_loss for both conditions | Primary comparison |
| Per-iteration eval loss (recurrent) | Does each iteration help? Diminishing returns? |
| Parameter count | Quantify the efficiency gain |
| Wall-clock time | Practical speed comparison |
| Gradient norms | Stability evidence |
| Activation RMS by iteration | Distribution health through iterations |

## Decision rule

| Gap (distinct - recurrent) | Interpretation | Next step |
|---|---|---|
| ≤ 0.03 | Strong evidence for Pathway 1 | Width scaling: can wider recurrent beat distinct at same params? |
| 0.03 – 0.05 | Moderate gap, worth investigating | Width scaling to see if more params closes the gap |
| > 0.05 | Significant tax on recurrence | Optimizer/stability investigation, or confidence in Pathway 1 drops |
| Unstable | Stability is the bottleneck | Muon, orthogonal init, gradient clipping experiments |

## Script

`runs/recurrent_depth_lm.py` — two conditions (distinct_4, recurrent_4), same training budget, same readout.

## Results

*(experiment running — PID 11360, log: `runs/recurrent_depth_run.log`)*
