# Hierarchical prediction via bottleneck latents

Serves [dictation 2026-05-23-5](../../../dictations/2026-05-23-5.md): "Block one tries to predict the next word, but block two tries to predict the features that predict the distribution over the next word."

## Status

**ACTIVE — closed-loop variant running.** Previous aux-only test was NULL (on wrong architecture). The critical gap was: aux-only prediction learns its target but doesn't affect the forward pass. A new closed-loop experiment feeds predictions BACK into block 0's computation, giving block 1 a forward role. Running now on WikiText-103, ctx=128.

## Phase 1: aux-only (NULL)

Tested with `token_injection="all"` (wrong architecture per [dictation 2026-05-23-7](../../../dictations/2026-05-23-7.md)). Every block saw tokens directly; predicting features between blocks added nothing.

| Seed | A: baseline | B: probe-only | C: hierarchical | C − A |
|------|-------------|---------------|-----------------|-------|
| 42 | 1.738 | 1.734 | 1.734 | -0.005 |
| 43 | 1.761 | 1.755 | 1.755 | -0.006 |
| 44 | 1.725 | 1.743 | 1.744 | +0.019 |
| **Avg** | **1.741** | **1.744** | **1.744** | **+0.003** |

The hierarchical loss DOES learn (0.94 → 0.41 cosine distance) — blocks CAN predict each other's future states. But this doesn't improve the task.

Artifact: [`experiments/fixed_multi_rate/artifacts/hierarchical_prediction/`](../../../experiments/fixed_multi_rate/artifacts/hierarchical_prediction/)

## Why aux-only is insufficient (diagnosis from 6 subsequent experiments)

The spectator problem on corrected architecture (`token_injection="block0"`) is NOT about:
- Gradient signal (aux losses proved blocks can learn, +0.003)
- Information access (temporal window k=8 with learned projection, +0.003)
- Readout competition (equal readout HURTS +0.024)
- Short context (ctx=128 makes it WORSE +0.014)

**Root cause:** Same-objective identical blocks with the corrected architecture have no reason to specialize. Block 0 already does the task optimally because it touches every token. More information for upper blocks doesn't help because they have nothing DIFFERENT to do with it.

Any auxiliary loss that doesn't affect the forward pass is just "blocks learn something on the side" — the spectator problem is a forward-pass architecture problem, not a training signal problem.

## Phase 2: closed-loop hierarchical prediction (RUNNING)

**Key difference from Phase 1:** Block 1's predictions are fed BACK into block 0's computation via MixAdd. Block 0 learns to USE the predictions. This gives block 1 a genuine forward role — block 0's output depends on block 1's prediction quality.

### Architecture

Two blocks: block 0 (rate=1), block 1 (rate=2). Block 0 only sees tokens.

- Block 1 receives block 0's state, processes via FFN, produces prediction of block 0's next 2 states
- Predictions stored in a 2-slot buffer
- At each timestep, block 0 receives the prediction for THIS timestep via a learnable MixAdd gate
- Block 0 computes: `x0 = prediction_mix(seed0, prior)` — learns how much to trust the prediction

### Loss

- Block 0: standard CE (next-token from final state)
- Block 1: cosine distance on LayerNorm'd states: `1 - cos(LN(predicted), LN(stopgrad(actual_state0)))`
- Total: `CE + 0.1 * pred_loss`

### Controls

- A_single: 1 block only (baseline)
- B_spectator: 2 blocks, shared CE, weighted readout (reconfirms spectator)
- C_closed_loop: 2 blocks, block 1 predicts, predictions fed back

### Early signal (100-step sanity check)

```
C_closed_loop:
  val_loss:          3.233
  ablated_val_loss:  4.130  ← predictions zeroed out
  pred_loss:         0.50 → 0.19 (decreasing)
  prediction_mix:    0.900

A_single:
  val_loss:          3.222
```

The ablation gap (+0.90 nats) proves block 0 IS USING the predictions — first time any upper block has shown clear forward contribution in this project. Full 20K-step results pending.

### What results would mean

- **C < A:** Closed-loop hierarchical prediction makes multi-block genuinely useful. Upper blocks have a valid forward role as "dynamics predictors." Opens path to local learning.
- **C ≈ A > B:** Prediction prevents spectator harm but no benefit. Mechanism works but doesn't help.
- **C ≈ B:** Closed-loop not enough. May need structural change.

## What this question settles (if positive)

- That the corrected architecture CAN support useful multi-block learning — specifically through role differentiation (different objectives per block)
- That blocks don't need to see tokens directly to be useful — they can model dynamics instead
- That the spectator problem is solvable through architectural means (closed-loop feedback), not just training signal tweaks

## What this doesn't settle

- Whether this extends to many blocks (4, 8, 16+)
- Whether stop-grad local learning can work on top of this
- Whether the prediction quality is sufficient for the mechanism to help at convergence (vs only helping transiently)
- Whether the parameter overhead (prediction head, MixAdd gate) is justified vs adding those params to block 0 directly
