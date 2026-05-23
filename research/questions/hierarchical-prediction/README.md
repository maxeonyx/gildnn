# Hierarchical prediction via bottleneck latents

Serves [dictation 2026-05-23-5](../../../dictations/2026-05-23-5.md): "Block one tries to predict the next word, but block two tries to predict the features that predict the distribution over the next word."

## Status

**ACTIVE — detached closed-loop rerun running.** Previous aux-only test was NULL (on wrong architecture). Closed-loop v1 did give block 1 a forward role, but then collapsed catastrophically because the prediction loss rewarded block 0 for becoming predictable. V2 keeps the same experiment and controls, but detaches `s0` from block 1's input so the prediction loss stays local to block 1.

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

## Phase 2: closed-loop hierarchical prediction

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

### v1 early signal (100-step sanity check, before collapse)

```
C_closed_loop:
  val_loss:          3.233
  ablated_val_loss:  4.130  ← predictions zeroed out
  pred_loss:         0.50 → 0.19 (decreasing)
  prediction_mix:    0.900

A_single:
  val_loss:          3.222
```

The ablation gap (+0.90 nats) proves block 0 IS USING the predictions — first time any upper block has shown clear forward contribution in this project. This was real, but it did not survive training.

### V1 result: catastrophic collapse

Full 20K-step training showed the closed-loop mechanism collapsing instead of helping.

| Condition | Val loss |
|---|---:|
| A_single | 1.670 |
| B_spectator | 1.664 |
| C_closed_loop | 3.142 |

Accuracy in `C_closed_loop` froze at `0.194` from step 1K onward. `pred_loss` went to ~0, not because block 1 learned rich dynamics, but because block 0 collapsed to an almost constant state that block 1 could predict trivially.

The failure mode was: block 1 initially helps, the auxiliary objective then finds an easier route than improving prediction quality, and the whole system falls into a degenerate regime where "be predictable" beats "be useful."

### Root cause analysis

In v1, block 1 was built from a live average of block 1's state and block 0's state:

`x1 = 0.5 * (s1 + s0)`

That leaves this gradient path alive:

```text
pred_loss
  → prediction_head
  → block1_ffn
  → x1
  → s0
  → block0_ffn
```

So the prediction loss was not actually local to block 1. It directly rewarded block 0 for producing states that were easier to predict. The easiest-to-predict state is a constant or near-constant one, so training created an attractor: block 0 drifts toward constant outputs, block 1 learns to predict that constant, `pred_loss` goes to zero, and CE can no longer recover useful token information.

### Fix: detach s0 from block 1 input

The fix is one line:

`x1 = 0.5 * (s1 + s0.detach())`

This cuts the bad path at `s0`. After the change, `pred_loss` still trains the prediction head and block 1's FFN, but no longer sends any "be predictable" pressure into block 0.

Gradient flow after the fix:

```text
pred_loss
  → prediction_head
  → block1_ffn
  → x1
  ✕ s0
```

This gives the mechanism the intended semi-local property:

- `pred_loss` is local to block 1
- `CE` is semi-local: it still trains both blocks through the closed-loop feedback path, because the prediction buffer `.copy_()` preserves autograd back into the prediction-producing path

So block 0 now learns only from task loss, while block 1 learns both to help CE indirectly and to make good local predictions.

### V2 (running)

V2 is the same experiment design, same controls, same dataset, same 20K-step run, with only the detach fix applied.

Expected behavior: no catastrophic collapse, block 0 learns normally from CE only, and block 1 is pressured to produce genuinely useful predictions rather than making block 0 constant.

### What results would mean

- **V2 C < A:** Detached closed-loop hierarchical prediction makes multi-block genuinely useful. Upper blocks have a valid forward role as dynamics predictors without collapsing the lower block.
- **V2 C ≈ A ≥ B:** The detach fix removes the collapse and spectator harm, but prediction feedback still does not buy much.
- **V2 C ≈ B:** Detaching fixes the failure mode but closed-loop prediction is still not enough to create a useful differentiated role.
- **V2 collapses again:** The bug was not just the gradient leak through `s0`; the mechanism likely has a deeper instability.

## What this question settles (if positive)

- That the corrected architecture CAN support useful multi-block learning — specifically through role differentiation (different objectives per block)
- That blocks don't need to see tokens directly to be useful — they can model dynamics instead
- That the spectator problem is solvable through architectural means (closed-loop feedback), not just training signal tweaks

## What this doesn't settle

- Whether this extends to many blocks (4, 8, 16+)
- Whether stop-grad local learning can work on top of this
- Whether the prediction quality is sufficient for the mechanism to help at convergence (vs only helping transiently)
- Whether the parameter overhead (prediction head, MixAdd gate) is justified vs adding those params to block 0 directly
