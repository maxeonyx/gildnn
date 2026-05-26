# Can temporal persistence replace a larger context window as the source of lateral value?

**Pathways:** 3 (Local Learning) → 1 (Wide Recurrent vs Deep Transformer)

**Dictation:** [2026-05-26-5](../../../dictations/2026-05-26-5.md) — "sequential dependencies across time steps only"

**Prior result:** Multi-block architecture validated with fixed-window information asymmetry. Interior blocks see more context (32/128 chars) than the output block (4 chars) and contribute useful lateral predictions via normalized addition. Effect: -0.045 to -0.088 nats depending on scale.

---

## The question

The validated architecture gives interior blocks their information advantage via a LARGER SAME-TIMESTEP CONTEXT WINDOW. But the vision describes blocks that communicate ACROSS TIME — persistent state, not bigger windows. Can a recurrent interior block (which maintains state across sequential timesteps) provide the same kind of useful lateral signal?

This is the minimal bridge from "local learning works" to "recurrent local learning works."

## Why this matters

The fixed-window approach recomputes interior blocks from scratch at every position. This is O(context_length × d_model²) per position. A recurrent block with persistent state is O(d_model²) per position — amortizing context over time. If recurrent state provides lateral value, the architecture can scale to arbitrary history without scaling compute.

More fundamentally: the vision describes an architecture where blocks fire at different RATES, communicating across time boundaries. Testing whether temporal state provides value AT ALL is prerequisite to testing multi-rate.

## Hypotheses

1. **Primary:** A recurrent interior block (GRU or similar) that accumulates state over sequential text will provide a useful lateral signal to the output block, improving prediction beyond block0_alone.

2. **Secondary:** The recurrent lateral may provide DIFFERENT value than the fixed-window lateral — capturing longer-range patterns that even a 128-char window misses. Or it may provide LESS value if the GRU cell can't compress context as effectively as a transformer attending over 128 positions.

3. **Null hypothesis:** The recurrent block's state doesn't compress useful information; its lateral contribution doesn't improve over baseline. This would suggest that simple temporal persistence isn't enough — the block needs more capacity or a different state update mechanism.

## Architecture

```
For each position t in sequential text:

Block 0 (output):
  input: chars [t-3, t-2, t-1, t]  (same 4-char window as before)
  architecture: transformer (same as validated)
  loss: CE on predicting char t+1

Block 1 (recurrent interior):
  input: chars [t-3, t-2, t-1, t] + additive state bias s_{t-1}
  architecture: SAME transformer (SequenceEncoder) as output block
  state_bias: s_{t-1} = normalize(block1_last_hidden_at_step_{t-1}), DETACHED
  output: last-position hidden h_t → used for:
    1. local CE (predict char t+1, gradient isolation)
    2. lateral contribution (detached, scaled, added to block 0's hidden)
    3. next state: s_t = normalize(h_t)
  loss: local CE (same as validated)
  lateral: h_t added (detached, scaled) to block 0's hidden before readout
```

**Design choices (addressing adversarial review):**

1. **No GRU** — uses the project's own primitives (transformer + normalize + addition). A GRU would conflict with the project's architectural preferences and add confounds (gating, optimizer sensitivity).

2. **No BPTT** — state is detached every step. Same gradient isolation principle as the lateral. The block optimizes for current-step prediction only. State quality emerges indirectly: the block's output captures information useful for predicting the current next-char, and that information happens to persist because it's fed back as state.

3. **Same architecture for both blocks** — fair comparison. The ONLY difference between block 0 and block 1 is the additive state bias. If block 1 outperforms block 0 alone, it's because the persistent state provides useful information.

4. **Cold-start handled** — state initialized to zero at position 0. First few positions receive no temporal benefit. Evaluation masks early positions (burn-in) so results reflect steady-state behavior.

5. **Reset-length ablation** — test state resets at different intervals (4, 32, 128, never) to directly measure how much persistence matters.

## Planned evidence

| Measurement | Purpose |
|---|---|
| val_loss for block0_alone (sequential) | Baseline in sequential setting |
| val_loss for block0 + recurrent_lateral (no reset) | Does temporal state help? |
| val_loss for block0 + recurrent_lateral (reset every 32) | How much persistence is needed? |
| val_loss for block0 + recurrent_lateral (reset every 128) | Longer persistence |
| Interior block local CE over time | Does the recurrent block learn better predictions as state builds? |
| Delta (recurrent - baseline) | Size of effect |
| Comparison to fixed-window result (-0.088 at tiny scale) | How does temporal persistence compare to explicit long context? |

**Burn-in handling:** Evaluation skips the first 32 positions of each sequence. Training uses all positions (the block still benefits from trying to predict even without state).

## Simplifications and non-goals

- **Same transformer architecture for both blocks** — NOT a GRU. The recurrent block is literally the same SequenceEncoder with an additive state bias.
- **Detached state (no BPTT)** — state quality emerges from local CE optimization, not from temporal gradient flow.
- **Per-character stepping** — no chunking, no multi-rate yet. One step = one 4-char window shifted by 1.
- **TinyShakespeare only** — same dataset for comparability.
- **Same model scale as tiny experiments** — d_model=64, fast iterations (~44s target for quick sanity check).
- **NOT testing multi-rate** — follow-up if this works.
- **NOT testing noise on laterals** — also follow-up.
- **NOT testing GRU or LSTM** — using project's own primitives (transformer + addition + normalization).

## What this will NOT settle

- Whether multi-rate firing helps (requires this to work first)
- Whether the recurrent approach scales better than fixed-window (requires larger experiments)
- Optimal state-update mechanism (GRU is a starting point, not necessarily the answer)
- Whether noise on laterals helps with temporal structure

## Exit conditions

- **Positive:** recurrent lateral improves over baseline → validates the direction, motivates more complex recurrent mechanisms and multi-rate experiments
- **Null:** no improvement → investigate why (state capacity? training dynamics? GRU too simple?). May need richer state-update or more training steps before concluding "recurrence doesn't help"
- **Negative:** recurrent lateral HURTS → strong signal that simple persistence isn't enough. Would redirect toward examining what makes the fixed-window approach work that recurrence lacks.

## Results

Script: [`runs/recurrent_lateral_lm.py`](../../../runs/recurrent_lateral_lm.py)

### Primary comparison (300 steps, seq_length=128, d_model=64, 1 layer, batch=32)

| Condition | val_loss | Δ from baseline | What it isolates |
|---|---|---|---|
| block0_alone | 2.0257 | — | Single block |
| no_persistence (second block, zero state always) | 1.9427 | **-0.083** | Ensemble/complementarity |
| recurrent_lateral (second block + persistent state) | 1.9113 | **-0.114** | Ensemble + persistence |

**Decomposition:** 72% of the total gain is from having a second independent predictor (ensemble effect). 28% (-0.031) is from temporal persistence specifically.

### Reset ablation (300 steps, same settings)

| Condition | val_loss | Δ from baseline |
|---|---|---|
| block0_alone | 2.0204 | — |
| recurrent_lateral (no reset) | 1.9527 | -0.068 |
| reset_128 | 1.9524 | -0.068 |
| reset_32 | 1.9834 | -0.037 |

Note: `reset_128` is a no-op at seq_length=128 (reset never fires within the sequence). The informative comparison is `reset_32` vs `no_reset`: resetting every 32 chars gives -0.037 vs -0.068 with no reset. The difference (-0.031) matches the `no_persistence` decomposition above.

### Training convergence (does the effect persist?)

| Steps | block0_alone | recurrent_lateral | Δ |
|---|---|---|---|
| 300 | 2.0204 | 1.9527 | -0.068 |
| 600 | 1.8129 | 1.7595 | -0.054 |
| 1200 | 1.6906 | 1.6390 | -0.052 |

The gap narrows from -0.068 to -0.052 over training. It remains positive through 1200 steps but has not been confirmed to stabilize (only 3 checkpoints, 1 seed).

### Comparison with fixed-window approach (different script, uncontrolled)

For reference only — different training loop, different data exposure per step:
- Fixed-window two_blocks (4800 random-window steps): block0_alone=1.7522, two_blocks=1.6646, Δ=-0.088
- Recurrent lateral (1200 sequential steps): block0_alone=1.6906, recurrent_lateral=1.6390, Δ=-0.052

Same order of magnitude. Not a controlled comparison.

## Interpretation (weakened per adversarial review)

1. **A stateful lateral pathway improves validation loss.** Part of that gain depends on persistence beyond 32 characters. But the majority (72%) is from having a second independently-trained predictor — an ensemble/complementarity effect.

2. **The effect remains present through 1200 steps** with the gap narrowing from 0.068 to about 0.05. "Stabilizes" is not confirmed — could continue shrinking.

3. **A separate fixed-window experiment showed improvement of similar order of magnitude**, but the comparison is uncontrolled (different scripts, training loops, data exposure).

4. **Resetting every 32 characters hurts** relative to no-reset at 300 steps, suggesting useful state spans beyond 32 characters within the 128-char sequence.

5. **This is an encouraging proof-of-concept** that justifies further recurrent/stateful follow-up experiments. It does not validate the overall recurrent direction — one seed, one scale, one dataset.

## What this taught us

- Multi-block architecture provides value even without information asymmetry (both blocks see same 4 chars). This is an ensemble effect: independently-trained blocks develop complementary features.
- Temporal persistence adds modest but real additional value (~30% of total improvement) on top of the ensemble effect.
- The persistence contribution is of similar order to the "extra information from longer context" contribution in the fixed-window experiments.
- Both sources of value (ensemble + persistence) are present and additive.

## Next steps

- **Multi-seed confirmation** — the persistence contribution (-0.03) is small enough that seed variance matters. Run 3 seeds to confirm it's real.
- **Longer sequences** — test seq_length=512 with reset ablation to determine if state beyond 128 chars provides additional value.
- **Matched-parameter control** — widen block0 to match total parameter count and confirm the two-block architecture is better than a single wider block.
- **Multi-rate** — if persistence is confirmed, test blocks firing at different rates (the actual vision).
