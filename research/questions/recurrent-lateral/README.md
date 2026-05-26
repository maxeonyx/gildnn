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

_(to be filled)_

## Next steps

_(to be filled after results)_
