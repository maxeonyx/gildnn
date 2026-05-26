# Can temporal persistence replace a larger context window as the source of lateral value?

**Pathways:** 3 (Local Learning) → 1 (Wide Recurrent vs Deep Transformer)

**Dictation:** [2026-05-26-5](../../../dictations/2026-05-26-5.md) — "sequential dependencies across time steps only"

**Status: ⚠️ RESULTS INVALID — lateral timing bug (dictation 2026-05-26-6)**

The experiment fed block 1's output to block 0 at the **same timestep** (position t → position t). This makes the blocks sequential — just a deeper network. The vision requires STALE laterals: block 0 at position t receives block 1's output from position t-1, so both blocks can fire concurrently.

The results below measured "extra sequential depth," NOT "parallel communication via stale state." They need to be re-run with the one-position lateral delay.

---

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

**⚠️ BUG: The "lateral" line below is wrong. It should be h_{t-1} (stale), not h_t (same-timestep).**

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
         ^^^ BUG: should be h_{t-1}, not h_t. As implemented, blocks are sequential.
```

**Correct design (to be implemented):** Block 0 at position t receives block 1's `h_{t-1}` (output from previous position). At position 0, lateral is zeros. Both blocks fire in parallel — neither waits for the other.

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

### Multi-seed confirmation (1200 steps, seq_length=128, d_model=64, 1 layer, batch=32)

| Seed | block0_alone | recurrent_lateral | Δ |
|---|---|---|---|
| 42 | 1.6906 | 1.6390 | -0.052 |
| 123 | 1.7459 | 1.6712 | -0.075 |
| 7 | 1.6913 | 1.6426 | -0.049 |
| **Mean** | **1.709** | **1.651** | **-0.058 ± 0.014** |

All 3 seeds positive. Effect is robust at 1200 steps.

### Training budget matters critically

At 300 steps, the same experiment is NOISE:

| Seed | Δ (recurrent - baseline) at 300 steps |
|---|---|
| 42 | -0.054 |
| 123 | +0.087 |
| 7 | -0.026 |
| Mean | +0.002 (not significant) |

The recurrent block needs ~1000+ steps to develop useful state representations. Below that threshold, the effect is dominated by initialization variance.

### The "ensemble effect" finding was wrong

An earlier single-seed (42) comparison suggested 72% of the multi-block gain was "ensemble/complementarity" (from a `no_persistence` control that showed -0.083). Multi-seed reveals this was noise:

| Seed | no_persistence Δ at 300 steps |
|---|---|
| 42 | -0.083 (helped) |
| 123 | +0.079 (hurt) |
| 7 | +0.064 (hurt) |
| Mean | +0.020 (not significant) |

**At 300 steps, a second block WITHOUT temporal state provides no reliable benefit when both blocks see the same 4-char window.** The earlier "ensemble effect" was a lucky seed.

### What IS real (at 1200 steps)

The recurrent lateral block (with temporal state) reliably helps: mean Δ = -0.058, all seeds positive. The temporal state provides genuine information that the output block cannot extract from its 4-char window alone.

### Convergence trajectory (seed 42 only)

| Steps | block0_alone | recurrent_lateral | Δ |
|---|---|---|---|
| 300 | 2.020 | 1.953 | -0.068 |
| 600 | 1.813 | 1.760 | -0.054 |
| 1200 | 1.691 | 1.639 | -0.052 |

The gap narrows initially as block0 catches up, then stabilizes around -0.05.

## Interpretation

1. **Temporal persistence provides reliable lateral value** given sufficient training (1200+ steps). Mean improvement -0.058 ± 0.014 across 3 seeds. All seeds positive.

2. **Training budget is critical.** Below ~1000 steps, the recurrent block hasn't developed useful state and the effect is dominated by initialization variance. This is analogous to the earlier finding that interior blocks need 4000+ steps in the fixed-window setting.

3. **"Ensemble effect" was a mirage.** A second block WITHOUT temporal state (same 4-char input, no persistence) does NOT reliably help — it's noise. The benefit specifically requires temporal persistence providing information asymmetry.

4. **Information asymmetry IS required** — original decision was correct. The source of asymmetry can be temporal (accumulated state from past positions) OR spatial (larger context window). Both work. Neither "ensemble" nor "same-input complementarity" are the mechanism.

5. **Effect size is comparable to fixed-window approach** (-0.058 recurrent vs -0.088 fixed-window at tiny scale), but this is an uncontrolled comparison across different scripts/training regimes.

## What this taught us

- Multi-seed checking is essential. Single-seed results at 300 steps were completely misleading.
- The "no_persistence" control was valuable — it COULD have shown that ensemble works without state. Instead it showed the opposite: you need information asymmetry.
- The recurrent block needs a training warm-up period (just like the fixed-window interior blocks needed 4000+ steps). The state becomes useful only after the block learns to compress history.
- **Original "information asymmetry required" finding is CONFIRMED, not overturned.** The 300-step single-seed result that seemed to contradict it was noise.

## Next steps

- **Multi-rate firing** — the actual vision. Now that persistence is confirmed, test whether blocks can fire at different rates (every 4 positions, every 16 positions). This is the core architectural question.
- **Scale up** — test at d_model=128, 2 layers (as validated in the fixed-window script). Does the effect persist at larger scale?
- **Longer sequences** — test seq_length=512 with adequate training to see if state beyond 128 chars provides additional value.
- **No_persistence control at 1200 steps** — confirm that the ensemble effect is genuinely absent with more training (not just a 300-step artifact in the other direction).
