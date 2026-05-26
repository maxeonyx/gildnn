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

### Previous results (INVALID — same-timestep bug)

The earlier results (mean Δ = -0.058, "confirmed") were measured with the lateral arriving at the SAME timestep (position t → position t). This made the blocks sequential (extra depth), not parallel. Those results are invalidated.

### Corrected experiment: stale lateral (one-position delay)

After fixing lateral timing (block 0 at position t receives block 1's output from position t-1):

**lateral_scale=1.0, 1200 steps, 3 seeds (params: --batch-size 32 --temperature 0.07):**

| Seed | block0_alone | recurrent_lateral | Δ |
|---|---|---|---|
| 42 | 1.7065 | 1.8421 | +0.136 |
| 123 | 1.7504 | 2.3058 | +0.555 |
| 7 | 1.6872 | 1.8110 | +0.124 |

**Stale laterals at scale=1.0 actively HURT.** The noise from an undertrained recurrent block overwhelms block 0.

**lateral_scale=0.2, 4800 steps, seed 42:**

| Condition | val_loss | Δ |
|---|---|---|
| block0_alone | 1.6112 | — |
| recurrent_lateral | 1.6280 | +0.017 (neutral) |

With reduced lateral scale and 4x more training: essentially neutral. But the TRAINING loss tells a different story:

| Condition | train CE @ 4800 | val_loss |
|---|---|---|
| block0_alone | 1.553 | 1.611 |
| recurrent_lateral | 1.480 | 1.628 |

The recurrent model trains BETTER (lower training loss) but validates WORSE — it overfits. The state captures temporal patterns in the training data that don't generalize.

## Interpretation

1. **Stale laterals from detached recurrent state DO NOT WORK** at this scale with this mechanism. At scale=1.0 they actively hurt; at scale=0.2 they're neutral.

2. **The "positive result" was entirely the timing bug.** Same-timestep lateral = extra sequential depth, which trivially helps. Parallel (stale) lateral = no benefit.

3. **The state IS capturing information** (proven by lower training loss), but it's not the RIGHT information for block 0. Block 1 is trained for local CE (predict its own next-char), NOT to produce messages useful for block 0 one step later. Without temporal credit assignment (BPTT), there's no gradient telling block 1 "your state was useful/useless to block 0."

4. **Why fixed-window works and stale-recurrent doesn't:** Fixed-window block 1 has IMMEDIATE, OBVIOUS extra information (128 chars vs 4 chars). The useful signal requires no temporal accumulation or communication learning. Stale recurrent block 1 must INVENT a useful long-range message from a d=64 vector updated step-by-step, with only local CE as its training signal.

5. **The overlap problem:** Block 0 at position t sees [t-3, t-2, t-1, t]. Block 1's stale output from t-1 was built from [t-4, t-3, t-2, t-1]. The overlap is 3/4 chars — the ONLY genuinely new information is whatever the state accumulated from positions 0..t-4. With no BPTT, that residue is extremely weak.

## What this taught us

- **The timing bug gave a false positive that masked a real negative.** Always check that architectural constraints (parallelism) are actually implemented.
- **Information asymmetry must be IMMEDIATE, not accumulated.** Giving block 1 more context (fixed-window) works. Asking block 1 to accumulate useful state step-by-step without temporal gradient does not.
- **Local CE trains blocks to predict well for themselves, not to communicate.** A block optimized for local prediction has no incentive to store information in its state that's useful for another block one step later.
- **State CAN capture information** (training loss proves this) but overfits to training sequences — doesn't generalize.
- **Information asymmetry IS required** — reconfirmed via the null result. Same-window blocks don't help each other regardless of temporal state.

## Implications for multi-rate firing

Multi-rate firing (block 1 fires every K positions) would help the **information geometry** — when K=4, block 1's lateral would be 4 steps stale, but it would have processed 4 unique windows between firings. This increases asymmetry.

BUT multi-rate alone does NOT solve the **communication learning problem**. If block 1 is still trained only with local CE and detached state, it still has no incentive to produce laterals useful for block 0.

**Possible solutions (not yet tested):**
1. **Give block 1 a larger window at firing time** — process the K accumulated positions together (= K*SHORT_CONTEXT effective context). This provides immediate info asymmetry like fixed-window.
2. **Truncated BPTT** — allow gradient to flow through a few steps of state, giving block 1 indirect feedback on state quality.
3. **Train block 1 with a communication objective** — predict something future-useful for block 0, not just its own local next-char.
4. **Multi-rate with context accumulation** — the slow block accumulates a buffer of positions and processes them all at once when it fires. This is essentially fixed-window but amortized.

## Exit condition assessment

The result is **between NULL and NEGATIVE:** stale laterals from detached recurrent state don't help (null at scale=0.2, hurt at scale=1.0). The mechanism is too weak without temporal credit assignment.

This does NOT kill the multi-rate direction — but it means multi-rate needs to provide information asymmetry through CONTEXT (accumulated buffer at firing time) rather than relying solely on recurrent state compression.
