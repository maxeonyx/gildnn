# Temporal Window: Does trajectory information create a niche for upper blocks?

**Status: PRE-REGISTERED (not yet run)**

**Pathway:** 3 (Local Learning) — upstream dependency. If upper blocks can't be made useful, local learning has no substrate to test on.

**Grounding:** Max in [dictation 2026-05-24-5](../../dictations/2026-05-24-5.md): "Block one should learn to predict something about block zero that block zero couldn't already know or wouldn't need to know therefore. For example, you know maybe block one's output is dependent on input from a longer time ago?"

---

## The problem

All attempts at making lateral-only blocks useful (`token_injection="block0"`) have failed:

| Config | Result vs A_single | Why it failed |
|--------|-------------------|---------------|
| Hardcoded 0.5 mixing (4-block) | +0.014 worse | Upper blocks are redundant delayed decoders |
| Zero-init gates (4-block) | +0.245 worse | Cold-start trap — gates stay near zero, blocks starve |

Upper blocks receive only block 0's current lateral state. Block 0 already has that information — it produced it. There's no computational niche for the upper block.

---

## Hypothesis

**Temporal window creates a computational niche** by giving the upper block explicit access to the TRAJECTORY of block 0's recent states — how its output has been moving over the last N steps. Block 0 cannot easily access this trajectory information within a single forward step. If the trajectory is useful for prediction, block 1 can learn to extract patterns from it that block 0 can't.

---

## Experimental design

### Key design choice: 2-block, readout_mode="last"

This is the cleanest test because:

1. **Block 1 is forced to be load-bearing** — it IS the readout block. No readout-collapse confound.
2. **Block 0 is purely a feature extractor** — its output feeds block 1 laterally but isn't read out directly.
3. **Only 1 lateral connection** — removes chain-scaling and multi-block interaction confounds.
4. **The question reduces to:** does seeing block 0's trajectory help block 1 produce better predictions than seeing only block 0's current state?

### Why NOT 4 blocks / readout_mode="all"?

With 4 blocks, failure is uninterpretable: is it the window, the chain, the readout averaging, or the topology? A null result teaches nothing. A 2-block positive can be followed up with 4-block scaling tests.

### Conditions

| # | Condition | `temporal_window` | Purpose |
|---|---|---|---|
| B0 | 2-block, no window | 0 | Baseline: block 1 sees only current lateral state |
| B4 | 2-block, window=4 | 4 | Block 1 sees 4-step trajectory of block 0 |
| B8 | 2-block, window=8 | 8 | Block 1 sees 8-step trajectory of block 0 |

**Shared config:**
- `num_blocks=2`
- `topology="upward"` (block 0 → block 1, no feedback)
- `readout_mode="last"` (block 1 produces output)
- `token_injection="block0"` (only block 0 gets tokens)
- `internal_steps=1`
- WikiText-103 ctx=128
- d_model and feedforward_dim chosen to make B0 match ~2.85M params (same as existing A_single). This means d_model ≈ 212, ff ≈ 848.
- 2 seeds (42, 43), 20K steps, batch 64, AdamW lr=3e-3, weight_decay=0.01
- Same training infrastructure as C_old ablation

**Parameter confound:** B4 adds `4 * d_model² + d_model ≈ 180K` params via `window_proj`. B8 adds ~360K. This is modest (~6-12% of 2.85M) and only matters if gains are small. If gains are large (≥0.015) AND trajectory ablation is positive, the confound is not credible. If gains are marginal, add a control: same-size projection fed duplicated current state instead of true history.

---

## Measurements

### Primary: val_loss comparison

- `Δ4 = val_loss(B0) - val_loss(B4)` — positive means window=4 helps
- `Δ8 = val_loss(B0) - val_loss(B8)` — positive means window=8 helps

### Secondary: trajectory-channel ablation

After training B4 and B8, evaluate with temporal history zeroed (replace `temporal_history` with zeros). Measure degradation.

**Why not upper-block ablation?** With `readout="last"`, block 1 is load-bearing by construction. Ablating it is tautologically catastrophic. The RIGHT mechanism test is: does the trained model actually USE the trajectory channel?

---

## Pre-registered interpretation

### Clear positive

- Mean `Δk ≥ 0.015` (at least one window size)
- Every seed positive by > 0.005
- Trajectory-channel ablation ≥ 0.02 nats

**Interpretation:** Temporal window creates a useful computational niche. Upper block extracts actionable trajectory information that the current-state-only lateral cannot provide.

**Next step:** 4-block scaling test with temporal window + local learning (bridge experiment design).

### Suggestive positive

- `0.005 ≤ mean Δk < 0.015`, all seeds positive

**Interpretation:** Potentially helpful but too small for strong claims. May be parameter-count confound.

**Next step:** Capacity control experiment (same projection, fed duplicated current state).

### Null

- `|mean Δk| < 0.005`

**Interpretation:** This temporal_window implementation does not create a useful niche at this scale.

**What this does NOT prove:** "trajectory information is useless" — only that this specific mechanism (linear projection of recent history) doesn't extract it usefully.

**Next step:** Consider attention-over-history instead of linear projection, or revisit the "corrected architecture" theory entirely.

### Negative

- `mean Δk ≤ -0.005`

**Interpretation:** Explicit history actively interferes. The projection may be adding noise that block 1 must learn to filter, with insufficient training budget to do so.

**Next step:** Investigate whether longer training or different window_proj init fixes this, or abandon this mechanism.

---

## What this experiment does NOT settle

- Whether temporal_window scales past 2 blocks
- Whether local learning works with temporal_window blocks
- Whether the optimal window size is 4 or 8 (only tests two points)
- Whether a different aggregation (attention vs linear projection) would be better
- The full "corrected architecture" (which also involves forced-equal readout + 4 blocks)

---

## External references

| What | Value | Source |
|------|-------|--------|
| A_single | val_loss 1.832 ± 0.007 | tied-depth experiment, 2 seeds |
| B_corrected (4-block, 0.5, block0) | ~1.86 | wikitext_103 ctx128 prior run |
| C_old (4-block, all tokens) | ~1.82 | wikitext_103 ctx128 prior run |
| Transformer baseline | 1.592 ± 0.003 | base-experiments, 2 seeds |

Matching/beating A_single (1.832) would be practical success. Matching C_old (~1.82) would be very strong. But the primary comparison is B0 vs B4/B8 — relative improvement from the temporal window mechanism.

---

## Artifacts (after running)

- Training logs: `experiments/temporal-window/artifacts/*.jsonl` (not yet created)
- Experiment script: `runs/temporal_window.py` (not yet written)
