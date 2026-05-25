# Temporal Window: Does trajectory information help upper blocks?

**Status: READY TO RUN (code verified on CPU; awaiting GPU)**

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

## Adversarial review findings (2026-05-25)

The original design (B0/B4/B8, 2 seeds) **failed adversarial review**. Key problems:

1. **Mixing-ratio confound:** Current code changes `current_lower` weighting when window is on (0.5 → 0.25). A positive result could be "different mixing ratio helps."
2. **Bias-channel confound:** `window_proj` has bias — zeroed history still produces nonzero contribution.
3. **Ablation unclean:** Zeroing `temporal_history` leaves bias path active.
4. **Missing control:** No condition rules out "extra parameters help" vs "trajectory specifically helps."
5. **Interpretation overclaimed:** "Clear positive" was stated as "niche proven" but the design couldn't distinguish trajectory from other explanations.

**Resolution:** Revised design below adds a same-capacity non-trajectory control, fixes the code path, and downgrades interpretation claims.

---

## Experimental design (revised)

### Key design choice: 2-block, readout_mode="last"

This is the cleanest test because:

1. **Block 1 is forced to be load-bearing** — it IS the readout block. No readout-collapse confound.
2. **Block 0 is purely a feature extractor** — its output feeds block 1 laterally but isn't read out directly.
3. **Only 1 lateral connection** — removes chain-scaling and multi-block interaction confounds.
4. **The question reduces to:** does seeing block 0's trajectory help block 1 more than a same-capacity non-trajectory input?

**Acknowledged limitation:** `readout_mode="last"` makes block 1 useful by fiat. A positive result means "a forced-readout upper block can use trajectory info," not "trajectory makes a voluntary upper block become useful." That's a narrower but still valuable claim.

### Conditions

| # | Condition | Auxiliary input to `window_proj` | Purpose |
|---|---|---|---|
| B0 | No auxiliary branch | — | Baseline: block 1 sees only current lateral state |
| H8 | True history, window=8 | `concat([x_{t-7}, ..., x_t])` | Block 1 sees 8-step trajectory of block 0 |
| C8 | Duplicated current state | `concat([x_t, x_t, ..., x_t])` (×8) | Same-capacity non-trajectory control |

**Primary comparison:** H8 vs C8 (isolates trajectory). **Secondary:** H8/C8 vs B0 (does any augmented input help?).

### Code changes required

1. **`window_proj` must use `bias=False`** — removes bias-channel confound
2. **Auxiliary branch must be ADDITIVE without reducing `current_lower` weight:**
   - B0: `block_input = 0.5 * state_input + 0.5 * current_lower`
   - H8/C8: `aux = window_proj(aux_input); block_input = 0.5 * state_input + 0.5 * current_lower + aux`
   - The `current_lower` path stays at 0.5 in ALL conditions — no mixing-ratio confound

3. **C8 uses the SAME `window_proj` weights** as H8 would — it's trained from scratch with duplicated-current input. (Separate training run, not a weight-sharing trick.)

### Shared config

- `num_blocks=2`
- `topology="upward"` (block 0 → block 1, no feedback)
- `readout_mode="last"` (block 1 produces output)
- `token_injection="block0"` (only block 0 gets tokens)
- `internal_steps=1`
- WikiText-103 ctx=128
- d_model and feedforward_dim chosen to make B0 match ~2.85M params (same as A_single). d_model ≈ 212, ff ≈ 848.
- **3 seeds** (42, 43, 44), 20K steps, batch 64, AdamW lr=3e-4, weight_decay=0.01
- Same training infrastructure as C_old ablation

### Parameter accounting

- B0: ~2.85M params
- H8/C8: ~2.85M + `8 * d_model * d_model` ≈ 2.85M + 360K ≈ 3.21M (12% more)
- This is the POINT — H8 and C8 have identical parameter counts. The comparison H8 vs C8 is parameter-matched. The comparison vs B0 is not, but that's a secondary question.

---

## Measurements

### Primary: val_loss comparison

- `Δ_trajectory = val_loss(C8) - val_loss(H8)` — positive means true history helps MORE than duplicated-current control
- `Δ_augmented = val_loss(B0) - val_loss(H8)` — positive means history augmentation helps vs no augmentation
- `Δ_control = val_loss(B0) - val_loss(C8)` — positive means even non-history augmentation helps

### Secondary: auxiliary-branch ablation

After training H8 and C8, evaluate with `aux = 0` (zero the projected output, not the raw history). This tests: does the trained model actually use the auxiliary branch?

**Why not zero the raw history?** Because `window_proj(zeros) ≠ 0` unless we use `bias=False`. With `bias=False` and zeroed input, `window_proj(zeros) = 0`, so post-projection zeroing and input-zeroing are equivalent. But post-projection zeroing is more explicit and general.

---

## Pre-registered interpretation

### Trajectory helps (clear)

- Mean `Δ_trajectory ≥ 0.015` (H8 beats C8)
- All 3 seeds show H8 < C8 in val_loss

**Interpretation:** True trajectory information helps more than a same-capacity non-trajectory auxiliary input. This is evidence that the temporal history mechanism provides genuinely useful information that the current-state-only path cannot.

**What this does NOT prove:** "trajectory creates a computational niche for upper blocks in general" — only for this forced-readout-last 2-block setup. Follow-up needed with voluntary readout.

**Next step:** 4-block scaling test with temporal window + readout_mode="all" to see if the niche makes the block voluntarily useful.

### Augmentation helps but trajectory doesn't matter

- `|Δ_trajectory| < 0.005` (H8 ≈ C8)
- `Δ_augmented ≥ 0.015` AND `Δ_control ≥ 0.015` (both beat B0)

**Interpretation:** An extra projected auxiliary input helps block 1, but true history is not specifically better than the same computation applied to duplicated current state. The benefit is capacity/expressivity, not trajectory.

**Next step:** This does NOT support the trajectory hypothesis. Consider whether block 1 simply needs more parameters, or whether a different trajectory mechanism (attention over history) is needed.

### Nothing helps

- `|Δ_augmented| < 0.005` AND `|Δ_control| < 0.005`

**Interpretation:** Neither true history nor duplicated-current augmentation helps. The forced-readout-last upper block cannot benefit from extra projected input at this scale/training budget.

**What this does NOT prove:** "trajectory information is useless in all forms" — only that this linear projection of recent history doesn't help here.

**Next step:** Consider whether the premise is wrong (maybe `readout_mode="last"` + `token_injection="block0"` is fundamentally too harsh a handicap for block 0), or whether attention-over-history would fare better.

### Negative

- `Δ_augmented ≤ -0.005` (H8 hurts vs B0)

**Interpretation:** The auxiliary branch actively interferes with learning. May indicate: projection adds noise that takes too long to learn to ignore, or the architectural change interacts badly with optimization at this training budget.

**Next step:** Investigate training dynamics. Does the auxiliary branch eventually become useful with longer training?

---

## What this experiment does NOT settle

- Whether temporal_window scales past 2 blocks
- Whether local learning works with temporal_window blocks
- Whether a voluntary upper block (readout_mode="all") would find this niche on its own
- Whether a different aggregation (attention vs linear projection) would be better
- The full "corrected architecture" question (which involves multiple blocks + equal readout)
- Whether the optimal window size is different from 8

---

## External references

| What | Value | Source |
|------|-------|--------|
| A_single | val_loss 1.832 ± 0.007 | tied-depth experiment, 2 seeds |
| B_corrected (4-block, 0.5, block0) | ~1.86 | wikitext_103 ctx128 prior run |
| C_old (4-block, all tokens) | ~1.76 | C_old ablation, 2 seeds |
| Transformer baseline | 1.592 ± 0.003 | base-experiments, 2 seeds |

The primary comparison is H8 vs C8 — does trajectory specifically help? Absolute performance vs A_single is context but not the discriminating question.

---

## Artifacts

- Training logs: `experiments/wikitext_103/artifacts/temporal_window/run.jsonl` (created at runtime)
- Experiment script: `runs/temporal_window.py` ✅ (sanity-checked on CPU, commit 95efdc8)
- Code changes: `core/model.py` — `bias=False` on `window_proj`, additive aux branch, `temporal_window_mode` parameter ✅ (commit 95efdc8)
