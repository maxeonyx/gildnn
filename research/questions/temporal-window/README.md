# Temporal Window: Does trajectory information help upper blocks?

**Status: COMPLETE ✅ — BRANCH 1 CONFIRMED. All 3 seeds concordant, mean Δ_trajectory = +0.042 (3× threshold). 4-block follow-up RUNNING (PID 20388, launched 02:55 NZST 2026-05-26).**

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

## Theory: what's in a trajectory that isn't in a snapshot?

### The decisive question

Does there exist predictive information in `(x_{t-7}, ..., x_{t-1})` conditional on `x_t`? Formally: is `I(next_token; T_t) > I(next_token; x_t)` where `T_t = [x_{t-7}; ...; x_t]`?

If yes, trajectory helps. If no, it's redundant.

### What H8 computes: a learned FIR filter

The linear history projection implements a learned multivariate finite impulse response (FIR) filter over block-0 states:

```
y_t = W[x_{t-7}; ...; x_t] = Σ_k A_k · x_{t-k}
```

This can compute: finite differences (velocity ≈ x_t - x_{t-1}), second differences (acceleration), weighted averages, exponential smoothing, lag-specific feature extraction — any linear temporal filter.

It **cannot** compute: nonlinear interactions across timepoints (e.g., "feature i was high at t-3 AND feature j is high at t"). But block 1's downstream nonlinearities CAN operate on the linear summaries, so only cross-time interactions destroyed by the projection bottleneck are truly lost.

### Why C8 is exactly current-state-only (not "approximately")

For duplicated-current control input `D_t = [x_t; x_t; ...; x_t]`:

```
W · D_t = Σ_k A_k · x_t = (Σ_k A_k) · x_t = A_eff · x_t
```

C8 collapses to a single learned linear projection of the current state. It lives on the diagonal subspace of R^(8d). This is not "approximately rank-1" — it is mathematically EXACTLY a reparameterized current-state-only model.

Consequence: anything that varies when `x_t` is held fixed but history changes is invisible to C8. All temporal diversity — velocity, trend, lag timing, motifs — is exclusively available to H8.

### Most plausible useful signals (ranked)

1. **Derivatives / direction of change** — same current state reached by drift vs jump vs reversal implies different continuations
2. **Trend / persistence** — "moving consistently toward X" vs "briefly visited X" vs "stationary at X"
3. **Lag timing** — WHEN a feature was active (1 step ago vs 7 steps ago) may predict next event
4. **Short temporal motifs** — rise-then-plateau, alternating sign, burst-then-decay
5. **Periodicity / oscillation** — mathematically possible but unlikely to be primary for language at window=8

### Why block 0's state might be insufficient

Block 0 is trained end-to-end to support prediction, so it has pressure to compress useful information into `x_t`. However:

- Block 0 is small (d=211) — compression is lossy
- Block 0 optimizes for its own readout pathway (but readout_mode="last" means it has no direct output)
- At 20K steps, block 0 may not have converged on an optimal compression
- Two different recent histories can map to similar `x_t` while differing in exactly the part that matters for the next token

### Interpretation constraints

A **positive result** (H8 > C8) means: at this scale and training budget, block 1 can exploit recent block-0 trajectory better than current-state-only access. It does NOT prove trajectory is fundamentally necessary in the limit — block 0 might learn to fold that information into `x_t` with more capacity or training.

A **negative result** (H8 ≈ C8) is less diagnostic: either current state is already sufficient, or the linear interface is too weak, or the window size is wrong, or the scale doesn't reward it.

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
- d_model and feedforward_dim chosen to make B0 match ~2.85M params (same as A_single). d_model=211, ff=848.
- **3 seeds** (42, 43, 44), 20K steps, batch 64, AdamW lr=3e-4, weight_decay=0.01
- Same training infrastructure as C_old ablation

### Parameter accounting

- B0: 2,851,382 params
- H8/C8: 3,207,550 params (identical — one `window_proj` of size Linear(8×211, 211, bias=False) = 356,168 extra params, 12.5% more)
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

- Training logs: `experiments/temporal-window/artifacts/run.jsonl` (created at runtime)
- Analysis script: `experiments/temporal-window/analyze.py`
- Experiment script: `runs/temporal_window.py` (sanity-checked on CPU, commit 95efdc8)

---

## Final results (all 3 seeds complete, 2026-05-26 02:55 NZST)

| Condition | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|-----------|---------|---------|---------|------------|
| B0 | 2.501 | 2.545 | 2.471 | 2.506 ± 0.037 |
| H8 | 2.370 | 2.369 | 2.390 | 2.376 ± 0.012 |
| C8 | 2.404 | 2.414 | 2.435 | 2.418 ± 0.016 |

**Key deltas:**
| Delta | Seed 42 | Seed 43 | Seed 44 | Mean |
|-------|---------|---------|---------|------|
| Δ_trajectory (C8-H8) | +0.034 | +0.045 | +0.045 | **+0.042** |
| Δ_augmented (B0-H8) | +0.132 | +0.176 | +0.081 | +0.130 |
| Δ_control (B0-C8) | +0.097 | +0.131 | +0.036 | +0.088 |

**Concordance:** All 3 seeds show H8 < C8 (positive Δ_trajectory). ✅
**Threshold:** Mean Δ_trajectory (+0.042) ≥ 0.015. ✅

**Pre-registered interpretation: TRAJECTORY CLEARLY HELPS.**

True history (H8) provides uniquely useful temporal diversity that duplicated-current (C8) cannot access. This is predicted exactly by the FIR filter / diagonal subspace theory: C8 collapses to a single effective linear projection of current state, while H8 can exploit lag-specific temporal features.

**Stability finding:** H8 is remarkably stable across seeds (std = 0.012) compared to B0 (std = 0.037) and C8 (std = 0.016). The temporal window provides not only better quality but more consistent optimization — likely because 8 distinct temporal samples create a better-conditioned loss surface than a single snapshot.

**What this does NOT prove:**
- "Trajectory creates a computational niche for voluntary upper blocks" — only for forced-readout. That's what the 4-block follow-up tests.
- "Trajectory is fundamentally necessary" — block 0 might learn to fold this info into its state with more capacity/training.

**Next step (now running):** 4-block voluntary readout experiment (PID 20388, ~3.6 hours). See "Follow-up" section below.

**Learning-curve divergence (C8-H8 gap over time, seed 42):**

| Step | C8 val_loss | H8 val_loss | Gap (C8-H8) |
|------|------------|------------|-------------|
| 1000 | 2.883 | 2.905 | -0.022 |
| 3000 | 2.734 | 2.734 | -0.001 |
| 5000 | 2.668 | 2.658 | +0.010 |
| 7000 | 2.614 | 2.601 | +0.012 |
| 9000 | 2.565 | 2.542 | +0.023 |
| 11000 | 2.535 | 2.514 | +0.021 |
| 15000 | 2.475 | 2.440 | +0.035 |
| 19000 | 2.415 | 2.380 | +0.035 |
| **20000** | **2.404** | **2.370** | **+0.034** |

**Prediction (pre-registered before final results):** C8 will finish around 2.39-2.44, giving Δ_trajectory = 0.02-0.07. ✅ **CONFIRMED** — actual C8 = 2.404, Δ = 0.034.

**Theoretical explanation for early C8 ≈ H8:** C8 has a "shared shortcut" advantage — all 8 submatrices receive the same gradient direction, converging faster on easy current-state patterns. H8's gradient must sort out which lags matter, which takes longer. Once easy patterns saturate, H8 can leave the diagonal subspace to exploit lag-specific features; C8 cannot.
- Code changes: `core/model.py` — `bias=False` on `window_proj`, additive aux branch, `temporal_window_mode` parameter ✅ (commit 95efdc8)

---

## Follow-up: 4-block voluntary readout (PRE-REGISTERED, conditional on branch 1)

**⚠️ This section applies ONLY if the 2-block experiment above confirms branch 1 (H8 > C8 clearly, Δ_trajectory ≥ 0.015, all seeds concordant).** Per the stop-loss rule, this is the final intended-architecture experiment allowed.

### Question

Does temporal trajectory information make upper blocks **voluntarily useful** in a 4-block `readout_mode="all"` regime? The 2-block result (if positive) proves a forced-readout block CAN use trajectory. This tests whether the mechanism scales — does it overcome the spectator problem that has defeated every prior 4-block intended-architecture attempt?

### Conditions

| Variant | temporal_window | temporal_window_mode | Purpose |
|---|---|---|---|
| A_all | 0 | — | Control: reproduce known spectator behavior |
| C8_all | 8 | "current" (duplicated) | Same-capacity non-trajectory control in 4-block voluntary regime |
| W8_all | 8 | "history" (true trajectory) | Main test: does trajectory rescue make blocks voluntarily useful? |

**Why include C8_all?** The 2-block result proves trajectory > capacity in the forced-readout regime. But forced-readout ≠ voluntary-readout dynamics. In a voluntary regime, the capacity boost from any extra branch might change the equilibrium differently. Including C8_all eliminates "any extra projected input helps voluntary blocks" as an alternative explanation.

### Shared config

- `num_blocks=4`
- `topology="upward"`
- `token_injection="block0"`
- `readout_mode="all"` (all blocks contribute — voluntary participation)
- `internal_steps=1`
- `rates=(1, 1, 1, 1)` — all rate-1, no staleness confound. Depth creates the propagation delay.
- `d_model=256`, `feedforward_dim=512` — matches C_old exactly for direct comparability
- WikiText-103 ctx=128, 20K steps, batch 64, AdamW lr=3e-4, weight_decay=0.01
- 3 seeds (42, 43, 44)
- Temporal window applied to **blocks 1, 2, and 3** (all upper blocks get history from the block below)

### Parameter accounting (verified via sanity check)

- A_all: 3,639,168 params (matches C_old exactly)
- C8_all / W8_all: 4,163,456 params (identical — one shared `window_proj` across all upper blocks)
- Difference: 524,288 = Linear(8×256, 256, bias=False) = one shared temporal projection
- Within-experiment comparison (W8 vs C8) is parameter-matched
- A_all vs W8/C8 differs by 14.4% — smaller than pre-estimated because the model uses ONE shared `window_proj` for blocks 1, 2, and 3 rather than per-block projections

**Architectural note:** All upper blocks apply the SAME learned FIR filter to their respective lower block's history. This is a shared inductive bias — the model learns one temporal feature extraction pattern applied at all depth levels. If this proves insufficient, per-block projections would be a natural follow-up (but requires model code change).

### Expected runtime

~24 min per variant-seed (based on C_old timing at same d/ff). 9 runs ≈ 3.6 hours total. Two-phase delegation required.

### Required measurements

**Primary:** val_loss by variant × seed, mean and std.

**Post-training per-block readout ablation:** For each trained model, evaluate with each block's readout contribution zeroed:
- Zero block k's contribution: `output = sum_{i≠k} w_i · s_i` (do NOT renormalize remaining weights)
- Record val_loss for each ablated block
- Key comparison: in A_all vs W8_all, which blocks show ablation cost ≥ 0.02?

**Cumulative ablation (secondary):**
- block 0 only
- blocks 0+1
- blocks 0+1+2
- full model
- For comparison to C_old where all 4 blocks were load-bearing.

### Success criteria

**Strong positive (viable at scale):**
1. W8_all beats A_all by ≥ 0.015 mean val_loss
2. W8_all beats C8_all by ≥ 0.015 mean val_loss
3. Same sign on all 3 seeds
4. In W8_all, ablating each of blocks 1, 2, 3 individually raises val_loss by ≥ 0.02 mean

Interpretation: trajectory information is not just usable under forcing; it makes upper blocks voluntarily load-bearing. The intended architecture IS viable when blocks get trajectory access.

**Partial positive:**
- W8_all beats controls, but only some of blocks 1-3 clear the 0.02 ablation threshold

Interpretation: trajectory creates SOME voluntary niche, but scaling is incomplete. Mechanism works but doesn't fully overcome spectator problem for all blocks.

**Null:**
- W8_all ≈ A_all, or blocks 1-3 are still spectators (ablation < 0.02) even with window

Interpretation: forced-readout benefit does NOT generalize to voluntary 4-block regime. The 2-block result was an artifact of forcing. Stop-loss fires — pivot to surrogate pathways immediately.

### Control validity check

If A_all does NOT reproduce spectator behavior (i.e., blocks 1-3 turn out to be useful WITHOUT the window), the experiment is invalid as a "rescue" test. This would be a surprising and informative result in itself — suggesting something about the configuration (d=256 vs smaller, or some other factor) makes the intended architecture work without trajectory help.

### What this does NOT settle (even if positive)

- Whether the mechanism works with heterogeneous rates (multi-rate firing creates additional staleness)
- Whether local learning (gradient truncation) works WITH temporal window
- Whether attention-based aggregation would be better than linear window projection
- Whether window=8 is optimal or scales with depth

### Theory note: temporal window × multi-rate synergy

In a multi-rate setup [1,2,4,8], upper blocks fire SLOWER than lower blocks. Between block i+1's firings, block i fires `rate[i+1]/rate[i]` times (typically 2×). This means block i+1's temporal window captures MORE distinct states (higher temporal diversity) than in the all-rate-1 case — the input block has evolved further between each window sample.

**Implementation detail (verified from `core/model.py`):** history windows track lower-block **firing events**, not wall-clock timesteps. The buffer only rolls when that block actually fires. Multi-rate does NOT fill windows with cached duplicates.

**Quantitative lookback with rates [1,2,4,8] and window=8:**

| Block | Sees | Tap spacing | Lookback horizon | New entries per firing |
|-------|------|-------------|------------------|-----------------------|
| 1 ← 0 | 8 distinct block-0 states | 1 timestep | 8 timesteps | 2 |
| 2 ← 1 | 8 distinct block-1 states | 2 timesteps | 16 timesteps | 2 |
| 3 ← 2 | 8 distinct block-2 states | 4 timesteps | 32 timesteps | 2 |

**FIR interpretation:** `window_proj` is a learned FIR filter on lower-block state trajectories. With multi-rate, higher blocks apply this filter to progressively **coarser-sampled, longer-horizon** temporal signals — a natural multiscale decomposition. Each level sees the trajectory on its own natural timescale.

**Limitation:** `window_proj` is **shared across all upper blocks** (one learned filter for all depths). Different timescales might benefit from different learned projections. Per-block projections are a natural follow-up if multi-rate × window proves beneficial but suboptimal.

**This is still hypothesis** (not tested). The 4-block follow-up uses uniform rates to isolate the temporal window effect. The multi-rate interaction would be a separate 2×2 experiment: {uniform rates, multi-rate [1,2,4,8]} × {no window, H8}.
