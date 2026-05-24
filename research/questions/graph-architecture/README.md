# Graph topology for the multi-block architecture

Serves [dictation 2026-05-23-5](../../../dictations/2026-05-23-5.md): "Originally my vision is a graph. I don't know if that graph helps, but I kind of think it might." Also [dictation 2026-05-24-1](../../../dictations/2026-05-24-1.md) (original Keep note describing the graph vision).

## Status

**Partially answered — width saturates at current target.** The N=3 star topology series (F, G, I) is complete through Phase 4. Key findings:

- F (shared loss, rates 2+4): **seed-sensitive** — coupling instability between helpers
- G (rate-4 only): **intrinsically too stale** — ablation gap = 0, pred_loss rises over training
- I (two rate-2 helpers, per-helper losses, ±phase offset): **width doesn't add benefit** — saturates at same -0.006 as one helper

Per-helper losses fix the coupling instability (both helpers survive in I). But the prediction TARGET is the bottleneck: predicting full block-0 state that block 0 already knows gives redundant information regardless of helper count. This is a target problem, not a topology problem.

**Next:** Phase 5 changes the prediction target (predict something block 0 doesn't know). If that breaks the saturation, graph scaling becomes worth revisiting. See [`local-learning-variants`](../local-learning-variants/README.md) for full details.

## Current architecture (as of 2026-05-24)

Single primary block processes tokens. Helpers predict block 0's future state and feed predictions back:

```
         Block 1 (rate=2)
        ╱ reads s0.detach()
Block 0 ←── predictions via additive gain
(rate=1) ←── predictions via additive gain
        ╲ reads s0.detach()
         Block 2 (rate=4)
```

- Block 0 is the ONLY block seeing token embeddings (corrected architecture per dictation 2026-05-23-7)
- Helpers read block 0's state (detached — no backward through the read)
- Helpers predict block 0's future via prediction heads
- Predictions feed back through additive zero-init gain (CE flows backward through this interface)
- This is a **star topology** with block 0 at the center

## The N=3 result (SEED-SENSITIVE — both seeds complete)

First topology test: does adding a second helper (2-spoke star vs 1-spoke) improve on N=2?

| Variant | Seed 42 | Seed 43 | Mean | Range |
|---------|---------|---------|------|-------|
| A_single | 1.669 | 1.672 | 1.670 | 0.003 |
| C_closed_loop | 1.660 | 1.669 | 1.665 | 0.009 |
| F_star_3block | **1.702** | **1.659** | **1.681** | **0.044** |

Seeds disagree: F hurts by 0.034 on seed 42, helps by 0.013 on seed 43. Both seeds reject helper 2 (gain_2 → 0). The difference is death speed: seed 43 killed helper 2 by step 5K, seed 42 took until step 15K.

**The finding is instability**, not consistent harm. The shared prediction loss at N=3 creates 15× more variance than baseline. This doesn't prove the objective-mismatch mechanism, but is consistent with transient coupling: while helper 2 is alive, it creates some interference; longer life → worse outcome.

## Scaling to N=8+: design analysis

The long-term goal is "many more parallel blocks" (dictation 2026-05-23-4). Key design considerations for scaling the star beyond N=3:

### Why star scales better than chain for this architecture

The one verified mechanism: **CE flows backward through the helper→block0 interface**, which teaches the helper WHAT to predict. A chain (0←1←2←...←7) weakens this for distant helpers — they're multiple hops from the task loss. The star keeps every helper exactly one hop from CE.

### Risk 1: Summed-prediction identifiability

Current design: all predictions are additive at block 0. The prediction cosine loss sees the SUM. With N=8 helpers, the individual contributions are underdetermined — helpers can produce redundant or cancelling predictions without the loss penalizing it. This is a real scaling concern.

Possible mitigations:
- Per-helper diagnostic (track each gain and ablation gap individually)
- Sparse/normalized aggregation instead of raw sum (softmax weights, top-k)
- Per-helper auxiliary objectives

### Risk 2: Rate schedule saturation

Powers-of-two rates: (2, 4, 8, 16, 32, 64, 128). At ctx=128, rate=128 means one update per sequence — that's barely a useful helper. **Cap the max rate** at something meaningful (16 or 32 at ctx=128).

Better N=8 approach: **phase offsets within rate bands**. Two helpers at rate=2 but firing on different timesteps (t%2==0 vs t%2==1). This gives differentiation without pushing rates to degenerate values. Example 7-helper allocation:

```
rate=2 phase=0, rate=2 phase=1,
rate=4 phase=0, rate=4 phase=2,
rate=8 phase=0, rate=8 phase=4,
rate=16 phase=0
```

### Risk 3: Prediction head parameter blow-up

Each prediction head is `Linear(d_model, rate * d_model)` — it predicts the next `rate` timesteps of block 0's state. At rate=16, that's `Linear(256, 4096)` = 1M params per head. Seven such heads dominates the parameter budget.

Mitigation: **factorized / low-rank prediction heads**. Helper outputs a small latent; a shared decoder maps (latent, horizon) → correction. This decouples helper count from head cost.

### Risk 4: Interface interference

Many additive corrections may create unstable co-adaptation (helpers learning to cancel each other's predictions). If N=3 shows interference, consider:
- Learned mixing weights (lightweight attention over helper outputs)
- Separate channels per helper (concatenate, then project down)

## Recommended scaling path (updated post-Phase 5 J)

**Phase 5 J CONFIRMED: the target bottleneck is broken.** J_older_window (predict mean of block 0 states from 5-8 steps ago) gives -0.010 with 53× less variance than the full-state target. The prediction target was the binding constraint on width — that constraint is now removed.

**The graph scaling path is REOPENED.** The key insight from J: width should scale via **temporal role differentiation** — each helper owns a different memory band (different offset), not duplicate prediction of the same thing. This directly addresses why I_phase_offset failed (two helpers predicting the same full-state target saturate regardless of phase diversity).

**Concrete next topology experiment (after J_far and J_fixed resolve):**
- N=3 with J-style targets: helper 1 at offset 5-8, helper 2 at offset 9-12 (or further, depending on J_far result)
- Each helper carries a different temporal band → tests whether bands COMPOSE (not redundant)
- Per-helper prediction losses (proven to fix coupling from I)
- If bands compose: scale to N=5+ with staggered offsets

**What Phase 4 (I) proved that still applies:**
- Per-helper losses are necessary at N≥3 (shared loss creates coupling instability)
- Rate-4 is too stale for current-state prediction, but might work with older-window target (untested)
- The star topology works mechanically — the failure was target redundancy, not topology

**Risk assessment for width + J target:**
- If J_far ≈ J (plateau): two helpers at different offsets carry genuinely different information → width likely helps
- If J_far << J (steep dropoff): the useful band is narrow → multiple helpers within it may be redundant
- Either way: J_fixed_embedding tells us whether the target needs to be self-generated or can be external (affects how we assign targets to N helpers)

## Historical context: earlier topology designs (2026-05-08 era)

The below topologies were designed for the OLD architecture (all blocks seeing tokens, 4-block chain writing to shared residual stream). That architecture is now abandoned — it produced the spectator problem. Kept for reference but **not directly applicable** to the current helper-prediction architecture.

<details>
<summary>Old topology alternatives (stale — click to expand)</summary>

### Dense Lower Pyramid Lattice

8 modules: 4×rate=1, 2×rate=2, 1×rate=4, 1×rate=8. Dense connectivity at fast level. Tests whether many small fast modules beat few large ones.

### Overlapping-Rate Diamond Graph

7 modules with same-rate pairs having different neighborhoods. Tests whether same-rate modules specialize given different input contexts.

### Slow-Hub Prior Graph

Star with fast modules reading from slow hubs. Tests whether the value of hierarchy is purely top-down priors.

### Adjacency-only probe

Same 4 modules, same rates, only change adjacency (chain vs skip connections). Cheapest test of topology effects. **Not applicable to current architecture** where there's only 1 primary block.

</details>

## Second-pass insights (still relevant)

From the earlier theory iteration, these observations remain valid:

1. "Overlapping rates" has multiple interpretations:
   - (A) Same rate, different neighborhoods
   - (B) Same rate, **different phase offsets** — probably closest to Max's words
   - (C) Stochastic/fuzzy rates

2. Sum aggregation from many sources can be unstable. Mean aggregation or normalized mixing bounds the activation scale regardless of fan-in.

3. Max explicitly said "I don't know if that graph helps." This remains uncertain. The N=3 star is the first real test.

## Epistemic note

The star topology is the only graph structure tested so far. Phase 5 J broke the target bottleneck — width scaling is no longer theoretically blocked. But it's still **experimentally untested** with the new target.

Current evidence base:
- N=2 star with full-state target: works modestly (-0.006, unreliable)
- N=2 star with older-window target (J): works better (-0.010, highly reliable)
- N=3 star with full-state target: saturates (width redundant — target was bottleneck)
- N=3 star with older-window targets at **different offsets**: UNTESTED — this is the next key topology experiment
- Everything else: untested theory

The temporal-role-differentiation hypothesis (different offsets per helper) is the strongest prediction but has zero experimental backing yet. It could fail if the gating mechanism can't integrate signals from multiple temporal bands simultaneously.
