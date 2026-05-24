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

## Recommended scaling path (updated with I results)

The original decision tree was:
- If N=3 wins → scale up
- If N=3 ties → try phase offsets or different input views
- If N=3 hurts → diagnose

**What actually happened:** F was seed-sensitive (coupling). G showed rate-4 is dead. I showed two rate-2 helpers (with or without phase offsets) saturate at the same benefit as one helper. Per-helper losses fix coupling. Width doesn't help.

**The bottleneck is the prediction target, not the graph.** Adding more nodes to the star can't help when every node predicts the same redundant thing. Graph topology scaling is **blocked until the prediction target changes** (Phase 5).

If Phase 5 breaks the saturation (benefit > -0.006 with a better target), the scaling path reopens:
1. N=3 with the new target (verify width helps now)
2. N=5 with rate/phase diversity
3. N=8 with factorized heads (Risk 3 still applies)

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

The star topology is the only graph structure tested so far. The findings tell us about width-scaling *under the current prediction target* — not about graph topology per se. Whether richer topologies (hierarchical, lateral connections between helpers) add value is completely unknown. The current evidence base:

- N=2 star with full-state target: works modestly (-0.006)
- N=3 star with full-state target: saturates at same -0.006 (width redundant)
- Everything else: untested theory

The interesting topology question reopens if Phase 5 breaks the target bottleneck — then multiple helpers predicting *different useful things* might compose.
