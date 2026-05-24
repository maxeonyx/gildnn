# Graph topology for the multi-block architecture

Serves [dictation 2026-05-23-5](../../../dictations/2026-05-23-5.md): "Originally my vision is a graph. I don't know if that graph helps, but I kind of think it might." Also [dictation 2026-05-24-1](../../../dictations/2026-05-24-1.md) (original Keep note describing the graph vision).

## Status

**Active — first topology test running.** N=3 star topology (2 helpers predicting block 0) is the first real graph structure being tested. Results pending. See [`local-learning-variants`](../local-learning-variants/README.md) for the experiment details.

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

## The N=3 result (pending)

First topology test: does adding a second helper (2-spoke star vs 1-spoke) improve on N=2?

- N=2 (variant C): val_loss 1.665, beats A_single (1.670) by 0.006
- N=3 (variant F_star_3block): running now

See `local-learning-variants/README.md` for decision rules.

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

## Recommended scaling path

**If N=3 wins (F < C):** Don't jump straight to N=8 with current design. Intermediate step:
1. N=5 with rate/phase design (test phase offsets)
2. Then N=8 with factorized heads

**If N=3 ties (F ≈ C):** Interpret as redundancy problem (both helpers doing the same thing). Try:
1. Phase offsets at same N=3
2. Or different input views (helper reads different temporal window of s0 history)

**If N=3 hurts:** Suspect interface interference. Try:
1. Normalized aggregation instead of raw sum
2. Or reduce to understanding why 2 helpers is worse than 1

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

The star topology is being tested first because it preserves the one mechanism we've verified (direct CE grounding for every helper). Whether richer topologies (hierarchical, lateral connections between helpers) add value is completely unknown. The current evidence base is: N=2 star works (modestly). Everything else is theory.
