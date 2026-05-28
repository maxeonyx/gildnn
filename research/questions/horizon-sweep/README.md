# Horizon Sweep: At what prediction horizon does Block 1 recurrence become load-bearing?

> Serves: [Pathway 3 (Local Learning)](../../../ROADMAP.md), specifically the multi-rate mechanism — testing whether longer prediction horizons create pressure for temporal integration.
> Source: 2-block ablation result (recurrence null at H1) and [multi-rate extension design](../stream-combining/README.md#multi-rate-extension-design-for-after-equal-rate-confirms).

---

## The question

At horizon 1, Block 1 predicts Block 0's next output 37% better than copy — but a no-recurrence model performs identically. Block 0's representation appears approximately Markov from a single timestep at this horizon.

The multi-rate architecture requires higher blocks to predict further ahead (rate-2 → +2, rate-4 → +4). If recurrence becomes necessary at these horizons, multi-rate naturally forces temporal integration. This experiment tests that prerequisite.

---

## Hypotheses

**H1 (confirmed):** At horizon 1, recurrence and no-recurrence are equal. Both beat copy by ~37%.

**H2 (to test):** A recurrent advantage emerges by horizon 2 or 4.

**H_null:** Frozen Block 0 remains approximately Markov across all tested horizons — a static transform of the current state suffices for predicting multiple steps ahead.

---

## Design

Shared Phase A: train Block 0 with CE (200 steps), freeze. Same as the completed 2-block experiment.

Phase B: Block 1 predicts A0_{t+N} for horizons N = 1, 2, 4. Add N=8 only if 2 and 4 are ambiguous. For each horizon, run recurrent vs no-recurrence Block 1 — architecture identical except the recurrence toggle. Reuse existing H=1 results.

Copy baseline per horizon: MSE(A0_t, A0_{t-N}). Gets worse as N grows (representations decorrelate).

Total new runs: 4 (H2-rec, H2-no-rec, H4-rec, H4-no-rec). Each: 800 steps, ~37 minutes. Sequential total: ~2.5 hours.

---

## Metrics

**Primary:** The recurrence gap = (no-recurrence eval MSE) − (recurrent eval MSE) at each horizon. Positive = recurrence helps. Zero = doesn't matter. Negative = likely optimization pathology, not architectural evidence.

**Secondary:** Each model's gain over its own horizon-matched copy baseline.

---

## Interpretation framework

- **Recurrence helps at H2:** Strongest support. Rate-2 blocks naturally need temporal integration. Validates multi-rate intuition.
- **Helps only at H4:** Partial. Natural timescale may be longer than planned rate-2.
- **Never helps, both beat copy:** Frozen teacher may already carry enough history. Weakens but doesn't kill multi-rate — co-training could differ.
- **Both approach copy as horizon grows:** Task gets hard. Capacity or data limitation at this scale.
- **Recurrent worse:** Optimization pathology. Investigate training dynamics before concluding anything architectural.

A positive result is cleaner evidence than a null. Null results here are ambiguous because they could reflect the frozen teacher's properties rather than architectural truth.

---

## Key nuance

This tests the **frozen teacher's Markov property**, not raw text temporal structure. Block 0 is CE-trained and may already compress history into its representation. A positive result (recurrence helps at H2+) is strong evidence that Block 1 state adds information beyond the frozen snapshot. A null result is weaker — co-trained predictive processing or a different teacher could still behave differently.

---

## Theory: why H=1 is null, and what this predicts for H=2/4

### Why no benefit at H=1

Block 0 was trained exactly to compress history into a state useful for next-token prediction. Its hidden state A0[t] IS the compressed history. Under MSE loss, Block 1's optimal prediction is E[A0[t+1] | available info]. Since A0[t] already encodes everything about the past relevant for predicting the next token/state, A0[t-1] is redundant — it's already been absorbed into A0[t] by Block 0's GRU dynamics.

The "specific transition" counter-argument (observing A0[t-1]→A0[t] reveals the token at t) is real in principle, but for one-step prediction, if that information mattered for predicting the next state, Block 0 had training pressure to encode it in A0[t] directly. Empirically, it did.

### Formal statement

The H=1 null means: past changes the conditional mean E[A0[t+1] | history] by essentially nothing beyond what E[A0[t+1] | A0[t]] already captures. This is weaker than "A0[t] is fully sufficient for the future" — it only constrains the mean, not the full conditional distribution.

### Critical insight for H=2+

**H=1 null does NOT imply H=2 null.** Here's why:

E[A0[t+2] | info] = E[ f(A0[t+1]) | info ] where f captures Block 0's dynamics.

If f is nonlinear, then two information sets with the same one-step conditional mean can still produce different two-step conditional means — because E[f(X)] ≠ f(E[X]) in general. Past history could matter for the *distribution* of A0[t+1] (higher moments) without affecting its mean, and those higher moments propagate through nonlinear dynamics to affect the H=2 mean.

### Predictions (pre-registered before running)

- **H=2:** Probably still little/no recurrence benefit. A0[t] is close to a predictive state and the dynamics should mostly factor through it. Any gain would be small.
- **H=4:** More likely to show benefit than H=2. Any state aliasing from L2 normalization compounds over longer horizons. Block 0 was trained for 1-step CE, not 4-step predictive sufficiency — features useful only at longer horizons may have been discarded.
- **Important:** Rising MSE with horizon does NOT by itself indicate recurrence should help. It may reflect irreducible uncertainty from unknown future tokens, not missing past context.

### What null at all horizons would mean

That the normalized Block 0 representation is already essentially history-sufficient — an approximate predictive-state representation. The difficulty of long-horizon prediction comes from future uncertainty (unknown tokens), not from missing past information. This would be a strong positive finding for the architecture: it means the information bottleneck works as intended.

### Conditions making recurrence most/least helpful

Most helpful: if normalization discards useful variables (aliasing), if Block 0's hidden state is bottlenecked, if there are latent regimes where the same A0[t] has different future kernels.

Least helpful: if A0[t] is close to a predictive-state representation, if the mapping A0[t] → E[A0[t+H]|A0[t]] is learnable without memory, if most uncertainty comes from unknown future tokens rather than current-state ambiguity.

---

## Controls

Required:
- Same frozen Phase A teacher for all conditions
- Same dataset split
- Same Block 1 width/depth (only recurrence toggle changes)
- Same optimizer and 800-step budget
- Horizon-specific copy baseline

If positive result found:
- 2-3 seeds at the first positive horizon
- Sequence-shuffle ablation: if recurrent advantage survives shuffled sequences, it's using extra function class, not temporal order

---

## What this does not settle

- Whether multi-rate itself helps (separate experiment)
- Whether co-trained predictive processing behaves differently from frozen-teacher
- Whether larger scale or different data changes the picture

---

## Status

**Implemented, waiting to launch.** Script: `runs/horizon_sweep.py`. Sanity-checked at H=1, 2, 4 — copy baselines correctly increase with horizon (H1: 0.002879, H2: 0.004434, H4: 0.005931). GPU currently running the 3-block stream combining experiment; horizon sweep launches after.

---

## Next steps

1. ~~Implement as a configurable-horizon variant~~ — DONE (`runs/horizon_sweep.py`)
2. Run H2 + H4 screen (single seed, 4 runs, ~2.5 hours) — NEXT when GPU is free
3. If positive: confirm with seeds + sequence-shuffle ablation
4. Feed results into multi-rate experiment design
