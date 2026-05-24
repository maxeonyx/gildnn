# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-24 evening)

**Phase 5 J (older-window prediction target) is confirmed and Max has endorsed the direction.** Block 1 predicting `mean(h_{t-8}..h_{t-5})` gives -0.010 with 53× less seed variance than the prior full-state approach. Per [dictation 2026-05-24-5](dictations/2026-05-24-5.md): "Block one should learn to predict something about block zero that block zero couldn't already know." Per [dictation 2026-05-24-7](dictations/2026-05-24-7.md): "Interesting, it's quite a dumb idea, but I like it."

**J_far_window (offset 12) running now — tracking toward "plateau" outcome.** Seed 42 finished: val_loss 1.663 (vs J's 1.660). Difference is 0.003 — essentially noise. Seed 43 in progress. If confirmed: the useful temporal offset is a broad band, which motivates multi-helper-different-offset experiments.

**What we have:**
- A working experiment framework (multi-seed, JSONL logs, CUDA graphs, ablation metrics)
- A confirmed mechanism: older-window prediction provides block 0 with useful missing context
- Evidence that strict-local collapses, semi-local works, and the target matters more than topology
- The graph scaling path reopened: multiple helpers with different temporal bands
- A 10.8x training speedup from CUDA graph compilation

**What we still need:**
- A full transformer baseline (partial run: 1.683 at 13K steps, on track but killed for GPU use)
- J_fixed_embedding: separates "older content" from "older hidden-state codes specifically"
- Multi-helper with different offsets: tests whether temporal bands compose
- J_strict_local: tests whether the good target rescues true locality (no CE through interface)

---

## The agent's working loop

This is the loop the agent should follow. Not a phase plan — a loop with exit conditions and redirect paths.

```
ORIENT → CHOOSE → THEORY → RUN → ANALYZE → CHECK → (loop or redirect)
```

**Orient:** Read PLAN.md, check time/reports, check active runs. Understand where we are.

**Choose:** Pick the cheapest useful next step that advances a ROADMAP pathway. If unsure, prefer:
- Pathways that haven't been tested at all (breadth > depth on marginal signals)
- Missing baselines (we need grounding before more custom work)
- Theory work over another experiment if the question isn't clear yet

**Theory:** Before running, write the hypothesis. What do we expect? What would increase/decrease confidence? Is this actually the cheapest test?

**Run:** Execute. Follow visibility and background-execution rules.

**Analyze:** What happened? What does it teach about the pathway?

**CHECK (exit conditions):**
- Is the result meaningful (>0.05 nats, or qualitatively informative)?
  - YES → record, continue on this pathway
  - NO → record what we learned, REDIRECT to a different pathway or a different approach
- Is the next experiment on this pathway still the cheapest useful thing?
  - YES → iterate
  - NO → redirect to whatever IS cheapest
- Am I amplifying a marginal signal?
  - YES → STOP. This is the primary failure mode. Record and redirect.

---

## What should happen next

Priority order (not a sequence — pick whichever is cheapest to do honestly right now):

1. **Analyse J_far results** — when seed 43 finishes. Then update decision framework and proceed.

2. **J_fixed_embedding** — most discriminating next experiment regardless of J_far outcome. Separates temporal-memory hypothesis (older content helps) from representation-specific hypothesis (older hidden-state codes specifically help).

3. **Dual-band width test (J_dual_band)** — two rate-2 helpers at offsets (8,12). Required control: `J_dual_same_band` with offsets (8,8). Tests whether temporal bands compose. Implementation needs per-helper prediction window offsets (small code change to loss plumbing).

4. **J_strict_local** — older-window target + strict-local (fully detached feedback). Tests whether a good target rescues true parallelism. Already implemented in code.

5. **Transformer baseline** on WikiText-103 at d=256, ctx=128. Non-negotiable for interpreting custom results. Partial run was on track (1.683 at 13K).

6. **Propagation-delay experiment** — the true architecture vision. Block 1 sees block 0's output from PREVIOUS timestep only. Never tested correctly.

---

## Completed work (reference)

| What | Result | Interpretation |
|---|---|---|
| **J_older_window** | **-0.010, std 0.00009** | Target was the bottleneck; older memory provides useful missing context |
| J_far_window (partial, s42) | -0.006 (1.663 vs 1.669) | Offset sensitivity is shallow — plateau from 8 to 12 |
| I_phase_offset | -0.006, width saturates | Two helpers predicting same target are redundant |
| G_rate4_only | ≈ A, ablation gap 0 | Rate-4 too stale for this architecture |
| F_star_3block | SEED-SENSITIVE | Shared loss coupling, rate-4 auto-rejected |
| E_grounded | Stable, +0.026 | Task-grounded strict-local: no collapse but no help |
| D_strict_local | COLLAPSE (+1.38) | Full-state local prediction fails without CE shaping |
| C_closed_loop (v3) | -0.006 (unreliable) | Semi-local mechanism works but target is redundant |
| CUDA graph training | 10.8x speedup | Working, in core/ |
| Async hardware measurement | 28% block concurrency | CUDA streams too high-level; needs custom CUDA |
| Backend choice | PyTorch + torch.compile | Decided |
