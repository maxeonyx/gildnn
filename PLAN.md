# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-25 midnight)

**J_fixed_embedding COMPLETE — token embeddings work as well as hidden states.** Δ=-0.012, std=0.0006. This is slightly BETTER than J_older_window's -0.010. The helper's value is temporal memory — "what tokens were here before" — not block 0's learned representations specifically. The target is external and doesn't depend on block 0's hidden state.

**Key implication:** Since the target is externally anchored (doesn't depend on block 0), it might work with strict-local training — removing the co-adaptation failure mode that killed D_strict_local.

**J_strict_local + J_fixed_strict_local running now** (ETA ~01:20 NZST). Tests both strict-local variants:
- J_strict_local: older-window target + no CE through interface
- J_fixed_strict_local: fixed-embedding target + no CE through interface

If J_fixed_strict_local works where J_strict_local fails, that confirms the external anchor is what makes strict-local viable → path to true parallelism.

**What we have:**
- A working experiment framework (multi-seed, JSONL logs, CUDA graphs, ablation metrics)
- A confirmed mechanism: older-window prediction provides block 0 with useful missing context
- Evidence that the helper's role is TEMPORAL MEMORY, not representation-specific
- Evidence that strict-local collapses, semi-local works, and the target matters more than topology
- The graph scaling path reopened: multiple helpers with different temporal bands
- A 10.8x training speedup from CUDA graph compilation

**What we still need:**
- J_strict_local + J_fixed_strict_local results (RUNNING NOW)
- J_dual_band: do temporal bands compose? (implemented, ready)
- Transformer baseline (partial run: 1.683 at 13K, on track but killed)
- Propagation-delay experiment (true architecture vision)

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

1. **Analyse J_strict_local + J_fixed_strict_local** — RUNNING (ETA ~01:20). This is the most important result pending. If J_fixed_strict_local works → path to true parallelism opens.

2. **Dual-band width test (J_dual_band)** — two rate-2 helpers at offsets (8,12). Required control: `J_dual_same_band` with offsets (8,8). Tests whether temporal bands compose. **Implementation DONE** — sanity-checked and committed.

3. **Transformer baseline** on WikiText-103 at d=256, ctx=128. Non-negotiable for interpreting custom results. Partial run was on track (1.683 at 13K).

4. **Propagation-delay experiment** — the true architecture vision. Block 1 sees block 0's output from PREVIOUS timestep only. Never tested correctly.

---

## Completed work (reference)

| What | Result | Interpretation |
|---|---|---|
| **J_older_window** | **-0.010, std 0.00009** | Target was the bottleneck; older memory provides useful missing context |
| **J_far_window** | **-0.007, std 0.0003** | Offset sensitivity is gentle inverted-U; plateau from 8 to 12 |
| **J_fixed_embedding** | **-0.012, std 0.0006** | Token embeddings ≥ hidden states; helper's role is temporal memory |
| I_phase_offset | -0.006, width saturates | Two helpers predicting same target are redundant |
| G_rate4_only | ≈ A, ablation gap 0 | Rate-4 too stale for this architecture |
| F_star_3block | SEED-SENSITIVE | Shared loss coupling, rate-4 auto-rejected |
| E_grounded | Stable, +0.026 | Task-grounded strict-local: no collapse but no help |
| D_strict_local | COLLAPSE (+1.38) | Full-state local prediction fails without CE shaping |
| C_closed_loop (v3) | -0.006 (unreliable) | Semi-local mechanism works but target is redundant |
| CUDA graph training | 10.8x speedup | Working, in core/ |
| Async hardware measurement | 28% block concurrency | CUDA streams too high-level; needs custom CUDA |
| Backend choice | PyTorch + torch.compile | Decided |
