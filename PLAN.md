# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-24)

The loop agent has been running "closed loop prediction" experiments (variants A through J) for several days. These tested whether a 2-block architecture with a prediction mechanism could outperform a single block.

**The honest assessment:** These experiments were a local optimization loop that lost connection to the project vision. The agent found a small signal (C: -0.006 nats) and spent many GPU-hours trying to amplify it (D, E, F, G, H, I, J) without asking whether the direction leads anywhere. Experiment J ("J_older_window") achieved -0.010 nats — statistically reliable but practically meaningless. The mechanism is architecturally incoherent (predicts past states, uses current input with zero propagation delay, negative gain emerged ad-hoc). It doesn't connect to any pathway in the roadmap in a principled way.

**What we DO have from this work:**
- A working experiment framework (training loop, evaluation, multi-seed, ablation, JSONL logging, CUDA graphs)
- Evidence that strict-local learning collapses with ungrounded targets
- Evidence that neighborhood-local (semi-local) helps slightly
- Evidence that the prediction TARGET matters more than the topology
- Evidence that multi-rate [1,2,4,8] provides an inductive bias (better val_loss per compute step)
- Evidence of ~28% block concurrency on RTX 3090 with simple CUDA stream API
- A 10.8x training speedup from CUDA graph compilation

**What we DON'T have:**
- A transformer baseline achieving expected published results on our dataset
- Any experiment with principled local learning (predictive coding, target propagation, etc.)
- Any experiment with real propagation delay (block N only sees block N-1's PREVIOUS output)
- Any custom CUDA work testing true async concurrency
- Any experiment at context lengths >128
- Any experiment where interior blocks DON'T see block 0's current output

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

1. **Transformer baseline** on WikiText-103 at d=256, ctx=128+. Non-negotiable — we can't interpret custom results without it. (Supports all pathways.)

2. **Propagation-delay experiment** — the actual architecture vision, never tested correctly. Block 0 gets tokens. Block 1 sees block 0's output from PREVIOUS timestep only. Block 1's local loss: predict own next input. (Pathway 4: Self-Prediction + Pathway 2: Local Learning.)

3. **Gradient radius sweep** — same architecture, vary stop-gradient window from k=1 through k=full. Find the knee. (Pathway 2: Local Learning, Step 2 from worked example.)

4. **Muon optimizer swap** — may fix the k=8/k=16 instability that was previously observed. Quick to test. (Pathway 7: Norm-Preserving.)

5. **Context length scaling** — ctx=256, 512. Needed for multi-rate to be meaningful. (Pathway 3: Multi-Rate.)

6. **Custom CUDA concurrency test** — can two independent matmuls actually overlap on the 3090? (Pathway 1: Async Execution.)

---

## Completed work (reference)

| What | Pathway | Result | Interpretation |
|---|---|---|---|
| Multi-rate [1,2,4,8] (corrected arch) | 3 | Spectator problem | Block 0 sees all tokens → no specialization pressure |
| Closed-loop C (semi-local) | 2, 4 | -0.006 nats, no ablation gap | Regularization only — not a real mechanism |
| Closed-loop D (strict local) | 2 | Collapsed (+1.38) | Strict local with ungrounded target is dead |
| Closed-loop E (grounded local CE) | 2 | Stable, +0.026 | Doesn't collapse but doesn't help |
| Closed-loop J (older window) | — | -0.010, ablation gap 0.18 | Load-bearing but architecturally incoherent. Not on any pathway. |
| CUDA graph training | infrastructure | 10.8x speedup | Working, in core/ |
| Async hardware measurement | 1 | 28% block concurrency | CUDA streams — too high-level, needs custom CUDA |
| Backend choice | infrastructure | PyTorch + torch.compile | Decided |
