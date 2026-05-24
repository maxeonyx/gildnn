# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-25)

**Process overhaul complete.** Max reviewed the project direction and found the loop agent was stuck amplifying marginal signals (the "closed loop prediction" / J experiment series) with no connection to the actual project thesis. Three documents restructured: VISION.md (requirements), ROADMAP.md (11 research pathways), PROCESS.md (nested loops with adversarial review gates).

The "closed loop prediction" experiment family (variants A through J, agent-coined name) is **done**. It produced some useful evidence about local learning failure modes but the mechanism itself is architecturally incoherent and the results are marginal. Do not continue it.

**What we have:**
- Working experiment infrastructure (training loop, eval, multi-seed, ablation, JSONL logs, CUDA graphs)
- Transformer baseline: val_loss 1.643, 186K params, TinyShakespeare ctx=32 (in base-experiments/)
- RNN baseline: val_loss 1.711, 186K params, TinyShakespeare ctx=32 (in base-experiments/)
- Evidence: strict-local collapses, semi-local barely helps, prediction target matters more than topology
- Evidence: multi-rate [1,2,4,8] provides inductive bias (better per-step val_loss)
- Evidence: CUDA Graph concurrency gives 28% speedup; stale reads don't hurt quality
- 10.8x CUDA graph training speedup in core/

**What we DON'T have:**
- The fundamental comparison (Pathway 1): wide recurrent vs transformer at matched compute
- Any experiment with real propagation delay
- Any experiment with self-prediction (computation compression)
- Any custom CUDA concurrency beyond the Graph approach
- Any experiment at ctx > 128

---

## The agent's working loop

Follow PROCESS.md. The short version:

```
PROCESS CHECK → PATHWAY SELECTION → [adversarial gate] →
CONCEPTUAL CLARIFICATION → [adversarial gate] →
EXPERIMENT DESIGN → [adversarial gate] →
RUN & ANALYZE → [adversarial gate] →
INTEGRATE / REPORT / UPDATE PLAN → back to PATHWAY SELECTION
```

Every gate is a separate subagent review that can send you back. See PROCESS.md for full details.

---

## What should happen next

Pick from this list based on cheapest honest test. These connect to specific roadmap pathways:

| Priority | Experiment | Pathway | Why |
|---|---|---|---|
| 1 | **Propagation-delay 2-block** — block 1 sees block 0's PREVIOUS output only | 1, 3 | The actual architecture vision. Genuinely new direction (breadth). Never tested correctly. |
| 2 | **Scale to WikiText-103 ctx=128** — does tied-depth still match baseline at real scale? | 1 | Critical for external validity. TinyShakespeare results may not generalize. |
| 3 | **Gradient radius sweep** — vary stop-gradient from k=1 to k=full | 3 (local learning) | Quantifies the locality constraint. Depends on propagation-delay architecture. |
| 4 | **Custom CUDA concurrency** — persistent kernels or fused dispatch | 2 (async) | Next step after the 28% CUDA Graph result. |
| 5 | **Muon optimizer swap** | 9 | De-prioritized: no instability found through 12 iterations on tiny rung. |
| 6 | **Dynamic depth (clean measurement)** | 5 | Preliminary probe showed heterogeneity but methodology was flawed. Needs clean redo. |

---

## Completed work (reference)

| What | Pathway | Result | Interpretation |
|---|---|---|---|
| Transformer baseline | all | val_loss 1.643, 186K params, TinyShakespeare ctx=32 | Done, in base-experiments/ |
| RNN baseline | all | val_loss 1.711, 186K params | Done, in base-experiments/ |
| **Tied-depth vs transformer (tiny rung)** | **1** | **Identical: mean diff 0.0002 nats** | **Weight sharing is free. Viable architecture.** |
| **Iteration scaling (6, 8, 12)** | **1** | **No instability; quality peaks ~8** | **Diminishing returns after 8. Limit is optimization/representational, not numerical.** |
| Per-token depth heterogeneity | 5 | Inconclusive | Heterogeneity exists but methodology flawed (non-standard eval frame, ε too loose). Hint only. |
| Multi-rate [1,2,4,8] | 8 | Spectator problem | Block 0 sees all tokens → no specialization |
| Closed-loop variants A-J | — | Marginal (-0.006 to -0.012) | Architecturally incoherent. Not on any pathway. Done. |
| CUDA graph training | infra | 10.8x speedup | In core/ |
| CUDA Graph concurrency | 2 | 28% speedup | Hardware CAN do concurrent execution |
| Stale-read quality cost | 2 | +0.005 ± 0.005 nats | Negligible (95% CI crosses zero) |
| Backend choice | infra | PyTorch + torch.compile | Decided |

---

## Roadmap evidence to suggest to Max

(Agent records evidence here; Max decides whether to update ROADMAP.md)

- **Pathway 1 (Wide Recurrent):** Weight sharing is free (3-iteration tied = 3-layer distinct). Iteration scaling shows no instability through 12, quality peaks around 8 on single seed. Diminishing returns flatten around depth 8–11. Pathway is alive and well-grounded at tiny rung.
- **Pathway 2 (Async):** Prior positive — 28% concurrency via CUDA Graphs, stale reads don't hurt. Next: custom CUDA.
- **Pathway 3 (Local Learning):** Prior negative — strict-local collapses. Prior weak-positive — neighborhood-local helps slightly BUT tested on wrong architecture. Needs redo with propagation delay.
- **Pathway 5 (Dynamic Depth):** Now enabled by Pathway 1 iteration scaling. Per-depth losses show clear variation — some tokens probably benefit more from extra iterations than others. First measurement (per-token variance) not yet done.
- **Pathway 8 (Multi-Rate):** Prior weak-positive — inductive bias confirmed at ctx=32. Needs longer context to be meaningful.
- **Pathway 9 (Norm-Preserving):** De-prioritized after iteration scaling showed no instability through 12 on tiny rung. May still matter at larger scale or higher iteration counts.
