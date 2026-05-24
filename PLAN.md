# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-25)

**Pathway 3 tested at tiny rung — spectator result.** The propagation-delay 2-block architecture was tested at TinyShakespeare ctx=32. Key findings:
- Hardcoded 0.5 lateral mixing is destructive (2-block worse than 1-block by +0.124 nats)
- Zero-init learnable gate fixes the interface, but block B is a spectator regardless (+0.002 nats ablation)
- The task is too easy for 1 block to benefit from a second block
- Local learning cannot be tested until a regime exists where B helps under full backprop
- **Next: scale to WikiText-103 ctx=128** where 1 block should be insufficient

**What we have:**
- Working experiment infrastructure (training loop, eval, multi-seed, ablation, JSONL logs, CUDA graphs)
- Transformer baseline: val_loss 1.643, 186K params, TinyShakespeare ctx=32 (in base-experiments/)
- RNN baseline: val_loss 1.711, 186K params, TinyShakespeare ctx=32 (in base-experiments/)
- **Pathway 1 confirmed:** tied-depth = distinct-layer at matched params (mean diff 0.0002 nats, 3 seeds)
- **Iteration scaling:** no instability through 12, quality peaks ~8, diminishing returns after that
- Evidence: strict-local collapses, semi-local barely helps, prediction target matters more than topology
- Evidence: multi-rate [1,2,4,8] provides inductive bias (better per-step val_loss)
- Evidence: CUDA Graph concurrency gives 28% speedup; stale reads don't hurt quality
- 10.8x CUDA graph training speedup in core/
- `.gitignore` now blocks `*.pt` files (model weights never committed)

**What we DON'T have:**
- Any experiment with real propagation delay (the actual architecture vision)
- Any experiment at scale beyond TinyShakespeare ctx=32 (~70K-186K params)
- Any custom CUDA concurrency beyond the Graph approach
- Self-prediction (computation compression)
- Clean dynamic-depth measurement (the probe was methodologically flawed)

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
| 1 | **Scale to WikiText-103 ctx=128** — does tied-depth match baseline at real scale? Multi-block useful? | 1, 3 | Critical for external validity AND for testing multi-block at a scale where 1 block is insufficient. |
| 2 | **Propagation-delay at scale** — rerun 2-block gated experiment at larger scale where single-block is insufficient | 3 | Can only test local learning where B contributes under full backprop first. |
| 3 | **Custom CUDA concurrency** — persistent kernels or fused dispatch | 2 (async) | Next step after the 28% CUDA Graph result. |
| 4 | **Gradient radius sweep** — vary stop-gradient from k=1 to k=full | 3 (local learning) | Depends on propagation-delay being useful first. |
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
| **Propagation-delay 2-block (tiny)** | **3** | **Spectator** | **Block B adds nothing at TinyShakespeare ctx=32. Hardcoded 0.5 mixing harmful; zero-init gate fixes ceiling but B stays closed.** |
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
- **Pathway 3 (Local Learning):** Prior negative — strict-local collapses. Propagation-delay 2-block tested at TinyShakespeare ctx=32: block B is a spectator even with full backprop — task too easy for 1 block. Hardcoded 0.5 lateral mixing is actively harmful (creates destructive coupling); zero-init learnable gate fixes the interface but B still adds nothing at this scale. **Local learning cannot be tested until we find a regime where B helps under full backprop.** WikiText-103 ctx=128 is the next test bed.
- **Pathway 5 (Dynamic Depth):** Now enabled by Pathway 1 iteration scaling. Per-depth losses show clear variation — some tokens probably benefit more from extra iterations than others. First measurement (per-token variance) not yet done.
- **Pathway 8 (Multi-Rate):** Prior weak-positive — inductive bias confirmed at ctx=32. Needs longer context to be meaningful.
- **Pathway 9 (Norm-Preserving):** De-prioritized after iteration scaling showed no instability through 12 on tiny rung. May still matter at larger scale or higher iteration counts.
