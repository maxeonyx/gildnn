# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-25)

**Gated multi-block experiment running.** Testing whether zero-init learnable gates fix the lateral spectator problem at WikiText-103 ctx=128. Prior session discovered existing ctx128_corrected data showing spectator persists with hardcoded 0.5 mixing at this scale. C_old (token_injection=all) proves blocks CAN help (+0.02 nats). The gated experiment discriminates: is the problem the interface or is lateral-only fundamentally weak?

**Background run active:** `experiments/gated_wikitext/run.py`, PID 23572, logging to `experiments/gated_wikitext/artifacts.ignore/run.jsonl`. A_single tracking existing results perfectly. Expected completion: ~04:20-04:30 NZST.

Key findings so far this session:
- PLAN.md priorities were wrong: the "scale to WikiText-103 ctx=128" experiment had ALREADY been partially done (ctx128_corrected results)
- Corrected priorities: gated multi-block is the discriminating test, not raw scaling
- Propagation-delay README updated with existing evidence

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
- A standard transformer baseline trained at WikiText-103 ctx=128 (script exists, only sanity-checked)
- Any test of the **gated** (zero-init) multi-block architecture at WikiText-103 ctx=128
- Any tied-depth (same block × N iterations) vs standard transformer comparison at scale
- Any custom CUDA concurrency beyond the Graph approach
- Self-prediction (computation compression)
- Clean dynamic-depth measurement (the probe was methodologically flawed)

**Critical existing evidence overlooked by prior sessions:**
- `experiments/wikitext_103/artifacts/ctx128_corrected/`: 4-block corrected (hardcoded 0.5 mixing, token_injection=block0) is +0.014 WORSE than single-block at WikiText-103 ctx=128. Same spectator pattern as TinyShakespeare.
- 4-block OLD (token_injection=all) is -0.021 BETTER — proving extra blocks CAN help at this scale, but only with fresh token injection.
- The hardcoded 0.5 mixing was the interface tested. The zero-init gate (proven to fix the interface at TinyShakespeare) has never been tested at this scale.

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
| 1 | **Gated corrected at WikiText-103 ctx=128** — 4-block with zero-init gate instead of hardcoded 0.5. Does fixing the interface make blocks useful? | 3 | Directly discriminates: was B's failure the bad interface, or is lateral-only fundamentally insufficient? C_old proves blocks CAN help at this scale. |
| 2 | **Transformer baseline at WikiText-103 ctx=128** — train `runs/transformer_baseline.py` | 1 | Missing control number. Can't compare tied-depth without knowing what standard transformer achieves. |
| 3 | **Tied-depth vs standard transformer at WikiText-103 ctx=128** | 1 | The actual Pathway 1 question at scale. Requires baseline first. |
| 4 | **Custom CUDA concurrency** — persistent kernels or fused dispatch | 2 (async) | Next step after the 28% CUDA Graph result. |
| 5 | **Gradient radius sweep** — vary stop-gradient from k=1 to k=full | 3 (local learning) | Depends on gated blocks being useful first. |
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
| **Multi-block corrected at WikiText-103 ctx=128** | **3** | **Spectator** | **4-block corrected (hardcoded 0.5) is +0.014 worse than single-block. 4-block old (token_injection=all) is -0.021 better. Blocks help when fed fresh tokens; lateral-only with 0.5 mixing fails.** |
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
