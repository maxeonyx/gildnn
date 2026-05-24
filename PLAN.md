# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-25)

**Gated experiment complete — negative result.** B_gated (4-block, zero-init gates) is +0.245 nats WORSE than A_single at WikiText-103 ctx=128. Cold-start problem: zero-init gates starve upper blocks of information. Gate 3 opened negatively (-0.115) for suppressive use only. Pathway 3 remains blocked.

**Tied-depth experiment RUNNING.** PID 12984, `runs/tied_depth.py`, 4 variants (A_single, tied_8iter, distinct_matched, distinct_rich), 2 seeds, 20K steps each. Logging to `experiments/wikitext_103/artifacts/tied_depth/run.jsonl`. Launched 04:36 NZST. Expected completion ~06:30-07:00 (tied_8iter is slow: 80K tok/s vs 330K for A_single).

**Progress (seed 42):**
- A_single: **1.832** (6 min) ✅
- tied_8iter: **1.798** (35 min) ✅ — 0.034 below A_single, closes 14% of transformer gap
- distinct_matched: **1.818** (41 min) ✅ — between tied and single. **Weight tying is a genuine inductive bias advantage** (0.020 nats over distinct at matched params)
- distinct_rich: RUNNING (4.69M params, 65% more than tied). Started ~06:03 NZST.
- Seed 43 runs for all 4 variants: queued after distinct_rich

ETA for full experiment: ~08:45-09:00 NZST.

**C_old ablation script READY.** `runs/c_old_ablation.py` committed. Trains C_lateral (upward) vs C_isolated (no lateral, same params). Added "isolated" topology to `core/model.py`. Launch after tied-depth finishes.

**Key insight this session:** The "fix the interface" hypothesis was incomplete. Zero-init gates are worse than hardcoded 0.5 because they completely starve upper blocks. The problem isn't just the mixing coefficient — it's initialization + information routing.

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
- **NEW: Gated WikiText-103 result** — cold-start problem with zero-init gates, blocks useless
- `.gitignore` now blocks `*.pt` files (model weights never committed)

**What we DON'T have:**
- A standard transformer baseline trained at WikiText-103 ctx=128 (**DONE: 1.592 ± 0.003**)
- Any tied-depth (same block × N iterations) vs standard transformer comparison at scale
- C_old eval ablations (does C_old's improvement come from lateral communication or just ensemble?)
- Any custom CUDA concurrency beyond the Graph approach
- Self-prediction (computation compression)
- Clean dynamic-depth measurement (the probe was methodologically flawed)

**Pathway 3 status:**
- Only token_injection=all (C_old) makes multi-block useful at this scale
- Lateral-only (token_injection=block0) fails with hardcoded 0.5 (+0.014 worse) AND with zero-init gates (+0.245 worse)
- Local learning is untestable until we have a regime where blocks help under full backprop
- Key open question: does C_old actually USE lateral connections, or is it just an ensemble?

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
| 1 | **Tied-depth experiment** — `runs/tied_depth.py` | 1 | **IN PROGRESS (PID 12984).** Can iteration close the transformer gap? |
| 2 | **C_old eval ablation** — retrain C_old config, then shuffle/ablate lateral connections and per-block readout contributions | 3 | Cheapest diagnostic: does C_old's improvement come from lateral communication, or just ensemble of token-fed blocks? |
| 3 | **Custom CUDA concurrency** — persistent kernels or fused dispatch | 2 (async) | Next step after the 28% CUDA Graph result. |
| 4 | **Dynamic depth (clean measurement)** | 5 | Preliminary probe showed heterogeneity but methodology was flawed. Needs clean redo. |
| 5 | **Local learning in C_old config** — if C_old ablations show lateral IS used | 3 | Stop-gradient + local CE on a regime where blocks are known useful. Only do after #2 confirms lateral matters. |

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
| **Gated multi-block at WikiText-103 ctx=128** | **3** | **Cold-start failure** | **4-block with zero-init gates is +0.245 worse than single-block. Gates starve upper blocks of signal — worse than hardcoded 0.5. Gate 3 opened negatively (suppressive). Experiment uninformative about original question due to cold-start confound.** |
| Multi-rate [1,2,4,8] | 8 | Spectator problem | Block 0 sees all tokens → no specialization |
| Closed-loop variants A-J | — | Marginal (-0.006 to -0.012) | Architecturally incoherent. Not on any pathway. Done. |
| CUDA graph training | infra | 10.8x speedup | In core/ |
| CUDA Graph concurrency | 2 | 28% speedup | Hardware CAN do concurrent execution |
| Stale-read quality cost | 2 | +0.005 ± 0.005 nats | Negligible (95% CI crosses zero) |
| Backend choice | infra | PyTorch + torch.compile | Decided |
| **Transformer baseline (WikiText-103 ctx=128)** | **1** | **val_loss 1.592 ± 0.003, 2.86M params, 2 seeds** | **External anchor for Pathway 1. 0.240 nats better than A_single.** |

---

## Roadmap evidence to suggest to Max

(Agent records evidence here; Max decides whether to update ROADMAP.md)

- **Pathway 1 (Wide Recurrent):** Weight sharing is free (3-iteration tied = 3-layer distinct). Iteration scaling shows no instability through 12, quality peaks around 8 on single seed. Diminishing returns flatten around depth 8–11. Pathway is alive and well-grounded at tiny rung.
- **Pathway 2 (Async):** Prior positive — 28% concurrency via CUDA Graphs, stale reads don't hurt. Next: custom CUDA.
- **Pathway 3 (Local Learning):** Confidence **decreased**. Every attempt at lateral-only multi-block has failed: hardcoded 0.5 (+0.014 worse), zero-init gates (+0.245 worse, cold-start trap). Only token_injection=all (C_old, -0.021 better) makes blocks useful — but that may not involve lateral communication at all (might just be an ensemble). **Local learning remains untestable until we confirm lateral communication is actually used in a working multi-block config.** Suggest roadmap update: note that Pathway 3 is blocked pending C_old ablation study; the original "propagation-delay → local learning" progression has not reached the point where local learning can be tested.
- **Pathway 5 (Dynamic Depth):** Now enabled by Pathway 1 iteration scaling. Per-depth losses show clear variation — some tokens probably benefit more from extra iterations than others. First measurement (per-token variance) not yet done.
- **Pathway 8 (Multi-Rate):** Prior weak-positive — inductive bias confirmed at ctx=32. Needs longer context to be meaningful.
- **Pathway 9 (Norm-Preserving):** De-prioritized after iteration scaling showed no instability through 12 on tiny rung. May still matter at larger scale or higher iteration counts.
