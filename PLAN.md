# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## Current state

**The multi-rate architecture is fundamentally broken.** Six WikiText-103 experiments now show:

1. Under corrected architecture (block0 only): upper blocks are spectators at ALL context lengths.
2. Under old architecture (all tokens): the ensemble benefit **collapses from -0.101 (ctx=32) to -0.020 (ctx=128).**
3. The mechanism "process less frequently" doesn't create useful hierarchy — it just means "miss more information."

### ctx=128 results (definitive)

| Variant | Mean val_loss | vs A |
|---|---|---|
| A_single | 1.838 | — |
| B_corrected (block0, rates 1,2,4,8) | 1.852 | +0.014 (HURTS) |
| C_old (all tokens, rates 1,2,4,8) | 1.818 | -0.020 (tiny) |

Compare ctx=32 where C_old was -0.101. The architecture becomes LESS useful at longer context.

### Why multi-rate fails at longer context

With rates=(1,2,4,8) and ctx=128:
- Block 0: 128 activations (sees everything)
- Block 1: 64 activations (misses half)
- Block 2: 32 activations (misses 3/4)
- Block 3: 16 activations (misses 7/8)

Slow blocks don't "model longer-range patterns" — they just **miss tokens**. At ctx=32 the damage was small enough that ensemble averaging helped. At ctx=128 the gaps are too large.

### Full diagnostic evidence (ctx=32, WikiText-103)

| Intervention | Result | What it ruled out |
|---|---|---|
| Baseline corrected | NULL (+0.003) | Not just small dataset |
| Aux prediction loss | NULL (+0.003) | Not gradient signal |
| Equal readout | HURTS (+0.024) | Not readout collapse |
| Temporal window (k=8) | NULL (+0.003) | Not missing trajectory info |
| ctx=128 | HURTS (+0.014) | Not short context |

**Conclusion:** The multi-rate diagonal architecture cannot support useful multi-block learning under any tested conditions. The corrected architecture (only block 0 sees tokens) is definitively broken. The old architecture (ensemble) loses its benefit at longer context.

## Active

Nothing running. GPU free.

## Next — fundamental rethink needed

The current architecture is exhausted. We need a genuinely different approach to multi-module parallel learning. Key constraint from Max's dictation: "how can I get local learning, i.e. enabling parallelism?"

Possible new directions (need fresh think session to evaluate):

- [ ] **Aggregation-based slow blocks** — Instead of "process every Nth step and skip the rest," slow blocks could AGGREGATE multiple steps (pool/attend over N consecutive fast-block outputs before processing). This is "summarize, then process" rather than "ignore, then process."
- [ ] **Heterogeneous specialization** — Different blocks get different computation types entirely (e.g., one does local ngram patterns, one does position-invariant features). Rather than identical blocks at different rates.
- [ ] **True mixture-of-experts** — Route different tokens to different blocks. Each block processes all tokens that route to it. This is the established approach to parallel specialization.
- [ ] **Transformer baseline** — Before pursuing any new architecture, know where we stand. Build a standard transformer at matched compute.

## Queue

- Named/typed tensor dimensions
- Loop management tooling
- Graph architecture idea from dictation

## Key completed findings

| Experiment | Result | Notes |
|---|---|---|
| ctx=128 corrected | **HURTS** (+0.014) | Spectator worse at longer context |
| ctx=128 ensemble | Tiny benefit (-0.020) | Was -0.101 at ctx=32; collapses |
| ctx=32 baseline | B wins (-0.101), C≈A | Spectator on corrected arch |
| ctx=32 local aux loss | **NULL** (+0.003) | Gradient isn't the problem |
| ctx=32 equal readout | **HURTS** (+0.024) | Information poverty confirmed |
| ctx=32 temporal window | **NULL** (+0.003) | History doesn't help |
| Bidirectional top-down | **HURTS** | TinyShakespeare |
| CUDA Graph training | 10.86× speedup | GraphTrainer in core/ |
| Self-prediction | NULL | Zero effect |
