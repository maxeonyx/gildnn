# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## Current state

**The multi-rate architecture is fundamentally broken.** Six WikiText-103 experiments confirm spectator collapse.

**Root cause (refined):** Not just "missed tokens." The temporal window (k=8, learned projection of block 0's recent outputs) was also NULL — proving the problem isn't information access. It's that **same-objective identical blocks have no reason to specialize.** Block 0 already does the job optimally; giving other blocks more information doesn't help because they have nothing DIFFERENT to do with it.

**Implication:** The next experiment must give blocks DIFFERENT ROLES, not just different information or rates.

## Active — closed-loop hierarchical prediction experiment

**Hypothesis:** If upper blocks predict block 0's FUTURE states (a different objective) and feed predictions BACK into block 0 (closed-loop), they become genuinely useful — not as token predictors but as dynamics modelers.

**Design (from 4 architectural think sessions):**

- 2 blocks: block 0 (rate=1), block 1 (rate=2)
- Block 0: sees tokens, predicts next token (CE from final state)
- Block 1: predicts block 0's next 2 states via prediction_head → `[batch, 2, d_model]`
- Closed loop: block 1's predictions fed back to block 0 via MixAdd at each step
- Loss: `CE + 0.1 * cosine_prediction_loss` (on LayerNorm'd targets, stop-grad)
- Readout: block 0 only (for variant C)
- Fixed-shape buffers for CUDA graph compatibility

**Variants:**
- A: single block (baseline)
- B: 2 blocks, shared CE, corrected arch (reconfirms spectator)
- C: 2 blocks, closed-loop hierarchical prediction

**Diagnostics:**
- val_loss
- prediction_mix coefficient over training
- pred_loss trajectory
- C ablation: zero out priors at eval → if CE unchanged, block 1 still spectator

**Decision rules:**
- C > A → hierarchical prediction genuinely helps! Next: add more blocks, test local learning
- C ≈ A > B → prediction prevents harm but no benefit (need richer mechanism)
- C ≈ B → closed-loop not enough either (may need structural change to residual stream)

## Why NOT aggregation (mean-pooling)

Temporal window (k=8) already tested a SUPERSET of mean-pooling — a learned linear projection of block 0's last 8 states. Result: NULL. Mean-pooling is a special case of learned projection and cannot outperform it. Skip.

## Queue

- Named/typed tensor dimensions
- Loop management tooling
- Graph architecture idea from dictation
- Transformer matched-compute baseline

## Key completed findings

| Experiment | Result | Notes |
|---|---|---|
| ctx=128 corrected | **HURTS** (+0.014) | Spectator worse at longer context |
| ctx=128 ensemble | Tiny benefit (-0.020) | Was -0.101 at ctx=32; collapses |
| ctx=32 baseline | B wins (-0.101), C≈A | Spectator on corrected arch |
| ctx=32 local aux loss | **NULL** (+0.003) | Gradient isn't the problem |
| ctx=32 equal readout | **HURTS** (+0.024) | Information poverty confirmed |
| ctx=32 temporal window | **NULL** (+0.003) | Learned projection of history doesn't help |
| Bidirectional top-down | **HURTS** | TinyShakespeare |
| CUDA Graph training | 10.86× speedup | GraphTrainer in core/ |
| Self-prediction | NULL | Zero effect |
