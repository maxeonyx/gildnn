# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## What just happened

**Strict-local COLLAPSES.** Phase 2 complete. Full-state prediction as a local objective doesn't work — the predictor learns task-irrelevant patterns, block 0 co-adapts, then the system collapses to the same degenerate state as v1.

| Variant | Mean val_loss | vs A |
|---------|--------------|------|
| A_single | 1.670 | — |
| C_closed_loop (semi-local) | 1.665 | -0.006 |
| **D_strict_local** | **3.047** | **+1.377 (COLLAPSE)** |

The decisive finding: **CE shaping through the feedback path is load-bearing.** Without task gradient reaching block 1, predictions aren't task-aligned and the system eventually collapses.

From dictation 2026-05-23-5: "It doesn't have to be totally local. We can be using backpropagation through a local neighborhood of blocks." The semi-local result IS backprop through a local neighborhood (the interface). It's legitimate.

## What's next

The full-state cosine prediction target is dead as a local objective. Three live paths:

### 1. Accept neighborhood-local as mainline, scale to N=3 (recommended)
- Semi-local with N=3 blocks: each block predicts its neighbor, CE propagates through interfaces
- Tests whether semi-local extends beyond 2 blocks
- If it does: the architecture parallelizes in proportion to N (each block pair is a "neighborhood")
- Max explicitly allows this: "backprop through a local neighborhood"

### 2. Task-grounded prediction target (fresh strict-local attempt)
- Replace "predict full hidden state" with a target that's task-relevant by construction:
  - Predict next-token logits / teacher distribution (local LM loss)
  - Predict a CE-grounded bottleneck latent (small projection that's trained to be CE-relevant)
  - Target propagation / synthetic gradients
- The fundamental problem with full-state: predicting all dimensions equally doesn't select for task-relevance
- A task-grounded target constrains the objective to only predict useful things

### 3. Per-dimension gate + semi-local (protector, not solution)
- Replace scalar gain with vector gate (zero-init Linear)
- Lets CE suppress nuisance prediction dimensions, keep useful ones
- Reduces co-adaptation risk but doesn't solve the fundamental target problem
- Worth combining with either path above

**Currently running:** E_grounded (path 2 at N=2). PID 24688, log at `experiments/wikitext_103/artifacts/closed_loop_prediction/run_grounded.jsonl`. Variants: A_single + E_grounded, 2 seeds, 20K steps, no-compile (CUDA graphs). Expected ~1.5 hours total. Started ~11:45am NZST.

**Decision rules:**
- **E does NOT collapse:** The problem was the target, not locality. Genuine strict-local learning works with task-grounded objectives. → Scale to N=3 with local CE on every block.
- **E collapses anyway:** Locality itself is too weak, even with good targets. → Accept neighborhood-local as mainline, test width scaling.
- **E matches A but no benefit:** Block 1's local CE grounds it, but its predictions don't help block 0. → The prediction mechanism itself might not be the right communication channel.

## Critical findings (carry forward)

1. **MixAdd sqrt formula at init=0.9 gives 31.6% coefficient, not 10%.** Root cause of v1/v2 collapse.
2. **Additive zero-init gate works.** No collapse. Model learns gain automatically. Negative gain = predictive coding.
3. **Semi-local is NOT genuinely local.** CE flows through feedback. Current v3 is global backprop through a narrow interface. ← CONFIRMED by strict-local collapse.
4. **Full-state cosine prediction is a bad local objective.** It doesn't select for task-relevant components. Predictor learns easy/average patterns, block 0 co-adapts, system collapses.
5. **Neighborhood-local is legitimate.** Max explicitly allows it. Interface-level CE shaping works.
6. **The spectator problem is solved by role differentiation.** B hurts on average (+0.006). Closed-loop gives block 1 a unique function.

## Queue

- Transformer matched-compute baseline (unfair at total params — backbone 263K vs 1.53M)
- Named/typed tensor dimensions
- Loop management tooling
- Graph architecture from dictation

## Key completed findings

| Experiment | Result | Notes |
|---|---|---|
| **Strict-local (D)** | **COLLAPSE** (+1.38) | CE shaping is load-bearing. Full-state local prediction fails. |
| **Closed-loop v3 (C)** | **C < A by 0.006** (2 seeds) | Mechanism active, net benefit tentative. Semi-local. |
| **Closed-loop v1** | **COLLAPSE** (+1.47) | pred_loss trained block 0 to be constant |
| **Closed-loop v2** | **COLLAPSE → NaN** | MixAdd 31.6% prior = stable collapsed fixed point |
| ctx=128 corrected | **HURTS** (+0.014) | Spectator worse at longer context |
| ctx=128 ensemble | Tiny benefit (-0.020) | Was -0.101 at ctx=32; collapses |
| ctx=32 baseline | B wins (-0.101), C≈A | Spectator on corrected arch |
| ctx=32 local aux loss | **NULL** (+0.003) | Gradient isn't the problem |
| ctx=32 equal readout | **HURTS** (+0.024) | Information poverty confirmed |
| ctx=32 temporal window | **NULL** (+0.003) | Learned projection of history doesn't help |
| Bidirectional top-down | **HURTS** | TinyShakespeare |
| CUDA Graph training | 10.86× speedup | GraphTrainer in core/ |
| Self-prediction | NULL | Zero effect |
