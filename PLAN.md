# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## Active — closed-loop prediction v3 (additive gain=0) RUNNING

**PID 20572**, log: `experiments/wikitext_103/artifacts/closed_loop_prediction/run_v3.jsonl`

### What happened in v1 and v2

**V1 (no detach):** Catastrophic collapse — val_loss stuck at 3.14 (vs A's 1.67). pred_loss gradient flowed into block 0, training it to be constant.

**V2 (s0.detach()):** STILL collapsed at 3.15, then went NaN at step 10K. Detach solved the wrong problem.

**Root cause (corrected):** Not gradient flow — forward-path instability. The MixAdd formula `sqrt(0.9)*seed + sqrt(0.1)*prior = 0.949*seed + 0.316*prior` gives **31.6% prior influence** (not 10% as intended). Combined with final-only CE supervision over 128 recurrent steps, this creates a stable collapsed fixed point. The untrained prior contaminates every timestep, and one CE signal at position 128 can't overcome 127 steps of corruption.

### V3 fix (current)

Replace MixAdd with **additive residual gate**:
```python
x0 = seed0 + prior_gain * layer_norm(prior_t)  # gain=0 at init
```

Block 0 starts IDENTICAL to A_single. Gain grows only if CE discovers predictions help. No contamination at initialization. LayerNorm keeps prior magnitude controlled.

### Expected behavior

- Steps 0-1K: C tracks A closely (gain ≈ 0, block 0 is effectively A_single)
- Steps 1K+: EITHER gain grows (predictions help → C < A) OR gain stays ≈ 0 (null result → C ≈ A)
- **NO collapse** regardless — gain=0 means block 0 always has a clean training path

### Decision rules

- **C < A (val_loss):** Predictions help! Gain learned to incorporate them. Proceed to strict-local test.
- **C ≈ A, gain ≈ 0:** Block 1's predictions don't help CE. Mechanism works but predictions aren't task-useful. Consider: is the prediction target wrong? Should block 1 predict something different?
- **C ≈ A, gain > 0 but ablation gap ≈ 0:** Predictions incorporated but cancel out. Strange. Investigate.

### If positive: next experiments

See `research/questions/local-learning-variants/README.md` for the full decision tree.

1. **Strict-local (Phase 2):** Also detach the feedback path. Block 1 trained ONLY by pred_loss. Key test: does purely local prediction still help?
2. **N=3 chain (Phase 3):** Three blocks, adjacent prediction, strict-local. Tests multi-hop grounding.

## Theoretical findings this session

1. **MixAdd sqrt formula gives 31.6% influence at init=0.9** — documented as root cause of collapse. The parameter value is NOT the coefficient.
2. **Current "semi-local" is NOT genuinely local** — CE flows through feedback path. Global backprop through a narrow interface. See `research/questions/local-learning-variants/README.md`.
3. **Strict-local is the real test of local learning.** First clean experiment that's genuinely more parallel than standard backprop.
4. **Final-only CE + 128-step recurrence = BPTT through 128 steps.** Even with inter-block locality, the temporal dimension is still global. Future question: temporal locality.
5. **Transformer baseline at matched total params is misleading.** With vocab=4980, embeddings eat 90% of params. A_single's backbone is only ~263K; a "matched" transformer backbone would be ~1.53M. See think agent analysis (not yet written up in question doc).

## Queue

- Transformer matched-compute baseline (complex fairness issues — see findings #5)
- Named/typed tensor dimensions
- Loop management tooling
- Graph architecture idea from dictation

## Key completed findings

| Experiment | Result | Notes |
|---|---|---|
| **Closed-loop v1** | **COLLAPSE** (+1.47) | pred_loss trained block 0 to be constant |
| **Closed-loop v2** | **COLLAPSE → NaN** | MixAdd 31.6% prior + final-only CE = stable collapsed fixed point |
| ctx=128 corrected | **HURTS** (+0.014) | Spectator worse at longer context |
| ctx=128 ensemble | Tiny benefit (-0.020) | Was -0.101 at ctx=32; collapses |
| ctx=32 baseline | B wins (-0.101), C≈A | Spectator on corrected arch |
| ctx=32 local aux loss | **NULL** (+0.003) | Gradient isn't the problem |
| ctx=32 equal readout | **HURTS** (+0.024) | Information poverty confirmed |
| ctx=32 temporal window | **NULL** (+0.003) | Learned projection of history doesn't help |
| Bidirectional top-down | **HURTS** | TinyShakespeare |
| CUDA Graph training | 10.86× speedup | GraphTrainer in core/ |
| Self-prediction | NULL | Zero effect |
