# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## Current state

**The spectator problem is confirmed and diagnosed.** Under the corrected architecture (only block 0 receives tokens), upper blocks become spectators. Three WikiText-103 experiments confirm this is **information poverty, not optimization failure:**

1. **Baseline:** corrected 4-block ≈ single block (1.844 vs 1.841). Old arch (all tokens) wins big (-0.101) but is just an ensemble.
2. **Aux loss:** giving upper blocks their own prediction loss — they learn (aux loss 8.5→2.9) but contribute nothing to the main model (+0.003).
3. **Equal readout:** forcing equal weight to all blocks HURTS (+0.024 vs single). Upper blocks genuinely lack useful complementary information.

**Root cause:** Block 0 sees the token directly and dominates. Upper blocks only see a delayed, compressed summary via the residual stream. They can decode it (aux losses decrease), but what they decode isn't MORE useful than what block 0 already provides to the readout.

## Active

Nothing running. GPU free.

## Next — architecture must change

The current shared-residual-stream model cannot support useful multi-block learning under the corrected architecture. Upper blocks need **exclusive useful information** that block 0 doesn't have. Three approaches, ranked by think agent analysis:

- [ ] **⚠ Temporal window architecture** — Give upper blocks access to a short history of lower-block states (last k time steps), not just the current residual stream value. This gives them something exclusive: trajectory information / temporal patterns that block 0's single-step processing can't capture. Requires changes to `core/model.py`.
- [ ] **Staggered/lagged input** — block i receives token embedding from t-i steps ago. Each block has exclusive temporal context. Risk: may collapse into variant B (each block independently solves with different lag).
- [ ] **Interface predictive-coding loss** — Instead of "predict next token," upper blocks predict the future incoming lower-level state or prediction error. Trains the communication protocol itself. Most aligned with Max's local-learning research question. But requires temporal window first.

## Queue (lower priority)

- **Transformer baseline** — Methodological debt. Per [dictation 2026-05-23-4](dictations/2026-05-23-4.md).
- **Many more blocks (16/32)** — Per [dictation 2026-05-23-4](dictations/2026-05-23-4.md). But think agent says likely to just confirm spectator pattern at scale.
- **Scale up context** — ctx=128+ after architecture works.
- **Graph architecture** — Per [dictation 2026-05-23-5](dictations/2026-05-23-5.md). Dense at low, sparse at high.
- Named/typed tensor dimensions — einops + jaxtyping.
- Loop management tooling.

## Key completed findings

All on WikiText-103 (ctx=32, d=256, 20K steps) unless noted.

| Experiment | Result | Notes |
|---|---|---|
| WikiText-103 baseline | B wins (-0.101), C≈A, D tiny edge | Spectator persists with more data |
| Local aux loss | **NULL** (+0.003) | Upper blocks learn but don't contribute |
| Equal readout | **HURTS** (+0.024) | Confirms information poverty |
| Bidirectional top-down | **HURTS** (+0.06-0.08) | TinyShakespeare |
| Architecture correction | **HURTS** (+0.034 vs single) | Upper blocks are spectators |
| Width scaling (d=256, 4-block, old arch) | **WINS** (-0.023, 2.5× faster) | Best TinyShakespeare result |
| Matched-FLOP multi-rate (old arch) | **WINS** (+0.019 avg) | 2/3 seeds better |
| CUDA Graph training | 10.86× speedup | GraphTrainer in core/ |
| Self-prediction | NULL | Zero task effect |
| Cosine LR | NULL | Dataset bottleneck |
