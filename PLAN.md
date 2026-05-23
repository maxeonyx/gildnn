# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## Current state

**TinyShakespeare is exhausted.** At d=256, single block achieves 1.712 — no architecture variant beats it. The dataset is saturated at this model capacity.

**WikiText-103 baseline DONE.** Key result: corrected architecture (block0-only injection) has a spectator problem even with larger data. Residual stream alone is insufficient for inter-block information flow.

| Variant | Mean val_loss (2 seeds) | vs A | Readout pattern |
|---|---|---|---|
| A: single block | 1.841 | — | — |
| B: 4-block old (all tokens) | **1.740** | **-0.101** | distributed |
| C: 4-block corrected (block0 only) | 1.844 | +0.003 | block0 dominant (58%) |
| D: 4-block multi-rate (block0 only) | 1.829 | -0.012 | block0 dominant (65%) |

**Implication:** Multi-block only helps when every block has direct token access (B). Under the corrected architecture, upper blocks become spectators. Multi-rate gives a tiny edge but doesn't solve the fundamental information flow problem.

## Active

Nothing running. GPU free.

## Next

The spectator problem is the central obstacle. Three approaches:

- [ ] **Local learning** — Give each block its own auxiliary loss, so upper blocks have gradient signal independent of the readout. This is the original research question (dictation 2026-05-23-5). Predictive coding theory work done (`research/questions/local-learning/README.md`). Now the spectator result makes it urgent — local loss is a natural solution to the vanishing-gradient-to-upper-blocks problem.
- [ ] **Wider architecture (more blocks)** — Per [dictation 2026-05-23-4](dictations/2026-05-23-4.md), "width" means many more parallel blocks. Test whether 8/16/32 blocks with block0-only injection still collapse. With more blocks, maybe some find useful niches even without direct token access.
- [ ] **Transformer baseline** — Per [dictation 2026-05-23-4](dictations/2026-05-23-4.md). Methodological debt — no matched-compute comparison exists. Important for paper-readiness but not blocking exploration.

## Queue (lower priority)

- **Scale up context** — ctx=128 or ctx=256 after architecture problem is solved.
- **Graph architecture** — Per [dictation 2026-05-23-5](dictations/2026-05-23-5.md). Dense at low level, sparse at high level.
- **CPU-parallel small runs** — Per [dictation 2026-05-23-4](dictations/2026-05-23-4.md).
- Named/typed tensor dimensions — einops + jaxtyping. Per [dictation 2026-05-22-14](dictations/2026-05-22-14.md).
- Loop management tooling — per [dictation 2026-05-20-14](dictations/2026-05-20-14.md).

## Key completed findings

All on TinyShakespeare unless noted. See `experiments/` for artifacts.

| Experiment | Result | Notes |
|---|---|---|
| **WikiText-103 baseline** | B wins (-0.101), C≈A, D tiny edge | Spectator problem persists with more data |
| Bidirectional top-down | **HURTS** (+0.06-0.08) | Single block wins on TinyShakespeare |
| Architecture correction (block0 only) | **HURTS** (+0.034 vs single) | Upper blocks are spectators |
| Width scaling (d=256, 4-block) | **WINS** (-0.023, 2.5× faster) | Best TinyShakespeare result (old arch) |
| Matched-FLOP parallel multi-rate | **WINS** (+0.019 avg) | 2/3 seeds clearly better (old arch) |
| Longer training | Parallel 2.6× faster to near-equal | Both overfit past 35K |
| CUDA Graph training | 10.86× speedup | GraphTrainer in core/ |
| Self-prediction | NULL | Zero task effect across all lambdas |
| Hierarchical prediction | NULL (wrong arch) | Needs re-test on corrected arch |
| Local learning / lateral gradients | INVALIDATED (wrong arch) | Must re-test with block0-only injection |
| Cosine LR | NULL | Dataset bottleneck, not LR schedule |
| Compute frontier | Parallel wins low-compute, sequential wins high-compute | Parallel plateaus past 4 blocks at d=128 |
