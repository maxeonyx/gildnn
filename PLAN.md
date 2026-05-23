# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## Current state

**TinyShakespeare is exhausted.** At d=256, single block achieves 1.712 — no architecture variant beats it. Multi-block, corrected injection, bidirectional top-down: all noise or worse. The dataset is saturated at this model capacity.

**Architecture correction done.** Only block 0 receives token embedding (`token_injection="block0"`, commit 93147c1). This makes upper blocks spectators at TinyShakespeare scale. The earlier "multi-block helps" result was an ensemble artefact of the wrong architecture (all blocks seeing tokens).

**WikiText-103 infrastructure ready.** Dataset downloaded (538M chars, vocab 4980), loader in `core/dataset.py`, experiment script at `runs/wikitext_baseline.py`.

## Active

- **[RUNNING] Bidirectional test** (PID 20984) — `topology="top_down_to_first"` + `readout_mode="first"` on TinyShakespeare. Partial results (6/12 from a previous crashed run) show it hurts (B_topdown_first: 1.779-1.801 vs A_upward_all: 1.719). Expected to confirm TinyShakespeare saturation.

## Next

- [ ] **⚠ WikiText-103 baseline** — THE TOP PRIORITY. 4 conditions × 2 seeds: A) single block, B) old 4-block (all tokens), C) corrected 4-block (block0 only), D) corrected multi-rate [1,2,4,8]. 20K steps as pilot — extend to 50K+ if curves still declining. ctx=32 (isolate dataset scale from context scale). `runs/wikitext_baseline.py` ready.
- [ ] **Local learning on WikiText-103** — Only test after multi-block shows value on larger data. Predictive coding theory work done (4 iterations, `research/questions/local-learning/README.md`).
- [ ] **Scale up context** — After WikiText-103 ctx=32 works, try ctx=128.

## Queue (lower priority)

- **Transformer baseline** — MISSING per [dictation 2026-05-23-4](dictations/2026-05-23-4.md). We have NO transformer comparison at matched compute. Methodological debt.
- **More parallel blocks** — Per [dictation 2026-05-23-4](dictations/2026-05-23-4.md), "width scaling" means MORE BLOCKS, not wider d_model. Stopped at 4 at d=128; untested at d=256 or larger data.
- **Graph architecture** — Per [dictation 2026-05-23-5](dictations/2026-05-23-5.md). Dense at low level, sparse at high level. Unexplored.
- **CPU-parallel small runs** — Per [dictation 2026-05-23-4](dictations/2026-05-23-4.md). Small runs on CPU while GPU does big runs.
- Named/typed tensor dimensions — continue converting to einops + jaxtyping. Per [dictation 2026-05-22-14](dictations/2026-05-22-14.md).
- Loop management tooling — per [dictation 2026-05-20-14](dictations/2026-05-20-14.md).

## Key completed findings

All on TinyShakespeare unless noted. See `experiments/fixed_multi_rate/artifacts/` for individual reports.

| Experiment | Result | Notes |
|---|---|---|
| Architecture correction (block0 only) | **HURTS** (+0.034 vs single) | Upper blocks are spectators |
| Bidirectional top-down | **HURTS** (partial, +0.06-0.08) | Stale state = noise |
| Width scaling (d=256, 4-block) | **WINS** (-0.023, 2.5× faster) | Best TinyShakespeare result overall |
| Matched-FLOP parallel multi-rate | **WINS** (+0.019 avg) | 2/3 seeds clearly better |
| Longer training | Parallel 2.6× faster to near-equal | Both overfit past 35K |
| CUDA Graph training | 10.86× speedup | GraphTrainer in core/ |
| Self-prediction | NULL | Zero task effect across all lambdas |
| Hierarchical prediction | NULL (wrong arch) | Needs re-test on corrected arch |
| Local learning / lateral gradients | INVALIDATED (wrong arch) | Must re-test with block0-only injection |
| Cosine LR | NULL | Dataset bottleneck, not LR schedule |
| Compute frontier | Parallel wins low-compute, sequential wins high-compute | Parallel plateaus past 4 blocks at d=128 |
