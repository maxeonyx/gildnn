# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## Now

**Parallel diagonal multi-rate: validated with clear regime boundaries.** Compute frontier shows:
- At LOW compute: parallel 4-block multi-rate is the best architecture (1.804 @ 123K FLOPs).
- At HIGH compute: sequential all-rate-1 dominates (1.741 @ 393K FLOPs).
- Parallel doesn't scale past 4 blocks at this width.
- The architecture's value: **same quality at ~half the per-token FLOPs** for moderate depths.

Architecture:
- All blocks parallel ✓
- Multi-rate [1,2,4,8] ✓  
- Lateral propagation (neighbor states) ✓
- All-block readout ✓
- Compute-efficient: 53% fewer block evals ✓
- **Quality at equal FLOPs: BETTER (+0.019 avg, 2/3 seeds clearly better, 1 tied)** ✓

Next steps:
- [ ] **Scale up: larger dataset** — The single most impactful next move. TinyShakespeare (250K tokens) is a ceiling; many experiments produce null results because the model memorizes it. Auxiliary objectives, graph topology, rate scheduling — all wash out. Need OpenWebText subset or WikiText-103. This enables all the interesting questions.
- [ ] **Matched-compute transformer baseline** — Existing baseline (1.643 @ 187K params, d=72 3-layer) already beats our best (1.738 @ 1.1M params) at roughly matched wall-time. Need a fair comparison using current infrastructure (same dataset loading, same eval). This is methodological debt but also motivation: our architecture wins on parallelism/locality, not parameter efficiency at tiny scale.
- [ ] **LOCAL LEARNING CONFIRMED — further integration** — `detach_lateral` flag added (commit b570129). The shared-adjoint algorithm works. Remaining: measure actual wall-clock speedup from smaller computation graph, make it default for new experiments.
- [ ] **Context scaling with more compute** — ctx=128 with [1,4,16,32] is worse than ctx=32 (1.796 vs 1.718). Root cause: insufficient compute per token at wider rates. Sequential 6-block at ctx=128 reaches 1.716 — BETTER than ctx=32 best. So ctx=128 CAN work with enough compute. Try: parallel 6-block or 8-block at ctx=128, or parallel 4-block d=384/512 at ctx=128.
- [ ] **More blocks at d=256** — Max's "width" vision means MORE PARALLEL BLOCKS, not just wider d_model. Compute frontier showed diminishing returns past 4 at d=128. Not tested at d=256. Try 6-8 blocks.
- [ ] **Adjacency probe** — Cheapest test of graph topology: same 4 blocks, same rates, just change who reads whom. Deferred until larger dataset makes results more interpretable.

Recent findings this session:
- [x] **Cosine LR** — DOES NOT HELP. Same overfitting pattern as constant LR (model memorizes TinyShakespeare before LR decays meaningfully). Dataset is the bottleneck, not LR schedule.
- [x] **Context scaling ctx=128** — Wide rate spacing [1,4,16,32] wins at ctx=128 (1.803 vs 1.812 for [1,2,4,8]). But ALL parallel configs worse than ctx=32 due to lower compute/token.
- [x] **Temporal window scaling** — window=16 is practical sweet spot at ctx=128 (1.799 vs 1.841 for window=4). Full attention (window=128) gives 1.796 but 2× slower. Confirms the bottleneck is compute/capacity not context access.
- [x] **d=384 dead spot** — RESOLVED: pure overfitting, not structural. µP predicts LR ∝ 1/width.

## Recently completed

- [x] **CUDA Graph training** — 10.86× confirmed, GraphTrainer in core/
- [x] **20K matched-FLOP** — 3-seed TIE (avg -0.001). Multi-rate is compute-equivalent.
- [x] **Literature backing** — 30+ papers. See `research/questions/literature-backing/README.md`.
- [x] **Backend decision** — PyTorch + torch.compile + manual CUDA Graphs.
- [x] **Core reintegration** — `core/model.py` + `core/training.py`
- [x] **Diagonal + multi-rate** — NEGATIVE. Raw: +0.063 at 20K. Scaled (ReZero): +0.007. Alphas grow but don't help. Signal helps early, hurts late. See `experiments/fixed_multi_rate/artifacts/diagonal_20k/` and `diagonal_scaled_20k/`.
- [x] **TRUE parallel diagonal (Max's architecture)** — WORKS. All-block readout: +0.010 (noise). Last-block-only: +0.093. Problem was readout bottleneck, not propagation. See `experiments/fixed_multi_rate/artifacts/parallel_diagonal_variants_20k/`.
- [x] **Multi-rate parallel diagonal** — +0.017 avg (3-seed, consistent). 53% fewer block evals. See `experiments/fixed_multi_rate/artifacts/parallel_diagonal_multirate_3seed/`.
- [x] ~~**Matched-FLOP parallel multi-rate** — **WINS by 0.036 avg**~~ BUG: control was also multi-rate, so parallel had 2.13× more FLOPs. Fixed: all-rate-1 control.
- [x] **Corrected matched-FLOP** — Parallel multi-rate WINS by 0.019 avg vs all-rate-1 sequential at equal per-token FLOPs (262K each). 2/3 seeds clearly better, 1 tied. See `experiments/fixed_multi_rate/artifacts/parallel_diagonal_matched_flop_fixed_3seed/`.
- [x] **8-block internal_steps=2** — NEGATIVE (-0.051 avg). Recurrence within blocks doesn't compensate for halved width.
- [x] **Compute frontier sweep** — Parallel wins at low compute (4-block @ 123K FLOPs beats seq-2-block @ 131K). Sequential dominates at high compute. Parallel plateaus past 4 blocks. See `experiments/fixed_multi_rate/artifacts/compute_frontier_sweep/`.
- [x] **Rate dilation sweep** — [1,2,4,16] ≈ [1,2,4,8], [1,2,4,32] slightly worse. Aggressive dilation is a wash at ctx=32 (rate=32 fires once at t=0, just a static prior). See `experiments/fixed_multi_rate/artifacts/rate_dilation_sweep/`.
- [x] **Self-prediction sweep** — NEUTRAL. Cosine alignment (adjacent fast→slow, d_aux=32) reduces aux loss 3× but has ZERO task effect across all lambdas [0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]. Blocks specialize naturally; alignment doesn't help. See `experiments/fixed_multi_rate/artifacts/self_prediction_sweep/`.
- [x] **Longer training** — Both models peak at ~35-40K then catastrophically overfit (no LR decay). Key finding: parallel peaks at 1.742 in 289s wall-time, sequential peaks at 1.731 in 739s. **Parallel is 2.6× faster to near-equal quality.** Gap is only 0.010 nats. See `experiments/fixed_multi_rate/artifacts/longer_training/`.
- [x] **Width scaling** — **MAJOR WIN.** Parallel 4-block d=256 achieves 1.718 in 198s vs sequential 6-block d=128 at 1.741 in 493s. Parallel wins in BOTH quality (-0.023) AND wall-time (2.5× faster). Width is the natural scaling axis for parallel architecture. See `experiments/fixed_multi_rate/artifacts/width_scaling/`.
- [x] **Hierarchical prediction** — **NULL.** Aux future-latent prediction learns (hier_loss drops 0.94→0.41) but adds ZERO task benefit. C≈B≈A across 3 seeds (+0.003 avg = noise). Same "learns something unused" pattern as self-prediction. Blocks specialize naturally from shared adjoint alone. See `experiments/fixed_multi_rate/artifacts/hierarchical_prediction/`.
- [x] **Local learning / lateral gradient test** — **H1 CONFIRMED.** Detaching lateral gradients costs nothing (avg -0.009 = slightly better). Parallel training is free for this architecture. `detach_lateral` flag added to model. See `research/questions/local-learning/README.md`.
- [x] **Async hardware measurement** — 28% block concurrency on this GPU at current scale. CUDA graph training is a separate speedup (10.86×) that's orthogonal to async — it's about fused execution, not parallelism. Async remains an open direction for many-block (50-100+) architectures per [dictation 2026-05-23-4](dictations/2026-05-23-4.md).

## Queue (lower priority)

- **Transformer baseline** — MISSING per [dictation 2026-05-23-4](dictations/2026-05-23-4.md). We have NO transformer comparison. Every claim about our architecture is currently ablation-only. Need a standard transformer at matched compute on the same dataset. This is methodological debt.
- **More parallel blocks** — Per [dictation 2026-05-23-4](dictations/2026-05-23-4.md), "width scaling" to Max means MORE BLOCKS in parallel, not wider d_model. "What is the optimal number of blocks for my GPU?" needs answering. We stopped at 4 because the compute frontier showed diminishing returns, but that was at d=128.
- **Graph architecture** — Per [dictation 2026-05-23-5](dictations/2026-05-23-5.md). Dense at low level, sparse at high level. Overlapping rates. Not just a horizontal chain. Unexplored.
- **Larger dataset** — Per [dictation 2026-05-23-2](dictations/2026-05-23-2.md). Max wants thousands of tokens of context, large datasets, real runs. TinyShakespeare is a ceiling.
- **CPU-parallel small runs** — Per [dictation 2026-05-23-4](dictations/2026-05-23-4.md). Small runs on CPU while GPU does big runs.
- Named/typed tensor dimensions — continue converting codebase to einops + jaxtyping style. Per [dictation 2026-05-22-14](dictations/2026-05-22-14.md). (Started: core/model.py done.)
- Loop management tooling — script to show recent agent messages, manage the autonomous loop. Per [dictation 2026-05-20-14](dictations/2026-05-20-14.md).
- Arbitrary-order sampling — deprioritized per [dictation 2026-05-22-6](dictations/2026-05-22-6.md). Prototype works (MSE 0.0195).
- Dynamic token count — not yet explored
- Complex-valued / orthogonal parameterization — exp-map was quality-negative
- Volume-preserving nonlinearities — highly speculative

## Done

- VISION.md rewrite from all dictations — per [dictation 2026-05-22-8](dictations/2026-05-22-8.md). Completed.
- Fixed multi-rate [1,1,2,4]: **POSITIVE** — 14.8% speedup, quality better. See `research/questions/fixed-multi-rate/README.md`.
- Fixed multi-rate [1,2,4,8]: **POSITIVE** — 20.7% speedup, quality better. Same README.
- Matched-FLOP 2K steps (3-seed): +0.018 LOSS. Superseded by 20K-step result (TIE).
- Matched-FLOP 20K steps (3-seed): **TIE** (-0.001 avg). Multi-rate is compute-equivalent.
- Arbitrary-order MNIST: **WORKING** — per-pixel MSE 0.0195. See `research/questions/arbitrary-order-sampling/README.md`.
- Dynamic depth: **POSITIVE** — 43% compute savings for 1% quality loss. See `research/questions/dynamic-depth/README.md`.
- Backend research: **DONE** — See `research/questions/backend-choice/README.md`.
- Thread 1 (18 experiments): architecture viable but dominated by transformers. See `research/SYNTHESIS.md`.
