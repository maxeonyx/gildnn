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
- [ ] **Cosine LR longer training RUNNING** (PID 6204, started ~4pm Sat May 23, ~5h total). 3 variants × 100K steps with warmup+cosine decay: parallel d=256, sequential d=128, sequential d=256. Log: `experiments/fixed_multi_rate/artifacts/cosine_lr_longer/run.jsonl`. Key question: does cosine decay prevent overfitting and let d=256 improve beyond 1.718?
- [ ] **Context scaling (script ready: `runs/context_scaling.py`)** — ctx=128 with rates=[1,4,16,32] (all blocks ≥4 firings), plus [1,2,8,32] and [1,2,4,8] baselines. Key question: does wider rate spacing become meaningful at longer context? Launch after cosine LR finishes.
- [x] **d=384 dead spot investigation** — RESOLVED: pure overfitting, not structural. d=384 peaks at 1.732 (step 14K) then overfits (train loss 1.26 → massive gap). µP predicts optimal LR ∝ 1/width (d=384 wants ~2e-4). Cosine LR should fix this.

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
- [x] **Async hardware** — CLOSED. 28% block concurrency, but irrelevant vs 10.86× whole-step graph.

## Queue (lower priority)

- Named/typed tensor dimensions — continue converting codebase to einops + jaxtyping style. Per [dictation 2026-05-22-14](dictations/2026-05-22-14.md). (Started: core/model.py done.)
- Loop management tooling — script to show recent agent messages, manage the autonomous loop. Per [dictation 2026-05-20-14](dictations/2026-05-20-14.md).
- Immediate dictation notification — OpenCode plugin/hook for real-time detection. Per [dictation 2026-05-22-14](dictations/2026-05-22-14.md), [dictation 2026-05-22-15](dictations/2026-05-22-15.md). (Partial: polling via `core.check_dictations` exists.)
- Muon optimizer — ~~swap in Muon and rerun window size ablation.~~ **DONE.** See `research/questions/muon-optimizer/README.md`.
- Arbitrary-order sampling — deprioritized per [dictation 2026-05-22-6](dictations/2026-05-22-6.md). Prototype works (MSE 0.0195).
- Self-prediction — NEGATIVE at 2K steps (+0.011-0.018 nats). Worth revisiting at 20K.
- Dynamic token count — not yet explored
- Complex-valued / orthogonal parameterization — exp-map was quality-negative
- Volume-preserving nonlinearities — highly speculative
- Declining batch size — easy bolt-on for any run

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
