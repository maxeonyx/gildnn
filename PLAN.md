# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## Now

**PARALLEL DIAGONAL MULTI-RATE BEATS SEQUENTIAL CONTROL.** At matched FLOPs: -0.036 nats avg (3-seed confirmed). Bigger blocks firing less often > smaller blocks every step.

Architecture validated:
- All blocks parallel ✓
- Multi-rate [1,2,4,8] ✓  
- Lateral propagation (neighbor states) ✓
- All-block readout ✓
- Compute-efficient: uses 53% fewer block evals, reinvests into block capacity ✓
- **Quality: BETTER than sequential** ✓

Next steps:
- [ ] **Scale up** — 8 blocks, larger d_model, longer training. Does the advantage grow?
- [ ] **Integrate into core/** — This is the architecture. Clean implementation in `core/model.py`, make it the default.
- [ ] **Self-prediction revisit** — Now that the base architecture works, can auxiliary losses (each block predicts its own next input) improve further?

## Recently completed

- [x] **CUDA Graph training** — 10.86× confirmed, GraphTrainer in core/
- [x] **20K matched-FLOP** — 3-seed TIE (avg -0.001). Multi-rate is compute-equivalent.
- [x] **Literature backing** — 30+ papers. See `research/questions/literature-backing/README.md`.
- [x] **Backend decision** — PyTorch + torch.compile + manual CUDA Graphs.
- [x] **Core reintegration** — `core/model.py` + `core/training.py`
- [x] **Diagonal + multi-rate** — NEGATIVE. Raw: +0.063 at 20K. Scaled (ReZero): +0.007. Alphas grow but don't help. Signal helps early, hurts late. See `experiments/fixed_multi_rate/artifacts/diagonal_20k/` and `diagonal_scaled_20k/`.
- [x] **TRUE parallel diagonal (Max's architecture)** — WORKS. All-block readout: +0.010 (noise). Last-block-only: +0.093. Problem was readout bottleneck, not propagation. See `experiments/fixed_multi_rate/artifacts/parallel_diagonal_variants_20k/`.
- [x] **Multi-rate parallel diagonal** — +0.017 avg (3-seed, consistent). 53% fewer block evals. See `experiments/fixed_multi_rate/artifacts/parallel_diagonal_multirate_3seed/`.
- [x] **Matched-FLOP parallel multi-rate** — **WINS by 0.036 avg** (3-seed). Bigger blocks firing less often beats smaller blocks every step. See `experiments/fixed_multi_rate/artifacts/parallel_diagonal_matched_flop_3seed/`.
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
