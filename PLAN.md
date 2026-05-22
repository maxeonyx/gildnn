# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## Now

- [ ] **Matched-FLOP comparison** — does multi-rate [1,2,4,8] beat a same-compute all-rate-1 model? Experiment code written (`experiments/fixed_multi_rate/matched_flop.py`), launch failed due to Start-Process issue. Needs retry when GPU is free (after midnight tonight).
- [ ] **Literature backing** — find published work supporting: multi-rate execution as inductive bias, persistent CUDA kernels for pipeline parallelism, MixAdd-style residual combination. Per [dictation 2026-05-22-13](dictations/2026-05-22-13.md).

## Recently completed

- [x] **Backend decision** — DECIDED: PyTorch deliberately. `torch.compile` for stable core/ paths, custom CUDA/Triton for async research. See `research/questions/backend-choice/README.md`.
- [x] **Core reintegration** — DONE. `core/model.py` has MixAdd, ResidualFeedForwardBlock, TemporalWindowAttention, MultiRateResidualModel. torch.compile-compatible forward paths. Old models moved to `legacy_models/`.
- [x] **Diagonal + multi-rate** — NEGATIVE (as currently implemented). Multi-seed confirmation: seed 42 gave -0.052, seed 43 gave 0.000, seed 44 DIVERGED (val_loss 18.4). Stabilized variant (gated/scaled) is open. See `research/questions/diagonal-multi-rate/README.md`.
- [x] **Async hardware investigation** — DONE. README corrected: PyTorch streams failed but hardware supports concurrent execution via persistent kernels, fused dispatch, or CUDA Graphs. See `research/questions/async-execution/README.md`.

## Queue (lower priority)

- **Async hardware investigation** — DONE. [dictation 2026-05-22-11](dictations/2026-05-22-11.md) corrected. README rewritten: PyTorch streams failed (real result), but hardware supports concurrent execution via persistent kernels, fused dispatch, or CUDA Graphs. Question reopened at hardware level. See `research/questions/async-execution/README.md`.
- Arbitrary-order sampling — deprioritized per [dictation 2026-05-22-6](dictations/2026-05-22-6.md). Prototype works (MSE 0.0195).
- Self-prediction — NEGATIVE (+0.011-0.018 nats)
- Dynamic token count — not yet explored
- Complex-valued / orthogonal parameterization — exp-map was quality-negative
- Volume-preserving nonlinearities — highly speculative
- Declining batch size — easy bolt-on for any run

## Done

- Fixed multi-rate [1,1,2,4]: **POSITIVE** — 14.8% speedup, quality better. See `research/questions/fixed-multi-rate/README.md`.
- Fixed multi-rate [1,2,4,8]: **POSITIVE** — 20.7% speedup, quality better. Same README.
- Arbitrary-order MNIST: **WORKING** — per-pixel MSE 0.0195. See `research/questions/arbitrary-order-sampling/README.md`.
- Dynamic depth: **POSITIVE** — 43% compute savings for 1% quality loss. See `research/questions/dynamic-depth/README.md`.
- Backend research: **DONE** — See `research/questions/backend-choice/README.md`.
- Thread 1 (18 experiments): architecture viable but dominated by transformers. See `research/SYNTHESIS.md`.
