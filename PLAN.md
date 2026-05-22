# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## Now

- [x] **Check 5k-step multi-rate results** — run killed externally at step 1000. Data sufficient: quality advantage consistent, speedup grows (8.7% → 12.4%). Result confirmed. Now running [1,2,4,8].
- [x] **VISION.md rewrite** — done. Rewritten from dictations, reviewed 3 rounds.

## Next (after current items)

- [x] **Scale multi-rate [1,2,4,8]** — DONE. 20.7% speedup (500-pass final timing), quality BETTER (-0.006 nats). Target cleared.
- [x] **Push rates further [1,2,4,8,16]** — DONE. ~22% speedup (checkpoints) but +0.016 quality cost. Rate-16 is where quality starts degrading. Sweet spot is [1,2,4,8] at 20.7% with quality BETTER.
- [ ] **Diagonal + multi-rate** — first probe shows -0.052 at step 2000 (suggestive, non-monotonic). Needs multi-seed or longer run to confirm. Question doc: `research/questions/diagonal-multi-rate/README.md`.
- [ ] **Backend decision** — JAX recommended (`research/questions/backend-choice/README.md`). Awaiting Max's input.
- [ ] **Core reintegration** — once backend is decided, rewrite core/ to be clean, compiled, and reusable.

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
