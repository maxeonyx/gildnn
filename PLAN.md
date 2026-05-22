# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## Now

- [ ] **Matched-FLOP comparison** — does multi-rate [1,2,4,8] beat a same-compute all-rate-1 model? Experiment code written (`experiments/fixed_multi_rate/matched_flop.py`), verified on CPU. Launch after midnight when GPU is free.

### Matched-FLOP decision tree

**What it tests:** Multi-rate [1,2,4,8] (d_model=128, 4 blocks) vs all-rate-1 model sized to match training-step wall-clock time (3 or 4 blocks with larger d_model). Same compute budget → which has better val_loss at 2000 steps?

**Success criteria (3-seed average, per process rule):**

| Outcome | Definition | Next action |
|---------|-----------|-------------|
| **Win** | Multi-rate val_loss ≤ control - 0.01 | Larger-scale confirmation: 8 blocks, longer training (5000+ steps). Multi-rate is genuinely better USE of compute. |
| **Tie** | Delta within ±0.01 | Multi-rate [1,2,4,8] is still preferable — same quality in same time but with architectural headroom for async. Write up as "multi-rate is compute-equivalent with structural benefits." Proceed to persistent-kernel async prototype. |
| **Loss** | Multi-rate val_loss > control + 0.01 | Reframe current results: multi-rate gives speedup by doing less work (not by doing BETTER work). The 4-block all-rate-1 model is quality-superior when given equal compute. Investigate why — is it the stale reads degrading optimization, or insufficient model capacity at rate-8? |

**Single-seed first:** Run one seed to get the ballpark. If clearly win or clearly loss (>0.03 delta), that's informative but NOT decisive. Run 2 more seeds to confirm.

**Artifacts needed:** `experiments/fixed_multi_rate/artifacts/matched_flop/report.json` with calibration results (which d_model was selected), training curves, and final metrics.

## Recently completed

- [x] **Literature backing** — DONE. 30+ papers supporting multi-rate, persistent kernels, and MixAdd. See `research/questions/literature-backing/README.md`.
- [x] **Backend decision** — DECIDED: PyTorch deliberately. `torch.compile` for stable core/ paths, custom CUDA/Triton for async research. See `research/questions/backend-choice/README.md`.
- [x] **Core reintegration** — DONE. `core/model.py` has MixAdd, ResidualFeedForwardBlock, TemporalWindowAttention, MultiRateResidualModel. torch.compile-compatible forward paths. Old models moved to `legacy_models/`.
- [x] **Diagonal + multi-rate** — NEGATIVE (as currently implemented). Multi-seed confirmation: seed 42 gave -0.052, seed 43 gave 0.000, seed 44 DIVERGED (val_loss 18.4). Stabilized variant (gated/scaled) is open. See `research/questions/diagonal-multi-rate/README.md`.
- [x] **Async hardware investigation** — DONE. README corrected: PyTorch streams failed but hardware supports concurrent execution via persistent kernels, fused dispatch, or CUDA Graphs. See `research/questions/async-execution/README.md`.

## Queue (lower priority)

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
