# Plan

Working file. Rewrite it as the state changes.

## Current state (2026-05-22)

**The project is at its natural conclusion.** All success criteria (floor AND strong) are met. The final synthesis is written (`research/SYNTHESIS.md`). 18 experiments completed. The residual-stream-across-time architecture and its proposed enhancements are comprehensively characterized.

Key findings:
- Architecture is viable but strictly dominated by transformers (+0.071 nats text, worse on images)
- Async stale-read semantics are quality-neutral (+0.005, not significant)
- Async execution provides NO wall-clock speedup on single GPU (definitively falsified)
- Muon/orthogonality is the stability mechanism (confirmed 3 ways)
- GRUs are 2-3× faster than transformers on this hardware

## What's left to do

1. **Process any new dictations** — none pending as of 11:36am May 22.
2. **Weekly report for week of May 19** — already exists (`research/weekly/2026-05-21.md`).

## If Max provides new direction

Process new dictations per PROCESS.md. Queue new experiments; don't bump the stack.

## Possible future directions (documented, not planned)

From `research/SYNTHESIS.md`:
- Multi-GPU async (requires different hardware)
- GRUs as cheap layers in standard architectures (exploit speed without recurrence)
- Larger scale test of the 0.071 gap
- Different tasks where temporal recurrence might matter more

These are "conditions under which negatives might not generalize" — not promising leads.

## Completed experiments

- **Local learning (residual)** — NEGATIVE. val 2.081 vs 1.644 end-to-end. See `research/questions/local-learning-residual/README.md`.
- **Attention-residual (depth-only)** — MARGINAL/INCONCLUSIVE. Transient 0.013 nat edge, regresses, 33% slower. See `research/questions/attention-residual/README.md`.
- **Async/selective execution** — NEGATIVE. 26-29% slower wall-clock despite fewer executions. See `research/questions/async-selective/README.md`.
- **GPU utilization study** — POSITIVE. GRU 1.8-2.8× faster training, 2.5-3.7× faster decode. See `research/questions/gpu-utilization/README.md`.
- **Causal triangle attention** — NEGATIVE. +0.028 nats, 1.8× slower. See `research/questions/causal-triangle-attention/README.md`.
- **Async volatile-memory prototype** — POSITIVE. Stale reads train within noise of sync. See `research/questions/async-volatile-memory/README.md`.
- **Self-prediction / compute compression** — NEGATIVE. +0.011 to +0.018 nats at every depth. See `research/questions/self-prediction-compute-compression/README.md`.
- **Async GRU combination** — QUALIFIED POSITIVE. Converges, +0.006 nats worse (single seed). See `research/questions/async-gru/README.md`.
- **Async GRU scale-up (100k data)** — POSITIVE on saturated data. Gap is noise at 5 seeds. See `research/questions/async-gru-scaleup/README.md`.
- **Multi-seed 900k async calibration** — CALIBRATED. +0.0055 ± 0.0046 nats (not significant). See `experiments/async_gru_corpus/`.
- **Broadcast channel** — NEGATIVE. +0.005 worse than plain async. See `experiments/broadcast_channel/`.
- **Image patches (MNIST)** — NEGATIVE. MSE 0.061 vs 0.050, 18× slower. See `research/questions/image-patches/README.md`.
- **Orthogonal parameterization (exp-map)** — STABILITY CONFIRMED, QUALITY NEGATIVE. val 1.972 vs 1.606. See `research/questions/orthogonal-parameterization/README.md`.
- **Partial detach at 900k** — NEUTRAL. Zero quality cost, zero speed benefit. See `experiments/partial_detach_900k/`.
- **900k scale-up** — DECISIVE. +0.071 nats vs transformer at matched params. See `experiments/scale_900k/`.
- **Muon optimizer** — POSITIVE. Enables k=8 window, 27% gap reduction. See `experiments/muon_window_ablation/`.
- **Async wall-clock (training)** — NEGATIVE. Sequential always fastest. See `research/questions/async-execution/README.md`.
- **Async wall-clock (inference)** — NEGATIVE. 0.50-0.86× (worse). See `research/questions/async-execution/README.md`.
