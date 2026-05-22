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

1. **Start on the queued research directions** — see below. These are genuinely new threads from Max, not extensions of the completed Thread 1/2 work.
2. **Weekly report for week of May 19** — already exists (`research/weekly/2026-05-21.md`).

## Queued research directions (from dictations, not yet started)

These were expressed by Max in dictations 2026-05-21-5 through 2026-05-21-10. They are new research threads independent of the completed Thread 1/2 work. Per the "queue not stack" rule, they join the queue — they don't invalidate the SYNTHESIS.

### Arbitrary-order sampling with cross-attention ([dictation 2026-05-21-10](dictations/2026-05-21-10.md))

From Max's master's work. A transformer with cross-attention to query positions, enabling prediction of any token given any subset in any order — not just next-token. Key properties:
- Single forward pass gives conditional marginals for ALL positions simultaneously (like a Gaussian process)
- Training: randomize prediction order, attention mask encodes "what's been sampled so far"
- Inference: choose sampling order strategically (predict easy parts first)
- Works on images (MNIST patches) and potentially text
- Connection to XLNet (permutation-based training)
- Open question from Max: "can we do this kind of query-based training with the current architecture?"

Max expressed high enthusiasm. This is already partially described in VISION.md (line 77, arbitrary-order image patches) but the cross-attention mechanism is new detail.

### Complex-valued networks / inherently orthogonal parameterization ([dictation 2026-05-21-5](dictations/2026-05-21-5.md))

Max's interest: parameterize weight matrices such that they're ALWAYS orthogonal regardless of parameter values — not pushed toward orthogonal (Muon) but constrained to be. Complex-valued matrices as rotation matrices. Open question — Max wasn't sure this is possible.

### Volume-preserving nonlinearities ([dictation 2026-05-21-6](dictations/2026-05-21-6.md))

Extensive conversation about replacing element-wise nonlinearities (which privilege a basis) with volume-preserving, basis-independent transformations. Key ideas:
- Hamiltonian flows: parameterize a scalar H(x) with a neural net, derive v = J∇H (automatically divergence-free)
- The flow module is trained SEPARATELY then frozen — used as infrastructure like a tokenizer
- "Corn syrup / Rubik's Cube" analogy — incompressible deformation, not rotation
- Would compose with unitary linear layers for a fully distribution-preserving architecture

Max called this "building my own ideal neural network training process."

### Declining batch size scheduling ([dictation 2026-05-21-9](dictations/2026-05-21-9.md))

Side direction: start with high batch size (via gradient accumulation), decline batch size AND learning rate together. Early = stable large steps, late = noisy small steps as regularization. Max noted this interacts with Muon vs Adam (Adam has internal rate adaptation; Muon doesn't).

## Possible extensions of completed work (lower priority)

From `research/SYNTHESIS.md`:
- Multi-GPU async (requires different hardware) — also relevant to [dictation 2026-05-22-3](dictations/2026-05-22-3.md) where Max says async wall-clock speedup is still the goal
- GRUs as cheap layers in standard architectures (exploit speed without recurrence)
- Larger scale test of the 0.071 gap
- Different tasks where temporal recurrence might matter more

These are "conditions under which negatives might not generalize" — not promising leads.

## Completed experiments

### Thread 2 (dynamic computation depth)

- **Dynamic depth / adaptive compute** — POSITIVE. 43% compute savings for 1% quality loss. Multi-exit training stable. See `research/questions/dynamic-depth/README.md`.

### Thread 1 (cortical column / residual-stream-across-time)

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
