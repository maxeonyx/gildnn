# Plan

Working file. Rewrite it as the state changes.

## Current state (2026-05-22, ~12:45pm)

Max redirected focus in dictations 2026-05-22-6 and 2026-05-22-7:
- **NOT arbitrary-order sampling** — that's interesting but not the core exploration right now
- **YES async diagonal residual design** — the core: modules on a residual stream with diagonal connections across time and depth, different update rates, potential wall-clock speedup
- **Clean core reintegration** — strong emphasis on keeping the codebase small and reusable
- **Backend decision needed** — Max wants compiled execution (not eager PyTorch). Options: JAX, torch.compile/Triton, or 2026-era alternatives. Decision must be deliberate.

Prior experiments showed async execution CAN'T speed up via CUDA streams on single GPU. But "diagonal residual connections across time, modules at different rates" is a broader design space than just CUDA stream parallelism.

## Active work

- **Background run:** fixed-multi-rate 5000-step comparison (PID 5512 in `runs/active.lock`). Expected ~15-30 min. Confirming that the 14.8% speedup result holds at convergence.

## Recent results

- **Fixed multi-rate: POSITIVE** — 14.8% wall-clock speedup, no quality loss. Blocks on schedule [1,1,2,4]; skipped blocks reuse cached output. First positive speed result. See `research/questions/fixed-multi-rate/README.md`.
- **Arbitrary-order MNIST: WORKING** — per-pixel MSE 0.0195 after 50 epochs. Mechanism validated but deprioritized per Max's direction.
- **Backend research: DONE** — JAX recommended for clean compiled code. Decision awaiting Max. See `research/questions/backend-choice/README.md`.

## Next steps (in priority order)

1. **Check 5k-step multi-rate results** — confirm quality and speedup hold at convergence.

2. **Scale multi-rate** — more aggressive rates, more blocks, larger model. Can we get >20% speedup? Does quality hold?

3. **Diagonal + multi-rate combination** — the diagonal residual (block1's output at time t feeds block2 at time t+1) naturally pairs with multi-rate. If block2 runs every 2nd step, the diagonal connection IS the stale-read mechanism.

4. **Core reintegration** — once backend is decided, rewrite core/ to be clean, compiled, and reusable.

5. **Backend decision** — awaiting Max's input on JAX vs alternatives.

## Queued research directions (lower priority, from earlier dictations)

### Arbitrary-order sampling ([dictation 2026-05-21-10](dictations/2026-05-21-10.md))

Explicitly deprioritized by Max in [dictation 2026-05-22-6](dictations/2026-05-22-6.md): "I don't want to be doing the arbitrary order sampling work now." Reference material: `maxeonyx/msc` and `maxeonyx/thesis` on GitHub. Work done: basic encoder-decoder prototype works on MNIST (overfit test passes, per-pixel MSE 0.032 at epoch 10 with random subsets — model IS learning).

### Self-prediction, dynamic depth, dynamic token count ([dictation 2026-05-22-7](dictations/2026-05-22-7.md))

Part of the async diagonal residual family. Prior results:
- Self-prediction: NEGATIVE (+0.011-0.018 nats at every depth)
- Dynamic depth: POSITIVE (43% compute savings for 1% quality loss)
- Dynamic token count: not yet explored

### Complex-valued / orthogonal parameterization ([dictation 2026-05-21-5](dictations/2026-05-21-5.md))

Inherently orthogonal weight matrices. Related prior result: exp-map parameterization confirmed stability but quality-negative.

### Volume-preserving nonlinearities ([dictation 2026-05-21-6](dictations/2026-05-21-6.md))

Hamiltonian flows as basis-independent nonlinearities. Highly speculative, requires separate training of flow module.

### Declining batch size ([dictation 2026-05-21-9](dictations/2026-05-21-9.md))

Side direction. Easy to bolt on to any training run.

## Completed experiments

### Thread 2 (dynamic computation depth)

- **Dynamic depth / adaptive compute** — POSITIVE. 43% compute savings for 1% quality loss. See `research/questions/dynamic-depth/README.md`.

### Thread 1 (cortical column / residual-stream-across-time)

18 experiments. See `research/SYNTHESIS.md` for the full picture. Key: architecture viable but dominated by transformers (+0.071 nats), async execution can't speed up on single GPU via CUDA streams.
