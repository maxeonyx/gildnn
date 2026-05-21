# Question: Can async execution give wall-clock speedup on a single GPU?

## Motivation

[Dictation 2026-05-22-3](../../dictations/2026-05-22-3.md): "As far as I know, we still haven't done the minimal async test, which is to prove async on versus async off gives a wall clock time speed up. The only point of it is to get wall clock time speed up."

All previous async experiments proved **stale reads don't hurt quality** (~0.005 nats, not significant). But none demonstrated the actual point: that async execution is FASTER. We proved the semantics are safe; now we test whether the mechanism delivers the promised speed.

## The core question

On a single RTX 3090, can we get measurable wall-clock improvement by running multiple model blocks on separate CUDA streams with relaxed inter-block dependencies?

## Hypothesis

**H0 (null):** Single-GPU async provides no meaningful wall-clock speedup because each block's kernels already saturate the GPU.

**H1 (alternative):** With many small blocks that individually under-occupy the GPU, CUDA stream overlap can give measurable forward-pass speedup.

## Architecture under test

Multiple processing blocks on a shared residual stream. Each block: read shared state → compute (linear projections + nonlinearity) → write delta to shared state.

- **Sync:** all blocks at timestep t must complete before any block starts timestep t+1.
- **Async:** block m at timestep t+1 can start as soon as block m's own previous computation finishes — it reads potentially stale shared state from other blocks.

## Experiment design

### What we measure

**Primary:** forward-pass wall-clock time (inference mode, CUDA events) for a fixed number of timesteps.

Why forward-only first: if forward doesn't speed up, training definitely won't. Forward isolates the systems question from autograd/optimizer overhead.

### Three variants (same total compute)

1. **Sync-sequential:** all blocks execute one at a time on the default stream. This is the naive baseline.
2. **Sync-parallel:** all blocks at the same timestep launched on separate CUDA streams, synchronized at each timestep boundary. This is the HONEST sync baseline — it already exploits within-timestep parallelism.
3. **Async-pipelined:** each block on its own stream, no per-timestep barrier. A block only waits for its OWN previous timestep to finish before reading (potentially stale) shared state and proceeding.

The comparison that matters is **variant 2 vs variant 3**. If async only beats sequential (variant 1), that proves nothing about the async MECHANISM — it just proves "using CUDA streams is better than not."

### Controlled variables

- Identical block implementations (same weights, same ops)
- Identical number of block invocations
- Identical tensor shapes
- Identical timestep count
- No gating, no skipping, no precision changes
- CUDA event timing (not Python timers)

### Parameter sweep

Sweep `d_model` and `num_blocks` to find the regime (if any) where blocks are small enough to not individually saturate the GPU:

- `num_blocks ∈ {4, 8, 16}`
- `d_model ∈ {64, 128, 256, 512}`
- `batch_size = 64`, `seq_len = 128`, `timesteps = 32`
- Block = Linear(d_model, 4*d_model) → GELU → Linear(4*d_model, d_model)

### Expected outcome

Honest prior: **most likely no speedup or single-digit %** from variant 3 vs variant 2. The GPU already extracts parallelism internally. But we need to run it to know.

## What this does NOT settle

- Whether async is useful for multi-GPU or multi-device
- Whether async can help at inference/generation time specifically (single-token sequential)
- Whether different-rate modules (time dilation) have value independent of speed
- The quality question (already answered: stale reads are fine)

## Results

**DECISIVE NEGATIVE.** Async-pipelined execution never outperforms sync on a single RTX 3090. The sequential baseline is always fastest.

### Full sweep (9 of 12 configs, remaining would be larger/slower)

| blocks | d_model | sequential (ms) | parallel (ms) | async (ms) | async vs parallel |
|--------|---------|-----------------|---------------|------------|-------------------|
| 4 | 64 | 24.78 ±0.99 | 30.35 ±2.80 | 36.11 ±3.52 | **0.84×** |
| 4 | 128 | 57.73 ±1.34 | 65.18 ±2.91 | 70.05 ±1.78 | **0.93×** |
| 4 | 256 | 199.19 ±7.26 | 210.10 ±3.24 | 216.27 ±3.75 | **0.97×** |
| 4 | 512 | 716.71 ±8.61 | 751.13 ±4.80 | 752.81 ±7.75 | **1.00×** |
| 8 | 64 | 50.59 ±0.63 | 57.94 ±4.86 | 73.86 ±5.19 | **0.78×** |
| 8 | 128 | 124.24 ±2.39 | 136.76 ±2.84 | 145.35 ±4.69 | **0.94×** |
| 8 | 256 | 405.00 ±7.11 | 423.96 ±5.57 | 440.44 ±11.69 | **0.96×** |
| 8 | 512 | 1453.63 ±9.38 | 1509.07 ±3.89 | 1505.17 ±7.24 | **1.00×** |
| 16 | 64 | 100.24 ±2.95 | 117.21 ±7.38 | 147.91 ±8.92 | **0.79×** |

batch_size=64, seq_len=128, timesteps=32, warmup=10, measure=50 iterations. CUDA event timing.

### Key findings

1. **Sequential is always fastest.** Multiple CUDA streams never help — even for 16 tiny blocks (d_model=64). The GPU already schedules internal parallelism; explicit stream management just adds overhead.

2. **Async is always slowest at small d_model.** The commit-stream serialization and event synchronization overhead costs more than any potential overlap benefit. At small blocks (d_model=64), async is 16-22% SLOWER than even the parallel baseline.

3. **At large d_model (512), all three converge.** When each block saturates the GPU, stream management overhead becomes negligible relative to compute. But there's still no speedup — the blocks are too large to overlap.

4. **There is no "sweet spot."** The theoretical hope was blocks small enough to not saturate the GPU but large enough that overlap would compensate for stream overhead. This regime does not exist on the RTX 3090 — either blocks are too small (overhead dominates) or too large (GPU saturates, no room for overlap).

### Why this happens

The RTX 3090 already handles intra-kernel parallelism (82 SMs). When you launch a matmul on a stream, cuBLAS uses ALL available SMs. A second stream's kernel can only overlap if the first doesn't use the full GPU — but even d_model=64 with batch=64 produces enough work to keep most SMs busy. Meanwhile, stream management (event recording, waiting, stream switching) adds real overhead: ~5-20% at small scale.

### Implications for the async vision

On a **single GPU**, async execution via CUDA streams cannot provide wall-clock speedup for this class of workload. The approaches that COULD help:

- **Multi-GPU:** blocks genuinely on different devices (eliminates contention)
- **Custom persistent kernels:** bypass the stream/event mechanism entirely with a single long-running kernel that schedules sub-blocks internally
- **Different hardware:** accelerators designed for independent module execution (neuromorphic, multi-chip)
- **Inference pipeline parallelism:** during autoregressive generation, pipelining across tokens rather than within a token might help if latency-bound

### Inference regime (batch_size=1, seq_len=1, timesteps=256)

Simulating autoregressive token-by-token generation — the regime where blocks process tiny tensors and the GPU should theoretically be under-occupied:

| blocks | d_model | sequential (ms) | parallel (ms) | async (ms) | async vs parallel |
|--------|---------|-----------------|---------------|------------|-------------------|
| 4 | 128 | 153.30 ±10.19 | 244.21 ±10.72 | 282.99 ±12.17 | **0.86×** |
| 4 | 256 | 176.03 ±10.86 | 272.05 ±9.96 | 335.79 ±30.81 | **0.81×** |
| 4 | 512 | 206.36 ±10.57 | 298.55 ±14.62 | 365.65 ±46.54 | **0.82×** |
| 8 | 128 | 341.24 ±69.79 | 508.44 ±59.96 | 1022.34 ±315.12 | **0.50×** |

**Even worse than training regime.** At tiny per-token tensors, stream/event overhead completely dominates. Sequential is 1.45-1.59× faster than parallel; async is 0.50-0.86× of parallel. The overhead of CUDA stream management (event recording, synchronization, stream switching) far exceeds any potential kernel overlap benefit when each kernel does microseconds of actual work.

This rules out inference pipelining as a path to async speedup on a single GPU.

## Next steps

The wall-clock speedup question is fully answered for single-GPU PyTorch in BOTH regimes (training and inference): **no.** The CUDA stream/event mechanism adds more overhead than any kernel overlap could recover.

Further work on async speedup would require:

1. Multi-GPU (genuinely separate devices, eliminates SM contention)
2. Custom persistent CUDA kernels (bypass stream/event overhead entirely)
3. Hardware designed for independent module execution

The async mechanism has potential value for other reasons (module independence, different update rates, conceptual architecture) but NOT for single-GPU speed on any workload tested.
