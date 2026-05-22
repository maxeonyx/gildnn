# Question: Can async execution give wall-clock speedup on a single GPU?

## Motivation

[Dictation 2026-05-22-3](../../../dictations/2026-05-22-3.md): "As far as I know, we still haven't done the minimal async test, which is to prove async on versus async off gives a wall clock time speed up. The only point of it is to get wall clock time speed up."

[Dictation 2026-05-22-11](../../../dictations/2026-05-22-11.md): Corrects the premature "decisive negative" conclusion — what was tested was PyTorch's stream API overhead, not the hardware's concurrent execution capabilities.

## The core question

On a single RTX 3090, can we get measurable wall-clock improvement by running multiple model blocks concurrently with relaxed inter-block dependencies?

## Architecture under test

Multiple processing blocks on a shared residual stream. Each block: read shared state → compute (linear projections + nonlinearity) → write delta to shared state.

- **Sync:** all blocks at timestep t must complete before any block starts timestep t+1.
- **Async:** block m at timestep t+1 can start as soon as block m's own previous computation finishes — it reads potentially stale shared state from other blocks.

## Experiment 1: PyTorch CUDA Streams

### Design

Three variants (same total compute):

1. **Sync-sequential:** all blocks execute one at a time on the default stream. Naive baseline.
2. **Sync-parallel:** all blocks at the same timestep launched on separate CUDA streams, synchronized at each timestep boundary. Honest sync baseline — already exploits within-timestep parallelism.
3. **Async-pipelined:** each block on its own stream, no per-timestep barrier. A block only waits for its OWN previous timestep to finish.

The comparison that matters is **variant 2 vs variant 3**. If async only beats sequential, that proves nothing about the async mechanism.

### Controlled variables

- Identical block implementations (same weights, same ops)
- Identical number of block invocations, tensor shapes, timestep count
- No gating, no skipping, no precision changes
- CUDA event timing (not Python timers)

### Parameter sweep

- `num_blocks ∈ {4, 8, 16}`, `d_model ∈ {64, 128, 256, 512}`
- `batch_size = 64`, `seq_len = 128`, `timesteps = 32`
- Block = Linear(d_model, 4*d_model) → GELU → Linear(4*d_model, d_model)

### Results: Training regime

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

### Results: Inference regime (batch_size=1, seq_len=1, timesteps=256)

Simulating autoregressive token-by-token generation — tiny tensors, theoretically under-occupied GPU:

| blocks | d_model | sequential (ms) | parallel (ms) | async (ms) | async vs parallel |
|--------|---------|-----------------|---------------|------------|-------------------|
| 4 | 128 | 153.30 ±10.19 | 244.21 ±10.72 | 282.99 ±12.17 | **0.86×** |
| 4 | 256 | 176.03 ±10.86 | 272.05 ±9.96 | 335.79 ±30.81 | **0.81×** |
| 4 | 512 | 206.36 ±10.57 | 298.55 ±14.62 | 365.65 ±46.54 | **0.82×** |
| 8 | 128 | 341.24 ±69.79 | 508.44 ±59.96 | 1022.34 ±315.12 | **0.50×** |

**Even worse than training regime.** Stream/event overhead completely dominates when each kernel does microseconds of actual work.

### What this experiment proves

PyTorch's CUDA stream/event API adds more overhead than it recovers for small kernels. **Confirmed experimentally.** ✅

### What this experiment does NOT prove

That the RTX 3090 hardware cannot run independent small kernels concurrently. The experiment tested one software mechanism (PyTorch streams + events), not the hardware's concurrent execution capabilities. The overhead is in the API layer — event recording, synchronization, stream switching — not in the hardware's ability to schedule work across SMs.

---

## Hardware Capabilities (RTX 3090 / GA102)

The RTX 3090 ([GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)) has architecture specifically designed for concurrent kernel execution:

**Compute resources:**
- 82 SMs, each with 128 CUDA cores, 4 warp schedulers, 128 KB L1/shared memory
- Compute capability 8.6, Hyper-Q with 128 concurrent hardware connections
- Different SMs CAN run different kernels simultaneously — this is a [documented hardware fact](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-kernel-execution)

**Memory hierarchy relevant to our workload:**
- 128 KB L1 per SM (configurable shared/L1 split)
- 6 MB L2 cache, shared across all SMs
- 24 GB GDDR6X @ 936 GB/s bandwidth
- L2 persistence API allows pinning data in L2

**For our ~500KB weight matrices:** they fit entirely in L2 cache. Multiple blocks reading the same shared state would hit L2, not GDDR6X. The L2 persistence API can explicitly pin them.

The hardware supports what we want. The failure was in the software mechanism we used to request it.

---

## Paths Forward

The stream experiment showed that PyTorch's per-kernel-launch overhead (~5µs per launch + event overhead) kills any benefit when kernels are small. The paths forward bypass this overhead entirely:

### A) Single fused kernel with blockIdx dispatch

Launch ONE kernel with enough thread blocks for all matmuls. Use `blockIdx` to determine which matmul each block handles. Zero concurrency overhead — just one launch. The hardware scheduler distributes blocks across SMs naturally.

This is the simplest approach. No inter-kernel coordination needed. For N independent matmuls of known size, you tile them into a single launch grid.

### B) Persistent CUDA kernels

A single kernel occupying all SMs that never exits. It internally schedules sub-tasks via atomics and shared memory. Zero launch overhead after initial start. Can implement pipeline parallelism by assigning SM ranges to pipeline stages.

References: [Gupta et al. "Revisiting Persistent Threads"](https://arxiv.org/abs/2012.13259), [NVIDIA CUTLASS 3.x](https://github.com/NVIDIA/cutlass) uses this pattern for its warp-specialized pipeline.

### C) CUDA Graphs

Encode parallel independent launches as graph nodes. Replay the entire graph with ~5µs total overhead (vs ~5µs PER kernel in the stream approach). The hardware scheduler distributes graph nodes across SMs naturally.

This is the least invasive change from the current code — you capture the existing stream-based execution as a graph, then replay it. The overhead that killed the stream experiment (per-launch cost) is amortized to a single graph-launch cost.

### D) MPS with SM partitioning (Linux only)

GA102 supports static SM partitioning via [Multi-Process Service](https://docs.nvidia.com/deploy/mps/index.html) at 4-SM granularity. You can assign different processes (or contexts) to non-overlapping SM sets, guaranteeing concurrent execution without hardware scheduling uncertainty.

Not available on Windows. Would require running on the Linux lab server.

---

## Conclusion

| What | Status |
|------|--------|
| PyTorch streams provide single-GPU async speedup | ❌ Closed — overhead exceeds benefit |
| RTX 3090 hardware supports concurrent small-kernel execution | ✅ Documented capability |
| Fused kernels / persistent kernels / CUDA Graphs can exploit this | ⏳ Open — not yet tested |

The question is **NOT closed.** What's closed is only the PyTorch-streams approach. The hardware can do what we want; we tested the wrong software mechanism.

The path to actual async speedup requires going below PyTorch: persistent kernels, fused batched kernels, or CUDA Graphs. These bypass the per-launch overhead that dominated our measurements.

## What this does NOT settle

- Whether async is useful for multi-GPU or multi-device
- Whether different-rate modules (time dilation) have value independent of speed
- The quality question (already answered elsewhere: stale reads are fine)
- Which of the paths forward (A-D) will actually deliver measurable speedup

## References

- [GA102 Whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)
- [CUDA Programming Guide — Concurrent Kernel Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-kernel-execution)
- [MPS Documentation](https://docs.nvidia.com/deploy/mps/index.html)
- [Persistent Threads paper (Gupta et al.)](https://arxiv.org/abs/2012.13259)
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
