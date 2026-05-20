# GPU Utilization: RNN vs Transformer Throughput

*Serving dictation: [2026-05-20-13](../../../dictations/2026-05-20-13.md)*

## Question

> "I tend to think that we can get significantly more FLOPs out of a GPU training an RNN than a transformer."
> "What's the largest parameter shape or architecture shape that just sits in cache, repeatedly processing data?"
> "From what I understand, in transformers memory bandwidth is a bottleneck, but I don't see why that should be the case for an RNN."

The hypothesis is about hardware utilization, not model quality. The follow-up questions are about *why* — how the 3090's internals shape what runs well on it. This document attempts to answer both.

---

## Hardware mental model: what the RTX 3090 actually is

The 3090 is an Ampere-architecture GPU. The relevant facts for this analysis:

- **82 Streaming Multiprocessors (SMs),** each with 128 CUDA cores (FP32) and 4 Tensor Core units
- **Tensor Cores:** the specialized matrix-multiply units introduced in Volta. On Ampere, each can execute a 16×8×16 matrix multiply (for FP16/BF16) in a single clock tick. This is where the real compute lives. For FP32, they use a TF32 approximation (10-bit mantissa instead of 23-bit) that doubles throughput with negligible accuracy loss.
- **Peak throughput:** ~35.6 TFLOP/s (FP16 Tensor Core) vs ~35.6 TFLOP/s (TF32 Tensor Core) vs ~17.8 TFLOP/s (FP32 CUDA cores, no Tensor Cores). Using Tensor Cores is roughly a 2× throughput multiplier at FP32.
- **Memory bandwidth:** 936 GB/s (GDDR6X, 384-bit bus). This sounds fast, but it's the bottleneck for memory-bound ops. At peak FLOP/s, the ratio of compute to bandwidth is ~38 FLOP/byte. Any operation that does fewer FLOPs per byte fetched is *memory-bandwidth limited.*
- **L2 cache:** 6 MB. A 1M-parameter model at FP32 is 4 MB of weights. A 10M-parameter model is 40 MB. At these sizes, **the weights do not fit in L2.** Each training step re-fetches parameters from GDDR6X VRAM.
- **Power cap:** intentionally limited. Benchmark runs show a 185W cap vs the stock 350W. This means the GPU is not running at full speed. The relative story (GRU vs transformer) is unchanged, but absolute utilization numbers will improve if the cap is raised or removed.

The single most important concept for understanding GPU performance is **arithmetic intensity:** the ratio of FLOPs performed to bytes fetched from memory. Operations above ~38 FLOP/byte are compute-bound; below that, they're memory-bandwidth limited. This is the lens for everything that follows.

---

## Why current utilization is low

Two separate issues: the data pipeline and the model itself.

### The data pipeline: ~10% of the problem

*Evidence: [task_a_synthetic_vs_real.json](../../../experiments/gpu_utilization/artifacts/profiling/task_a_synthetic_vs_real.json)*

Comparing GRU 1M, ctx=32, batch=512 with synthetic (random tensor) vs real DataLoader:

| Mode | Wall ms/step | GPU active ratio | GPU util mean | Power (W) |
|---|---:|---:|---:|---:|
| Synthetic | 12.6 | 99.6% | 82.9% | 173.6 |
| Real DataLoader | 14.1 | 76.8% | 63.3% | 152.4 |

The 11.8% slowdown breaks down as:
- **CPU DataLoader wait:** 3.07 ms/step — the GPU is idle while PyTorch's DataLoader prepares the next batch on CPU
- **PCIe host-to-device copy:** 0.10 ms/step — negligible

The DataLoader is the bottleneck, not PCIe transfer. The GPU spends 3ms stalled, waiting for the CPU to deliver data. This is addressable: `pin_memory=True`, `num_workers>0`, and prefetching (e.g. `persistent_workers=True`) should close most of this gap. But even fixing this completely only recovers ~10% of step time. The GPU will still not be at theoretical maximum utilization.

The deeper issue is the model's arithmetic intensity, not the pipeline.

### Why the model itself isn't fully utilizing the GPU

*Evidence: [task_b/summary.json](../../../experiments/gpu_utilization/artifacts/profiling/task_b/summary.json)*

At 1M params, profiling 6 active steps each:

| Model | CUDA time share | Kernel launches/step | Kernel events/step | Tensor Core kernels |
|---|---:|---:|---:|---|
| Transformer 1M | 51.0% | 224 | 287 | None |
| GRU 1M | 68.7% | 246 | 532 | 6 distinct kernels |
| GRU 10M | 51.5% | 118 | 202 | 6 distinct kernels |

"CUDA time share" is the fraction of total profiled time spent in GPU kernels (vs CPU overhead, kernel launch overhead, synchronization). The transformer at 1M spends *half its time* outside GPU kernels. GRU at 1M is better — nearly 70% in kernels — but still leaves 30% on the table. At 10M, GRU drops back to 51%, not because it got worse but because each kernel invocation does more work (fewer but heavier calls).

The transformer's 49% CPU overhead comes from:
1. **Many small GEMMs:** at hidden_size=128, each attention GEMM is tiny. Small matrix multiplies don't saturate the SMs and have proportionally high kernel-launch overhead.
2. **LayerNorm and softmax:** these are elementwise/reduction ops with very low arithmetic intensity. They're memory-bandwidth limited regardless of batch size.
3. **Fragmented work:** 287 kernel events per step, all competing for SM scheduling slots.

---

## Why GRU wins: tensor cores, weight reuse, and kernel consolidation

The most striking finding in the profiler data: **GRU uses Tensor Core kernels; the transformer (at 1M) does not.**

GRU's top 6 kernels by device time are all named `cutlass_80_tensorop_s1688*` — the `tensorop` string is CUTLASS's naming for Tensor Core paths on Ampere. The `s1688` refers to the TF32 (single-precision via 16×8×8 tiles) Tensor Core operation.

```
GRU top kernel: cutlass_80_tensorop_s1688gemm_128x128_32x3_nt_align4
  count: 36 invocations
  self_device_time: 78,926 µs  ← 40% of all GRU GPU time
```

The transformer's top kernels are `ampere_sgemm_*` — the CUDA-core SGEMM path, not Tensor Cores. Why? At hidden_size=128 and batch=512, the attention GEMMs are shaped as `[512×32, 32×32]` (batch × seq, seq × head_dim). The 32-wide dimension is too small for the Tensor Core's minimum tile size — CUTLASS falls back to SIMT SGEMM.

**This is the key hardware lesson:** Tensor Cores have minimum problem sizes. Below roughly 64×64 in each matrix dimension, cuBLAS/CUTLASS won't dispatch to them. GRU's weight matrices are larger in the dimensions that matter (hidden_size=232 per layer, batched across the sequence), so it hits the threshold. The transformer's small attention heads and short context at 1M params don't.

Why does GRU also benefit from weight reuse? Each GRU layer has weight matrices `W_r, W_z, W_n` (reset, update, new gates) of shape `[hidden, input+hidden]`. At every timestep, the *same weights* are applied to the current hidden state and input. Over a sequence of length T, those weights are used T times — each time fetched once from L2/VRAM, then reused for all the gate computations that timestep. The per-token FLOP/byte ratio is higher than a transformer's per-token attention, which recomputes fresh Q, K, V projections each position.

**Measured consequence:** the top GRU kernel does 78,926 µs of GPU work in 36 invocations = ~2.2ms per call. That's a long, heavy, well-saturated kernel. The transformer's top kernels are all under 1ms each, many under 0.5ms. Long kernels = better SM occupancy = higher utilization.

---

## Sequence length story

*Evidence: [task_c_sequence_sweep.json](../../../experiments/gpu_utilization/artifacts/profiling/task_c_sequence_sweep.json)*

Both architectures at 1M params, batch tuned per run:

| ctx | Transformer tok/s | GRU tok/s | GRU advantage |
|---:|---:|---:|---:|
| 32 | 528,524 | 1,405,358 | **2.66×** |
| 64 | 530,393 | 1,161,539 | **2.19×** |
| 128 | 542,033 | 1,210,277 | **2.23×** |
| 256 | 515,027 | 1,241,538 | **2.41×** |
| 512 | 600,383 | 1,217,781 | **2.03×** |
| 1024 | 276,087 | 1,199,027 | **4.34×** |

GRU stays in a flat band of 1.2–1.4M tok/s across the entire range. The transformer oscillates, then collapses at ctx=1024. The 4.34× advantage at ctx=1024 is not because GRU got faster — it's because the transformer's attention becomes quadratically expensive and falls off a cliff.

The theoretical crossover point where transformers *should* win (due to parallelism across the sequence dimension) appears to be well above ctx=1024 for a 1M-parameter model at this scale. This is consistent with Max's intuition: **at long sequence lengths, GRU has the hardware advantage**, not the transformer. The transformer's parallelism advantage is in *training quality* (attention can look at any position in one step), not throughput on this hardware at this scale.

Note: The transformer's throughput non-monotonicity (goes up at 512, collapses at 1024) is a batch size artifact — the autotuner picks different architectures at different context lengths. At ctx=256 it has 1 layer (hidden=272), at ctx=1024 it has 8 layers (hidden=96). This is an experiment design limitation: the fair comparison should fix architecture across contexts. The trend at 1024 is real regardless.

---

## What "fits in cache" actually means

Max's question: *"What's the largest parameter shape that just sits in cache, repeatedly processing data?"*

This is the right intuition pointed at the wrong level. L2 cache is 6 MB. A 1M-parameter model at FP32 is 4 MB of weights — *plus* activations, gradients, and optimizer state, which multiply that by 3–4× in training. Nothing "sits in cache" during a training step at these sizes; the working set is far larger than L2.

The more useful reframe: **you want high arithmetic intensity, not cache residency.**

Consider what happens during a single GRU forward step for one token in a sequence:
- Fetch the weight matrices from VRAM (let's say hidden_size=256, so W_r is 256×512 = 128K floats = 512 KB)
- Apply them to the input/state vector: 2 × 256 × 512 = 262K FLOPs
- That's 262K FLOPs / 512KB = ~0.5 FLOP/byte

That seems low — but the key is batch size. With batch=512 sequences running in parallel, the same 512 KB of weights is applied to 512 different input vectors simultaneously. The FLOPs scale with batch size; the memory fetch does not:
- FLOPs: 512 × 262K = 134M
- Bytes fetched: 512 KB (weights, fetched once and reused across the batch)
- Arithmetic intensity: 134M FLOPs / 512 KB ≈ **256 FLOP/byte**

At 256 FLOP/byte vs the 3090's ~38 FLOP/byte memory bandwidth ratio, this is comfortably compute-bound. The weights are not "in cache" in any L2 sense, but they're reused across the batch in a way that makes the memory bandwidth cost irrelevant. This is exactly what GEMM operations do well.

**For transformers, why is memory bandwidth a bottleneck?** Attention's core operation — computing attention scores across Q, K, V — involves matrices whose size scales quadratically with sequence length. At long context, the activation tensors dominate memory traffic, not the weights. Flash attention addresses this by restructuring the computation to maximize reuse within each SM's SRAM, but it's fundamentally fighting a harder arithmetic intensity problem than a weight-dominated computation.

**What model shape would fit the 3090 best?** *(Inferred, not measured.)*
- Wide hidden layers (≥512) to ensure GEMMs are large enough to hit Tensor Core thresholds
- Moderate depth (fewer layers means fewer kernel-launch overhead events per step)
- Large batch size to amortize weight fetches across many sequences — this is what provides the effective arithmetic intensity
- Fewer distinct matrix shapes so cuBLAS can cache tiling configurations

GRU at hidden_size=232 with batch=512 is already close to this shape. The evidence shows it hits Tensor Cores; the transformer at hidden_size=128 does not. Scaling GRU width (not depth) is the most likely path to better utilization — though this is prediction, not measurement.

---

## The gradient accumulation argument for scaling RNNs

*(This is a theoretical argument Max asked to explore, not a measured result.)*

The standard objection to RNNs at scale is that training is inherently sequential in time — you cannot parallelize the forward pass across a long sequence because step T depends on the hidden state from step T-1. Transformers can process all positions in parallel, which is why they scale to long contexts in training.

But here's the gradient accumulation argument:

1. At short-to-medium context lengths (say, ctx=128), GRU trains at ~2–3× the token throughput of a transformer on this hardware.
2. If you have N independent RNN training streams running in parallel (N different initialization seeds, N different data shuffles, or N separate sequences from the corpus), each stream sees a short context efficiently.
3. Accumulate gradients across these N streams before taking an optimizer step. This is logistically equivalent to a larger batch size across sequence chunks — the same operation distributed data parallelism does across GPUs, but here applied to temporal segments of the same sequences.
4. The effective "sequence length" a single model sees per optimizer step can be N × ctx, even though no single stream processed sequences longer than ctx.

This doesn't give transformers' global attention across the full N×ctx window — the model only sees local context within each stream's segment. But for many sequence modeling tasks, local context may be sufficient, and the training throughput advantage can be substantial.

The measured 2–4× throughput advantage already exists with a single stream. With multiple streams, the advantage multiplies. Whether this translates to comparable training efficiency (loss per compute-second) depends on task-specific questions about how much context the model needs. That question is not answered here.

---

## What to do next (measured claims only)

**Definitely:**
1. **Fix the DataLoader.** Add `pin_memory=True`, `num_workers=2`, `persistent_workers=True`. Expected recovery: ~10% of step time (3ms out of 14ms). Quick win.
2. **Increase the power cap or remove it.** Currently capped at 185W vs 350W stock. This limits clock speed. The relative GRU/transformer story is unchanged, but absolute utilization will improve. [MEASURED: current peak is 184W.]
3. **Profile GRU at larger hidden sizes** to find the point where Tensor Core utilization is maximum. The 1M-param GRU at hidden=232 already hits Tensor Cores; a larger hidden (e.g. 512–1024) with fewer layers may be even better-utilized.

**Probably:**
4. **Try TF32 explicitly.** PyTorch on Ampere enables TF32 by default for CUDA matmuls, but disables it for cuDNN (which handles RNN ops). `torch.backends.cudnn.allow_tf32 = True` may unlock a 2× throughput improvement for GRU. [Not yet measured in this benchmark.]
5. **Try `torch.compile`.** Fuses elementwise ops, reduces kernel launch overhead. The transformer's 49% CPU overhead is the most likely to improve.

**Open:**
6. **Transformer at larger hidden sizes.** The transformer's failure to hit Tensor Cores is likely an artifact of small hidden_size=128. A 1M-param transformer with hidden=512 and fewer layers/heads would produce larger GEMMs and may change the picture.
7. **The gradient accumulation argument.** Needs an experimental design, not just profiling.

---

## Original throughput benchmark

*Benchmark script: [`experiments/gpu_utilization/benchmark.py`](../../../experiments/gpu_utilization/benchmark.py). Full results: [`experiments/gpu_utilization/artifacts/results.json`](../../../experiments/gpu_utilization/artifacts/results.json).*

| Family | Size | Ctx | Params | Batch | Train tok/s | ms/step | Peak VRAM | Decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| transformer | ~1M | 32 | 936,017 | 256 | 710,637 | 11.53 | 0.23 GB | 1,087 |
| transformer | ~1M | 256 | 996,945 | 32 | 543,433 | 15.07 | 0.33 GB | 1,087 |
| lstm | ~1M | 32 | 994,225 | 512 | 1,742,851 | 9.40 | 0.41 GB | 2,028 |
| lstm | ~1M | 256 | 994,225 | 256 | 1,570,337 | 41.73 | 1.49 GB | 2,028 |
| gru | ~1M | 32 | 1,003,233 | 512 | 1,313,351 | 12.47 | 0.49 GB | 2,684 |
| gru | ~1M | 256 | 1,003,233 | 512 | 1,265,393 | 103.58 | 3.60 GB | 2,922 |
| transformer | ~10M | 32 | 9,792,449 | 128 | 106,523 | 38.45 | 0.46 GB | 951 |
| transformer | ~10M | 256 | 9,993,153 | 8 | 91,705 | 22.33 | 0.31 GB | 1,078 |
| lstm | ~10M | 32 | 10,008,161 | 256 | 150,886 | 54.29 | 1.04 GB | 648 |
| lstm | ~10M | 256 | 10,008,161 | 512 | 156,958 | 835.08 | 12.95 GB | 721 |
| gru | ~10M | 32 | 10,004,545 | 128 | 194,254 | 21.09 | 0.55 GB | 3,678 |
| gru | ~10M | 256 | 10,004,545 | 512 | 252,836 | 518.41 | 11.48 GB | 3,590 |

### Training throughput

At 1M parameters: LSTM 2.5×, GRU 1.8× the transformer at ctx=32. At 10M: GRU 2.5–2.8×, LSTM 1.4–1.7×. The RNN advantage holds at longer context — at ctx=1024, GRU is 4.34× faster than the transformer (see sequence sweep above).

### Decode throughput

- 1M: both RNN types decode at ~2× the transformer rate
- 10M: **GRU decodes at 3.7× the transformer** (3,678 vs 951 tok/s at ctx=32)
- 10M: **LSTM falls *below* the transformer** (648 vs 951 tok/s) — unexplained, possibly an artifact

RNN decode is O(1) in sequence length (no KV cache). The transformer's KV-cache-growing decode will widen this gap further at longer context.

### VRAM

Transformer is the most memory-efficient at these sizes (~0.23–0.46 GB vs 0.4–12.95 GB for RNNs). This is somewhat misleading: RNNs use more VRAM *because* they need larger batches to saturate the GPU. At fixed batch size or at batch=1 inference, the picture would be different.

---

## Caveats

- **Power-capped hardware.** Max of 185W vs 350W stock. All absolute numbers are artificially depressed. Relative comparisons hold.
- **Float32 only.** Mixed precision (fp16/bf16) with explicit Tensor Core paths would likely boost the transformer disproportionately. The transformer's failure to hit Tensor Cores at 1M/ctx=32 may not persist at larger sizes or with explicit AMP.
- **Batch-optimized throughput, not production serving.** Batch sizes tuned per-cell for max throughput. Batch=1 inference would differ substantially.
- **Quality not measured.** Throughput says nothing about how well these models learn. Not measured here.
- **LSTM decode collapse at 10M is unexplained.** Don't treat the LSTM 10M decode number as settled.
- **Sequence sweep uses different transformer architectures at different contexts** (the autotuner picks the best config per context). This muddies the crossover analysis.
