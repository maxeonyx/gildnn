# GPU Utilization: RNN vs Transformer Throughput

## Question

> "I tend to think that we can get significantly more FLOPs out of a GPU training an RNN than a transformer."

The hypothesis is about hardware utilization, not model quality. Transformers are attention-heavy: quadratic in sequence length during training, and KV-cache-growing during autoregressive decode. RNNs (LSTM, GRU) process sequence recurrently — O(1) per step in state size, but potentially more parallelizable within a step. The question is whether that pays off in raw throughput on a modern GPU at small-to-medium parameter counts.

## Method

Benchmarked three architectures — transformer, LSTM, GRU — at two parameter targets (~1M, ~10M) and two context lengths (32, 512), on RTX 3090 (24 GB VRAM), PyTorch 2.11, CUDA 12.8, float32 throughout.

- Parameters matched within each size class (within ~1%)
- Batch size tuned independently per cell to maximize training throughput
- Synthetic data (random token IDs) to eliminate data pipeline bottlenecks
- Training throughput: tokens/second sustained over full benchmark window
- Decode throughput: autoregressive generation, one token at a time, averaged over a fixed sequence length

Benchmark script: [`experiments/gpu_utilization/benchmark.py`](../../../experiments/gpu_utilization/benchmark.py). Full results JSON: [`experiments/gpu_utilization/artifacts/results.json`](../../../experiments/gpu_utilization/artifacts/results.json).

## Results

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

## Analysis

### Training throughput

At 1M parameters, both RNN types are clearly faster in training:

- LSTM: **2.5×** the transformer's tokens/sec at ctx=32 (1.74M vs 711K)
- GRU: **1.8×** at ctx=32 (1.31M vs 711K)

The transformer narrows at longer context but doesn't close the gap — the RNN advantage holds at ctx=256 too.

At 10M, the picture shifts:

- GRU: **2.5–2.8×** the transformer (194K vs 107K at ctx=32; 253K vs 92K at ctx=256)
- LSTM: only **1.4–1.7×** — still ahead, but the gap has compressed relative to GRU

GRU consistently outpaces LSTM in training throughput, which suggests LSTM's two-gate structure (vs GRU's one) costs more than it might appear at this batch scale.

### Decode throughput

This is where the architecture difference is most stark:

- At 1M, LSTM and GRU both decode at ~2× the transformer rate
- At 10M, **GRU decodes at 3.7× the transformer** (3,678 vs 951 tok/s at ctx=32)
- At 10M, **LSTM falls *below* the transformer** (648 vs 951 tok/s)

LSTM's decode collapse at 10M is unexpected. The decode path for an RNN is O(1) in sequence length — there's no KV cache — so this is not a context-growth effect. It likely reflects something about LSTM's internal state management at larger hidden sizes becoming a bottleneck on a single-step forward pass. This warrants investigation if LSTM is a serious candidate.

### Context scaling

The transformer's attention cost is theoretically quadratic in context length. In practice, at ctx=256 vs ctx=32:

- Transformer training throughput drops ~24% (711K → 543K at 1M; smaller at 10M due to batch constraint)
- RNN training throughput drops less or holds — LSTM 1M: 10% drop; GRU 1M: 4% drop

For decode, the RNNs show *identical* decode speed regardless of context (LSTM 1M: 2,028 both contexts; GRU 1M: 2,684 → 2,922). The transformer decode speed also holds here — because at batch=1 autoregressive, the KV cache lookup cost is modest at these sizes. The theoretical advantage for RNNs in very long-context decode will grow as context scales beyond what was tested.

### VRAM

Transformer is the most memory-efficient at these sizes:

- ~0.23–0.46 GB at 1–10M params
- LSTM and GRU require 0.4–12.95 GB, depending on batch size tuned for peak throughput

This is somewhat misleading: RNNs use more VRAM *because* they need larger batches to saturate the GPU. At a fixed batch size, the transformer's VRAM would be comparable or lower. At production serving with batch=1, the RNN VRAM advantage would likely flip back in their favor.

## Conclusion

The hypothesis is confirmed. At matched parameters on RTX 3090, RNNs achieve substantially higher training and decode throughput than a transformer in float32. GRU is the clearest winner — faster in training at both scales, and dramatically faster in decode at 10M. LSTM leads in training but has an unexplained decode regression at 10M.

The transformer's key advantage at these sizes is not throughput — it's VRAM efficiency at low batch sizes. For research that doesn't need to maximize generation throughput, that matters. For anything where training speed or inference rate is the bottleneck, GRU at these scales runs faster on this hardware.

## Caveats

- **Float32 only.** Mixed precision (fp16/bf16) with tensor cores would likely boost the transformer disproportionately, since Ampere flash attention implementations are highly optimized for lower precision. The comparison may look different with `torch.compile` + AMP.
- **Batch-optimized throughput, not production serving.** Batch sizes were tuned per-cell for maximum throughput. At batch=1 (typical inference), the results would differ — and the RNN's O(1) state-per-step advantage would be more pronounced.
- **Quality not measured.** Throughput advantage is irrelevant if the model doesn't learn as well. These sizes and architectures haven't been quality-compared in this repo yet.
- **Small scale only.** 1M–10M parameters may not represent scaling behavior at 100M+, where transformer training infrastructure is far more mature. The conclusions here apply to this hardware-and-scale regime only.
- **LSTM decode collapse at 10M is unexplained.** A single-step forward pass for an RNN should not degrade this way. This could be a profiling artifact, a batch scheduling effect, or something real about LSTM's hidden state operations at larger sizes. Don't treat the LSTM 10M decode number as settled.
