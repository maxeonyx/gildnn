# Burn + Candle Operator Support (verified from source)

## Methodology
- Cloned **Candle** at commit `df618f80083d6f95de44a2cf52044543fe79301b` and **Burn** at commit `87aa0be058624e9f1bc860482fd7cd144b5e1152` on the public `main` branches to inspect operator coverage.
- Read the relevant Candle GPU backend implementations (`candle-core`, `candle-nn`, `candle-kernels`) and the Burn Candle backend glue (`crates/burn-candle`) to confirm the exact ops exposed to Burn.
- Focused on primitives required for a decoder-style transformer (linear projections, softmax-based attention, MLP activations, normalization, tensor reshaping/indexing, dropout, embeddings, and einsum-style contractions).

## Transformer-Critical Operator Inventory

| Operator block | Candle GPU implementation evidence | Burn Candle binding |
| --- | --- | --- |
| Dense matmul & linear layers | `Tensor::matmul` delegates to backend `Storage::matmul`, and the CUDA path calls cuBLAS strided batched GEMM for BF16/F16/F32/F64 (`candle-core/src/tensor.rs`, `candle-core/src/cuda_backend/mod.rs`). Linear modules wrap matmul and optional bias (`candle-nn/src/linear.rs`). | `float_matmul` keeps tensors contiguous before invoking Candle `broadcast_matmul`, powering `burn::tensor::module::linear` (`crates/burn-candle/src/ops/tensor.rs`, `crates/burn-tensor/src/tensor/module.rs`). |
| Softmax / attention reduction | `SoftmaxLastDim` custom op dispatches CUDA kernels loaded from `candle-kernels::REDUCE` and Metal pipelines (`candle-nn/src/ops.rs`). Flash-attention kernels live under `candle-flash-attn`. | Burn’s `activation::softmax` composes exp/sum ops backed by Candle kernels, and attention helpers rely on those reductions (`crates/burn-tensor/src/tensor/activation/base.rs`, `crates/burn-candle/src/ops/tensor.rs`). |
| Elementwise activations | Unary ops such as ReLU/GELU/Sigmoid map to CUDA kernels defined in `candle-kernels/src/unary.cu` with dispatch code in `candle-core/src/op.rs`. | `ActivationOps` for the Candle backend call the corresponding Candle tensor methods (e.g., `tensor.relu()`, `tensor.gelu()`) (`crates/burn-candle/src/ops/activation.rs`). |
| LayerNorm / RMSNorm | `LayerNorm` custom op issues specialized CUDA kernels via `layernorm` entry points and Metal shaders (`candle-nn/src/ops.rs`). High-level modules wrap these ops (`candle-nn/src/layer_norm.rs`). | Burn maps LayerNorm/RMSNorm through tensor ops and module helpers that call Candle’s custom op via the backend (`crates/burn-candle/src/ops/tensor.rs`, `crates/burn-tensor/src/tensor/ops/modules/base.rs`). |
| Dropout | Training dropout in Candle samples device RNG and applies scaling on-device (`candle-nn/src/ops.rs`). | Burn’s `Dropout` module draws Bernoulli masks with `float_random` and rescales activations using Candle elementwise ops (`crates/burn-nn/src/modules/dropout.rs`, `crates/burn-candle/src/ops/tensor.rs`). |
| Residual adds & scalar ops | Binary ops dispatch to CUDA kernels declared in `candle-kernels/src/binary.cu` via `BinaryOp` handlers (`candle-core/src/tensor.rs`, `candle-core/src/op.rs`). | Backend `float_add/sub/mul/div` forward to Candle broadcast ops (`crates/burn-candle/src/ops/tensor.rs`). |
| Reshape, permute, transpose | View/reshape utilities are metadata-only, while permutations fall back to contiguous copies handled per backend (`candle-core/src/tensor.rs`). | Burn exposes reshape/permute through backend helpers (`crates/burn-candle/src/ops/base.rs`, `crates/burn-candle/src/ops/tensor.rs`). |
| Indexing (gather, scatter, narrow) | CUDA backend implements gather/scatter/index_select kernels (`candle-core/src/cuda_backend/mod.rs`) used by tensor indexing APIs. | Burn uses these for attention masks and KV updates via `float_gather`, `float_scatter`, and `float_select_assign` (`crates/burn-candle/src/ops/tensor.rs`). |
| Embeddings | Embedding lookups/scatter gradients reside in `candle-nn/src/embedding.rs` atop tensor gather/scatter. | Burn’s generic `ModuleOps::embedding` relies on backend select/select_assign, which the Candle backend implements with GPU kernels (`crates/burn-tensor/src/tensor/ops/modules/base.rs`, `crates/burn-candle/src/ops/tensor.rs`). |
| Einsum / batched contractions | `Tensor::einsum` supports GPU execution through CUDA helpers in `candle-core` (einsum lowering + kernels). | Burn exposes einsum via the backend tensor API for custom attention patterns (`crates/burn-candle/src/ops/tensor.rs`). |

## KV-Cache and Memory Notes
- Candle’s CUDA backend keeps tensors on the device; `float_empty/zeros` allow preallocating KV buffers, while `float_narrow`, `float_scatter`, and `float_select_assign` update cache slices without host copies (`crates/burn-candle/src/ops/tensor.rs`).
- Mixed-precision caches (FP16/BF16) are supported because Candle’s CUDA kernels operate across those dtypes (`candle-core/src/cuda_backend/mod.rs`, `candle-kernels/src/unary.cu`).
- When cache lengths exceed a single allocation, tensors can be concatenated with `Tensor::cat`/`float_cat`, which Candle handles on device.

## Practical Conclusions
- All primitives needed for a standard transformer block—linear projections, softmax attention, GELU/SILU activations, residual add, normalization, dropout, reshape, gather/scatter, embeddings, and einsum—are present in Candle’s GPU backends and exposed through Burn’s Candle backend.
- Training and inference will run on compiled CUDA (via cuBLAS and custom kernels) or Metal (via precompiled shaders). Expect first-call kernel loading but no interpreter overhead during steady-state execution.
- KV-cache extensions rely on manual tensor management but are fully GPU-resident; dynamic sequence growth is achievable through scatter-based updates without reallocating per step.
- Missing pieces remain advanced fused attention kernels (e.g., FlashAttention variants beyond what `candle-flash-attn` provides) and distributed multi-GPU orchestration, but a “vanilla” transformer stack is implementable today.
