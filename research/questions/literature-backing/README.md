# Literature Backing for Research Directions

This document backs our three experimental directions with prior work. Purpose: confirm we're building on real foundations, identify exactly where our contributions sit, and flag what's genuinely novel vs. incremental. Requested in [dictation 2026-05-22-13](../../../dictations/2026-05-22-13.md).

---

## 1. Multi-Rate Execution

**Our result:** fixed deterministic block rates [1,2,4,8] give 20.7% speedup with *better* quality than uniform execution.

### Foundational

| Paper | Key idea | Relation to us |
|-------|----------|----------------|
| **Clockwork RNN** (Koutník et al. 2014) [arXiv:1402.3511](https://arxiv.org/abs/1402.3511) | Partition RNN modules by fixed clock rates — some tick every step, others every 2, 4, 8. | Direct ancestor. We apply the same principle to transformer blocks. |
| **Stochastic Depth** (Huang et al. 2016) [arXiv:1603.09382](https://arxiv.org/abs/1603.09382) | Randomly skip residual blocks during training; acts as regularizer. | Explains *why* skipping improves quality — implicit ensemble + regularization. |
| **Mixture-of-Depths** (Raposo et al. 2024, DeepMind) [arXiv:2404.02258](https://arxiv.org/abs/2404.02258) | Per-layer learned router decides which tokens get compute. 50% faster, matches quality. | Most recent large-scale validation that not-all-tokens-need-all-layers works. Uses learned routing; we use fixed schedule. |

### Supporting

| Paper | Key idea | Relation to us |
|-------|----------|----------------|
| **LayerDrop** (Fan et al. 2020) [arXiv:1909.11556](https://arxiv.org/abs/1909.11556) | Structured dropout over entire layers enables elastic inference depth. | Training-time technique; our fixed rates are both train and inference. |
| **Adaptive Computation Time** (Graves 2016) [arXiv:1603.08983](https://arxiv.org/abs/1603.08983) | Learned halting — variable compute per position. | Pioneered "not all positions need equal compute." We use position-independent block-level rates instead. |
| **Token Merging / ToMe** (Bolya et al. 2023) [arXiv:2210.09461](https://arxiv.org/abs/2210.09461) | Progressively merge similar tokens to reduce sequence length at deeper layers. | Orthogonal efficiency approach; could compose with multi-rate. |
| **Hierarchical Multiscale RNN** (Chung et al. 2017) [arXiv:1609.01704](https://arxiv.org/abs/1609.01704) | Hard boundary detection for multi-timescale processing. | Learned boundaries vs. our fixed schedule. More complex, less predictable compute. |

### What's novel in our approach

Fixed deterministic rates showing *better* quality, not just equivalent. The literature explores learned routing (MoD), stochastic skipping (Stochastic Depth), and adaptive halting (ACT) — all add complexity to decide what to skip. We skip on a fixed schedule and get a regularization benefit. The closest ancestor (Clockwork RNN) predates transformers and wasn't tested at scale. The specific finding that mandatory structured skipping acts as a beneficial regularizer in transformers appears underexplored.

---

## 2. Persistent CUDA Kernels for Intra-GPU Pipelining

**Our goal:** run multiple independent small models concurrently on one consumer GPU by scheduling them as a persistent kernel, avoiding PyTorch stream overhead.

### Key references

| Paper / Project | Key idea | Relation to us |
|-----------------|----------|----------------|
| **CUTLASS 3.x** (NVIDIA) [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | Production persistent kernels with warp-specialized SM scheduling. | Implementation reference — shows how to keep SMs occupied without re-launching. |
| **Stream-K** (Osama et al. 2023, NVIDIA) [arXiv:2301.03598](https://arxiv.org/abs/2301.03598) | Persistent GEMM with dynamic work division across SMs, eliminating wave quantization. | Proves persistent scheduling works for irregular workloads. We'd extend to heterogeneous model stages. |
| **FlashAttention-2** (Dao 2023) [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) | Persistent kernel scheduling for attention — occupancy-aware tiling. | Shows persistent patterns applied to ML-specific ops beyond GEMM. |
| **FlashAttention-3** (Shah et al. 2024) [arXiv:2407.08691](https://arxiv.org/abs/2407.08691) | Extends FA2 with warp specialization and Hopper-specific features. | Latest evolution of persistent ML kernels. |
| **Nimble** (Kwon et al. 2020) [arXiv:2012.02732](https://arxiv.org/abs/2012.02732) | Kernel fusion and ahead-of-time scheduling to reduce launch overhead. | Tackles the same problem (launch overhead) from the framework side. We go below. |
| **Triton** (Tillet et al. 2019) [arXiv:1907.00598](https://arxiv.org/abs/1907.00598) | Python-level GPU kernel authoring with block-level abstractions. | Likely implementation path — persistent kernels in Triton are possible but non-trivial. |

### Context (pipeline parallelism)

| Paper | Key idea | Relation to us |
|-------|----------|----------------|
| **GPipe** (Huang et al. 2019) [arXiv:1811.06965](https://arxiv.org/abs/1811.06965) | Micro-batch pipeline parallelism across GPUs. | We want intra-GPU pipelining of the same conceptual structure. |
| **PipeDream** (Narayanan et al. 2019) [arXiv:1806.03377](https://arxiv.org/abs/1806.03377) | Asynchronous pipeline with weight stashing to hide bubble overhead. | Async scheduling ideas transfer; weight stashing less relevant for inference. |

### What's novel in our approach

Prior persistent kernel work targets single-model efficiency (FlashAttention) or single-op throughput (Stream-K). Pipeline parallelism work targets multi-GPU clusters. We're applying persistent kernel scheduling to run *multiple independent small models* as pipeline stages on a *single consumer GPU*. The problem is: PyTorch streams add too much overhead for small kernels, but the hardware supports concurrent execution — we need to go below the framework to exploit it. This specific application (intra-GPU model-level pipelining via persistent kernels) doesn't appear in the literature.

---

## 3. MixAdd — Norm-Preserving Learned Residual Mixing

**Our mechanism:** `output = x * sqrt(σ(m)) + delta * sqrt(1 - σ(m))` — learnable per-block gate, norm-preserving on both branches.

### Foundational

| Paper | Key idea | Relation to us |
|-------|----------|----------------|
| **Highway Networks** (Srivastava et al. 2015) [arXiv:1505.00387](https://arxiv.org/abs/1505.00387) | `T·H(x) + (1-T)·x` — OG learned gating for residual mixing. | Direct ancestor. Linear interpolation but no norm-preservation guarantee. |
| **ReZero** (Bachlechner et al. 2021) [arXiv:2003.04887](https://arxiv.org/abs/2003.04887) | `x + α·F(x)` with learnable α initialized to 0. Simplifies training dynamics. | Learnable scaling on delta only. No gate on residual, no norm control. |
| **NFNets** (Brock et al. 2021, DeepMind) [arXiv:2102.06171](https://arxiv.org/abs/2102.06171) | Remove normalization layers entirely; use analytical variance-preserving residual scaling. | Norm-preservation via fixed computed scales. Not learned, not adaptive per-input. |

### Supporting

| Paper | Key idea | Relation to us |
|-------|----------|----------------|
| **DeepNet** (Wang et al. 2022) [arXiv:2203.00555](https://arxiv.org/abs/2203.00555) | `α·x + β·F(x)` with depth-dependent constants for 1000-layer transformers. | Fixed α,β derived analytically. Shows residual scaling matters enormously at depth. |
| **Signal Propagation in Transformers** (Noci et al. 2022) [arXiv:2209.15399](https://arxiv.org/abs/2209.15399) | Theoretical analysis of how residual connections affect signal propagation. | Provides the *why* — unconstrained residuals cause rank collapse or explosion at depth. |
| **minGRU** (Feng et al. 2024) [arXiv:2410.01201](https://arxiv.org/abs/2410.01201) | `(1-z)·h + z·candidate` — minimal gated unit, surprisingly powerful. | Same interpolation structure, applied to sequence state. Validates that learned linear mixing is sufficient. |
| **GRU** (Cho et al. 2014) [arXiv:1406.1078](https://arxiv.org/abs/1406.1078) | Update gate: `(1-z)·h + z·h̃`. | The original update-gate interpolation. MixAdd is this pattern applied to residual connections with sqrt for norm control. |
| **Mamba / S4** (Gu & Dao 2023–24) [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | Input-dependent gating of state retention in structured state spaces. | Shows input-dependent retention gating is broadly useful; MixAdd gates residual retention similarly. |

### What's novel in our approach

Three properties combined:

1. **Learned** — gate adapts during training (unlike NFNets' fixed scales, DeepNet's constants)
2. **Norm-preserving** — sqrt factors on *both* branches maintain expected norm (unlike Highway's linear interpolation, ReZero's delta-only scaling)
3. **Per-block adaptive** — each block learns its own mixing ratio (unlike global depth-dependent formulas)

Highway Networks had (1) but not (2). ReZero had learnable scalars but only on delta and no norm guarantee. NFNets had (2) but not (1). The specific combination — `sqrt(gate)` on both branches ensuring `E[||output||²] = E[||x||²]` when inputs are uncorrelated — appears novel. The closest match is GRU's update gate, but GRU uses linear interpolation `(1-z)·a + z·b` which doesn't preserve norms unless inputs happen to have equal magnitude.
