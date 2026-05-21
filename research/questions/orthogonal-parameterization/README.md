# Inherently Orthogonal Weight Parameterization

**Question:** Does parameterizing weight matrices to be orthogonal by construction (not pushed there by the optimizer) help the residual-stream-across-time architecture? Can we eliminate the need for Muon by making orthogonality a structural guarantee?

**Status:** Theory complete. Experiment designed.

**Grounded in:** [dictations/2026-05-21-5.md](../../../dictations/2026-05-21-5.md), [dictations/2026-05-21-6.md](../../../dictations/2026-05-21-6.md), [dictations/2026-05-21-7.md](../../../dictations/2026-05-21-7.md)

---

## Motivation

From [dictation 2026-05-21-5](../../../dictations/2026-05-21-5.md):

> *"Muon is a way of trying to approximate orthogonal matrices via singular value decomposition at the optimizer level, but it's pushing them towards orthogonal. But something else I'm interested in is — let's say we could have a complex-valued network, then parameterize our weight matrices in such a way that no matter where we move the weights, the matrix itself stays orthogonal."*

The [Muon experiment](../residual-stream-across-time/README.md#muon-optimizer-orthogonal-stability-hypothesis-confirmed) confirmed that orthogonal pressure fixes gradient explosion in the residual-stream-across-time architecture. But Muon is optimizer-level — it pushes TOWARD orthogonal without guaranteeing it. What if the parameterization itself guarantees orthogonality?

---

## Approaches (from literature review)

| Method | Mechanism | Cost | Coverage |
|--------|-----------|------|----------|
| **Exponential map** (expRNN) | W = exp(A), A skew-symmetric | O(n³) via matrix_exp | Full SO(n) |
| **Cayley transform** | W = (I+A)⁻¹(I−A), A skew-symmetric | O(n³) for inverse | Almost all of SO(n) |
| **Householder reflections** | Product of n reflections | O(n²) sequential | Full O(n) |
| **Givens rotations** | Product of plane rotations | O(n²) parallel | Full SO(n) |

At our scale (d_model ≈ 100-200), `torch.linalg.matrix_exp` is cheap. **Exponential map is the best first choice.**

### How it works

Instead of storing a weight matrix W directly, store a skew-symmetric matrix A (upper triangle only, negate to get lower). The actual weight used in the forward pass is W = exp(A). Since A is skew-symmetric, exp(A) is guaranteed orthogonal for any values of A. Standard gradient descent on A's parameters works normally.

Parameters: d(d-1)/2 free parameters for a d×d orthogonal matrix (the upper triangle of A).

---

## Experiment design

**Question:** Does enforced orthogonality (via exp map) help the residual-stream-across-time architecture compared to:
1. Unconstrained weights + AdamW (the original approach that diverged at k≥8)
2. Unconstrained weights + Muon (the approach that fixed stability)

**Task:** Character-level TinyShakespeare, same as all previous experiments.

**Architecture:** Residual-stream-across-time (mix-add, temporal attention k=8) with:
- All weight matrices (proj_in, proj_out, Q/K/V/O in temporal attention) parameterized via exp map
- Standard AdamW optimizer (no Muon needed — orthogonality is structural)
- ~185K params to match baseline

**Controls:**
- Transformer baseline: 190K params, val 1.535 on 900k data
- Residual-stream-time + Muon: 185K params, val 1.606 on 900k data

**Hypothesis:** If orthogonality is the key factor, structural orthogonality should:
1. Train stably at k=8+ without Muon (no divergence)
2. Potentially achieve better loss than Muon (exact vs approximate orthogonality)
3. Allow standard AdamW optimizer

**Negative hypothesis (also interesting):** If Muon's near-orthogonal pressure is enough and exact orthogonality over-constrains the network, the exp-map version might train stably but converge to WORSE loss. This would indicate the network benefits from some non-orthogonal component.

---

## Volume-preserving nonlinearities (future)

From [dictation 2026-05-21-6](../../../dictations/2026-05-21-6.md): Max's deeper vision is unitary linear layers + Hamiltonian-flow nonlinearities. The flow is a separately-trained, frozen module — "like a tokenizer."

This is **deferred** because:
- Computing ∇H per activation application is expensive (~10-50x vs ReLU)
- Need to validate the simpler version (orthogonal weights + standard GELU) first
- If orthogonal weights alone don't help, the more complex version is unlikely to help either

---

## Results

### Exp-map orthogonal RNN (182K params, 900k TinyShakespeare, 5 epochs)

| Epoch | Train Loss | Val Loss | Orth Error (‖WᵀW - I‖) |
|-------|-----------|----------|------------------------|
| 1 | 2.327 | 2.175 | 0.0045 |
| 2 | 2.048 | 2.064 | 0.0044 |
| 3 | 1.961 | 2.018 | 0.0045 |
| 4 | 1.912 | 1.994 | 0.0045 |
| 5 | 1.883 | **1.972** | 0.0045 |

Training time: **108 minutes** (vs ~3 min for Muon version, ~2 min for transformer).

### Comparison

| Model | Params | Val Loss | Gap to Transformer |
|-------|--------|----------|-------------------|
| Transformer | 190K | 1.535 | — |
| Residual-stream + Muon | 185K | 1.606 | +0.071 |
| Residual-stream + exp-map | 183K | **1.972** | **+0.437** |

### Interpretation

**The stability hypothesis is fully confirmed:** structural orthogonality prevents divergence at k=8 with plain AdamW. No Muon needed. This is the third independent confirmation that orthogonality is THE mechanism for stability in repeated-weight architectures.

**But exact orthogonality massively over-constrains the network.** The +0.366 gap between exp-map and Muon is much larger than the +0.071 gap between Muon and transformer. The network needs non-orthogonal degrees of freedom to be expressive.

Why this happens:
1. **Square FFN:** orthogonal matrices must be square, so the feedforward can't expand (d→4d→d becomes d→d→d). This eliminates the capacity bottleneck that FFN expansion provides.
2. **Reduced expressiveness:** orthogonal transformations preserve norms but can only rotate/reflect — they can't scale or project. The network can't selectively amplify or suppress dimensions.
3. **matrix_exp cost:** 42ms per weight refresh, making training 30-50x slower. Not viable for rapid iteration.

### Conclusion

**Muon is the right approach.** It provides enough orthogonal pressure for stability while leaving enough freedom for expressiveness. Exact orthogonality is too much of a constraint.

This also answers Max's question from [dictation 2026-05-21-5](../../../dictations/2026-05-21-5.md): "can we parameterize weight matrices so they stay orthogonal?" — Yes, and it works for stability, but it kills expressiveness. The optimizer-level approach (Muon) is strictly better.

### On the volume-preserving direction

The deeper vision (unitary layers + Hamiltonian-flow nonlinearities) might address the expressiveness gap — the flow could provide the nonlinear capacity that orthogonal layers alone lack. But given that the simpler version (orthogonal + GELU) underperforms so badly, and that Hamiltonian flows are expensive to compute, this direction seems unlikely to produce a practical architecture within the project's timebox.

Artifacts: [`experiments/orthogonal_rnn/artifacts/`](../../../experiments/orthogonal_rnn/artifacts/).

---

*Last updated: 2026-05-22. Experiment complete — negative on quality, positive on stability mechanism.*
