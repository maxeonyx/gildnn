# Backend Choice: PyTorch

**Status: DECIDED** — per [dictation 2026-05-22-12](../../../dictations/2026-05-22-12.md).

## Motivation

[Dictation 2026-05-22-7](../../../dictations/2026-05-22-7.md): Max wants compiled execution, deliberately chosen — not defaulted to eager PyTorch out of inertia. [Dictation 2026-05-22-12](../../../dictations/2026-05-22-12.md): "nothing should be blocked on me — choose something."

## Decision

**PyTorch** as the project backend, with a three-tier execution policy:

- **Experiments:** eager PyTorch. Flexibility for rapid iteration, debugging, weird shapes.
- **Reusable core/:** `torch.compile` for stable forward paths. Target `fullgraph=True` where possible.
- **Async execution research:** custom CUDA kernels via Triton or raw CUDA. This is the eventual target — neither framework's compiler solves it.

## Why PyTorch over JAX

The project's core research goal is **async execution via persistent CUDA kernels** (per [dictation 2026-05-22-11](../../../dictations/2026-05-22-11.md), [VISION.md](../../../VISION.md)). This is fundamentally a custom GPU programming problem. Neither JAX's XLA nor PyTorch's standard APIs solve it — you need to go below both.

Given that:

- JAX's advantage (strict compilation) is irrelevant for the hardest part of the work
- PyTorch + Triton provides a more natural path to custom kernel development
- The existing Windows setup works NOW with 9 days left in the timebox
- Linux migration (required for JAX GPU) is real overhead for no payoff on the core goal
- JAX's XLA is actually *less* flexible for custom low-level CUDA than PyTorch + Triton

JAX would be the right choice if the project were "clean compiled standard neural nets." It isn't — it's "explore weird async execution patterns on a single GPU."

## Why not IREE

Architecturally the cleanest compiler story. But immature for research, no ecosystem, requires Linux, no payoff inside the timebox.

## Rejected alternatives

| Backend | Why rejected |
|---|---|
| JAX | Requires Linux migration, XLA less flexible for custom kernels, no payoff for the core async goal |
| IREE | Immature, no ecosystem, requires Linux |
| Mojo | Too early, partial CUDA support |
| Rust (Candle/Burn) | Not viable for iterative research |
| tinygrad | Too risky, sparse ecosystem |

## Compilation policy for core/

Three tiers:

1. **torch.compile path:** Model forward functions in `core/` should work with `torch.compile(fullgraph=True)`. No in-graph side effects, no dynamic shapes where avoidable. This catches "accidentally eager" code.
2. **Eager-OK path:** Debug tracing, logging, dataset loading, evaluation harnesses. These stay eager deliberately.
3. **Custom kernel path (future):** For the real async work — persistent CUDA kernels, Triton custom ops. This is where the hard research lives. Neither `torch.compile` nor JAX's XLA solve this; it requires going below the framework.

## Revisit conditions

Reopen this decision if:

- Project continues past the current timebox AND Linux migration happens naturally
- A clean JAX + custom kernel story emerges (e.g., Pallas becomes as flexible as Triton)
- The project pivots away from the async CUDA kernel direction
