# Backend Choice: Compiled Execution

## Motivation

[Dictation 2026-05-22-7](../../../dictations/2026-05-22-7.md): Max wants compiled execution rather than defaulting to eager PyTorch. "I want the decision to be made deliberately."

## Options (assessed May 2026)

| Backend | CUDA/3090 | Custom arch | Research iteration | Maturity | Verdict |
|---|---|---|---|---|---|
| **JAX** | Yes | Excellent (functional) | Good (cached JIT) | Mature | **Principled choice** |
| **torch.compile** | Yes | Good (Inductor/Triton) | Best (eager fallback) | Massive ecosystem | Pragmatic fallback |
| Mojo | Partial | Immature for research | Unknown | Early | Not ready |
| tinygrad | Yes | Hackable | DIY | Sparse | Too risky |
| Rust (Candle/Burn) | Yes | Limited | Slow cycle | Thin | Not viable for research |
| MLX | No CUDA | — | — | — | **Eliminated** |

## Recommendation

**JAX** — compiled by default (not bolted on), functional paradigm enforces clean reusable code, custom architectures are first-class, single-GPU CUDA well-supported. Smaller community but higher quality. Equinox library gives PyTorch-like ergonomics while staying functional.

**torch.compile** is acceptable if JAX's functional constraint feels too limiting.

**Rust:** Revisit in 2-3 years. Not yet viable for iterative research.

## Key tradeoff

JAX requires rewriting core/ from scratch. The functional paradigm (no mutation, pure functions) is a significant style change. But it aligns with Max's values: principled, clean, compiled, not "whatever everyone uses."

The migration path: continue PyTorch for experimental probes, write anything promoted to core/ in JAX. Don't rewrite existing experiments.

## Decision status

**Open.** Awaiting Max's confirmation before committing to a rewrite.
