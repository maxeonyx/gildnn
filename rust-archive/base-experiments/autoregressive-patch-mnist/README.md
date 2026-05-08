# Autoregressive Patch MNIST

This follow-up experiment keeps the autoregressive transformer intact but replaces raw pixels with discrete VAE patch tokens. The goal is to understand whether compressing $28\\times28$ digits into a small vocabulary of latent codes improves sample quality and convergence speed.

## Planned Scope
- Train (or import) a tiny discrete VAE that maps MNIST patches to a codebook of ≤64 entries.
- Replace the pixel tokenizer with VAE encoder outputs plus learned positional embeddings for the patch grid.
- Start with the same tiny transformer from `base-experiments/autoregressive-mnist`; only the token pipeline changes.
- Compare likelihood metrics and qualitative samples against the pixel baseline once both are stable.

## Guardrails
- No alternate architectures in this crate—focus on the VAE + transformer combo only.
- Reuse the deterministic reporting flow from the pixel baseline so artefacts stay comparable.
- Document patch reconstructions alongside samples to make VAE errors visible.

See `report.md` for the walkthrough template that the harness should fill once the implementation lands.
