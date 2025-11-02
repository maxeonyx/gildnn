# Base System Roadmap

The base implementation grows through focused experiment crates rather than numbered stages. Each crate locks down one new capability, keeps earlier work runnable, and feeds improvements back into the shared core.

- **hello-world-mnist-classifier** — Project skeleton, supervised loop, deterministic reporting.
- **autoregressive-mnist** — Pixel-level autoregressive baseline with a tiny transformer on downsampled digits.
- **autoregressive-patch-mnist** — Swap pixel tokens for discrete-VAE patch codes while keeping the transformer the same.
- **autoregressive-mnist-sampling-orders** — Train and sample with shuffled pixel orderings using 2D positional embeddings.
- **character-language-model** — Parallel text baseline covering recurrent and transformer variants on character data.
- **tokenizer-language-model** — Introduce subword tokenization while reusing the shared language-model harness.
- **byte-language-model** — Extend the language pipeline to raw byte sequences with flexible vocabulary handling.
- **byte-image-model** — Explore byte-level image generative models sharing infrastructure with the byte language path.
- **unified-byte-multimodal** — Aim for a single byte-driven model that can ingest text or images through configuration alone.

Whenever a new idea combines multiple bullets above, split it into separate experiment crates so we can validate one change at a time.
