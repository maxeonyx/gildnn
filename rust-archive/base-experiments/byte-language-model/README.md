# Byte Language Model

Objective:
- Transition the language modeling stack to operate directly on byte sequences without losing compatibility with token-based workflows.

Key Requirements:
- Implement UTF-8 byte streaming datasets with deterministic train/validation/test splits.
- Handle sequence packing, attention masking, and positional encoding for long byte sequences.
- Provide evaluation scripts that report both perplexity (bits-per-byte) and human-readable decodings when possible.

Integration Notes:
- Share embedding layers and output projections across tokenized and byte pipelines via configurable vocabulary objects.
- Allow mixed-batch training experiments where subsets of data use token vocabularies and others use byte vocabularies.
- Keep the configuration schema stable so prior experiments can still be reproduced.
