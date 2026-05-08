# Tokenizer-Based Language Model

Objective:
- Introduce modular tokenization while keeping the shared training and evaluation loop intact.

Key Requirements:
- Integrate third-party or custom tokenizers (SentencePiece/BPE) with reproducible vocabulary training scripts.
- Support reversible text encoding/decoding pipelines for evaluation and sampling.
- Manage embeddings, tied weights, and padding masks through configuration rather than ad-hoc code.

Integration Notes:
- Reuse character-level datasets by adding a preprocessing step that generates tokenized corpora.
- Ensure the same model definitions can swap between character and tokenized inputs via dependency injection.
- Preserve logging, checkpointing, and CLI ergonomics established in earlier stages.
