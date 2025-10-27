# Character Language Model (RNN & Transformer)

Objective:
- Expand the shared framework to text data with both recurrent and transformer implementations at the character level.

Key Requirements:
- Provide dataset loaders for small corpora (e.g., Tiny Shakespeare) with UTF-8 normalization.
- Implement lightweight RNN and transformer modules that conform to the base model API.
- Add metrics such as bits-per-character and perplexity while retaining image metrics.

Integration Notes:
- Factor embeddings, positional encodings, and output heads so they can be shared across architecture types.
- Ensure config-driven selection between RNN and transformer paths without code duplication.
- Maintain compatibility with prior MNIST demos by reusing the same trainer entrypoint and logging system.
