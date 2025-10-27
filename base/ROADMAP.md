# Base System Roadmap

The base implementation will evolve through a numbered sequence of working demos. Each stage keeps all prior demonstrations
functional while refactoring the shared code to increase flexibility. The goal is one adaptable training stack that can swap
modalities (text or image), input encodings (tokens, bytes, or patches), and core architectures (recurrent or transformer) with
minimal duplication.

1. **Hello World MNIST Classifier**  
   Establish the project skeleton with a straightforward supervised training loop on MNIST image classification. Focus on
   dataset loading, batching, optimizer setup, checkpointing, and evaluation utilities that will remain in place for later
   stages.

2. **Hello World MNIST Pixel Generator**  
   Reuse the same training loop to model MNIST pixels autoregressively. This forces the base code to support generative loss
   functions, sequential sampling hooks, and configuration toggles without breaking the classifier path.

3. **Character Language Model (RNN & Transformer)**  
    Extend the shared model abstraction so both a minimal recurrent baseline and a transformer variant can be instantiated for
    character-level corpora. Ensure data pipelines can stream UTF-8 bytes while still supporting simple character vocabularies.

4. **Tokenizer-Based Language Model**  
   Integrate a pluggable tokenization module (e.g., SentencePiece or Byte-Pair Encoding) that feeds into the same training
   harness. Exercise weight sharing, embedding management, and evaluation metrics (perplexity) alongside the character pipeline.

5. **Byte Language Model**  
   Switch the language modeling path to consume pure byte sequences. Solidify the interface for specifying vocabularies and
   positional encodings so tokenized and byte-based flows coexist cleanly.

6. **Patch-Based Image Model**  
   Add a ViT-style patch embedding option while retaining the earlier convolutional/classifier utilities. Confirm that the
   architecture registry can instantiate both spatial (patch) and sequential (token/byte) backbones through a common factory.

7. **Byte Image Model**  
   Build a byte-oriented image autoencoder or autoregressive model using the same data handling stack as the byte language
   model. Harmonize normalization, sequence length management, and decoding utilities across modalities.

8. **Unified Byte Multi-Modal Model**  
   Demonstrate a single architecture that can ingest byte sequences representing either text or images. This stage requires
   configuration-driven dataset selection, modality tokens, and evaluation routines that can branch without diverging the
   implementation.
