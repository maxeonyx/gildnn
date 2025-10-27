# Patch-Based Image Model

Objective:
- Equip the base implementation with patch embedding pipelines suitable for transformer-style image modeling.

Key Requirements:
- Implement image preprocessing that slices inputs into fixed-size patches with position embeddings.
- Provide both supervised (classification) and generative (reconstruction) objectives for patch sequences.
- Support hybrid models that combine convolutional stems with patch transformers under the same interface.

Integration Notes:
- Reuse training/evaluation scripts by adding modality-specific configuration blocks (e.g., `modality: image_patch`).
- Align the logging system to compare patch-level metrics with earlier pixel-based baselines.
- Maintain compatibility with byte-level models by isolating modality-specific preprocessing in dedicated modules.
