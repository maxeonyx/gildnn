# Hello World MNIST Pixel Generator

Objective:
- Adapt the base infrastructure to support autoregressive generation over image pixels while keeping the classifier operational.

Key Requirements:
- Reuse data loaders but flatten or sequence pixels for autoregressive ordering.
- Support both teacher-forced training and ancestral sampling routines.
- Introduce configuration switches for loss functions (cross-entropy vs. negative log-likelihood) and sampling temperature.

Integration Notes:
- Extend the model registry with a simple masked MLP or PixelRNN variant compatible with the classifier backbone interface.
- Ensure evaluation scripts can run classification and generation without manual code changes.
- Validate that checkpoints can be resumed regardless of whether they originated from classifier or generator runs.
