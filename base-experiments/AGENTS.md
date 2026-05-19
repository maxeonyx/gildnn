# base-experiments/

Standard model training that reproduces expected published results on the project's primary datasets.

## Purpose

This directory proves our implementation is correct. Each subdirectory trains a standard architecture (feedforward, RNN, transformer) and achieves validation loss matching what we expect from published results on the same dataset.

These are not custom or experimental architectures. They are the known-good reference points that all experimental results are compared against.

## Structure

Each model type gets its own subdirectory: `base-experiments/<model>/`

Each directory contains:
- Training script (uses components from `core/`)
- A README documenting: what published result we're targeting, what we achieved, the exact config
- Minimal saved artifacts proving the result (loss curves, final metrics)

## Relationship to core/

Base experiments use `core/` components (datasets, training loops, model building blocks). If a base experiment needs new shared infrastructure, add it to `core/` first, then use it here. The flow is:

1. Build reusable component in `core/`
2. Use it in `base-experiments/` to reproduce a known result
3. Use it in `experiments/` to try something new, comparing against the baseline

## What "expected results" means

Before training a baseline, research what validation loss a model of this size should achieve on this dataset. Document the source (paper, blog post, reference implementation). Then verify our implementation matches within a reasonable margin.

A baseline is not done until it achieves the expected quality. If it doesn't, the implementation has a bug — fix it before moving on.
