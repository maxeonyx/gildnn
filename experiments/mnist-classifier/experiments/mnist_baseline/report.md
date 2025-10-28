# MNIST Baseline Experiment

This notebook captures the canonical MNIST baseline that we use to grow the gildnn core. It exercises the end-to-end pipeline on a familiar task so we can harden our experiment ergonomics before exploring research questions.

## Experiment Context

We treat this run as the "hello world" checkpoint for our base implementation. The goal is not to push accuracy, but to validate that the shared experiment harness can train, evaluate, and report on a simple classifier with deterministic behaviour.

## MNIST Task Overview

The MNIST dataset pairs $28\times 28$ grayscale digit images with labels from 0-9. Below we surface compact glimpses of the training and test splits to ground the experiment.

### Training split glimpse

<!-- OUTPUTSLOT:dataset-train-row start -->
<!-- OUTPUTSLOT:dataset-train-row end -->

### Test split glimpse

<!-- OUTPUTSLOT:dataset-test-row start -->
<!-- OUTPUTSLOT:dataset-test-row end -->

## Hypothesis: Input to Output Mapping

Our baseline hypothesis is straightforward: given a handwritten digit image from the test distribution, the model should predict the correct class. We illustrate this with 50 random test examples (sampled using the committed RNG seed) where each panel pairs digits with their labels.

<!-- OUTPUTSLOT:hypothesis start -->
<!-- OUTPUTSLOT:hypothesis end -->

## Configuration

<!-- OUTPUTSLOT:configuration start -->
<!-- OUTPUTSLOT:configuration end -->

## Metrics

<!-- OUTPUTSLOT:metrics start -->
<!-- OUTPUTSLOT:metrics end -->

## Training Progress Snapshots

To understand how learning evolves, we capture predictions on a fixed set of 10 held-out digits throughout training. Each image stacks the input digit, the expected label, and the model's guess with colour-coded backgrounds indicating correctness.

<!-- OUTPUTSLOT:training-progress start -->
<!-- OUTPUTSLOT:training-progress end -->

## Final Evaluation on Held-Out Examples

The following panels revisit the 50-digit hypothesis set, now annotated with the model's final predictions. Green backgrounds indicate correct classifications while red highlights mistakes.

<!-- OUTPUTSLOT:final-evaluation start -->
<!-- OUTPUTSLOT:final-evaluation end -->
