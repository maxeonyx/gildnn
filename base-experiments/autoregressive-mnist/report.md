# Autoregressive MNIST (Pixel Transformer)

We reset this experiment to focus on a single, tiny decoder-only transformer that models rasterised, downsampled MNIST pixels. The goal is to establish one clean baseline that we can trust before introducing patches or flexible sampling orders in follow-up experiments.

## Introduction

### 1. Peek at the Dataset

We work with MNIST digits downsampled to $14\\times14$ to keep sequences shorter. We also quantize the number of colors from 256 to 16 colors.

<!-- OUTPUTSLOT:dataset-examples start -->
Short explanation

Each panel:
10 examples of MNIST digits plus from test set
The same 10 examples, downsampled & quantized MNIST digits plus from test set
Labels

The same as above, but the training set.
<!-- OUTPUTSLOT:dataset-examples end -->

### 2. Data pipeline

With the image pixel color as one-hot vectors in raster order, prepend a single start token, and shift the sequence to produce `(tokens[:-1], tokens[1:])` pairs. The table below illustrates the first few positions.

<!-- OUTPUTSLOT:token-trace start -->
One train set dataset example image.
Two-color (more/less red x more/less green) represntation of 2D positional encodings as an image.

The same example, with pixels ordered in sequence.
The positional encoding flattened out like the image.
The same again but offset, showing the correspondence between the n / n+1 input/output pairs
<!-- OUTPUTSLOT:token-trace end -->

### 3. Transformer model

The simplified model uses:

- Embedding width 64, feed-forward width 128
- 3 decoder blocks, 4 attention heads
- Learned positional embeddings sized to the downsampled grid

<!-- OUTPUTSLOT:model-architecture start -->
Diagram of the model architecture, with labels showing embedding / residual dimensions, # tokens, size of hidden layers, one residual block template times how many of them there are, and output heads.
<!-- OUTPUTSLOT:model-architecture end -->

### 4. Evaluations

#### Training snapshots

Sampling snapshots help sanity-check progress long before final evaluation.

<!-- OUTPUTSLOT:train-snapshots start -->
Explanation of the snapshots that will be produced during the training.

The first of these snapshots is shown here prior to the first training step.
<!-- OUTPUTSLOT:train-snapshots end -->

#### Final Evaluation

Use a larger test batch (e.g., 256 digits) for the last panel so variance settles and we can compare runs.

<!-- OUTPUTSLOT:final-batch start -->
Explanation of the final evaluation This serves as a check that we have implemented the visualization correctly, before we train the actual model, preventing errors where we crash at the end of training and e.g. lose our results.
<!-- OUTPUTSLOT:final-batch end -->

### 5. Checkpoints

<!-- OUTPUTSLOT:checkpoints start -->
The model's name is <superb-random-name>. Checkpoints are saved at <clickable link to directory>
<!-- OUTPUTSLOT:checkpoints end -->

## Training

<!-- OUTPUTSLOT:training-configuration start -->
Number of steps, batch size, learning rates (schedules if any), snapshot schedule, checkpoint schedule (distinct from snapshot schedule).
<!-- OUTPUTSLOT:training-configuration end -->

### Training progress:


<!-- OUTPUTSLOT:training-progress start -->
Snapshots should be appended and written here as training progresses (incrementally).

Each snapshot should have input / output examples on both train and test sets.
<!-- OUTPUTSLOT:training-progress end -->

## Final evaluation

Explanation of how to interpret the final evaluation visualizations below.

<!-- OUTPUTSLOT:final-evaluation start -->
Visualizations from the final model, similar to the training snapshots but with higher batch sizes, and more ablations across things like amount of conditioning given, different inference scenarios, etc.

Also include a final stats section
<!-- OUTPUTSLOT:final-evaluation end -->

## Discussion

To be written by the user after they read and understand the final results.
