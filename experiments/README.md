# Research Experiments

This directory is for the actual experiments which are the eventual goal of this repository, and will be started after the base experiments are done, to ensure we have a working, flexible ML implementation before starting the more complicated stuff.

Like the base experiments, when writing new experiments, refactor out shared functionality from any other experiments (including base ones) to build the flexible `core` implementation, and run all experiments in `test` mode to test that they all still work.

The most important thing is that we take only one small step at a time, and make sure it's thoroughly working before moving on. We then want to reuse code while ensuring it's still testable on old experiments, to reduce the risk of bugs that leave our experiments with indeterminate results.

## Roadmap

Not all of these need to be done in order:

- Parallel depth: propagate out across training steps, processing different timesteps in parallel, for pipelined inference. Recurrent "cortical columns".
  - Have some depth process "aggregated" activations from previous depth, to time dilate and form a kind of hierarchical time representation.
  - Allow information to propagate in *both directions* for top-down + bottom-up (where top-down info comes from further back in time). I personally think this is extremely biologically plausible.
  - Have a single central attention that goes across depth rather than across time, and then is provided as input to all columns
- Loss prediction: Have the model predict the loss of various output heads, so that at inference time we can choose the best one.
- Dynamic depth: a model that transcends "RNN" and "Transformer Depth" - we can take out samples at any layer in the residual stream, so that at inference time we can scale down the computation. We can also have the model *not* produce a token, in order to scale *up* the computation.
  - "Eventual" loss prediction: Have the model somehow predict the loss "if we extended the computation time indefinitely". For some maths questions for example, we might need thousands of tokens to compute the right answer.
  - Self-prediction - for steps where the model predicts more computation would help, have the model attempt to predict it's own future activation state earlier.
  - Computation-time loss term: incentivise giving a good answer with a shorter residual stream depth.
- Live graphics visualization: Allow constructing a rendering pipeline which renders visualizations of the weights & activations of the model during training, sharing the exact GPU memory with the training, zero-copy where possible.
- And more, see the subdirectories.

## Note

The subdirectories here may need to be broken up further into smaller experiments.
