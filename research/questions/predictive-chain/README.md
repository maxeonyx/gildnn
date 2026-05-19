# Question: Predictive Chain

## What this bounded unit asks

Can a tiny 3-node recurrent line on the existing fixed-window character next-token task learn both the main task and local next-input prediction targets, with saved message rollouts that make the internal chain inspectable?

## Artifacts

- Dataset: [`raw_text.txt`](raw_text.txt)
- Run config: [`artifacts/config.json`](artifacts/config.json)
- Environment proof: [`artifacts/environment.json`](artifacts/environment.json)
- Model summary: [`artifacts/model_summary.json`](artifacts/model_summary.json)
- One-batch overfit metrics: [`artifacts/overfit_metrics.json`](artifacts/overfit_metrics.json)
- One-batch overfit predictions: [`artifacts/overfit_predictions.txt`](artifacts/overfit_predictions.txt)
- One-batch auxiliary traces: [`artifacts/overfit_auxiliary_traces.json`](artifacts/overfit_auxiliary_traces.json)
- One-batch rollout examples: [`artifacts/overfit_rollout_examples.json`](artifacts/overfit_rollout_examples.json)
- Tiny-data run metrics: [`artifacts/tiny_run_metrics.json`](artifacts/tiny_run_metrics.json)
- Tiny-data auxiliary traces: [`artifacts/tiny_auxiliary_traces.json`](artifacts/tiny_auxiliary_traces.json)
- Tiny-data generated samples: [`artifacts/tiny_samples.json`](artifacts/tiny_samples.json)
- Tiny-data rollout examples: [`artifacts/tiny_rollout_examples.json`](artifacts/tiny_rollout_examples.json)

## Scope

- 3 nodes in a line: `A -> B -> C`
- recurrent cell: `GRUCell`
- task head: next-character classification from the final hidden states of all three nodes
- auxiliary heads:
  - A predicts the next token embedding
  - B predicts A's next message
  - C predicts B's next message
- message-target gradients are detached in this bounded unit

## Results

### Baseline (aux_weight=0.001)

The task head overfits cleanly: accuracy `1.0`, task loss `0.000254`, total loss `0.000998`. But the auxiliary predictive losses remain non-trivial: A `0.5106`, B `0.1983`, C `0.0354`. See [`artifacts/aux_weight_0.001/`](artifacts/aux_weight_0.001/).

Tiny full-dataset run generates clean text (e.g. `hello world.\nsmall text.\nhello world...`). Final accuracy `0.9793`.

### Strong aux pressure (aux_weight=1.0)

With equal weighting, aux losses drop dramatically: A `0.0086` (was 0.51), B `0.0392` (was 0.20), C `0.0371` (roughly unchanged). The task head is completely unharmed: accuracy `1.0`, task loss `1.27e-07`. See [`artifacts/aux_weight_1.0/`](artifacts/aux_weight_1.0/).

Tiny run also unharmed: accuracy `0.9793`, task loss slightly better at `0.0322`. Generated samples identical quality.

### Key finding

**Local predictive pressure does not conflict with task performance** at this scale. The aux losses at low weight were high because the optimizer wasn't trying, not because the targets are impossible. When pressured, nodes A and B become highly predictable to their neighbors while still serving the downstream task equally well.

Node C is the exception — its aux loss was already low and doesn't improve much with stronger weighting. This may be because C's prediction target (B's next message) is inherently more variable, or because C's own downstream contribution is less constrained.

Full comparison: [`artifacts/compare_auxiliary_weights.json`](artifacts/compare_auxiliary_weights.json).

## What this bounded unit does not settle

- whether attention between neighbors helps
- whether async or desynchronized execution is viable
- whether loss-prediction heads for halting are useful
- whether this scales or generalizes beyond the tiny task
- whether gradient coupling should be reduced, increased, or handled differently

## Status

Two runs completed. The core finding — local predictive learning is compatible with task learning — opens the door to the next question: does forcing predictability change the *representations* in a way that matters when we scale up or add more nodes? This folder is the first experimental unit for Thread 1.
