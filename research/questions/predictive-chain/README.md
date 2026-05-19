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

## Current result

The task head overfits the one-batch memorization check cleanly. In the saved overfit run, accuracy reaches `1.0`, task loss falls to `0.000254`, and the weighted total loss falls to `0.000998` ([`artifacts/overfit_metrics.json`](artifacts/overfit_metrics.json)).

The auxiliary predictive losses do not collapse in the same run. They finish at A `0.510554`, B `0.198283`, and C `0.035449` even after the task is fully memorized ([`artifacts/overfit_metrics.json`](artifacts/overfit_metrics.json)). This is interesting evidence rather than a failure: at this scale, predicting a neighbor's next message appears harder than solving the downstream next-character task itself.

C's auxiliary loss being the lowest is directionally sensible. C receives B's message, which is already somewhat processed, so its local prediction target may be easier than the earlier-node targets.

The tiny full-dataset run still learns the corpus structure well enough to generate clean repeating samples, for example from the `"hello"` prompt: `hello world.\nsmall text.\nhello world...` ([`artifacts/tiny_samples.json`](artifacts/tiny_samples.json)). Final tiny-run metrics are total loss `0.034652`, task loss `0.033067`, and accuracy `0.979328` ([`artifacts/tiny_run_metrics.json`](artifacts/tiny_run_metrics.json)).

## What this bounded unit does not settle

- whether attention between neighbors helps
- whether async or desynchronized execution is viable
- whether loss-prediction heads for halting are useful
- whether this scales or generalizes beyond the tiny task
- whether gradient coupling should be reduced, increased, or handled differently

## Status

This bounded unit succeeded at the narrow implementation goal and produced a useful asymmetry: the task head is easy to overfit here, while the local predictive targets remain materially harder. That makes the auxiliary losses part of the result, not just an optimization nuisance. This folder is still only the first bounded unit for Thread 1.
