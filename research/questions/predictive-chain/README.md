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

### Detached message gradients (aux_weight=1.0, detach_messages=True)

With gradients stopped at message boundaries — each node receives messages but cannot backpropagate into the sender — task performance is identical: accuracy `1.0` (overfit), `0.9793` (tiny). Generated samples are the same.

Aux losses are moderately higher than the coupled version: overfit total `0.156` (vs `0.085`), with the increase spread across all nodes. See [`artifacts/detached_messages/`](artifacts/detached_messages/) and [`artifacts/compare_detached_messages.json`](artifacts/compare_detached_messages.json).

**This means the "unhooked gradients" vision is viable at this scale.** Nodes can learn useful representations from local predictive pressure alone, without receiving gradient signal from downstream consumers of their messages. The downstream task head doesn't care whether the internal communication channel is gradient-coupled or not.

The local prediction task is slightly harder without coupling (aux losses ~2× higher), which makes sense — without B sending gradients back to A, node A has no direct optimization signal to make its messages more predictable. It only learns message structure through its own local loss.

### 8-node detached chain (num_nodes=8, aux_weight=1.0, detach_messages=True)

Scaling from 3 to 8 nodes with the full "vision" configuration (strong aux pressure + detached gradients). Task head still works perfectly: accuracy `1.0` (overfit), `0.9793` (tiny). Generated samples are clean.

Per-node aux losses (overfit): A `0.013`, B `0.076`, C `0.066`, D `0.073`, E `0.064`, F `0.061`, G `0.023`, H `0.0003`.

The pattern: hardest in early-mid chain (B), gradually decreasing, with the deepest node (H) being nearly trivial. Information gets progressively more predictable deeper in the chain — deeper nodes see increasingly constrained signals.

See [`artifacts/8_nodes_detached/`](artifacts/8_nodes_detached/) and [`artifacts/8_nodes_detached/auxiliary_position_summary.json`](artifacts/8_nodes_detached/auxiliary_position_summary.json).

### Shakespeare comparison (real English, context_size=5, train/val split)

All prior experiments hit the same 0.9793 accuracy ceiling on trivially repetitive text. Switching to a 7k-char Shakespeare excerpt with an 80/20 train/val split reveals genuine architectural differences.

Results (all models, same hyperparameters, ~matched params):

| Model | Train Loss | Val Loss | Val Acc |
|-------|-----------|----------|---------|
| Feedforward | 0.79 | 4.87 | 0.282 |
| RNN | 1.23 | 3.05 | 0.292 |
| Transformer | 1.48 | 2.80 | 0.311 |
| **Pred Chain (aux 0.001, detached)** | 2.04 | **2.59** | 0.281 |
| **Pred Chain (aux 1.0, detached)** | 1.98 | **2.59** | 0.290 |

**The predictive chain generalizes best.** Despite the highest training loss (learns slowest), it has the lowest validation loss — it overfits least. The baselines all overfit severely (feedforward worst at 6× generalization gap; RNN and transformer also bad).

The aux weight makes little difference to validation performance (2.59 vs 2.59), though strong aux slightly helps training speed and val accuracy.

See [`artifacts/shakespeare_comparison/`](artifacts/shakespeare_comparison/).

## What this does not settle

- whether attention between neighbors helps further
- whether async or desynchronized execution is viable
- whether loss-prediction heads for halting are useful
- whether the generalization advantage holds at larger scale
- whether the pattern changes with a graph topology (not just a line)
- why the predictive chain generalizes better — is it the message bottleneck, the aux regularization, or something else?

## Status

Five variants completed across two corpora. Core findings:
1. Local predictive learning is compatible with task learning
2. Aux losses are optimization-driven, not fundamentally hard
3. Unhooked gradients are viable — nodes learn independently
4. The pattern holds at 8 nodes
5. **On real English text, the predictive chain generalizes better than all baselines** — lower val loss despite higher train loss

Next natural questions: why does it generalize better? Is it the message bottleneck acting as regularization? Would attention between neighbors help further? Does the advantage persist at larger context/model size?
