# Text baseline trust anchors

Run from repo root:

```powershell
& .\.venv\Scripts\python.exe -m base_experiments.transformer
& .\.venv\Scripts\python.exe -m base_experiments.rnn
```

Primary artifacts live in:

- [`../base-experiments/transformer/artifacts/`](../base-experiments/transformer/artifacts/)
- [`../base-experiments/rnn/artifacts/`](../base-experiments/rnn/artifacts/)

## Comparison frame (fixed for all future experiments)

All experimental results in this project compare against these anchors using the following fixed frame:

| Dimension | Value |
|-----------|-------|
| **Dataset** | TinyShakespeare (`experiments/corpora.ignore/tinyshakespeare_input.txt`) |
| **Train slice** | first 100,000 characters |
| **Val slice** | characters [100000, 120000) |
| **Context length** | 32 characters |
| **Parameter budget** | ~186K (tolerance: ±10%) |
| **Primary metric** | Best validation cross-entropy loss (nats/char) |
| **Secondary metrics** | Final val loss, val accuracy, training time |
| **Compute metric** | Wall-clock training time on RTX 3090 (indicative, not authoritative) |

**Anchor values:**

| Model | Params | Best Val Loss | Training Time |
|-------|--------|---------------|---------------|
| Transformer | 186,805 | **1.632** | 73s |
| Vanilla RNN | 186,125 | **1.706** | 78s |

Future experiments must report at minimum: parameter count, best val loss, and training time against this frame. Differences in dataset, split, or context length invalidate comparisons.

**External reference gap:** No verified published result exists for this exact setup (~186K params, 32-char context, this split). The nanoGPT README reports 1.4697 val loss for a much larger transformer — useful as a sanity ceiling but not a direct comparison target. This gap is acknowledged, not hidden.

## Transformer baseline

## Exact config

- corpus: `experiments/corpora.ignore/tinyshakespeare_input.txt`
- train characters: `100000`
- validation characters: `20000`
- validation slice: `[100000, 120000)`
- objective: character-level next-character prediction from a fixed `32`-character context
- transformer: `d_model=72`, `num_layers=3`, `num_heads=4`, `feedforward_dim=256`
- batch size: `256`
- epochs: `13`
- optimizer: `AdamW`
- learning rate: `0.003`
- gradient clip norm: `1.0`
- seed: `42`
- vocab size: `61`
- parameter count: `186805`

Inline evidence from [`model_summary.json`](../base-experiments/transformer/artifacts/model_summary.json), [`config.json`](../base-experiments/transformer/artifacts/config.json), and [`corpus_summary.json`](../base-experiments/transformer/artifacts/corpus_summary.json):

```json
{
  "parameter_count": 186805,
  "context_size": 32,
  "d_model": 72,
  "feedforward_dim": 256,
  "num_heads": 4,
  "num_layers": 3,
  "train_characters": 100000,
  "val_characters": 20000,
  "val_start": 100000,
  "val_stop": 120000,
  "vocab_size": 61
}
```

## Why trust this run at all?

The baseline does not start training until four cheap checks pass. Evidence from [`correctness_checks.json`](../base-experiments/transformer/artifacts/correctness_checks.json):

```json
[
  {
    "name": "dataset_round_trip",
    "sample": "First Citizen:\\nBefore we proceed"
  },
  {
    "name": "forward_shape",
    "expected_shape": [8, 61],
    "actual_shape": [8, 61]
  },
  {
    "name": "known_tiny_batch_loss",
    "loss": 4.498300552368164,
    "expected_range": [3.1108738641733114, 5.110873864173311]
  },
  {
    "name": "one_batch_memorization",
    "final_loss": 0.00011668281513266265,
    "final_accuracy": 1.0
  }
]
```

What they prove:

- round-trip: the vocabulary mapping is self-consistent
- forward shape: the model returns one distribution over the vocab per input window
- tiny-batch loss range: logits and targets line up well enough for cross-entropy to be sane
- one-batch memorization: gradients flow, optimization works, and the model can fit an easy case instead of silently failing

One real bug was caught during this work: the naive contiguous `100K/20K` split can put unseen characters into validation (`E`, `H`, `U`). The runner now refuses that bad setup and picks the first later validation slice whose vocabulary is covered by the training slice.

## Achieved loss

From [`final_metrics.json`](../base-experiments/transformer/artifacts/final_metrics.json) and [`training_history.json`](../base-experiments/transformer/artifacts/training_history.json):

```json
{
  "runtime_seconds": 103.54640960000688,
  "final_val_loss": 1.647757887840271,
  "final_val_accuracy": 0.512870592948718,
  "best_val_loss": 1.632211,
  "best_val_accuracy": 0.515825,
  "best_epoch": 12
}
```

Epoch trace:

```text
epoch 0  val_loss 4.264182
epoch 1  val_loss 1.942437
epoch 2  val_loss 1.809755
epoch 3  val_loss 1.736840
epoch 4  val_loss 1.700192
epoch 5  val_loss 1.674973
epoch 6  val_loss 1.675177
epoch 7  val_loss 1.643299
epoch 8  val_loss 1.644804
epoch 9  val_loss 1.633977
epoch 10 val_loss 1.651702
epoch 11 val_loss 1.647328
epoch 12 val_loss 1.632211
epoch 13 val_loss 1.647758
```

This is better than the repo's earlier internal transformer evidence of about `1.73` validation loss at about `187K` parameters on the same broad `100K/20K` setup, and it stays in the same rough band as the previously saved scale-up baselines in `research/questions/`.

## Generated sample

From [`sample.txt`](../base-experiments/transformer/artifacts/sample.txt):

```text
First Citizen:
Before we proceed the people the people,
And they say the people, the people,
And they say the people, the people,
And they say the people, the people,
And they say the people, the people,
And they say the people, the people,
And they say the people, the people,
And they say the people, the people,
And they say the people, the people,
```

This is not good writing. It is useful evidence that the sampling path runs end to end and produces Shakespeare-like local texture instead of crashing or emitting garbage tokens.

## Honest comparison to external references

What exists:

- a larger external TinyShakespeare anchor is already noted in `PROCESS-PLAN.ignore.md`: nanoGPT reports validation loss `1.4697` on TinyShakespeare for a much larger 6-layer, 384-dimension transformer
- internal repo evidence already existed around `1.73` validation loss for a `~187K` parameter transformer on the `100K/20K` setup

What does not yet exist here:

- no verified published external target for this exact `~187K` parameter, `context=32`, `100K/20K` fixed-window setup
- no multi-seed stability run for this exact baseline
- no saved best-checkpoint weights yet; the saved `model_state.pt` is the final epoch, while the best validation loss occurred at epoch `12`
- no claim that this split exactly matches anyone else's public benchmark protocol

So the honest claim is narrower: this repo now has a reproducible, internally checked transformer trust anchor at `186805` parameters with best validation loss `1.632211` and final validation loss `1.647758` on this saved setup. That is enough to anchor future experiments here, but not enough to pretend we have matched a canonical published benchmark.

## RNN baseline

The RNN uses the same corpus slice, fixed-window dataset, optimizer family, logging shape, correctness checks, and artifact layout as the transformer baseline. The only intended change is the model family: a plain `nn.RNN` with `tanh`, not a GRU or LSTM.

## Exact config

- corpus: `experiments/corpora.ignore/tinyshakespeare_input.txt`
- train characters: `100000`
- validation characters: `20000`
- validation slice: `[100000, 120000)`
- objective: character-level next-character prediction from a fixed `32`-character context
- RNN: `embedding_dim=64`, `hidden_dim=368`, `num_layers=1`, `nonlinearity=tanh`
- batch size: `256`
- epochs: `13`
- optimizer: `AdamW`
- learning rate: `0.003`
- gradient clip norm: `1.0`
- seed: `42`
- vocab size: `61`
- parameter count: `186125`

Inline evidence from [`../base-experiments/rnn/artifacts/model_summary.json`](../base-experiments/rnn/artifacts/model_summary.json), [`../base-experiments/rnn/artifacts/config.json`](../base-experiments/rnn/artifacts/config.json), and [`../base-experiments/rnn/artifacts/corpus_summary.json`](../base-experiments/rnn/artifacts/corpus_summary.json):

```json
{
  "parameter_count": 186125,
  "context_size": 32,
  "embedding_dim": 64,
  "hidden_dim": 368,
  "num_layers": 1,
  "nonlinearity": "tanh",
  "train_characters": 100000,
  "val_characters": 20000,
  "val_start": 100000,
  "val_stop": 120000,
  "vocab_size": 61
}
```

## Why trust this run at all?

The RNN baseline uses the same four cheap gates before training. Evidence from [`../base-experiments/rnn/artifacts/correctness_checks.json`](../base-experiments/rnn/artifacts/correctness_checks.json):

```json
[
  {
    "name": "dataset_round_trip",
    "sample": "First Citizen:\\nBefore we proceed"
  },
  {
    "name": "forward_shape",
    "expected_shape": [8, 61],
    "actual_shape": [8, 61]
  },
  {
    "name": "known_tiny_batch_loss",
    "loss": 4.19972038269043,
    "expected_range": [3.1108738641733114, 5.110873864173311]
  },
  {
    "name": "one_batch_memorization",
    "final_loss": 2.7499507268657908e-05,
    "final_accuracy": 1.0
  }
]
```

What they prove is the same as for the transformer: the dataset mapping is self-consistent, the model returns one distribution per input window, cross-entropy is wired sanely, and gradients are good enough to fit an easy batch instead of failing silently.

## Achieved loss

From [`../base-experiments/rnn/artifacts/final_metrics.json`](../base-experiments/rnn/artifacts/final_metrics.json) and [`../base-experiments/rnn/artifacts/training_history.json`](../base-experiments/rnn/artifacts/training_history.json):

```json
{
  "runtime_seconds": 77.96514240000397,
  "final_val_loss": 1.7204482463689952,
  "final_val_accuracy": 0.5055588942307693,
  "best_val_loss": 1.7056,
  "best_val_accuracy": 0.503155,
  "best_epoch": 8
}
```

Epoch trace:

```text
epoch 0  val_loss 4.144862
epoch 1  val_loss 1.896119
epoch 2  val_loss 1.798121
epoch 3  val_loss 1.742980
epoch 4  val_loss 1.737920
epoch 5  val_loss 1.732899
epoch 6  val_loss 1.709892
epoch 7  val_loss 1.728668
epoch 8  val_loss 1.705600
epoch 9  val_loss 1.726252
epoch 10 val_loss 1.717653
epoch 11 val_loss 1.719537
epoch 12 val_loss 1.721171
epoch 13 val_loss 1.720448
```

At roughly the same parameter budget, the plain RNN is worse than the transformer here. Best validation loss `1.7056` is `0.073389` above the transformer's best `1.632211`. Final validation loss `1.720448` is `0.072690` above the transformer's final `1.647758`. That is useful evidence in itself: on this exact `100K/20K`, `context=32`, `~186K` setup, the transformer buys a real quality gain over a vanilla Elman RNN.

## Generated sample

From [`../base-experiments/rnn/artifacts/sample.txt`](../base-experiments/rnn/artifacts/sample.txt):

```text
First Citizen:
Before we proceed the people, they are and the people, they are and the people, they are and the people, they are and the people, they are and the people, they are and the people, they are and the people, they are and the people, they are and the people, they are and the people, they are and the people, they are and the people, they ar
```

This is worse writing than the transformer sample, which matches the loss gap. It is still useful as an end-to-end proof that sampling runs and that the model has learned local character texture rather than emitting nonsense.

## Honest comparison to external references

What exists:

- this repo now has two internally checked baselines on the same saved setup: transformer `186805` params / best val `1.632211`, and RNN `186125` params / best val `1.7056`
- the RNN result is directionally consistent with the expectation that a plain recurrent baseline should lag a similarly sized transformer on TinyShakespeare

What does not yet exist here:

- no verified published external target for this exact `~186K` parameter vanilla RNN with `context=32` and this fixed `100K/20K` split
- no multi-seed stability run for this exact RNN baseline
- no saved best-checkpoint weights yet; the saved `model_state.pt` is the final epoch, while the best validation loss occurred at epoch `8`

So the honest claim here is also narrow: this repo now has a reproducible, internally checked vanilla RNN trust anchor at `186125` parameters with best validation loss `1.7056` and final validation loss `1.720448` on the saved setup. It is clearly worse than the transformer's `1.632211` best val at almost the same parameter budget, which makes it a useful second baseline rather than a competitor for the default reference point.
