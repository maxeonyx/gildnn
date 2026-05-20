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
| Transformer | 186,805 | **1.643** | 212s |
| Vanilla RNN | 186,125 | **1.711** | 161s |

Future experiments must report at minimum: parameter count, best val loss, and training time against this frame. Differences in dataset, split, or context length invalidate comparisons.

**External reference gap:** No verified published result exists for this exact setup (~186K params, 32-char context, this split). The nanoGPT README reports 1.4697 val loss for a much larger transformer — useful as a sanity ceiling but not a direct comparison target. This gap is acknowledged, not hidden.

## Training progression samples

All rows below use the same prompt and sample length: prompt `First Citizen:\\nBefore we proceed`, 200 generated characters. Evidence from [`../base-experiments/transformer/artifacts/progression_samples.json`](../base-experiments/transformer/artifacts/progression_samples.json) and [`../base-experiments/rnn/artifacts/progression_samples.json`](../base-experiments/rnn/artifacts/progression_samples.json).

| Epoch | Transformer | RNN |
|---|---|---|
| 0 | val 4.264182<br><br>First Citizen:<br>Before we proceedTBMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM | val 4.144862<br><br>First Citizen:<br>Before we proceedYbztQ-UbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUbMUb |
| 1 | val 1.944690<br><br>First Citizen:<br>Before we proceed to the see to the seeven to the good to the seeven the good to the seever the seeven the so mare to the seeven to the seeven to the seeven to the seeven to the seeven to the seeven to the seeven to t | val 1.896089<br><br>First Citizen:<br>Before we proceed them them them them them them them them them them them them them them them them them them them them them them them them them them them them them them them them them them them them them them them them |
| 3 | val 1.724200<br><br>First Citizen:<br>Before we proceeds the proud the proud<br>The proud the proud the proud.<br><br>CORIOLANUS:<br>So the proud the proud the proud the proud<br>The proud the proud the proud.<br><br>CORIOLANUS:<br>So the proud the proud the proud the proud<br>The | val 1.736296<br><br>First Citizen:<br>Before we proceed, and the people,<br>So more the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the th |
| 7 | val 1.652433<br><br>First Citizen:<br>Before we proceed the consul.<br><br>MENENIUS:<br>He have the consul.<br><br>MENENIUS:<br>He have the consul.<br><br>MENENIUS:<br>He have the consul.<br><br>MENENIUS:<br>He have the consul.<br><br>MENENIUS:<br>He have the consul.<br><br>MENENIUS:<br>He have the consul.<br><br> | val 1.717554<br><br>First Citizen:<br>Before we proceed to the gods good for the gods good for the gods good for the gods good for the gods good for the gods good for the gods good for the gods good for the gods good for the gods good for the gods good fo |
| 13 | val 1.643463<br><br>First Citizen:<br>Before we proceed, and they have they shall the people.<br><br>COMINIUS:<br>They are the people.<br><br>COMINIUS:<br>They are the people.<br><br>COMINIUS:<br>They are the people.<br><br>COMINIUS:<br>They are the people.<br><br>COMINIUS:<br>They are the people.<br><br> | val 1.733100<br><br>First Citizen:<br>Before we proceed and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the |

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
  "runtime_seconds": 211.93036529998062,
  "final_val_loss": 1.6434634251472278,
  "final_val_accuracy": 0.5111678685897436,
  "best_val_loss": 1.643463,
  "best_val_accuracy": 0.511168,
  "best_epoch": 13
}
```

Epoch trace:

```text
epoch 0  val_loss 4.264182
epoch 1  val_loss 1.944690
epoch 2  val_loss 1.817977
epoch 3  val_loss 1.724200
epoch 4  val_loss 1.699524
epoch 5  val_loss 1.675895
epoch 6  val_loss 1.669888
epoch 7  val_loss 1.652433
epoch 8  val_loss 1.644267
epoch 9  val_loss 1.644418
epoch 10 val_loss 1.655656
epoch 11 val_loss 1.653315
epoch 12 val_loss 1.665134
epoch 13 val_loss 1.643463
```

This is better than the repo's earlier internal transformer evidence of about `1.73` validation loss at about `187K` parameters on the same broad `100K/20K` setup, and it stays in the same rough band as the previously saved scale-up baselines in `research/questions/`.

## Generated sample

From [`sample.txt`](../base-experiments/transformer/artifacts/sample.txt):

```text
First Citizen:
Before we proceed, and they have they shall the people.

COMINIUS:
They are the people.

COMINIUS:
They are the people.

COMINIUS:
They are the people.

COMINIUS:
They are the people.

COMINIUS:
They are the people.

COMINIUS:
They are the people.

COMINIUS:
They are the people.

COMINIUS:
They are the p
```

This is not good writing. It is useful evidence that the sampling path runs end to end and produces Shakespeare-like local texture instead of crashing or emitting garbage tokens.

## Honest comparison to external references

What exists:

- a larger external TinyShakespeare anchor is already noted in `PROCESS-PLAN.ignore.md`: nanoGPT reports validation loss `1.4697` on TinyShakespeare for a much larger 6-layer, 384-dimension transformer
- internal repo evidence already existed around `1.73` validation loss for a `~187K` parameter transformer on the `100K/20K` setup

What does not yet exist here:

- no verified published external target for this exact `~187K` parameter, `context=32`, `100K/20K` fixed-window setup
- no multi-seed stability run for this exact baseline
- no saved best-checkpoint weights yet; the saved `model_state.pt` is the final epoch, which is also the best validation checkpoint in this rerun
- no claim that this split exactly matches anyone else's public benchmark protocol

So the honest claim is narrower: this repo now has a reproducible, internally checked transformer trust anchor at `186805` parameters with best validation loss `1.643463` and final validation loss `1.643463` on this saved setup. That is enough to anchor future experiments here, but not enough to pretend we have matched a canonical published benchmark.

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
  "runtime_seconds": 160.77953600001638,
  "final_val_loss": 1.733099754040058,
  "final_val_accuracy": 0.5099158653846154,
  "best_val_loss": 1.711203,
  "best_val_accuracy": 0.50651,
  "best_epoch": 8
}
```

Epoch trace:

```text
epoch 0  val_loss 4.144862
epoch 1  val_loss 1.896089
epoch 2  val_loss 1.793392
epoch 3  val_loss 1.736296
epoch 4  val_loss 1.726801
epoch 5  val_loss 1.737113
epoch 6  val_loss 1.714296
epoch 7  val_loss 1.717554
epoch 8  val_loss 1.711203
epoch 9  val_loss 1.742093
epoch 10 val_loss 1.728699
epoch 11 val_loss 1.728786
epoch 12 val_loss 1.713650
epoch 13 val_loss 1.733100
```

At roughly the same parameter budget, the plain RNN is worse than the transformer here. Best validation loss `1.711203` is `0.067740` above the transformer's best `1.643463`. Final validation loss `1.733100` is `0.089636` above the transformer's final `1.643463`. That is useful evidence in itself: on this exact `100K/20K`, `context=32`, `~186K` setup, the transformer buys a real quality gain over a vanilla Elman RNN.

## Generated sample

From [`../base-experiments/rnn/artifacts/sample.txt`](../base-experiments/rnn/artifacts/sample.txt):

```text
First Citizen:
Before we proceed and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people, and the people,
```

This is worse writing than the transformer sample, which matches the loss gap. It is still useful as an end-to-end proof that sampling runs and that the model has learned local character texture rather than emitting nonsense.

## Honest comparison to external references

What exists:

- this repo now has two internally checked baselines on the same saved setup: transformer `186805` params / best val `1.643463`, and RNN `186125` params / best val `1.711203`
- the RNN result is directionally consistent with the expectation that a plain recurrent baseline should lag a similarly sized transformer on TinyShakespeare

What does not yet exist here:

- no verified published external target for this exact `~186K` parameter vanilla RNN with `context=32` and this fixed `100K/20K` split
- no multi-seed stability run for this exact RNN baseline
- no saved best-checkpoint weights yet; the saved `model_state.pt` is the final epoch, while the best validation loss occurred at epoch `8`

So the honest claim here is also narrow: this repo now has a reproducible, internally checked vanilla RNN trust anchor at `186125` parameters with best validation loss `1.711203` and final validation loss `1.733100` on the saved setup. It is clearly worse than the transformer's `1.643463` best val at almost the same parameter budget, which makes it a useful second baseline rather than a competitor for the default reference point.
