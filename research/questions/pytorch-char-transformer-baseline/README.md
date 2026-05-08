# Question: PyTorch Character Transformer Baseline

## What this bounded unit asks

Can an ordinary transformer clear the same tiny character-level next-token ladder as the feedforward sanity check in this repo: fixed 5-character context, one-batch overfit first, then a tiny-data run with inspectable outputs, while keeping the task framing fixed enough that the main changed variable is model family?

## Artifacts

- Dataset: [`raw_text.txt`](raw_text.txt)
- Run config: [`artifacts/config.json`](artifacts/config.json)
- Environment proof: [`artifacts/environment.json`](artifacts/environment.json)
- Model summary: [`artifacts/model_summary.json`](artifacts/model_summary.json)
- One-batch overfit metrics: [`artifacts/overfit_metrics.json`](artifacts/overfit_metrics.json)
- One-batch overfit predictions: [`artifacts/overfit_predictions.txt`](artifacts/overfit_predictions.txt)
- Tiny-data run metrics: [`artifacts/tiny_run_metrics.json`](artifacts/tiny_run_metrics.json)
- Tiny-data generated samples: [`artifacts/tiny_samples.json`](artifacts/tiny_samples.json)

## What stayed fixed vs the feedforward reference

- same raw text content
- same character-level next-token objective
- same context size: `5`
- same fixed first `16` windows for the one-batch overfit stage
- same tiny-data full-dataset run framing
- same seed: `7`
- same prompts for saved outputs: `"hello"`, `"small"`, `" text"`

## Key observed evidence

The ordinary transformer cleared the one-batch memorization bar on CUDA:

```json
{
  "final_loss": 0.0009553811978548765,
  "final_accuracy": 1.0,
  "reached_memorization_bar": true
}
```

The tiny-data run also completed end to end on the same local PyTorch CUDA path:

```json
{
  "device": "cuda",
  "torch_version": "2.11.0+cu128",
  "cuda_device_name": "NVIDIA GeForce RTX 3090",
  "final_loss": 0.03202458471059799,
  "final_accuracy": 0.9793282151222229
}
```

The changed variable was made explicit rather than hidden:

```json
{
  "model_family": "ordinary_transformer",
  "transformer_parameter_count": 9107,
  "feedforward_reference_parameter_count": 9435,
  "parameter_delta_vs_feedforward": -328,
  "parameter_ratio_vs_feedforward": 0.965236
}
```

Saved outputs are inspectable and on-pattern for the tiny corpus:

```json
{
  "hello": "hello world.\nsmall text.\nhello world.\nsmall text.\nhello world.\nsmall text.\nhello worl",
  "small": "small text.\nhello world.\nsmall text.\nhello world.\nsmall text.\nhello world.\nsmall text",
  " text": " text.\nhello world.\nsmall text.\nhello world.\nsmall text.\nhello world.\nsmall text.\nhel"
}
```

## What this shows

This repo now has an ordinary transformer baseline for the same tiny bounded task as the earlier feedforward sanity check: on the already-proven local PyTorch CUDA path, a small one-layer transformer can overfit one fixed batch, complete the tiny-data run, and produce saved inspectable outputs.

## What this does not show

- It does not settle the backend/framework choice.
- It does not say the transformer is better than the feedforward reference in any meaningful broad sense.
- It does not say anything useful yet about scaling, generalization, or the long-term architecture.

## Remaining open edge

The tiny-data run landed at the same 0.9793 full-dataset accuracy as the feedforward reference rather than exact memorization of the whole tiny dataset. That keeps the comparison narrow and tidy, but it also means this baseline should not be over-read as stronger than the bounded evidence actually supports.

## Next discriminating step

Build the ordinary RNN baseline on the same tiny character-level task, unless cleanup of the current comparison first reveals a more basic issue that should be fixed before adding the next model family.
