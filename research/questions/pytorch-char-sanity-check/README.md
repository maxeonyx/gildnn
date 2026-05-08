# Question: PyTorch Character Sanity Check

## What this bounded unit asks

Can the already-proven local PyTorch path run the smallest useful character-level next-token sanity check in this repo: fixed 5-character context, tiny feedforward model, one-batch overfit first, then a tiny-data run with inspectable outputs?

## Artifacts

- Dataset: [`raw_text.txt`](raw_text.txt)
- Run config: [`artifacts/config.json`](artifacts/config.json)
- Environment proof: [`artifacts/environment.json`](artifacts/environment.json)
- One-batch overfit metrics: [`artifacts/overfit_metrics.json`](artifacts/overfit_metrics.json)
- One-batch overfit predictions: [`artifacts/overfit_predictions.txt`](artifacts/overfit_predictions.txt)
- Tiny-data run metrics: [`artifacts/tiny_run_metrics.json`](artifacts/tiny_run_metrics.json)
- Tiny-data generated samples: [`artifacts/tiny_samples.json`](artifacts/tiny_samples.json)

## Key observed evidence

One-batch overfit reached the stated bar on CUDA:

```json
{
  "final_loss": 0.00018824258586391807,
  "final_accuracy": 1.0,
  "reached_memorization_bar": true
}
```

Tiny-data run completed end to end on the same path:

```json
{
  "device": "cuda",
  "torch_version": "2.11.0+cu128",
  "cuda_device_name": "NVIDIA GeForce RTX 3090",
  "final_loss": 0.03174847364425659,
  "final_accuracy": 0.9793282151222229
}
```

Saved generated outputs are inspectable and visibly match the tiny training pattern:

```json
{
  "hello": "hello world.\nsmall text.\nhello world.\nsmall text.\nhello world.\nsmall text.\nhello worl",
  "small": "small text.\nhello world.\nsmall text.\nhello world.\nsmall text.\nhello world.\nsmall text",
  " text": " text.\nhello world.\nsmall text.\nhello world.\nsmall text.\nhello world.\nsmall text.\nhel"
}
```

## What this shows

This repo now has a minimal end-to-end PyTorch character sanity check for this bounded unit only: a tiny feedforward next-token model with 5-character context can overfit one batch, train on a tiny text sample, and produce saved outputs on the already-proven local CUDA path.

## What this does not show

- It does not settle the backend/framework choice.
- It does not compare PyTorch against any alternative.
- It does not say anything useful yet about scaling, generalization, or the long-term architecture.

## Remaining open edge

The tiny-data run plateaued at 0.9793 full-dataset accuracy rather than exact memorization. That is enough for this bounded unit because the contract only required end-to-end completion plus inspectable outputs after the separate one-batch overfit proof, but it is still a real loose end if this script is kept as a long-lived reference.

## Next discriminating step

Use the same tiny character-level task as the reference point for the first ordinary transformer baseline, unless cleanup of this script reveals a more basic blocker that should be fixed first.
