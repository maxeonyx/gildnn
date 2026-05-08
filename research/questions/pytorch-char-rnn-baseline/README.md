# Question: PyTorch Character RNN Baseline

## What this bounded unit asks

Can an ordinary RNN clear the same tiny character-level next-token ladder as the existing feedforward and transformer references in this repo: fixed 5-character context, one-batch overfit first, then a tiny-data run with inspectable outputs, while keeping enough of the tiny task surface fixed to make a narrow bounded comparison without treating it as a controlled single-variable experiment?

## Artifacts

- Dataset: [`raw_text.txt`](raw_text.txt)
- Run config: [`artifacts/config.json`](artifacts/config.json)
- Environment proof: [`artifacts/environment.json`](artifacts/environment.json)
- Model summary: [`artifacts/model_summary.json`](artifacts/model_summary.json)
- One-batch overfit metrics: [`artifacts/overfit_metrics.json`](artifacts/overfit_metrics.json)
- One-batch overfit predictions: [`artifacts/overfit_predictions.txt`](artifacts/overfit_predictions.txt)
- Tiny-data run metrics: [`artifacts/tiny_run_metrics.json`](artifacts/tiny_run_metrics.json)
- Tiny-data generated samples: [`artifacts/tiny_samples.json`](artifacts/tiny_samples.json)
- Verification stderr capture: [`artifacts/verification_stderr.txt`](artifacts/verification_stderr.txt)

## What stayed fixed vs the feedforward and transformer references

- same raw text content
- same character-level next-token objective
- same context size: `5`
- same fixed first `16` windows for the one-batch stage
- same tiny-data run framing
- same seed: `7`
- same prompts for saved outputs: `"hello"`, `"small"`, `" text"`
- same CUDA-backed local PyTorch path

## Explicit RNN-specific handling

The hidden state was reset on every forward call, with no carry across unrelated windows or batches:

```json
{
  "hidden_state_handling": "reset_to_zero_per_forward_call_no_carry_between_windows_or_batches"
}
```

That keeps this bounded unit on the same fixed-window comparison surface as the existing feedforward and transformer references.

## Key observed evidence

The ordinary RNN cleared the one-batch memorization bar on CUDA:

```json
{
  "final_loss": 0.0008715996518731117,
  "final_accuracy": 1.0,
  "reached_memorization_bar": true
}
```

Saved predictions recovered the fixed batch exactly:

```text
context | target | prediction
hello |   |  
ello  | w | w
llo w | o | o
lo wo | r | r
o wor | l | l
 worl | d | d
world | . | .
orld. | \n | \n
rld.\n | s | s
ld.\ns | m | m
d.\nsm | a | a
.\nsma | l | l
\nsmal | l | l
small |   |  
mall  | n | n
all n | e | e
```

The tiny-data run also completed end to end on the same local PyTorch CUDA path:

```json
{
  "final_loss": 0.03424537926912308,
  "final_accuracy": 0.9793282151222229,
  "hidden_state_handling": "reset_to_zero_per_forward_call_no_carry_between_windows_or_batches"
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

The changed variable is still only loosely controlled rather than perfectly matched-capacity:

```json
{
  "model_family": "ordinary_rnn",
  "rnn_parameter_count": 7451,
  "feedforward_reference_parameter_count": 9435,
  "transformer_reference_parameter_count": 9107,
  "parameter_ratio_vs_feedforward": 0.789719,
  "parameter_ratio_vs_transformer": 0.818162
}
```

The NumPy warning seen during verification is now captured rather than asserted from memory:

```text
C:\Users\maxeo\gildnn\.venv\Lib\site-packages\torch\_subclasses\functional_tensor.py:307: UserWarning: Failed to initialize NumPy: No module named 'numpy' [...]
```

## What this shows

This repo now has an ordinary RNN baseline for the same tiny bounded task as the earlier feedforward and transformer references: on the already-proven local PyTorch CUDA path, a simple `nn.RNN` can overfit one fixed batch, complete the tiny-data run, and produce saved inspectable outputs without changing the task into a cross-window state-carry setup.

## What this does not show

- It does not settle the backend/framework choice.
- It does not say the RNN is better than the feedforward or transformer references in any broad sense.
- It does not say anything useful yet about scaling, generalization, or long-range state use.

## Remaining open edges

- This is not a matched-capacity bake-off: the RNN has materially fewer parameters than the feedforward and transformer references.
- The tiny-data run landed at the same saved `final_accuracy` value of `0.9793282151222229` as the other two references rather than `1.0`. See `../pytorch-char-reference-comparison/README.md` when asking what that shared value means across the three tiny references: on current evidence it is treated as a bounded fixed-window task-surface plateau, not as an unexplained RNN-specific issue. This folder still does not archive the final per-window prediction table.
- The verification capture shows a real local NumPy warning on this path, but this bounded unit did not investigate whether that matters beyond these successful runs.

## Next discriminating step

See `../pytorch-char-reference-comparison/README.md` for what the three tiny references now jointly show. No separate `final_accuracy` follow-up starts from this folder on current evidence; tighten the harness only if a later bounded question genuinely needs stronger controlled-comparison claims.
