# Question: PyTorch Character Reference Comparison

## What this bounded unit asks

What can this repo honestly say when comparing the feedforward sanity check, ordinary transformer baseline, and ordinary RNN baseline on the same tiny fixed-window character-level next-token task, without pretending this is a controlled or matched-capacity experiment?

## Artifacts

- Feedforward reference: [`../pytorch-char-sanity-check/README.md`](../pytorch-char-sanity-check/README.md)
- Transformer reference: [`../pytorch-char-transformer-baseline/README.md`](../pytorch-char-transformer-baseline/README.md)
- RNN reference: [`../pytorch-char-rnn-baseline/README.md`](../pytorch-char-rnn-baseline/README.md)
- Backend proof reused by all three: [`../backend-validation/README.md`](../backend-validation/README.md)

For post-integration reruns of these three entrypoints, run from repo root via `python -m experiments.<name>`.

## Key observed evidence

All three references cleared the one-batch overfit bar: [`feedforward`](../pytorch-char-sanity-check/artifacts/overfit_metrics.json), [`transformer`](../pytorch-char-transformer-baseline/artifacts/overfit_metrics.json), [`rnn`](../pytorch-char-rnn-baseline/artifacts/overfit_metrics.json).

```json
{
  "feedforward_final_accuracy": 1.0,
  "transformer_final_accuracy": 1.0,
  "rnn_final_accuracy": 1.0,
  "all_reached_memorization_bar": true
}
```

All three tiny runs ended with the same saved `final_accuracy` field: [`feedforward`](../pytorch-char-sanity-check/artifacts/tiny_run_metrics.json), [`transformer`](../pytorch-char-transformer-baseline/artifacts/tiny_run_metrics.json), [`rnn`](../pytorch-char-rnn-baseline/artifacts/tiny_run_metrics.json).

```json
{
  "feedforward_saved_final_accuracy": 0.9793282151222229,
  "transformer_saved_final_accuracy": 0.9793282151222229,
  "rnn_saved_final_accuracy": 0.9793282151222229
}
```

The three saved prompt continuations also match exactly: [`feedforward`](../pytorch-char-sanity-check/artifacts/tiny_samples.json), [`transformer`](../pytorch-char-transformer-baseline/artifacts/tiny_samples.json), [`rnn`](../pytorch-char-rnn-baseline/artifacts/tiny_samples.json).

```json
{
  "hello": "hello world.\nsmall text.\nhello world.\nsmall text.\nhello world.\nsmall text.\nhello worl"
}
```

The comparison stays narrow because changed variables remain visible rather than hidden: [`feedforward config`](../pytorch-char-sanity-check/artifacts/config.json), [`transformer config`](../pytorch-char-transformer-baseline/artifacts/config.json), [`rnn config`](../pytorch-char-rnn-baseline/artifacts/config.json), [`feedforward env`](../pytorch-char-sanity-check/artifacts/environment.json), [`transformer env`](../pytorch-char-transformer-baseline/artifacts/environment.json), [`rnn env`](../pytorch-char-rnn-baseline/artifacts/environment.json), [`transformer summary`](../pytorch-char-transformer-baseline/artifacts/model_summary.json), [`rnn summary`](../pytorch-char-rnn-baseline/artifacts/model_summary.json).

```json
{
  "feedforward_overfit_learning_rate": 0.05,
  "transformer_overfit_learning_rate": 0.02,
  "rnn_overfit_learning_rate": 0.05,
  "post_integration_git_sha": "9c40d58...",
  "feedforward_reference_parameter_count": 9435,
  "transformer_parameter_count": 9107,
  "rnn_parameter_count": 7451,
  "rnn_hidden_state_handling": "reset_to_zero_per_forward_call_no_carry_between_windows_or_batches"
}
```

## What this shows

This repo now has a narrow three-way reference comparison on one tiny fixed-window task. Across the saved artifacts, the three references share the same raw text, core task-surface fields, sample prompts, and recorded local PyTorch CUDA environment surface; all three clear the one-batch overfit bar, end with the same saved `final_accuracy` value, and save the same short prompt continuations.

The comparison was rerun after the minimal `core/` integration slice using the supported module-entrypoint contract from repo root (`python -m experiments.<name>`). Those reruns preserved the same bounded evidence surface, but they were captured from a dirty pre-commit working tree rather than an exact committed clean tree, so they are useful as behavior-preservation checks rather than final clean-state provenance.

## What this does not show

- It does not show a controlled single-variable comparison. Learning rates and git SHAs differ across the saved runs.
- It does not show a matched-capacity comparison. The RNN is materially smaller, and the feedforward parameter count is only carried indirectly here through later model summaries.
- It does not show anything useful yet about scaling, generalization, long-range memory, or cross-window state use. In this bounded RNN reference, hidden state is reset per forward call with no carry across windows or batches.
- It does not settle the backend/framework choice.
- It does not rank the three model families or show meaningful qualitative separation between them on this toy corpus.

## Remaining open edges

- The same saved `final_accuracy` value appears in all three tiny-run metric files, and current evidence supports a bounded task-surface explanation for that shared plateau: all three scripts compute that field on the same fixed-window next-token surface over the same saved `window_count = 387`, and the shared tiny corpus has one ambiguous 5-character context, `mall `, split evenly between next-token targets `n` and `t` (`8` each). On this surface, that makes an eight-error plateau a strong interpretation rather than an unexplained architecture result.
- The exact final mistake locations are still not directly archived in the repo. This interpretation is grounded in the shared metric definition, shared denominator, and shared corpus structure, but not in a saved final per-window prediction table.
- The local NumPy warning captured in the RNN question folder is real but still only captured, not investigated.

## Next discriminating step

No separate `final_accuracy` follow-up is needed on current evidence. Reopen that question only if a future rerun under the same setup materially breaks this interpretation, or if a later comparison genuinely needs the final per-window predictions saved. If stronger controlled-comparison claims are needed instead, tighten the harness first rather than over-reading this comparison.
