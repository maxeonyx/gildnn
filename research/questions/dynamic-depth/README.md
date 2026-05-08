# Question: Dynamic Depth

## What we're asking

Does adaptive computation depth work on a standard recurrent model or transformer — using the same weights multiple times per output token, with a learned halting criterion based on loss prediction?

This is a completely separate research direction from the cortical column work. It starts on a standard model and only potentially connects to Thread 1 much later, if both make sense independently.

## Current state of knowledge

- The mechanism: run multiple forward passes per token (1, 2, 4, ... iterations — exponential schedule). Record loss at each depth. Train a loss-prediction head to predict those losses. At inference, use the head to decide when to stop.
- Dynamic hierarchical encodings might be relevant — the representation at each depth could encode different levels of abstraction, and the halting criterion might interact with that.
- This is well-studied territory. The goal here is not novelty but understanding whether it works on our tasks and hardware.
- Must be validated on a standard model before being combined with cortical columns.

## Open sub-questions

- What training schedule for depth rollouts works best? (exponential? uniform? curriculum?)
- How well does the loss-prediction head calibrate?
- Is there a meaningful quality/compute tradeoff, or does the model just learn to always use max depth?
- How does this interact with the RNN statefulness in the arbitrary-order patch task?
- If it works independently, does it compose with the cortical column architecture?

## Relevant prior work

There is prior work in this space (Adaptive Computation Time, PonderNet, Universal Transformers). Read only if a paper has results directly relevant to our setup — not for coverage.

## First bounded unit

The broad question above remains the research direction. The first bounded unit is deliberately narrower.

- **Host surface:** one standard model only, the existing tiny transformer.
- **Task surface:** the existing tiny fixed-window character next-token task.
- **Comparator:** the ordinary one-pass version on the same host and task surface.
- **Bounded question:** does repeated within-token reuse of the same model weights produce a readable depth-related outcome surface on this bounded task, relative to the one-pass comparator?
- **What this unit is not:** it is not a claim that dynamic depth works in general, not a full learned-halting or loss-prediction study, not a compute/quality benchmark, not a backend decision, and not a step toward cortical-column composition.
- **Loss-prediction / learned halting:** not assumed to be in scope for this first unit. That must be explicitly decided before coding rather than absorbed from the broader question framing.

The broader sub-questions above remain open even if this first bounded unit succeeds, fails, or closes ambiguously.

### Unit 01 artifacts

- Scope and summary: [`comparison_scope.json`](artifacts/comparison_scope.json), [`comparison_summary.json`](artifacts/comparison_summary.json)
- Depth-1 comparator: [`config`](artifacts/depth_1/config.json), [`environment`](artifacts/depth_1/environment.json), [`overfit_metrics`](artifacts/depth_1/overfit_metrics.json), [`tiny_run_metrics`](artifacts/depth_1/tiny_run_metrics.json), [`tiny_samples`](artifacts/depth_1/tiny_samples.json), [`depth_surface`](artifacts/depth_1/depth_surface.json)
- Depth-2 run: [`config`](artifacts/depth_2/config.json), [`environment`](artifacts/depth_2/environment.json), [`overfit_metrics`](artifacts/depth_2/overfit_metrics.json), [`tiny_run_metrics`](artifacts/depth_2/tiny_run_metrics.json), [`tiny_samples`](artifacts/depth_2/tiny_samples.json), [`depth_surface`](artifacts/depth_2/depth_surface.json)
- Depth-4 run: [`config`](artifacts/depth_4/config.json), [`environment`](artifacts/depth_4/environment.json), [`overfit_metrics`](artifacts/depth_4/overfit_metrics.json), [`tiny_run_metrics`](artifacts/depth_4/tiny_run_metrics.json), [`tiny_samples`](artifacts/depth_4/tiny_samples.json), [`depth_surface`](artifacts/depth_4/depth_surface.json)

The clean rerun provenance for all three saved depths points at commit `04f25e3...` with `git_working_tree_clean: true` and `git_status_short: []` in the saved environment artifacts.

### What Unit 01 shows

Unit 01 produced a readable depth-related outcome surface, but it came back near-null on this toy task. The saved summary shows the same final accuracy, mismatch count, and prompt continuation across depths 1, 2, and 4: [`comparison_summary.json`](artifacts/comparison_summary.json).

```json
{
  "depth_1": {
    "final_loss": 0.03202458471059799,
    "final_accuracy": 0.9793282151222229,
    "mismatch_count": 8
  },
  "depth_2": {
    "final_loss": 0.03172789886593819,
    "final_accuracy": 0.9793282151222229,
    "mismatch_count": 8,
    "loss_delta_vs_depth_1": -0.0002966858446598053
  },
  "depth_4": {
    "final_loss": 0.031890515238046646,
    "final_accuracy": 0.9793282151222229,
    "mismatch_count": 8,
    "loss_delta_vs_depth_1": -0.00013406947255134583
  }
}
```

The final mismatch surface is identical across all three saved depths: each ends with the same eight `mall ` contexts predicting `t` where the target is `n`: [`depth_1`](artifacts/depth_1/depth_surface.json), [`depth_2`](artifacts/depth_2/depth_surface.json), [`depth_4`](artifacts/depth_4/depth_surface.json).

```json
{
  "shared_final_mismatch_context": "mall ",
  "shared_final_target": "n",
  "shared_final_prediction": "t",
  "shared_final_mismatch_count": 8
}
```

The saved `hello` continuation also matches exactly across depths 1, 2, and 4: [`depth_1`](artifacts/depth_1/tiny_samples.json), [`depth_2`](artifacts/depth_2/tiny_samples.json), [`depth_4`](artifacts/depth_4/tiny_samples.json).

```json
{
  "hello": "hello world.\nsmall text.\nhello world.\nsmall text.\nhello world.\nsmall text.\nhello worl"
}
```

Depth 4 did show one transient instability spike during training before recovering: [`depth_4/tiny_run_metrics.json`](artifacts/depth_4/tiny_run_metrics.json).

```json
{
  "step": 1200,
  "loss": 0.437334,
  "accuracy": 0.875969
}
```

The narrowest honest reading is: on this one tiny transformer host surface and this one tiny fixed-window character task, repeated within-token reuse at depths 2 and 4 did not produce a meaningful final discrete behavior change relative to depth 1. The saved outcome surface is readable, but it is near-null apart from tiny final-loss shifts and one transient depth-4 instability spike.

### What Unit 01 does not show

- It does not show that dynamic depth works in general.
- It does not show that learned halting or loss prediction is useful.
- It does not show a meaningful compute/quality tradeoff.
- It does not show that deeper reuse improves final task behavior on this toy task.
- It does not show anything yet about composition with cortical-column ideas.

### Next discriminating step

No immediate dynamic-depth follow-up is required by this result alone. Reopen this question only if a later bounded unit can sharpen a real open edge — for example, by testing whether the near-null surface persists on a slightly less toy standard-model setup, or by making the stopping/loss-prediction question explicit as its own bounded unit rather than smuggling it into this one.

## Status

One bounded unit is now complete. The broader dynamic-depth question remains open.
