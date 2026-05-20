# Self-prediction compute compression

## Question

Does adding a detached self-prediction auxiliary loss make shallow recurrent steps approximate the model's own final-depth output distribution well enough to improve quality at lower depth or improve the dynamic halting quality/compute frontier?

Relevant dictations: [`2025-05-08-5`](../../../dictations/2025-05-08-5.md), [`2026-05-20-11`](../../../dictations/2026-05-20-11.md), [`2026-05-20-12`](../../../dictations/2026-05-20-12.md).

There is real dictation tension here. The older dictation says the objective should be predicting the next input / latent, not the model's own output. The newer dictation explicitly asks whether the network should try to predict its own outputs and internal latents in order to compress computation into fewer steps. This experiment treats the newer self-prediction idea as a narrow testable branch, not as settled project direction.

## Locked simplification

- Existing weight-shared GRU dynamic-depth scaffold
- Same TinyShakespeare `100K/20K`, `ctx=32`, char-level frame as the existing dynamic-depth experiment
- Same `embedding_dim=32`, `hidden_dim=128`, `max_depth=8`, optimizer, seed, and halting analysis frame
- Same multi-exit task loss and same loss-prediction head
- Only changed variable: auxiliary KL from shallow logits to **detached** final-depth logits
- This stage tests logits only, not latent-state distillation

## Planned evidence

- one-batch overfit trace showing the auxiliary wiring is actually detached and trainable
- fixed-depth validation comparison at depths `1, 2, 3, 4, 8`
- dynamic halting frontier comparison using the same threshold sweep method as the baseline dynamic-depth report

## Non-goals

- latent self-prediction
- larger models or longer runs
- any claim about other distillation targets
- any claim about transformer-scale behavior

## Results

### Wiring check

Artifacts: [`baseline summary`](../../../experiments/self_prediction/artifacts/tiny_rung_w01/baseline/summary.json), [`self-prediction summary`](../../../experiments/self_prediction/artifacts/tiny_rung_w01_self_only/self_prediction/summary.json), [`self-prediction overfit probe`](../../../experiments/self_prediction/artifacts/overfit_probe_w01/self_prediction/summary.json).

The detached-target check passed in both variants: shallow logits receive gradient from the KL term, while the final-depth target logits do not.

| Variant | Detached target check | Shallow grad norm | Final-target grad | Initial aux KL | Final aux KL | Final task loss | Final accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline control | pass | `0.001386` | `null` | `0.001585` | `0.000496` | `0.000213` | `1.0` |
| Self-prediction (`KL weight = 0.1`) | pass | `0.001386` | `null` | `0.001585` | `0.000265` | `0.000222` | `1.0` |

Inline evidence:

```json
{
  "baseline_detach_probe": {
    "checked": true,
    "shallow_grad_norm": 0.001385680865496397,
    "final_grad_norm": null
  },
  "self_prediction_detach_probe": {
    "checked": true,
    "shallow_grad_norm": 0.001385680865496397,
    "final_grad_norm": null
  }
}
```

So the negative result is not a wiring bug in the obvious sense. The KL loss is connected the intended way and it does go down on the overfit rung.

### Stronger auxiliary weight failed the ladder gate

The first attempt used `self_prediction_weight = 0.5`. That run failed the required one-batch overfit gate, so it was not a valid experiment rung to compare against the baseline. To complete the tiny rung honestly, I reduced the auxiliary weight to `0.1` and reran the self-prediction variant. That weaker setting overfit cleanly, but still produced a negative result.

This matters because the direction already looks bad in two ways at tiny scale:

- stronger self-prediction pressure broke the ladder gate
- weaker self-prediction pressure ran, but underperformed the baseline

### Fixed-depth comparison

Artifacts: [`baseline summary`](../../../experiments/self_prediction/artifacts/tiny_rung_w01/baseline/summary.json), [`self-prediction summary`](../../../experiments/self_prediction/artifacts/tiny_rung_w01_self_only/self_prediction/summary.json).

At every checked depth, the self-prediction variant is worse.

| Depth | Baseline val loss | Self-pred val loss | Delta |
|---|---:|---:|---:|
| 1 | `1.825240` | `1.843521` | `+0.018280` |
| 2 | `1.729182` | `1.746262` | `+0.017080` |
| 3 | `1.714955` | `1.727869` | `+0.012914` |
| 4 | `1.714244` | `1.725665` | `+0.011420` |
| 8 | `1.741272` | `1.759157` | `+0.017885` |

Inline evidence:

```json
{
  "baseline_fixed_depth_val": {
    "1": 1.825240135192871,
    "2": 1.7291821241378784,
    "3": 1.71495521068573,
    "4": 1.7142443656921387,
    "8": 1.74127197265625
  },
  "self_prediction_fixed_depth_val": {
    "1": 1.8435205221176147,
    "2": 1.7462621927261353,
    "3": 1.7278692722320557,
    "4": 1.725664734840393,
    "8": 1.7591568231582642
  }
}
```

This is the cleanest part of the answer. If the mechanism were helping compress useful computation into early steps, the shallow fixed-depth losses should have improved. They did not.

### Dynamic halting frontier

Artifacts: [`baseline summary`](../../../experiments/self_prediction/artifacts/tiny_rung_w01/baseline/summary.json), [`self-prediction summary`](../../../experiments/self_prediction/artifacts/tiny_rung_w01_self_only/self_prediction/summary.json).

The halting frontier is also worse. At matched approximate depths, the self-prediction variant has higher validation loss throughout.

| Approx baseline avg depth | Baseline val loss | Approx self-pred avg depth | Self-pred val loss |
|---|---:|---|---:|
| `1.000` | `1.825240` | `1.000` | `1.843521` |
| `1.840` | `1.813746` | `1.799` | `1.837160` |
| `2.773` | `1.797997` | `2.762` | `1.821014` |
| `3.629` | `1.779623` | `3.557` | `1.812158` |
| `4.592` | `1.768703` | `4.567` | `1.796308` |
| `5.645` | `1.754853` | `5.603` | `1.785043` |

Recommended operating points from the same threshold-selection rule:

| Variant | Threshold | Val loss | Avg depth | Compute savings |
|---|---:|---:|---:|---:|
| Baseline control | `1.064727` | `1.754853` | `5.645` | `29.4%` |
| Self-prediction (`KL weight = 0.1`) | `0.253540` | `1.763711` | `7.432` | `7.1%` |

Inline evidence:

```json
{
  "baseline_threshold_recommended": {
    "threshold": 1.064727,
    "val_loss": 1.7548531293869019,
    "val_avg_depth": 5.64453125,
    "compute_savings": 0.29443359375
  },
  "self_prediction_threshold_recommended": {
    "threshold": 0.25354,
    "val_loss": 1.7637109756469727,
    "val_avg_depth": 7.432091236114502,
    "compute_savings": 0.07098859548568726
  }
}
```

This is worse in exactly the direction the experiment was supposed to improve: higher loss while also using more depth.

## Conclusion

### Mechanism answer

**Negative.** Self-prediction auxiliary loss in this form — KL from shallow logits to detached deep logits — does **not** improve the quality/compute frontier on this tiny matched rung.

- At every fixed depth, the self-prediction variant is slightly worse.
- The halting frontier is also worse: higher validation loss at comparable depth, and a worse recommended operating point.
- Stronger self-prediction pressure (`weight = 0.5`) was bad enough to fail the one-batch ladder gate.

### Interpretation

The simplest reading is that the existing multi-exit training already extracts most of what the shallow steps can usefully learn at this scale. Explicitly distilling shallow logits from the model's own deep logits appears to add noise rather than signal.

That does **not** settle the broader self-prediction idea completely. What this experiment argues against is a very specific version:

- output-logit distillation
- same-model shallow-to-deep KL
- weight-shared GRU dynamic-depth scaffold
- tiny matched TinyShakespeare rung

So this is one data point against the newer self-prediction direction from [`2026-05-20-12`](../../../dictations/2026-05-20-12.md), especially when read against the older caution in [`2025-05-08-5`](../../../dictations/2025-05-08-5.md) that predicting the model's own output may be the wrong target.

### What remains open

Still untested:

- latent-space distillation rather than output-logit distillation
- whether a different target, architecture, or scale changes the sign
- whether the useful self-prediction target is internal state prediction rather than output prediction

For now though, the narrow answer is clean: **this self-prediction logits-KL auxiliary loss is a negative result.**
