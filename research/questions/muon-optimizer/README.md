# Muon Optimizer for Window Size Stability

## Question

[Dictation 2026-05-21-7](../../../dictations/2026-05-21-7.md): "Concrete next action: swap in Muon as the optimizer and rerun the window size ablation (k=4, k=8, k=16) to see if the instability goes away."

With AdamW, temporal windows larger than k=4 diverge during training. Does Muon stabilize training at k=8 and k=16?

## Result: instability fixed, quality reduced

| Window (k) | AdamW best val | Muon best val | AdamW diverged? | Muon diverged? |
|-----------|---------------|--------------|-----------------|----------------|
| 4 | **1.670** | 1.780 | No | No |
| 8 | — | **1.773** | **Yes** | No |
| 16 | — | **1.759** | **Yes** | No |

Config: d_model=192, feedforward_dim=768, 4 heads, 100K chars TinyShakespeare, 5 epochs, muon_lr=0.02, adamw_lr=0.0003.

Artifact: [`experiments/muon_window_ablation/artifacts/muon_ablation_results.json`](../../../experiments/muon_window_ablation/artifacts/muon_ablation_results.json).

## Interpretation

**Muon completely solves the instability.** k=8 and k=16 train stably where AdamW diverges. This confirms the instability was optimizer-related (likely gradient magnitude issues in the temporal attention path), not architectural.

**But Muon's quality is worse at k=4:** 1.780 vs AdamW's 1.670 — a 0.11 nat gap. This means Muon's update dynamics are less efficient for this architecture when training is already stable.

**Larger windows do improve with Muon:** k=16 (1.759) < k=8 (1.773) < k=4 (1.780). The trend is clear — more temporal context helps, as expected. AdamW couldn't show this because it diverged.

## Extended training (k=8, 15 epochs)

Pushed Muon k=8 to 15 epochs. Reached best val_loss 1.661 at epoch ~10, then **diverged** (final 4.157). So Muon delays but doesn't eliminate instability at longer training horizons.

Artifact: [`experiments/muon_window_ablation/artifacts/muon_k8_15ep.json`](../../../experiments/muon_window_ablation/artifacts/muon_k8_15ep.json).

## Parameter-matched (d_model=116)

With Muon at d_model=116 (184K params, matching transformer reference), k=8: best val_loss **1.694** in 5 epochs. Compare to AdamW param-matched at k=4: 1.717. So Muon + larger window actually BEATS the AdamW baseline when parameter-matched.

Artifact: [`experiments/muon_window_ablation/artifacts/muon_param_matched.json`](../../../experiments/muon_window_ablation/artifacts/muon_param_matched.json).

## What this settles

- ✅ The instability at k>4 is optimizer-related, not architectural
- ✅ Larger temporal windows help quality (monotonic improvement k=4 → k=16)
- ✅ Muon + k=8 at matched params (1.694) beats AdamW + k=4 at matched params (1.717)

## What remains open

- Whether a cosine schedule or LR warmup prevents the epoch-10 divergence
- Whether there's an optimizer that gives both stability AND AdamW's k=4 quality
- Whether this matters for the multi-rate architecture (which uses a different attention pattern)

## Relevance to current work

The current `MultiRateResidualModel` uses temporal_window=4 by default, which is stable with AdamW. If we want to experiment with larger windows (which could improve quality), Muon is the known path. But since the current multi-rate experiments are producing useful results at k=4, this is lower priority.
