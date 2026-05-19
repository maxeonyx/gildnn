# Dynamic Computation Depth

## Question

Does a weight-shared recurrent model trained with losses at every depth learn a useful loss-prediction signal, and at inference does that signal allocate different amounts of compute to different characters?

## Setup

- Model: single `GRUCell` applied repeatedly (weight-shared across depth), task head at every depth, loss-prediction head
- Training: compute loss at all depths 1–8, sum task losses + loss-prediction MSE
- Loss-prediction head: predicts what the task loss would be at the current depth
- Halting: at inference, stop when predicted loss drops below threshold

## Results

### Small corpus (7K chars, 72K params)

| Variant | Val Loss | Avg Depth |
|---|---|---|
| Fixed depth 1 | 4.348 | 1.0 |
| Fixed depth 8 | 4.066 | 8.0 |
| Dynamic (initial) | 4.385 | 1.35 |

The model allocates different depths to different characters — rare/hard tokens get more compute ("cataracts" → depth 6, "war-proof" → depth 5). But the loss-prediction head is poorly calibrated on validation (correlation 0.08, MSE 33.2). The halting threshold, calibrated on training loss, doesn't transfer well.

### Large corpus (100K chars train, 20K val, context=32)

Scaling up dramatically improves loss-prediction calibration:
- Correlation: 0.08 → **0.50**
- MSE: 33.2 → **1.92**

Pareto frontier (threshold → val_loss, avg_depth):

| Val Loss | Avg Depth | Compute Savings |
|---|---|---|
| 1.722 | 8.0 | 0% (baseline) |
| 1.730 | 5.7 | 29% |
| **1.738** | **4.56** | **43%** |
| 1.755 | 3.16 | 60% |
| 1.786 | 1.30 | 84% |

**Practical operating point: 43% compute reduction for 1% quality loss.** The tradeoff is smooth and continuous — you can dial compute vs quality to any desired point.

### What the model thinks is "hard"

Systematic feature correlation analysis (η = correlation ratio, ρ = Spearman rank):

| Rank | Feature | Effect Size | Direction |
|---|---|---|---|
| 1 | Position in word | η = 0.46 | Word-initial chars get max depth |
| 2 | Character identity | η = 0.37 | Uppercase rare letters deepest |
| 3 | After punctuation | r = −0.23 | Post-punctuation is *shallow* |
| 4 | Bigram novelty | ρ = −0.19 | Novel bigrams are shallower |
| 5 | Word frequency | ρ = 0.16 | Rare words slightly deeper |
| 6 | Local entropy | ρ = −0.12 | High-entropy regions shallower |

**The dominant signal is position in word.** First character after a space gets depth 8.0 uniformly. Middle/end of words get depth 3.8–4.5. The model concentrates compute at word boundaries where it must commit to a word identity, then coasts through predictable continuations.

Post-punctuation tokens (`\n`, space after `.`) are among the shallowest — the opposite of the initial qualitative impression. These positions have low uncertainty (new line = likely a character name in Shakespeare).

⚠️ **Reproducibility note:** The analysis ran on a retrained model (same hyperparams, different random seed). Qualitative patterns differ from the first run's depth annotations, suggesting the depth routing is not fully stable across seeds. The position-in-word effect is robust; finer-grained character-level patterns may not be.

See `artifacts/improved/depth_analysis/analysis.md` for full tables.

## Key findings

1. **Multi-exit training works stably** — the model trains at all depths simultaneously without instability
2. **Deeper is better, up to a point** — loss improves monotonically from depth 1 to depth ~7, then plateaus
3. **The loss-prediction head works when given enough data** — poor calibration on 7K chars, good calibration on 100K chars
4. **Adaptive compute is practical** — 43% savings for 1% quality degradation at the recommended operating point
5. **Depth allocation is structured but seed-dependent** — word-initial positions robustly get more depth; finer patterns vary across seeds

## What this does not settle

- Whether combining dynamic depth with the predictive chain adds value
- Whether the loss-prediction head can be improved (e.g., predict improvement rather than absolute loss)
- Whether PonderNet-style probabilistic halting is better than threshold-based
- Whether this works for a deeper base model (e.g., transformer layers)
- What the optimal max_depth is (we only tried 8)
- Whether depth routing is stable across random seeds (evidence suggests it's not at fine grain)

## Artifacts

- First experiment: `artifacts/comparison_summary.json`, `artifacts/depth_annotation.txt`
- Improved (Pareto + large corpus): `artifacts/improved/`
  - Small corpus: `artifacts/improved/small_7k/`
  - Large corpus: `artifacts/improved/large_100k/`
- Feature analysis: `artifacts/improved/depth_analysis/`
