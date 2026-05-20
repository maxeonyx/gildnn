# Local learning with stop-gradient residual boundaries

This question serves the block-boundary part of [dictation 2026-05-20-10](../../../dictations/2026-05-20-10.md), the earlier local-learning motivation in [dictation 2026-05-20-9](../../../dictations/2026-05-20-9.md), and Thread 1 in [VISION.md](../../../VISION.md): if a module is a single residual block on a shared `d_model` stream, what happens when gradients are cut at block boundaries and each block gets its own local predictive objective?

## Status

Stages 0-3 are now done for the attention-enabled residual-block version. The clarified architecture is implemented in [`experiments/local_learning_residual/`](../../../experiments/local_learning_residual/). Stage 1 mechanical trust checks are saved in [`mechanical_trust.json`](../../../experiments/local_learning_residual/artifacts/stage1_checks/mechanical_trust.json). Stage 2 overfit evidence is under [`artifacts/stage2/`](../../../experiments/local_learning_residual/artifacts/stage2/). Stage 3 tiny-run evidence is under [`artifacts/stage3/`](../../../experiments/local_learning_residual/artifacts/stage3/).

Current answer, still only from the reduced Stage 3 rung: the stop-gradient local-learning variants train mechanically and can memorize one batch, but they are much worse than the matched end-to-end controls on short TinyShakespeare runs.

## Exact question

On the fixed TinyShakespeare `100K/20K`, char-level, `context=32`, `~186K` parameter frame, does adding stop-gradient boundaries between residual blocks, plus one local prediction head per block, help or hurt relative to matched ordinary residual stacks trained end to end?

The primary comparison matrix is:

| Family | Blocks | `d_model` | Heads | `ff_hidden` | Exact params | Role |
|---|---:|---:|---:|---:|---|
| End to end | 1 | 72 | 4 | 1057 | 185,942 | Control |
| End to end | 3 | 72 | 4 | 254 | 185,935 | Control |
| End to end | 6 | 72 | 4 | 53 | 185,707 | Control |
| Local learning | 3 | 72 | 4 | 218 | 186,043 | Primary variant |
| Local learning | 6 | 72 | 4 | 17 | 185,923 | Primary variant |

Diagnostic-only width, not part of the main table unless needed later:

| Family | Blocks | `d_model` | Heads | `ff_hidden` | Exact params | Role |
|---|---:|---:|---:|---:|---|
| Local learning | 1 | 72 | 4 | 1021 | 185,978 | Optional diagnostic |

## Clarified architecture

The old recurrent-stack local-learning experiment is superseded. The architecture here follows the current clarified reading:

- token embedding: `vocab_size -> d_model`
- position embedding: `context_len -> d_model`
- each block: `LayerNorm(d_model) -> causal multi-head self-attention -> residual add -> LayerNorm(d_model) -> Linear(d_model, ff_hidden) -> GELU -> Linear(ff_hidden, d_model) -> residual add`
- final readout: `LayerNorm(d_model) -> Linear(d_model, vocab_size)`
- local variant only: one `Linear(d_model, d_model)` local head per block
- shared residual width everywhere: `d_model = 72`

For the local variant, the residual stream is detached between blocks. That means the LM loss reaches only the top block and the final head. Lower blocks are trained only by their own local objectives. Each block is now causal sequence-mixing rather than position-wise only, so the stack can actually use the 32-token context for language modelling.

The implemented local target is the **next block's incoming residual delta** for every non-final block. Concretely, local head `k` reads block `k`'s post-block residual state and predicts the delta block `k+1` will add at the next boundary. The final block has no deeper block to predict, so its local head falls back to predicting its own delta. This preserves one local head per block while keeping the “next incoming residual delta” idea live for the interior boundaries that matter.

## Simplifications and non-goals

This is an isolated boundary-mechanism test. It deliberately does **not** test:

- async or desynchronized execution
- recurrence or persistent hidden state
- attention residual paths
- graph topology or routing structure
- selective updates / surprisal scheduling
- anything beyond this exact parameter budget, dataset slice, and task frame

It is also not trying to prove that local learning is globally good or bad. A positive or negative result here only narrows this one branch of the design tree.

## What evidence every run must save

Stage 1 establishes the minimum evidence contract. Meaningful runs in later stages must save at least:

- exact config
- git SHA and working-tree state
- exact parameter count
- training history with best epoch marked
- runtime on this machine
- fixed-prompt sample text
- Stage 1 gradient-boundary evidence link
- per-block residual statistics where relevant

The report should embed the discriminating parts inline rather than forcing Max to click through raw artifacts.

## Stage 1 mechanical trust snapshot

The Stage 1 script runs exact parameter counts, forward shape checks, local target shape/value checks, explicit gradient-routing checks, and residual-stream statistics for all planned variants. Evidence file: [`mechanical_trust.json`](../../../experiments/local_learning_residual/artifacts/stage1_checks/mechanical_trust.json).

### Gradient-boundary proof summary

Local variant: each local loss updates only its own block and local head; LM loss updates only the last block plus LM head.

```json
{
  "local_3b": {
    "block_1_local_loss": {"block_1": ">0", "block_2": 0, "block_3": 0, "lm_head": 0},
    "block_2_local_loss": {"block_1": 0, "block_2": ">0", "block_3": 0, "lm_head": 0},
    "block_3_local_loss": {"block_1": 0, "block_2": 0, "block_3": ">0", "lm_head": 0},
    "lm_loss": {"block_1": 0, "block_2": 0, "block_3": ">0", "lm_head": ">0"}
  }
}
```

End-to-end control: LM loss reaches every block.

```json
{
  "end_to_end_6b": {
    "lm_loss": {
      "block_1": ">0",
      "block_2": ">0",
      "block_3": ">0",
      "block_4": ">0",
      "block_5": ">0",
      "block_6": ">0",
      "lm_head": ">0"
    }
  }
}
```

### Residual-statistics snapshot

Representative Stage 1 batch RMS values:

| Variant | Block | Incoming residual RMS | Predicted delta RMS | True target delta RMS |
|---|---:|---:|---:|---:|
| Local 3-block | 1 | see artifact | see artifact | see artifact |
| Local 3-block | 2 | see artifact | see artifact | see artifact |
| Local 3-block | 3 | see artifact | see artifact | see artifact |
| Local 6-block | 1-6 | see artifact | see artifact | see artifact |

The exact numbers are in the artifact because this README is still pre-result, but the key Stage 1 gate is already satisfied: the target deltas are non-degenerate, the predictions have the right shape, and the detach boundaries are real rather than assumed.

## Stage 2 — overfit one batch

All promoted variants passed the overfit gate.

| Variant | Steps to pass | Final LM loss | Final local loss total | Final accuracy |
|---|---:|---:|---:|---:|
| End-to-end 3-block | 13 | 0.019988 | 0.000000 | 1.000000 |
| Local 3-block | 214 | 0.019125 | 7.090687 | 1.000000 |
| Local 6-block | 220 | 0.018004 | 5.004693 | 1.000000 |

Evidence from [`stage2/end_to_end_3b/final_metrics.json`](../../../experiments/local_learning_residual/artifacts/stage2/end_to_end_3b/final_metrics.json), [`stage2/local_3b/final_metrics.json`](../../../experiments/local_learning_residual/artifacts/stage2/local_3b/final_metrics.json), and [`stage2/local_6b/final_metrics.json`](../../../experiments/local_learning_residual/artifacts/stage2/local_6b/final_metrics.json).

The local heads are learning something nontrivial rather than collapsing to near-zero. Example from the 3-block local run:

```json
{
  "block_1": {"zero_predictor_mse": 1.005127, "learned_mse": 0.463314},
  "block_2": {"zero_predictor_mse": 8.272040, "learned_mse": 5.652506},
  "block_3": {"zero_predictor_mse": 8.272040, "learned_mse": 0.974867}
}
```

That is weakly or strongly better than zero on every block, and much better on the deepest block. The 6-block local run shows the same direction on every block as well.

## Stage 3 — tiny full-frame runs

These are short 4-epoch runs on the standard `100K/20K`, `context=32` TinyShakespeare frame.

| Variant | Params | Best val LM loss | Final val LM loss | Val accuracy | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| End-to-end 1-block | 185,942 | 1.731269 | 1.731269 | 0.486478 | 13.34 |
| End-to-end 3-block | 185,935 | 1.691263 | 1.691263 | 0.497947 | 25.93 |
| End-to-end 6-block | 185,707 | 1.707826 | 1.707826 | 0.491136 | 45.67 |
| Local 3-block | 186,043 | 2.581370 | 2.581370 | 0.288862 | 29.12 |
| Local 6-block | 185,923 | 2.540320 | 2.540320 | 0.290966 | 50.90 |

The ranking is already clear at this reduced rung: all end-to-end controls are far ahead of both local variants, and the gap is large enough that this is not just noise from a tiny difference. The local variants are not dead in the strict sense — they do improve from random initialization, and they generate vaguely word-like text — but they are badly behind.

Representative samples from epoch 4:

| Variant | Sample |
|---|---|
| End-to-end 3-block | ```text
First Citizen:
Before we proceed the prompt the people,
And the people seek and the people,
And the people seek and the people,
And the people seek and the people,
And the people seek and the peop
``` |
| Local 3-block | ```text
First Citizen:
Before we proceed the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
``` |
| Local 6-block | ```text
First Citizen:
Before we proceed wo mo lo lo the whe the  oo eat hat hat he the whe he the he the the  an eat eat eat he the the he the  ail wo the the the he the the he the he eat eat eat eat eat you he the whe he the the he the he
``` |

Evidence from [`stage3/end_to_end_3b/progression_samples.json`](../../../experiments/local_learning_residual/artifacts/stage3/end_to_end_3b/progression_samples.json), [`stage3/local_3b/progression_samples.json`](../../../experiments/local_learning_residual/artifacts/stage3/local_3b/progression_samples.json), and [`stage3/local_6b/progression_samples.json`](../../../experiments/local_learning_residual/artifacts/stage3/local_6b/progression_samples.json).

## What changed in understanding so far

- The original FFN-only residual block interpretation was not actually a language model over context. Adding causal attention fixed that and made Stage 2 possible.
- With causal attention added, the detach-boundary local-learning variants are mechanically healthy.
- Mechanical health is not enough: on this first reduced TinyShakespeare rung, stop-gradient local learning is strongly negative relative to matched end-to-end stacking.
- The 6-block local variant is not obviously deader than the 3-block local variant on this rung, but neither is remotely competitive with the end-to-end controls.

## Open tensions

These remain open on purpose:

1. **Last-block target tension.** “Predict the next incoming residual delta” is clean for interior blocks but underspecified for the last block. Stage 1 uses a fallback: the final block predicts its own delta so the top block still has a local head and a local-loss path.
2. **Fairness tension.** Parameter matching is explicit, but local heads still reallocate parameters away from trunk width. If later results are very close, a same-trunk-width diagnostic may still be needed.
3. **Depth-collapse tension.** The `6`-block local variant is mechanically valid at `ff_hidden=163`, but that does not yet prove it deserves promotion to the standardized training table.
4. **Embedding-gradient tension.** Detach boundaries are between residual blocks, not between embeddings and block 1. Whether that is the best boundary convention is not settled here.

## What this does not settle

Even after the full comparison, this experiment will still not settle async execution, graph structure, attention residuals, recurrence, or whether any broader cortical-column picture is good. It only asks what stop-gradient boundaries plus local heads do in this one minimal shared-residual-block setting.
