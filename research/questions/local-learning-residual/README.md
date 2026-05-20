# Local learning with stop-gradient residual boundaries

This question serves the block-boundary part of [dictation 2026-05-20-10](../../../dictations/2026-05-20-10.md), the earlier local-learning motivation in [dictation 2026-05-20-9](../../../dictations/2026-05-20-9.md), and Thread 1 in [VISION.md](../../../VISION.md): if a module is a single residual block on a shared `d_model` stream, what happens when gradients are cut at block boundaries and each block gets its own local predictive objective?

## Status

Stage 0 and Stage 1 are now in place. The report contract exists, the clarified architecture is implemented in [`experiments/local_learning_residual/`](../../../experiments/local_learning_residual/), and the Stage 1 mechanical trust checks are saved in [`mechanical_trust.json`](../../../experiments/local_learning_residual/artifacts/stage1_checks/mechanical_trust.json).

This README is still pre-result. It defines the comparison and records the mechanical evidence that the implementation matches the intended stop-gradient wiring. It does not yet answer whether local learning helps language modelling quality.

## Exact question

On the fixed TinyShakespeare `100K/20K`, char-level, `context=32`, `~186K` parameter frame, does adding stop-gradient boundaries between residual blocks, plus one local prediction head per block, help or hurt relative to matched ordinary residual stacks trained end to end?

The primary comparison matrix is:

| Family | Blocks | `d_model` | `ff_hidden` | Exact params | Role |
|---|---:|---:|---:|---:|---|
| End to end | 1 | 72 | 1203 | 185,944 | Control |
| End to end | 3 | 72 | 400 | 185,941 | Control |
| End to end | 6 | 72 | 199 | 185,719 | Control |
| Local learning | 3 | 72 | 364 | 186,049 | Primary variant |
| Local learning | 6 | 72 | 163 | 185,935 | Primary variant |

Diagnostic-only width, not part of the main table unless needed later:

| Family | Blocks | `d_model` | `ff_hidden` | Exact params | Role |
|---|---:|---:|---:|---:|---|
| Local learning | 1 | 72 | 1167 | 185,980 | Optional diagnostic |

## Clarified architecture

The old recurrent-stack local-learning experiment is superseded. The architecture here follows the current clarified reading:

- token embedding: `vocab_size -> d_model`
- position embedding: `context_len -> d_model`
- each block: `LayerNorm(d_model) -> Linear(d_model, ff_hidden) -> GELU -> Linear(ff_hidden, d_model)` plus residual add
- final readout: `LayerNorm(d_model) -> Linear(d_model, vocab_size)`
- local variant only: one `Linear(d_model, d_model)` local head per block
- shared residual width everywhere: `d_model = 72`

For the local variant, the residual stream is detached between blocks. That means the LM loss reaches only the top block and the final head. Lower blocks are trained only by their own local objectives.

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

## Open tensions

These remain open on purpose:

1. **Last-block target tension.** “Predict the next incoming residual delta” is clean for interior blocks but underspecified for the last block. Stage 1 uses a fallback: the final block predicts its own delta so the top block still has a local head and a local-loss path.
2. **Fairness tension.** Parameter matching is explicit, but local heads still reallocate parameters away from trunk width. If later results are very close, a same-trunk-width diagnostic may still be needed.
3. **Depth-collapse tension.** The `6`-block local variant is mechanically valid at `ff_hidden=163`, but that does not yet prove it deserves promotion to the standardized training table.
4. **Embedding-gradient tension.** Detach boundaries are between residual blocks, not between embeddings and block 1. Whether that is the best boundary convention is not settled here.

## What this does not settle

Even after the full comparison, this experiment will still not settle async execution, graph structure, attention residuals, recurrence, or whether any broader cortical-column picture is good. It only asks what stop-gradient boundaries plus local heads do in this one minimal shared-residual-block setting.
