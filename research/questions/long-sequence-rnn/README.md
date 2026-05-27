# Question: Does Long-Sequence RNN Training Outperform Fixed-Window?

## What this asks

Does a sequential RNN trained with truncated BPTT on long sequences (1000+ tokens) outperform the same architecture trained with independent fixed windows? And does it approach/beat a transformer at its sweet spot (128-256 token attention) at matched parameter count?

This is Pathway 1 tested at the RNN's natural operating point for the first time. All prior experiments in this project used fixed-window sampling — which tests the architecture as a windowed transformer surrogate, not as a sequential stateful model.

## Which goals this serves

- **Pathway 1 (Wide Recurrent vs Deep Transformer)** — [ROADMAP.md](../../../ROADMAP.md): "Can a wide shallow network run many times match a deep transformer?" This tests the RNN side of that comparison properly.
- **Dictation [2026-05-27-1](../../../dictations/2026-05-27-1.md):** "The correct comparison is: transformer at its sweet spot (attention over 128-256 tokens) vs our RNN at its sweet spot (thousands of tokens, truncated BPTT), at matched parameter count."

## Why prior results may be invalid

All multi-block/lateral experiments used SHORT_CONTEXT=4 for the output block. The negative results about stale laterals, multi-rate, and sequential processing were all tested at this tiny scale. Max identifies: "They might be completely wrong once you give the architecture sequences long enough for the hidden state to actually accumulate useful temporal structure."

The key insight: a 4-character window gives the hidden state essentially NO time to accumulate useful temporal patterns. The RNN's advantage is EXACTLY that it carries state across many tokens — testing it at 4 tokens is like testing a transformer with ctx=1.

## Architecture: what "sequential RNN" means here

The project has two different uses of "recurrent":
1. **Recurrent over depth** (RecurrentDepthLM) — same weights applied N times to the same token's window. This is a shared-layer transformer, NOT a sequential RNN.
2. **Recurrent over time** — processes tokens one by one, maintaining hidden state across the sequence. THIS is what the dictation asks for.

For this experiment, we need architecture (2): a model that processes tokens sequentially with persistent state across the full sequence, trained with truncated BPTT.

## Truncated BPTT

Standard operational meaning:
- Forward pass: process a LONG sequence (e.g., 2048-8192 tokens), carrying hidden state throughout
- Backward pass: split into chunks of K tokens (e.g., K=128 or K=256). Detach state at chunk boundaries. Backprop only through each chunk.
- Effect: the model SEES thousands of tokens of context (via hidden state) but gradient only flows through K tokens at a time.

This means:
- Per-step cost: O(hidden_dim²) — independent of sequence length (Max's point)
- Memory cost: O(K × hidden_dim) — bounded by BPTT chunk, not full sequence
- Information flow: state accumulates over full sequence, even though gradient is truncated

## The comparison

| Model | Context mechanism | Training | Per-step cost |
|---|---|---|---|
| Transformer (baseline) | Attention over window (128-256) | Independent windows | O(ctx × d²) |
| RNN (window-trained) | Hidden state from fixed window | Independent windows | O(d²) |
| **RNN (TBPTT)** | **Hidden state from full sequence** | **Truncated BPTT** | **O(d²)** |

The hypothesis: RNN (TBPTT) > RNN (window) because it sees more context. The question: RNN (TBPTT) vs Transformer — which wins at matched params?

## Hypotheses

**H1:** RNN trained with TBPTT (seq_len=2048, chunk=128) achieves lower val_loss than the same RNN trained with independent windows (ctx=128).

**H2:** RNN trained with TBPTT at matched parameter count approaches or beats the transformer baseline at its sweet spot (ctx=128-256).

**H3 (stretch):** The benefit increases with sequence length — going from seq=2048 to seq=8192 further improves the TBPTT-trained RNN.

## Exit conditions

| Outcome | Interpretation | Next step |
|---|---|---|
| H1 confirmed (TBPTT > window by >0.05 nats) | Long-sequence state IS useful | Scale up; test project architecture at this sweet spot |
| H1 not confirmed (TBPTT ≈ window) | Either the task has no long-range structure, or the RNN can't exploit it | Try on WikiText-103; or the direction is weak |
| H2 confirmed (RNN ≈ transformer at matched params) | The thesis holds at the RNN's sweet spot | Major result; proceed to project architecture variants |
| H2 not confirmed (transformer still wins) | RNN can't match transformer even at its own sweet spot | Re-evaluate Pathway 1 thesis strength |

## Cheapest honest test (the ladder)

### Rung 1: GRU with TBPTT vs window (TinyShakespeare)

Use the existing GRU architecture from `base-experiments/rnn/` but change the training:
- **Window baseline:** GRU, ctx=128, independent random windows (same as existing but larger ctx)
- **TBPTT variant:** Same GRU, seq_len=2048, bptt_chunk=128
- Both: 186K params (matching existing baseline), TinyShakespeare
- Duration: <5 minutes each

Exit: is TBPTT better than window-based? If yes, proceed. If no, try WikiText-103 (rung 2).

### Rung 2: GRU with TBPTT (WikiText-103)

Same comparison but on WikiText-103 which has genuine long-range structure (article-level coherence).
- Matched params to existing transformer baseline (~2.86M params)
- Duration: ~30 minutes each

Exit: does the larger dataset's long-range structure make TBPTT more beneficial?

### Rung 3: Project architecture with TBPTT

If rungs 1-2 show TBPTT helps, build a single-block version of the project architecture (additive residual, normalized, weight-tied readout) as a sequential model with TBPTT.
- This is the real test of Pathway 1

## What needs to be built

1. **TBPTT training loop** — process long sequences in chunks, carry state, detach at boundaries
2. **Sequential dataset** — yields long contiguous character sequences (not random windows)
3. **GRU model wrapper** — simple: embedding → GRU → tied readout. Already mostly exists in `core/model.py`
4. **Evaluation** — measure val_loss with full sequential state (not windowed)

Existing code that can be reused:
- `core/model.py` — has GRU model definition
- `core/run_utils.py` — logging, lock management, device handling
- `core/dataset.py` — corpus loading (needs sequential iterator, not random windows)
- Tied readout from `core/tied_readout.py`

## Non-goals

- Multi-block lateral architecture (that's AFTER this basic test works)
- Dynamic depth / halting (orthogonal mechanism)
- Comparison on image/patch tasks (future work)
- Novel architecture design (use standard GRU first as honest control)

## Results

_Placeholder — to be filled after experiment runs._

## Next steps

_Placeholder — depends on which exit condition is met._
