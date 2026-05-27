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

### Rung 1: TinyShakespeare, 188K-param GRU

**Training comparison (best validation loss):**

| Mode | Best val_loss | At step | Training tokens seen |
|---|---|---|---|
| TBPTT carry (400 steps) | 1.596 | 175 | 5.7M |
| Chunk-reset (400 steps) | 1.595 | 200 | 6.6M |
| Window (6400 steps, token-matched) | 1.591 | 425 | 0.9M |

All three modes reach the same best validation loss (~1.59). State-carrying provides **zero benefit** over chunk-reset. The window baseline reaches the same floor with 6× fewer tokens processed.

**Reset-sweep evaluation (hidden state reset every N tokens):**

| Mode | N=1 | N=8 | N=32 | N=128 | N=512 | N=2048 | full |
|---|---|---|---|---|---|---|---|
| TBPTT carry | 3.153 | 1.938 | 1.679 | 1.616 | 1.602 | 1.597 | 1.596 |
| Chunk-reset | 2.653 | 1.767 | 1.637 | 1.603 | 1.597 | 1.595 | 1.595 |
| Window (200 steps only) | 2.676 | 1.831 | 1.755 | 1.735 | 1.730 | 1.729 | 1.729 |

Key observations:
- Both TBPTT and chunk-reset models benefit significantly from short-range state (N=1→128: ~1.5 nats improvement).
- **Almost no benefit beyond 128 tokens** (N=128→full: 0.020 nats for TBPTT, 0.008 nats for chunk-reset).
- The gradient horizon is 128 (bptt_chunk=128). Neither model learns to use state beyond that.
- The TBPTT model is MORE dependent on state (worse at N=1: 3.15 vs 2.65) — it expects context but doesn't exploit long-range context any better.
- Window model not at its peak (only trained 200 steps here; needs 425+ for best val) — this comparison is about TBPTT vs chunk-reset.

### Interpretation

**H1 is NOT confirmed on TinyShakespeare.** Training with TBPTT (state carry) provides no benefit over chunk-reset (same training but state cleared every 128 tokens). The GRU hidden state provides useful SHORT-range context (~128 tokens) but does not accumulate useful LONG-range temporal structure.

Possible reasons:
1. TinyShakespeare (100K chars) has no genuine long-range structure beyond ~128 characters
2. The model (188K params) is too small to represent complex long-range dependencies
3. Truncated BPTT with chunk=128 can't teach the model to use information beyond 128 (gradient doesn't reach)
4. The GRU architecture may be fundamentally limited in long-range information propagation (forgetting gate)

This does NOT kill the direction — it's the expected "try a harder task" case. WikiText-103 has real article-level coherence.

## Next steps

Per exit conditions: "H1 not confirmed → try on WikiText-103 (rung 2); or the direction is weak."

### Rung 2: WikiText-103

WikiText-103 has 100M+ characters with genuine article-level coherence (paragraphs reference earlier paragraphs, consistent topics across thousands of tokens). This is where long-range state SHOULD help if it helps anywhere.

Changes from Rung 1:
- **Dataset:** WikiText-103 (already supported by `core/dataset.py`)
- **Model size:** Scale to ~2.86M params (matching transformer baseline) to have capacity for long-range patterns
- **Article boundaries:** Reset state at article boundaries so we don't blur cross-article dependencies
- **Longer gradient horizon?** Consider bptt_chunk=256 or 512 to give gradient more reach

Key question: does the reset-sweep result change on a corpus with real long-range structure? If not, the GRU architecture itself may be the limiting factor (forgetting gate kills information too quickly).
