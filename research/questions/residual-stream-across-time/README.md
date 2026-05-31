# Residual stream across time

**Status:** Answered

**Question:** Does the cellular automaton need a temporal residual in its state update, or is full state replacement enough?

**Relevant files:** [VISION.md](../../../VISION.md), [ROADMAP.md](../../../ROADMAP.md), [core/automaton.py](../../../core/automaton.py)

## Why this question mattered

The original automaton update effectively replaced state each fire. But [VISION.md](../../../VISION.md) says the block output should **merge** with the stream, not overwrite it: “a shared/tied combining function merges block output with the lateral stream at each node.” This is also a [Pathway 1](../../../ROADMAP.md#pathway-1-wide-recurrent-vs-deep-transformer) question: if recurrence is meant to substitute for depth, the recurrent state must preserve useful information across repeated applications.

## Setup

- Model: [`CellularAutomaton`](../../../core/automaton.py)
- Dataset: TinyShakespeare, character-level (~1M chars)
- Fixed: `d_stream=256`, `chunk_size=128`, `seq_len=128`, `batch_size=32`, `lr=3e-4`, `1000` training steps
- Sweep: 3 state update rules × multiple `steps_per_token`
- Scope: `n_levels=1` to isolate the temporal update rule itself

## State update rules tested

| Rule | Code shape | Intended effect |
|---|---|---|
| Replace | `state = output` | Original behavior: each fire discards previous state |
| Raw add | `state = state + output` | Preserve state by accumulation |
| Normalized residual | `state = normalize(state + output)` | Preserve state without unbounded norm growth |

## Results

| Update rule | spt=1 | spt=2 | spt=4 | spt=8 |
|---|---:|---:|---:|---:|
| `state = output` (replace) | 2.20 | — | — | 3.29 |
| `state = state + output` (raw add) | 3.26 | — | — | 3.29 |
| `state = normalize(state + output)` | 2.23 | 2.29 | 2.37 | 2.68 |

Reference baselines on the same task:

| Model | Cross-entropy |
|---|---:|
| Random over 65 chars | ~4.17 |
| GRU (820K params) | 1.58 |
| Tied transformer (854K params, ctx=128) | 1.67 |

## Findings

1. **State replacement breaks at high `steps_per_token`.** At `spt=8`, replace gives CE `3.29`: barely above random. Replacing the whole state eight times per token destroys information faster than the model can rebuild it.
2. **Raw additive residual is worse than replacement.** At `spt=1`, raw add is already bad (`3.26`). The state norm grows, the accumulated state becomes sticky, and the MLP output stops being able to redirect it.
3. **Normalized residual fixes the high-`spt` failure mode.** At `spt=8`, normalized residual improves CE from `3.29` to `2.68`. Keeping the state on the unit sphere preserves information without letting magnitude explode.
4. **Extra recurrent steps still hurt in the single-level case.** Even with the best update rule, performance degrades monotonically: `2.23 → 2.29 → 2.37 → 2.68` as `spt` goes `1 → 2 → 4 → 8`.
5. **That monotonic degradation is not evidence against the multi-level design.** With `n_levels=1`, extra steps apply the same autonomous MLP with no new information arriving between steps. The result is autonomous drift. In the multi-level architecture, laterals arrive between steps; that is what gives extra steps something useful to do.
6. **The remaining gap to GRU is not explained by attention.** The GRU also has no attention. More likely causes are: no gating, frozen embeddings, and pervasive L2 normalization constraining state to the sphere.

## Interpretation

This question is answered cleanly: **the automaton needs merge, not replace.** The minimal merge that worked here was unit-sphere residual accumulation. That directly supports the wording in [VISION.md](../../../VISION.md): block output should be combined with the stream, not substituted for it.

It also sharpens the role of the wider architecture. Repeating a single autonomous block many times is not useful by itself. The point of extra recurrent steps is to give the model time for **inter-level communication**. Laterals from higher levels are the memory mechanism; without them, more steps mostly create drift.

## What this does and does not settle

Settled here:

- Pure replacement is the wrong temporal update rule.
- Unnormalized additive residual is also wrong.
- A normalized residual is the best of the three tested options.

Still open:

- Does normalized residual help or hurt the full multi-level model at convergence?
- Would a learned gate such as `α * state + (1 - α) * output` close more of the gap to GRU?
- Is the remaining `2.23` vs `1.58` gap fixable without attention, or is a feedforward-only recurrent block fundamentally limited?

## Code change

The implementation change was committed in `8a6f09a` as:

```python
current_states = torch.where(fire_mask, l2_normalize(current_states + output), current_states)
```

This is now the tested state-update rule in [`core/automaton.py`](../../../core/automaton.py).
