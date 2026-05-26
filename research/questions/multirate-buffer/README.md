# Can a slow block with larger context, firing periodically and holding its output between firings, help the output block?

**Pathway:** 8 (Multi-Rate Processing)

**Dictations:** [2026-05-22-10](../../../dictations/2026-05-22-10.md) — "multi-rate means that a certain part of the network is attempting to predict further in the future"; [2026-05-23-2](../../../dictations/2026-05-23-2.md) — "It's a static prior, not a useful slow feature extractor."

**Status: negative result — the sequential regime breaks the lateral mechanism, and the held-output version is catastrophically worse.**

The fixed-window lateral mechanism works when the slow block is freshly computed at each evaluation position (window-based training, independent random samples). This experiment tested the more realistic version: what if the slow block fires periodically and its output is held between firings? Answer: it doesn't work, and the reason isn't "needs more tuning" — the sequential training regime is structurally incompatible with the lateral mechanism. Adjacent positions overlap so heavily that the lateral becomes a near-constant bias rather than a position-specific signal.

---

## Prior result this was trying to amortize

The fixed-window regime in [`runs/tied_readout_lm.py`](../../../runs/tied_readout_lm.py) already showed that a slow block with more context can help block 0 **when recomputed fresh at every evaluation position**:

| Setup | block0 val_loss | slow-block condition | Δ |
|---|---:|---:|---:|
| 4-char output block alone | 1.7522 | — | — |
| + 128-char slow block, fresh same-timestep lateral | 1.6646 | 128-char context | **-0.088** |

That result matters because it proves the lateral mechanism itself can work. The question here was narrower: can the same idea survive when the slow block fires less often and its output is held between firings?

## The question

Can a slow block with larger context, firing periodically and holding its output between firings, help the output block?

This is the minimal "context buffer" version of Pathway 8. It keeps the same basic local-learning setup, but replaces fresh slow-block recomputation with periodic firing plus reuse.

## Why this matters

- Fresh recomputation at every position is the expensive version.
- Multi-rate firing is closer to the eventual architecture.
- If the held-output version worked, it would be evidence that multi-rate can preserve the useful part of the fixed-window result while reducing slow-block compute.

## Script

[`runs/multirate_buffer_lm.py`](../../../runs/multirate_buffer_lm.py)

## Conditions

| Condition | Slow context | Refresh interval | Role |
|---|---:|---:|---|
| `block0_alone` | — | — | Baseline |
| `direct_ctx32_refresh1` | 32 | 1 | Control: same sequential regime, but fresh slow recomputation every position |
| `buffered_ctx32_refresh8` | 32 | 8 | Target: slow block fires every 8 positions, held between firings |

Follow-up: same control idea with `slow_context=128` and `seq_length=256`.

## What the script actually implements

The refresh schedule chooses firing positions, then reuses the most recent slow hidden state until the next firing:

```python
valid_output_indices = torch.nonzero(eval_mask, as_tuple=False).squeeze(-1)
relative_positions = valid_position_ends - (slow_context - 1)
fire_mask = relative_positions.remainder(refresh_interval) == 0
fire_output_indices = valid_output_indices[fire_mask]
ages = relative_positions.remainder(refresh_interval)
```

At evaluation/training time, the held slow state is copied across all positions since the last refresh:

```python
fire_hidden_by_position = fire_hidden.view(batch_size, schedule.fire_output_indices.shape[0], -1)
held_hidden = fire_hidden_by_position[:, schedule.valid_to_fire_slot, :]
lateral[:, schedule.valid_output_indices, :] = held_hidden
```

That is the whole test: same lateral pathway, but the slow block only updates periodically.

## Results

### Run 1 — wrong scale, therefore not the real test

`lateral_scale=0.2` was too weak for this tiny setup. The control already went the wrong direction:

| Condition | val_loss | Δ vs baseline | Read |
|---|---:|---:|---|
| `direct_ctx32_refresh1` | — | +0.111 | Worse; scale too weak to provide useful signal |

This run only showed that the initial scale choice was wrong. The real test is the corrected scale below.

### Run 2 — corrected scale (`lateral_scale=1.0`, `slow_context=32`)

| Condition | val_loss | Δ vs `block0_alone` | Ablated val_loss | Ablation Δ |
|---|---:|---:|---:|---:|
| `block0_alone` | 1.6154 | — | — | — |
| `direct_ctx32_refresh1` | 1.6154 | +0.0000 | 1.9064 | +0.2910 |
| `buffered_ctx32_refresh8` | 2.2051 | +0.5897 | — | — |

Two things happened at once:

1. **The direct control tied baseline exactly.** Fresh recomputation at every position did **not** recover the fixed-window win.
2. **The buffered target collapsed.** Holding the slow output for 8 positions made validation loss catastrophically worse.

The ablation result on `direct_ctx32_refresh1` matters: removing the lateral makes loss much worse (`+0.291`), so block 0 is using the lateral. It is just not getting a net benefit from it.

### Age buckets for the buffered condition (`buffered_ctx32_refresh8`)

| Age since refresh | val_loss | Read |
|---:|---:|---|
| 0 | 1.9402 | Already much worse than baseline even when the slow state is fresh |
| 1 | 2.1880 | Immediate degradation after one held step |
| 2-7 | 2.22-2.26 | Stays catastrophically bad while the held state gets older |

This is not a "mostly good except when stale" pattern. Even age 0 is bad. Staleness makes it worse, but the failure starts earlier: the sequentially-trained slow lateral is already the wrong kind of signal.

### Run 3 — larger slow context does not rescue sequential mode

The obvious rescue attempt was: maybe ctx=32 is just too short, so give the slow block the same 128-char context that worked in the fixed-window regime.

| Condition | slow_context | seq_length | val_loss | Δ vs baseline |
|---|---:|---:|---:|---:|
| `block0_alone` | — | 256 | 1.6410 | — |
| `direct_ctx128_refresh1` | 128 | 256 | 1.6795 | +0.0385 |

Even with 128-char slow context, the sequential regime is still worse.

## What this means

The key negative result is stronger than "buffering didn't work": **the sequential training regime is fundamentally incompatible with this lateral mechanism, regardless of context size.**

- In the fixed-window regime, samples are independently drawn. The slow lateral changes meaningfully from sample to sample, so block 0 has to use its **content**.
- In the sequential regime, adjacent positions differ by only one character. With `slow_context=128`, consecutive slow windows overlap by **127/128**. The slow lateral is nearly constant from one position to the next.
- Block 0 therefore experiences the slow input mostly as a **static bias**, not as a position-specific message.

This explains all three observations at once:

1. `direct_ctx32_refresh1` ties baseline instead of helping.
2. `buffered_ctx32_refresh8` gets much worse once that nearly-static signal is also held stale.
3. `direct_ctx128_refresh1` still fails even though 128-char context helped in the window-based regime.

## Regime comparison

| Regime | Slow block context | Slow computation pattern | Sample-to-sample variation seen by block 0 | Outcome |
|---|---:|---|---|---|
| [`tied_readout_lm.py`](../../../runs/tied_readout_lm.py) fixed-window | 128 | Fresh every evaluation position | High: independent sampled windows | **Works** (`Δ=-0.088`) |
| [`multirate_buffer_lm.py`](../../../runs/multirate_buffer_lm.py) sequential direct | 32 / 128 | Fresh every position, but on adjacent windows | Low: adjacent windows heavily overlap | Fails (`Δ=0.000`, then `+0.039`) |
| [`multirate_buffer_lm.py`](../../../runs/multirate_buffer_lm.py) buffered | 32 | Fires every 8 positions, held between | Very low: adjacent windows overlap and held state adds staleness | Catastrophic (`Δ=+0.590`) |

## Implications

1. **The lateral mechanism requires window-based training to work.**
2. **Multi-rate with held output does not work in this form, because it forces sequential processing.**
3. **The only demonstrated working training setup is fresh slow-block computation at each evaluation position.**
4. **"Multi-rate" at inference may still be possible via caching**, but that would be an inference optimization layered on top of a window-trained mechanism, not a sequential-training solution.
5. **This combines with the negative result in [`research/questions/recurrent-lateral/README.md`](../recurrent-lateral/README.md): all tested forms of temporal reuse fail without temporal credit assignment.** Recurrent stale laterals fail; buffered stale laterals fail; the common pattern is that reused temporal state is not being trained to communicate useful position-specific information.

This closes the "context buffer in sequential mode" idea for Pathway 8. Combined with the [recurrent-lateral negative result](../recurrent-lateral/README.md), the pattern is clear: every tested form of temporal reuse of a locally-trained block's output fails. The common cause is lack of temporal credit assignment — the slow block has no gradient signal telling it what would be useful for block 0 in the future. The one thing that works is fresh per-position computation with independent sampling. Multi-rate as an inference-time caching strategy (train window-based, cache at inference) remains untested.

## What this settles

- It rules out this specific "slow block fires periodically and its output is held" training scheme as a useful way to recover the fixed-window lateral benefit.
- It shows that simply giving the slow block more context is not enough once training is sequential.

## What this does not settle

- Whether a window-based training regime plus inference-time caching can approximate multi-rate compute without sacrificing the useful lateral signal.
- Whether temporal reuse could work with temporal credit assignment.
- Whether chunked or buffered schemes could work if the slow block processed genuinely new accumulated context at firing time rather than near-duplicate adjacent windows.
