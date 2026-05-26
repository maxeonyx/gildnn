# Question: Capstone Generation — Recurrent Depth with Dynamic Halting

## What this asks

Can the recurrent depth model with dynamic halting produce coherent, readable English text at a scale where the outputs are interesting — and does the halting mechanism allocate depth meaningfully (more thinking on hard tokens, less on easy ones)?

## Which vision goals this serves

- **Interactive inference** (VISION.md) — "Type at it, see what it generates"
- **Dynamic computation** (Pathway 5, now resolved mechanistically)
- **Wide recurrent architecture** (Pathway 1 — THE thesis: width + recurrence substitutes for depth)

## Why this experiment exists

Pathway 5 validated the halting mechanism at d=128/256 on TinyShakespeare. But TinyShakespeare is ~1M chars — too small for interesting generation. This experiment scales to WikiText-103 (100M+ chars) where the model can learn diverse English patterns, and asks: at what scale does the model produce text worth reading?

## The simplification

Character-level, single shared block (not multi-rate). No transformer comparison (that's a separate experiment). The comparison is dynamic-halting vs fixed-depth at the same architecture.

## Hypotheses

1. **d=256 on WikiText-103/20K steps** produces noticeably better text than d=256 on TinyShakespeare (dataset quality matters more than model size at this scale)
2. **Halting depth varies meaningfully by token type** — punctuation, rare chars, and word boundaries get more depth; common letter sequences get less
3. **Dynamic halting matches fixed-depth quality** while using fewer average iterations (speedup without quality loss)

## Planned evidence

| Artifact | What it shows |
|---|---|
| Generated text samples (temperature 0.8) | Coherence level at each scale |
| Per-token depth histogram | Whether halting varies or is mostly constant |
| Depth-annotated samples | Which specific tokens trigger deeper thinking |
| Speed comparison (dynamic vs fixed-8) | Computational savings from halting |

## Ladder position

- ✅ Overfit one batch (done in d=128 sessions)
- ✅ Tiny model, tiny data (d=128/10K on TinyShakespeare — Pearson 0.572)
- ✅ Inspect actual outputs (d=256/TinyShakespeare — semi-coherent Shakespeare)
- → **Scale in steps** (d=256 on WikiText-103, then d=512 if needed)

## Experiment design

**Phase 1 (cheapest test):** d=256, context=256, n_heads=4, ff_dim=1024, iterations=8, WikiText-103 char-level, 20K steps. ~15-20 min.
- Exit: if generated text shows coherent English words/phrases → proceed to depth analysis
- Exit: if still gibberish → scale to 50K steps or d=512

**Phase 2 (if phase 1 succeeds):** Generate depth-annotated samples. Measure speed. Write the capstone report.

**Phase 3 (stretch):** d=512 if d=256 isn't interesting enough. Or more steps.

## Non-goals

- Beating a transformer baseline (separate question)
- Multi-rate / local learning (different architecture)
- Academic-scale training
- Novel sampling strategies

## Results

_To be filled as experiments run._
