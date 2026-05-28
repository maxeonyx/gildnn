# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-28, 16:20 NZST)

**⚠️ DICTATION 15: The model is a cellular automaton.** Blocks are 100% stateless feedforward. No GRU, no carry state, no temporal attention. Information propagates through the grid over many timesteps by bouncing between blocks. ParallelDiagonalModel must be deleted — it implements an RNN, which is wrong.

**Piece tests are done. Build the real thing.**

**~1.5 days remain in timebox. GPU: BUSY (H=4 horizon sweep finishing, PID 19988, irrelevant to new direction but let it complete).**

---

## ⚠️ DICTATION 15 — the architecture

The model is a **cellular automaton / message-passing graph**:

- Each **node** = a combining function
- Each **vertical edge** (through time, in place) = a block — stateless feedforward
- Each **lateral edge** (left-right) = propagation, which adds noise

Information bounces around between blocks over time. Higher-level blocks have context from further back because lower block outputs get merged onto the stream and flow rightwards. Multiple timesteps per token. The noise on laterals creates the need for prediction and the information hierarchy.

**Delete ParallelDiagonalModel.** It's an RNN with GRU and carry state — not Max's architecture. Start from scratch.

**Stop toys.** No more 2-block experiments. No sequence lengths of 4 or 32. Build something that looks like the final model: many blocks, many timescales, optional steps per token, delayed propagation, long sequences.

---

## What to build NOW

A cellular automaton model:
- Grid: [timesteps × blocks]. Many timesteps per token (configurable). Many blocks.
- Nodes: combining function (tempered PoE, validated in piece test 06)
- Vertical edges: stateless feedforward MLP/Linear (a "block")
- Lateral edges: propagation with added noise (validated in piece test 07)
- Grounding: bottom row reads token embeddings. CE loss on bottom block only.
- Prediction: each block predicts the next arrival at its node (W₂² loss, validated in piece test 05)
- Multi-rate: higher blocks fire less often (configurable rates per level)
- Scale: train on TinyShakespeare (100K chars), long sequences (2048+), enough steps to see learning

This should be one clean script. Not a framework. Not abstractions. Just the grid computation, the losses, and training.

---

## What's been validated (still useful for the new model)

| Finding | Evidence | How it informs the new model |
|---|---|---|
| W₂² = MSE on (μ,σ) | Piece test 01, 05 | Use as prediction loss directly |
| Tempered PoE combining | Piece test 06 (26/26 checks) | Use as the node combining function |
| Stream carries actual distributions | Piece test 04 | Nodes carry distributions, surprise is side-channel |
| Noise on laterals forces prediction | Piece test 07 | Lateral propagation adds noise — this IS the mechanism |
| Prediction beats copy 37-43% | 2-block, 3-block experiments | Feedforward prediction works (consistent with stateless blocks!) |
| Recurrence doesn't help | H=1, H=2, H=4 sweep | **EXPECTED** — blocks are stateless. The null confirms the design. |
| Combining modestly helps | 3-block stream combining (Case A) | Combined stream is more useful than raw — combining works |
| Local learning strictly better | A/B test: detach=True wins | Blocks must be independent — consistent with cellular automaton |

---

## What was wrong (delete/ignore)

- `core/model.py` — ParallelDiagonalModel. RNN with GRU. **Delete.**
- All framing around "when does recurrence become load-bearing" — blocks are stateless, recurrence is wrong concept
- The "horizon sweep" tested GRU vs feedforward inside blocks. Answer: feedforward. Which is just... the design.
- Multi-rate as "forcing mechanism for recurrence" — no. Multi-rate creates temporal hierarchy in the GRID, not memory in blocks.

---

## Immediate next steps

1. **Delete ParallelDiagonalModel** — remove `core/model.py` (or gut it)
2. **Build cellular automaton model** — new `core/automaton.py` or similar
3. **Run it at scale** — many blocks (8-16), many timesteps per token (4-16), long sequences (2048), TinyShakespeare
4. **Daily + weekly reports** (due after 4pm today) — use back-chain format from dictation 15

---

## Report format (dictation 15)

Back-chain from the goal:
- Why is the model not yet doing the full architecture?
- If it did the full architecture now, what would happen?
- Why would that result be indeterminate or built on shaky foundations?
- Therefore it's doing this step → therefore that step

At every step, justify why we're not just running the real thing. If you can't justify it, run the real thing.

---

## Key references

- `dictations/2026-05-28-15.md` — the cellular automaton clarification
- `dictations/2026-05-27-10.md` — the predictive processing course correction
- `dictations/2026-05-27-11.md` — stream accumulation
- `research/questions/stream-combining/README.md` — combining validated
- `research/questions/horizon-sweep/README.md` — recurrence null (confirms stateless is correct)
- `ROADMAP.md` Pathway 3 — local learning via distributional predictive coding
