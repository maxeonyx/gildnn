# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## ⚠️ PRIORITY REDIRECT: Assemble the actual architecture (dictation 2026-05-27-2)

**Stop training models that aren't Max's.** A flat GRU with TBPTT is not his model. A standard transformer is not his model. Testing those tells us nothing about whether his design works.

**Assemble the pieces now.** The individual pieces have been poked at for days. Put them together. If a piece turns out broken during assembly, fix it then. Stop endlessly testing pieces in isolation against standard baselines.

**What "sanity check a piece" means:** Does it learn at all? Is the loss going down? Is the architecture not completely broken? That takes 30-60 seconds. NOT training to convergence and analyzing the result.

### Max's architecture (from dictation)

1. Multi-block grid — `ParallelDiagonalModel` in `core/model.py` ✅
2. Blocks as temporal edges — stale laterals with one-timestep delay ✅
3. Stale laterals — `detach_lateral=True`, topology="upward" ✅
4. Noise creating information hierarchy — ❌ NOT IMPLEMENTED
5. Higher blocks predict further ahead with local predictive loss — ❌ KEY MISSING PIECE
6. Distributional stream with weight-tied readout as combining mechanism — partially (tied readout exists, distributional stream doesn't)

### What "assembly" means concretely

Use `ParallelDiagonalModel(num_blocks=3, rates=(1,2,4), topology="upward", detach_lateral=True)` with:
- Per-block local loss: block i at rate R predicts R tokens ahead using `tied_logits`
- Long-context sequential training with TBPTT (ctx=128+ minimum, seq=2048)
- Optionally: noise injection per block level (higher blocks see more noise → forced to be robust)

The TBPTT infrastructure just built (`runs/rnn_tbptt.py`) is useful for the training loop — it just needs to drive `ParallelDiagonalModel` instead of a flat GRU.

---

## What's done this session

1. ~~GRU TBPTT experiment (Rung 1 on TinyShakespeare)~~ **DONE** — state carry provides zero benefit
2. ~~Reset-sweep evaluation~~ **DONE** — confirms no useful state beyond 128 tokens for flat GRU
3. ~~WikiText-103 script built~~ **DONE** — `runs/rnn_tbptt_wiki.py` (sanity-checked but NOT run — redirected by dictation)
4. ~~Rung 1 results documented~~ **DONE** — `research/questions/long-sequence-rnn/README.md`

---

## What's next

**~3 days remain in timebox. GPU: FREE.**

### THE PRIORITY: Assemble and train Max's architecture at long sequences

Build a training script that:
1. Uses `ParallelDiagonalModel` (multi-block grid, stale laterals, rates)
2. Trains with TBPTT on long sequences (reuse infrastructure from `runs/rnn_tbptt.py`)
3. Gives each block a local CE loss predicting `rate` steps ahead
4. Quick sanity check (30-60 seconds): does loss go down?
5. If yes: scale up and run properly

### After: noise injection, distributional stream (if time permits)

---

## Architecture (validated, from dictation 2026-05-26-5)

- Shared normalized token embeddings (weight-tied readout)
- Normalized block outputs (L2-norm before addition)
- Addition-based lateral combination (lateral_scale=0.2 needed even with normalization)
- CE local loss for interior blocks (cosine/L2 worse)
- Temperature = 0.07–0.10

---

## Closed pathways (summary)

| Pathway | Status | One-line finding |
|---|---|---|
| 1 (Recurrent Depth) | VALIDATED | Shared-weight iteration works. Capstone generation demonstrates at scale. |
| 3 (Local Learning) | VALIDATED | Window-based + fresh lateral works. Co-training self-organizes. |
| 5 (Dynamic Depth) | **RESOLVED** | Regression halt head works. Calibrated. Integrated. Generates readable text. |
| 8 (Multi-Rate) | CLOSED (at 4-char context) | Sequential regime incompatible at ctx=4. **May work at long sequences (re-evaluate during assembly).** |

---

## Key references

- `VISION.md` — stakeholder requirements (DO NOT EDIT)
- `ROADMAP.md` — research pathways (DO NOT EDIT)
- `PROCESS.md` — experiment discipline and loop
- `dictations/2026-05-27-2.md` — THE directive: assemble now
- `dictations/2026-05-27-1.md` — use long sequences, TBPTT
- `core/model.py` — `ParallelDiagonalModel` (the grid architecture)
- `core/tied_readout.py` — weight-tied readout + local loss
- `runs/rnn_tbptt.py` — TBPTT training infrastructure (reusable)
- `research/questions/long-sequence-rnn/README.md` — Rung 1 flat-GRU results (negative)
