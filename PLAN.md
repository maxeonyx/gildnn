# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current operational state (2026-05-27, 08:30 NZST)

**GPU: BUSY** — d=512 capstone training (PID 18932/6364, `experiments/capstone_d512/`)
- Config: d=512, ctx=256, iter=8, n_heads=8, ff=2048, WikiText-103, 20K steps
- Parameters: 5,881,537 (2.7× larger than d=256)
- Progress at step 1000: val_loss=2.33, avg_depth=7.18 (expected early)
- Estimated finish: ~09:30 NZST (~70 min total)
- **Question:** Does halting depth variation increase with model size? Does text quality become sentence-level coherent?

**Capstone d=256: COMPLETE.** Full report: `research/questions/capstone-generation/README.md`

**Daily report 2026-05-27:** NOT YET WRITTEN (due after 4pm). Today's content: Pathway 5 resolution + capstone generation + d=512 scale-up.

---

## What's done today

1. ~~Integration~~ **DONE** — `core/recurrent_depth.py`
2. ~~Choose direction~~ **DONE** — Capstone Generation
3. ~~Generation infrastructure~~ **DONE** — `core/generation.py`, `runs/generate_text.py`
4. ~~--save-checkpoint default~~ **DONE**
5. ~~Capstone training d=256~~ **DONE** — 20K steps, 35 min, val_loss 1.62
6. ~~Generation samples~~ **DONE** — halting patterns confirmed interpretable
7. ~~Speed comparison~~ **DONE** — 10% real-time speedup (memory-bound regime)
8. ~~Capstone report~~ **DONE** — `research/questions/capstone-generation/README.md`
9. ~~Fix bugs~~ **DONE** — checkpoint format compat, report-path dir handling, main() def
10. **ACTIVE: d=512 scale-up** — launched, ETA 09:30

---

## What's next

**GPU is free. ~3 days remain.**

### Immediate

- **Daily report** (after 4pm)

### Options for remaining time

**A. Scale to d=512 or more steps** — the capstone text is word-level coherent but not sentence-level. A bigger model or more training would produce genuinely interesting text for interactive use. Cost: 2-4 hours at d=512, or ~25 min for 50K more steps at d=256.

**B. Pathway 4 (Active Compression)** — Can the recurrent depth model learn to frontload computation (make early depths more informative)? Would enlarge the oracle ceiling. Natural successor experiment.

**C. Synthetic task validation** — dictation 2026-05-26-3 says "run whatever architecture across multiple datasets and tasks." Test the recurrent depth model on a CFG/grammar task where depth allocation might be even more interpretable (strict nesting → depth should correlate with nesting level).

**D. Process / cleanup** — Any remaining stale docs, missing baselines, or infrastructure improvements.

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
| 8 (Multi-Rate) | CLOSED | Sequential regime incompatible; CE-trained laterals position-specific |

---

## Key references

- `VISION.md` — stakeholder requirements (DO NOT EDIT)
- `ROADMAP.md` — research pathways (DO NOT EDIT)
- `PROCESS.md` — experiment discipline and loop
- `research/questions/capstone-generation/README.md` — **NEW** capstone result
- `research/questions/dynamic-depth/README.md` — full Pathway 5 write-up
- `core/tied_readout.py` — validated multi-rate lateral architecture
- `core/recurrent_depth.py` — validated recurrent depth + halting architecture
- `core/generation.py` — text generation with halting annotations
- `runs/capstone_train.py` — WikiText-103 training script
- `runs/generate_text.py` — generation CLI
- `research/daily/2026-05-26.md` — yesterday's report
