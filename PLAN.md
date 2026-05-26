# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current operational state (2026-05-27, 09:10 NZST)

**GPU: BUSY** — d=512 capstone training (PID 18932/6364, `experiments/capstone_d512/`)
- Config: d=512, ctx=256, iter=8, n_heads=8, ff=2048, WikiText-103, 20K steps
- Parameters: 5,881,537 (2.7× larger than d=256)
- Progress at step 11000: val_loss=1.74, avg_depth=6.79
- Estimated finish: ~09:46 NZST
- **Question:** Does halting depth variation increase with model size? Does text quality become sentence-level coherent?

**Capstone d=256: COMPLETE.** Full report: `research/questions/capstone-generation/README.md`

**Pathway 4 opportunity: MEASURED.** Per-depth analysis shows 95% of quality achieved by depth 4 (only 0.056 nats lost vs full depth 8). Report: `research/questions/computation-compression/README.md`

**Grammar experiment: INFRASTRUCTURE READY + REPORT-FIRST DONE.** Question README: `research/questions/grammar-depth/README.md`

**Daily report 2026-05-27:** NOT YET WRITTEN (due after 4pm). Today's content: d=512 scale-up + grammar experiment + Pathway 4 measurement.

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

**GPU busy with d=512 until ~09:30. ~3 days remain.**

### Immediate

- Wait for d=512 → generate samples → compare to d=256 → update capstone report
- **Daily report** (after 4pm)

### Options for remaining time (after d=512)

**A. Scale to d=512 or more steps** — ALREADY RUNNING.

**B. Pathway 4 (Active Compression)** — Opportunity confirmed: only 0.056 nats lost by stopping at depth 4. Cheapest test: explicit self-prediction head at depth 2 predicting depth 8's output. Does it improve depth-2 quality? See `research/questions/computation-compression/README.md`.

**C. Synthetic grammar task** — ✅ READY TO RUN. Infrastructure built + report-first done.
  - Data: `data/grammar/` (typed recursive brackets, 2M chars, max depth 5) + `data/grammar-flat/` (flat control)
  - Training: `runs/grammar_train.py` (d=64, ctx=64, iter=8, 5K steps default)
  - Analysis: `experiments/grammar_depth/analyze_depth.py` (teacher-forced depth-by-nesting table)
  - **Hypothesis:** Closing delimiters at deeper nesting use more halting depth.
  - **Baseline:** Flat control (same chars, depth-1 only)
  - **Exit:** Closers show increasing halt depth with nesting depth AND flat control doesn't show same pattern.

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
- `research/questions/capstone-generation/README.md` — capstone result
- `research/questions/grammar-depth/README.md` — grammar experiment design (report-first)
- `research/questions/computation-compression/README.md` — Pathway 4 opportunity measurement
- `research/questions/dynamic-depth/README.md` — full Pathway 5 write-up
- `core/tied_readout.py` — validated multi-rate lateral architecture
- `core/recurrent_depth.py` — validated recurrent depth + halting architecture
- `core/generation.py` — text generation with halting annotations
- `runs/capstone_train.py` — WikiText-103 training script
- `runs/grammar_train.py` — grammar training script
- `runs/generate_text.py` — generation CLI
- `research/daily/2026-05-26.md` — yesterday's report
