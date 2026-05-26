# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## ⚠️ PRIORITY REDIRECT: Long-sequence RNN testing (dictation 2026-05-27-1)

**All experiments so far use wrong sequence lengths.** `SHORT_CONTEXT = 4` everywhere. The architecture is an RNN — per-step cost is O(weights × batch), independent of history length. We should be testing with **thousands of tokens**.

**The fundamental question (Pathway 1) has never been tested at the RNN's sweet spot.** We've been hobbling the RNN to match the transformer's interface.

**What must change:**
1. Implement truncated BPTT: forward through entire long sequence (1000+ tokens), backprop through K steps (32-64)
2. Test whether the architecture leverages temporal structure at realistic sequence lengths
3. Re-evaluate stale lateral / recurrence negatives — those were all with 4-char windows and may be artifacts
4. Correct comparison: transformer at 128-256 token attention vs RNN at thousands of tokens, matched params

**This is the priority.** Pathway 5 (halting/dynamic depth) work is paused. Grammar experiment is paused. The next experiment must be long-sequence RNN unrolling with truncated BPTT.

**d=512 run:** Crashed at step 15000/20000 (no checkpoint saved). Do NOT restart it — wrong priority.

**GPU: FREE** (crash killed all processes, lock file stale).

---

## What's done (prior sessions, still valid)

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

**~3 days remain in timebox. GPU: FREE.**

### THE PRIORITY: Long-sequence truncated BPTT experiment

Design and run an experiment that:
1. Processes sequences of 1000+ tokens (character-level, WikiText-103)
2. Uses truncated BPTT (backprop through K=32-64 steps, forward through full sequence)
3. Tests whether recurrent hidden state accumulates useful temporal information
4. Compares against transformer baseline at matched params (transformer gets 128-256 token attention window)

This is Pathway 1 (Wide Recurrent vs Deep Transformer) tested properly for the first time.

### After that: re-evaluate recurrence/lateral negatives

If long-sequence RNN works, revisit stale laterals and multi-rate at realistic sequence lengths. The previous negatives may be artifacts of 4-char windows.

### Paused (valid work, lower priority now)

- Grammar experiment (infrastructure ready, `runs/grammar_train.py`)
- d=512 capstone (crashed, no checkpoint — would need full restart)
- Pathway 4 active compression

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
