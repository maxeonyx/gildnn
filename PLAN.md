# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## ⚠️ PRIORITY REDIRECT: Long-sequence RNN testing (dictation 2026-05-27-1)

**All experiments so far use wrong sequence lengths.** `SHORT_CONTEXT = 4` everywhere. The architecture is an RNN — per-step cost is O(weights × batch), independent of history length. We should be testing with **thousands of tokens**.

**The fundamental question (Pathway 1) has never been tested at the RNN's sweet spot.** We've been hobbling the RNN to match the transformer's interface.

**The direction:** Test the architecture at its natural sweet spot — long sequences where hidden state accumulates temporal structure. Truncated BPTT is the expected training mechanism (forward through full sequence, backprop through a limited window). The RNN advantage over transformers should show up here if it exists.

Previous negatives about recurrence/stale laterals were all tested with 4-char windows. They may be artifacts of having no temporal structure to leverage.

**This is the priority.** Pathway 5 (halting/dynamic depth) and grammar work are paused.

**d=512 run:** Crashed at step 15000/20000 (no checkpoint saved). Do NOT restart — wrong priority.

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

### THE PRIORITY: Test the architecture with long sequences

The architecture is an RNN. Its advantage over transformers is that it can process arbitrarily long sequences without quadratic cost. We have never tested this. All prior experiments used tiny context windows (4 chars). This is Pathway 1 tested properly for the first time.

### After that: re-evaluate recurrence/lateral negatives

The previous negatives (stale laterals, multi-rate, sequential regime) may be artifacts of 4-char windows. Revisit at realistic sequence lengths if the long-sequence direction works.

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
| 8 (Multi-Rate) | CLOSED (at 4-char context) | Sequential regime incompatible; CE-trained laterals position-specific. **May need re-evaluation at long sequences.** |

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
