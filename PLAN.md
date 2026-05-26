# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current operational state (2026-05-27, 07:35 NZST)

**GPU: BUSY** — capstone training (PID 17016/23076, `experiments/capstone_generation/`)
- Config: d=256, ctx=256, iter=8, WikiText-103, 20K steps
- Progress at step 1000: loss=2.22, avg_depth=6.68/8 (halting already active)
- Estimated finish: ~08:15 NZST

**Direction chosen: E (Capstone Generation)** — scale the recurrent depth model to produce readable text with visible halting behavior. Connects to Pathways 1+5.

**Daily report 2026-05-27:** NOT YET WRITTEN (due after 4pm). Today's report covers Pathway 5 resolution + capstone launch.

---

## What's next

1. ~~Integration~~ **DONE**
2. ~~Choose direction~~ **DONE** — Capstone Generation (option E)
3. ~~Generation infrastructure~~ **DONE** — `core/generation.py`, `runs/generate_text.py`
4. ~~--save-checkpoint default~~ **DONE** — now defaults to True
5. **ACTIVE: Capstone training** — WikiText-103, d=256, 20K steps
6. **After training:** Generate samples with halting annotations, write capstone report
7. **If text quality insufficient:** Scale to d=512 or more steps
8. **Daily report** (after 4pm)

---

## Open directions (post Pathway 5)

The two validated architectures serve different vision requirements:

| Architecture | Sequential inference? | Multi-rate? | Dynamic depth? | Local learning? |
|---|---|---|---|---|
| `core/tied_readout.py` (multi-rate laterals) | ❌ window-only | ✓ | ❌ | ✓ |
| `core/recurrent_depth.py` (shared-weight iteration) | ✓ | ❌ | ✓ | ❌ |

Neither alone satisfies the full vision. Options:

**A. Pathway 4 (Active Compression)** — natural successor to Pathway 5. Can the recurrent depth model learn to frontload computation, making early depths more informative? This enlarges the oracle ceiling and compounds with the halt head. Operates on the sequential model.

**B. Sequential inference for multi-rate laterals** — the deeper problem. Window-trained laterals collapse in sequential mode (K=2 → +0.43 loss). Solving this would unblock the full vision architecture. Requires fundamentally different local objectives (not CE per-position).

**C. Combining both architectures** — can the recurrent depth model have multi-rate lateral inputs at each depth? Or: can the lateral model iterate its blocks? These are unexplored compositions.

**D. Scaling the recurrent depth model** — bigger dataset, bigger model, interactive demo. The mechanism works and is integrated; how does it perform at a scale where it generates interesting text?

---

## What Max wants tested (dictation 2026-05-26-2)

**The question:** Does adding predictive-loss-trained interior blocks improve over a single CE-connected block?

**Status:** VALIDATED at d=128 scale (Δ=-0.045 with two blocks). Architecture in `core/tied_readout.py`. The multi-rate architecture works with window-based training; Pathway 8 closure means sequential inference remains an open problem.

---

## Architecture (validated, from dictation 2026-05-26-5)

- Shared normalized token embeddings (weight-tied readout)
- Normalized block outputs (L2-norm before addition)
- Addition-based lateral combination (lateral_scale=0.2 needed even with normalization)
- CE local loss for interior blocks (cosine/L2 worse)
- Temperature = 0.07–0.10

---

## Pathway 5 results (Dynamic Depth / Early Exit) — RESOLVED

### Summary

Regression halt head (predict remaining loss gain) achieves dynamic early exit beating fixed-depth baselines. Key numbers:
- d=128/10K: Pearson 0.572, ε=0.02 speedup 1.38×, oracle efficiency 60%
- d=256/50K: Pearson 0.561, needs affine calibration to beat fixed-6
- After calibration: ε=0.02 speedup 1.39× at loss 0.014 (vs fixed-6: 1.33× at 0.02)

Integrated as `core/recurrent_depth.py`. Full write-up: `research/questions/dynamic-depth/README.md`

---

## Closed pathways (summary)

| Pathway | Status | One-line finding |
|---|---|---|
| 1 (Recurrent Depth) | INCONCLUSIVE | Δ=-0.011, p≈0.11 at matched params. Per-FLOP distinct wins. |
| 3 (Local Learning) | VALIDATED | Window-based + fresh lateral works. Co-training self-organizes. |
| 5 (Dynamic Depth) | **RESOLVED** | Regression halt head works. Calibration fixes scale gap. Integrated. |
| 8 (Multi-Rate) | CLOSED | Sequential regime incompatible; CE-trained laterals position-specific |

---

## Key references

- `VISION.md` — stakeholder requirements (DO NOT EDIT)
- `ROADMAP.md` — research pathways (DO NOT EDIT)
- `PROCESS.md` — experiment discipline and loop
- `research/questions/dynamic-depth/README.md` — full Pathway 5 write-up
- `core/tied_readout.py` — validated multi-rate lateral architecture
- `core/recurrent_depth.py` — validated recurrent depth + halting architecture
- `runs/halting_regression.py` — regression halt head training script
- `runs/calibration_check.py` — post-training calibration analysis
- `research/daily/2026-05-26.md` — yesterday's report
