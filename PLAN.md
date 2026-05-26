# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current operational state (2026-05-27, 06:50 NZST)

**GPU: FREE.**

**Pathway 5: RESOLVED this session.** Full arc:
- Oracle measurements → binary BCE halt head (FAIL) → regression MSE halt head (SUCCESS at d=128)
- Scale-up to d=256/50K: Pearson scales (0.561) but strict-threshold speedup doesn't (calibration mismatch)
- Calibration experiment: affine post-hoc fix → ε=0.02 speedup jumps from 1.30 to 1.39, beating fixed-6
- **Integrated into `core/recurrent_depth.py`** — tested, all 5 piece tests pass

**Daily report 2026-05-27:** NOT YET WRITTEN (due after 4pm). Today's report covers the full Pathway 5 resolution — the most significant experimental arc of the project so far.

---

## What's next

1. ~~Integration~~ **DONE** — `core/recurrent_depth.py` with `SharedRecurrentCore`, `HaltHead`, `RecurrentDepthLM`, calibration support
2. **Choose next research direction** — see "Open directions" below
3. **Daily report** (after 4pm)
4. **Process note:** Consider making `--save-checkpoint` default for future experiment runs (calibration experiment was initially blocked by missing checkpoint)

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
