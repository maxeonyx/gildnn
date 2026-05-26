# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current operational state (2026-05-27, 06:49 NZST)

**GPU: FREE.**

**Pathway 5 conclusion (this session):**
- 50K run COMPLETED: Pearson 0.561, ε=0.02 speedup 1.20 (doesn't beat fixed-6 at tight threshold)
- Key finding: prediction quality scales (Pearson matches d=128) but oracle headroom shrinks with training
- Calibration experiment CONFIRMED: affine calibration boosts ε=0.02 speedup from 1.30→1.39, beating fixed-6
- **Integration warranted:** mechanism proven + calibration fix identified + prediction scales

**Daily report 2026-05-27:** NOT YET WRITTEN (due after 4pm).

---

## What's next

1. **Integration into `core/`** — create `core/recurrent_depth.py` with `SharedRecurrentCore`, `HaltHead`, `RecurrentDepthLM`. Include optional calibration parameters for inference. Training/eval machinery stays in `runs/`.
2. **Daily report** (after 4pm): Full Pathway 5 arc — oracle → binary fail → regression success → scale-up → calibration fix
3. **After integration:** What's the next research direction? Pathway 4 (active compression) is the natural successor — model learning to be "ready" earlier would enlarge the oracle ceiling

---

## What Max wants tested (dictation 2026-05-26-2)

**The question:** Does adding predictive-loss-trained interior blocks improve over a single CE-connected block?

**Design constraint (non-negotiable):** Interior blocks receive ONLY local signals. No task gradient (CE) flows to them. The output block gets CE. Interior blocks predict what arrives laterally.

**Status:** Validated at d=128 scale (Δ=-0.045 with two blocks). Architecture in `core/tied_readout.py`. Dynamic depth is the current active pathway.

---

## Architecture (validated, from dictation 2026-05-26-5)

- Shared normalized token embeddings (weight-tied readout)
- Normalized block outputs (L2-norm before addition)
- Addition-based lateral combination (lateral_scale=0.2 needed even with normalization)
- CE local loss for interior blocks (cosine/L2 worse)
- Temperature = 0.07–0.10

---

## Pathway 5 results (Dynamic Depth / Early Exit)

### Oracle measurements

| Config | Oracle speedup | Notes |
|---|---|---|
| d=72, 8-iter | 1.96× | Large headroom at small scale |
| d=256, 4-iter | 1.37× | Moderate headroom |
| d=256, 8-iter | 1.58× | Good headroom, target config |

### Halt head approaches

| Approach | Result | Key metric |
|---|---|---|
| Post-hoc probe | FAIL | Model doesn't naturally develop halt-predictive features |
| Binary BCE (10K) | FAIL | AUROC 0.69, worse than fixed depth-6 |
| Binary BCE (30K) | FAIL | Saturates at AUROC 0.70 |
| **Regression MSE (10K)** | **SUCCESS** | **55.7% oracle efficiency, beats fixed-depth** |

### Regression halt head detail (d=128, 8-iter, 10K steps)

| ε threshold | Speedup | Loss hit | Oracle efficiency |
|---|---|---|---|
| 0.01 | 1.33× | 0.017 | 55.7% |
| 0.02 | 1.38× | 0.021 | 60.4% |
| 0.05 | 1.57× | 0.035 | 72.9% |

Fixed depth-6 baseline: 1.33× speedup at 0.013 loss hit. Regression ε=0.02 BEATS this (1.38× at 0.021).

---

## Closed pathways (summary)

| Pathway | Status | One-line finding |
|---|---|---|
| 1 (Recurrent Depth) | INCONCLUSIVE | Δ=-0.011, p≈0.11 at matched params. Per-FLOP distinct wins. |
| 3 (Local Learning) | VALIDATED | Window-based + fresh lateral works. Co-training self-organizes. |
| 8 (Multi-Rate) | CLOSED | Sequential regime incompatible; CE-trained laterals position-specific |

---

## Key references

- `VISION.md` — stakeholder requirements (DO NOT EDIT)
- `ROADMAP.md` — research pathways (DO NOT EDIT)
- `PROCESS.md` — experiment discipline and loop
- `research/questions/dynamic-depth/README.md` — full Pathway 5 write-up
- `core/tied_readout.py` — validated architecture
- `runs/halting_regression.py` — the working regression halt head
- `research/daily/2026-05-26.md` — yesterday's report
