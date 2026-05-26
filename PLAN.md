# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current operational state (2026-05-27, 05:30 NZST)

**GPU: BUSY.** Scale-up experiment running (PID 12060, d=256/8-iter/20K steps, ETA ~05:50).

**Active run:**
- Script: `runs/halting_regression.py --d-model 256 --steps 20000`
- Log: `experiments/tinyshakespeare/artifacts/halting_regression_d256/run.jsonl`
- Report: `experiments/tinyshakespeare/artifacts/halting_regression_d256/report.json`
- Lock: `runs/active.lock`

**Key discovery this session:** Regression halt head (predict remaining gain as continuous value) dramatically outperforms binary classification (predict safe/not-safe). Same model, same architecture, same training time — only the loss matters. 55.7% oracle efficiency at d=128/8-iter/10K steps.

**Daily report 2026-05-27:** NOT YET WRITTEN (due after 4pm).

---

## What's next

1. **Check d=256 scale-up results** — does regression halting improve at larger scale? (Oracle ceiling is 1.58× there)
2. **If yes:** Integration into `core/` as a first-class architecture feature
3. **Daily report** (after 4pm): Pathway 1 resolution → direction switch → oracle → binary fail → regression success → scale-up
4. **Open question:** What does learned halting mean for the vision? The model can learn "am I done?" — this connects to adaptive computation and efficient inference

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
