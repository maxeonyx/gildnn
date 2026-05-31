# Plan

Working notes. Final day (2026-05-31), project concludes at midnight NZST.

---

## Final state (12:20 NZST, May 31)

### CORRECTION: "CE 2.67 purely-local" was mislabeled

The earlier hierarchical-targets result (CE 2.67) used the standard band-0 CE readout — making it a hybrid (CE on band 0, local on bands 1-7). The truly purely-local version gives CE 3.28 at step 100. No purely-local objective has beaten the control (2.70). See corrected daily report.

### Active experiment
`systemctl --user status gildnn-hier-1000` — hierarchical targets purely-local (detached), 1000 steps (~45 min total). At step 100: CE 3.28. Watching for convergence toward or plateau above the control's 2.70 wall.

### Corrected results table

| Experiment | CE | Steps | Notes |
|-----------|-----|-------|-------|
| GRU baseline (batch=32) | **1.35** | 1000 | 820K params, 8s |
| GRU baseline (batch=4) | **1.74** | 1000 | Fair comparison, 11s |
| Transformer (batch=32) | **1.84** | 1000 | 825K params, 8s |
| Transformer (batch=4) | **2.32** | 1000 | Fair comparison, 12s |
| Hier targets hybrid (CE + local) | **~2.67** | 100 | ≈ control, not purely-local |
| Control: our arch, no local loss | **2.70** | 100 | = 1000-step result |
| Our arch (default, 1000 steps) | **2.69** | 1000 | 16M params, 220s |
| Band0-local-loss + CE on band 0 | **3.04** | 100 | Local prediction + CE |
| Band0-local-loss (detached) | **3.24→3.41** | 100→500 | Purely local, CE stagnates |
| Hier targets purely-local (detached) | **3.28** | 100 | Band 0 untrained, 1000-step run active |
| Combined (band0+hier+attn+detach) | **3.35** | 100 | Worse than parts individually |

### Key conclusion (corrected)
No purely-local objective has matched or beaten the control's CE (2.70). All tested purely-local objectives produce CE 3.2-3.4 from a detached readout. The 1000-step run will determine if this is a convergence delay or a fundamental ceiling.

### What remains
- [ ] Wait for 1000-step purely-local hierarchical targets result
- [ ] Run matched detached control (attention-readout + detach-readout, default local loss) for comparison
- [ ] Update weekly with final correction and 1000-step result
- [ ] Final commit

### Architecture spec (reference)

**192-module 2D grid:**
- 8 rate bands × 24 positions = 192 modules
- Topology: 2D grid, horizontal ring (wrap), vertical open boundary
- Neighbors: 4-neighborhood (left, right, up, down)
- Rates by band: `[1, 2, 4, 8, 16, 32, 64, 128]`
- d_stream=96, d_hidden=384 → ~16M params
- Communication: detached noisy neighbor reads from shared buffer
- Token injection: band 0 only
- Output: band 0 CE only (one CE head)
- Local loss: OPEN QUESTION — see `research/questions/local-objectives/README.md`
- Chunk: 128 tokens × 8 steps/token = 1024 microsteps

### Key references
- `VISION.md` — what the model should DO and BE
- `ROADMAP.md` — research pathways
- `PROCESS.md` — how work is done
- `research/weekly/2026-05-31.md` — final project summary
- `research/daily/2026-05-31.md` — today's report
- `research/questions/local-objectives/README.md` — local objectives exploration
- `dictations/` — authoritative source of Max's intent
- `core/automaton_graph.py` — the model
