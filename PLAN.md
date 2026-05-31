# Plan

Working notes. Final day (2026-05-31), project concludes at midnight NZST.

---

## Final state (12:02 NZST, May 31)

### All experiments complete

Queue drained successfully. Full results:

| Experiment | CE | Steps | Notes |
|-----------|-----|-------|-------|
| GRU baseline (batch=32) | **1.35** | 1000 | 820K params, 8s |
| GRU baseline (batch=4) | **1.74** | 1000 | Fair comparison, 11s |
| Transformer (batch=32) | **1.84** | 1000 | 825K params, 8s |
| Transformer (batch=4) | **2.32** | 1000 | Fair comparison, 12s |
| Hierarchical targets (detached) | **2.67** | 100 | Best purely-local |
| Control: our arch, no local loss | **2.70** | 100 | Matches 1000-step result |
| Our arch (default, 1000 steps) | **2.69** | 1000 | 16M params, 220s |
| Band0-local-loss + CE on band 0 | **3.04** | 100 | Local prediction + CE |
| Band0-local-loss (detached) | **3.24→3.41** | 100→500 | CE stagnates while prediction improves |
| Combined (band0+hier+attn+detach) | **3.35** | 100 | Worse than parts individually |
| Per-band CE + attention | **3.03** | 2000 | Rejected (not local) |

### Key findings from final queue
1. **Batch-4 baselines close the fairness argument**: GRU at batch=4 still 1.74, nearly a full nat better than any config of our architecture
2. **Combined features hurt**: adding everything together (3.35) is worse than hierarchical-targets alone (2.67). The detached readout can't exploit cascading representations.
3. **Architecture hits a wall early**: control at 100 steps (2.70) ≈ 1000 steps (2.69) — diminishing returns after step ~50

### Project complete
All planned experiments run. Reports updated. See `research/weekly/2026-05-31.md` for the final project summary.

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
