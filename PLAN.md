# Plan

Working notes. Final day (2026-05-31), project concludes at midnight NZST.

---

## Current state (11:55 NZST, May 31)

### Queue running
`systemctl --user status gildnn-ablation-queue` — auto-draining experiments. Check `runs/queue-logs/` for results.

Remaining in queue:
1. Combined: `--band0-local-loss --hierarchical-targets --attention-readout --detach-readout` (100 steps) — RUNNING NOW
2. Transformer baseline batch_size=4 (1000 steps)
3. GRU baseline batch_size=4 (1000 steps)
4. Control: default config, no local loss (100 steps)

### Results collected today

| Experiment | CE | Steps | Notes |
|-----------|-----|-------|-------|
| GRU baseline (batch=32) | **1.35** | 1000 | 820K params, 8s |
| Transformer baseline (batch=32) | **1.84** | 1000 | 825K params, 8s |
| Our arch (default) | **2.69** | 1000 | 16M params, 220s |
| Hierarchical targets (detached) | **2.67** | 100 | Best purely-local |
| Band0-local-loss + CE on band 0 | **3.04** | 100 | Local prediction + CE |
| Band0-local-loss (detached) | **3.24→3.41** | 100→500 | CE stagnates while prediction improves |
| Per-band CE + attention | **3.03** | 2000 | Rejected (not local) |

### Key insight
Per dictation 07: "Aux loss numbers are completely meaningless. The question is whether it's the right incentive, not whether it's learning well given its incentive." Stop reporting prediction loss.

### What's done
- [x] Queue mechanism (`experiments/queue_runner.py`)
- [x] Baselines (transformer, GRU)
- [x] Band0-local-loss implementation
- [x] 500-step detached result (negative: CE stagnates)
- [x] Horizontal vs vertical coupling insight
- [x] Daily report updated
- [x] Weekly report updated

### What remains
- [ ] Collect remaining queue results (combined, batch-4 baselines, control)
- [ ] Update daily/weekly with final results
- [ ] Final commit and push
- [ ] Pretrained embeddings: Max asked about it. For char-level with 65 tokens, the answer is "doesn't apply cleanly" — see daily report.

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
