# Plan

Working notes. Final state as of project end (2026-05-31).

---

## Project status: concluded

The project ran May 20–31, 2026. The architecture is built and individual pieces are validated. The local learning rule — the core thesis — remains an open question.

### What's built
- `core/automaton_graph.py` — 192-module 2D graph cellular automaton (8 bands × 24 positions, d=96, d_hidden=384, ~16M params)
- `core/triton_forward.py` — fused forward kernel, 1.94x speedup (forward only)
- `experiments/automaton_graph/train.py` — training script with streaming TBPTT, checkpointing, multi-scale input, per-band CE, attention readout
- Diagnostics: `lag_probe.py`, `band_similarity.py`, `band_utility.py`, `grad_audit.py`
- Infrastructure: systemd user services, lock files, loop script

### What was validated
| Piece | Result |
|-------|--------|
| Multi-rate execution | 20.7% wall-clock speedup (time/quality tradeoff) |
| Width scaling | 4-block d=256 beats 6-block sequential |
| Detached laterals | 3.85 vs 4.07 nats, strictly better |
| Tempered PoE combining | 26/26 property checks |
| Stateless feedforward blocks | Horizon H=1, H=2 nulls |
| 192-module graph | CE 4.17→2.69 in 1000 steps |
| Triton forward | 1.94x |

### What was NOT solved
The local learning rule. The core open question: what local objectives produce good representations in a multi-module system? (See [dictation 2026-05-31-04](dictations/2026-05-31-04.md) — this is explicitly an exploration space, not a settled architecture.)

Three boundary conditions bracket the answer:
- **InfoNCE** (local neighbor prediction): orthogonal to token prediction (cosine 0.013). Bands 2-7 never specialize.
- **Per-band CE** (each band predicts tokens at own horizon): rapid specialization, but rejected as "unprincipled hack" — broadcasts global objective to every level ([dictation 2026-05-31-01](dictations/2026-05-31-01.md)).
- **Predict-next-inputs** (band 0 predicts neighbor_sum + token_emb): genuinely local, CE 3.24 at 100 steps via detached attention. One candidate, not the answer.
- **Hierarchical targets** (band k predicts mean of band k-1): CE 2.67 at 100 steps. Another candidate.

### Key open question
What local neighborhood objective is both genuinely local AND task-aligned enough to create multi-timescale representations? This is an exploration space — many things should be tried, not one settled on. (See [dictation 2026-05-31-04](dictations/2026-05-31-04.md).)

Candidates explored so far: InfoNCE (failed), predict-next-inputs (promising), hierarchical-targets (promising). Unexplored: delta prediction, mutual information maximization, contrastive neighbor distinction, phase-amplitude coupling.

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
- Local loss: OPEN QUESTION — InfoNCE orthogonal, per-band CE rejected
- Chunk: 128 tokens × 8 steps/token = 1024 microsteps

### Key references
- `VISION.md` — what the model should DO and BE
- `ROADMAP.md` — research pathways
- `PROCESS.md` — how work is done
- `research/weekly/2026-05-31.md` — final project summary
- `dictations/` — authoritative source of Max's intent
- `core/automaton_graph.py` — the model
