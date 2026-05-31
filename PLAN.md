# Plan

Working notes. Final day (2026-05-31), project concludes at midnight NZST.

---

## Final state (13:09 NZST, May 31)

### Baselines — DONE
Dictation-09 asks for baselines. They were already run at 11:48 today (before the dictation was written). Evidence: `runs/queue-logs/20260531-114828-968102-ablation-queue.log` (transformer, 1.84) and `runs/queue-logs/20260531-114840-371122-ablation-queue.log` (GRU, 1.35). Both scripts live in `experiments/baselines/`.

### CORRECTION: "CE 2.67 purely-local" was mislabeled

The earlier hierarchical-targets result (CE 2.67) used the standard band-0 CE readout — making it a hybrid (CE on band 0, local on bands 1-7). The truly purely-local version gives CE 3.28 at step 100. No purely-local objective has beaten the control (2.70). See corrected daily report.

### Active experiment — COMPLETED
Hierarchical targets purely-local (detached), 1000 steps. **Result: CE 3.28 → 3.52 (WORSE over training).**

Local prediction learning actively hurts token classification. As modules specialize for inter-band prediction, they move AWAY from representations the attention head can exploit. The ~3.28 at step 100 was incidental token info from initialization; training destroys it.

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
| Hier targets purely-local (detached) | **3.28→3.52** | 100→1000 | Worsens with training |
| Combined (band0+hier+attn+detach) | **3.35** | 100 | Worse than parts individually |

### Key conclusion (scoped)
**The tested local objectives (predict neighbors/lower-bands/inputs) are anti-correlated with token classification on this architecture.** All tested purely-local objectives produce CE 3.2-3.5 from a detached readout, and CE WORSENS over training (3.28 → 3.52 at 1000 steps). The modules learn their local tasks well (prediction loss 4.17 → 1.97) while becoming less useful for token prediction.

**Scope of this result:** Only 3 local objectives tested, all from the same "predict neighbor state" family. One architecture, one communication structure (detached laterals). This does NOT prove local learning is impossible — it shows this specific family doesn't work here. Different objective classes or different architectures remain untested.

### What remains
- [x] 1000-step purely-local hierarchical targets result — DONE (negative: CE worsens)
- [x] Update daily report with 1000-step result
- [x] Update weekly with definitive negative conclusion
- [x] Theoretical analysis: formal argument for why tested objective family fails
- [x] Final commit

---

## If this project is picked up again

### The gating question (answer this FIRST)

**"With ordinary global training, does this architecture family buy any real advantage that a much simpler recurrent/transformer baseline does not?"**

Concretely: can a globally-trained multi-rate 192-module graph show a quality/compute or dynamic-depth advantage at matched wall-clock or FLOPs vs a GRU/transformer? If no, the 192-module graph should be demoted from "core architecture" to "interesting failed branch."

### Pathway priority (after local-learning negative results)

1. **Pathway 1 (Wide Recurrent vs Deep Transformer)** — the fundamental thesis. Never directly compared on this architecture. If weight-tied recurrence + width can't buy a real advantage under ordinary global training, nothing else matters.

2. **Pathway 5 (Dynamic Depth / Early Exit)** — strongest near-term value story. If different tokens genuinely need different iteration counts, that's an architecture win regardless of local learning.

3. **Pathway 8 (Multi-Rate at long context)** — only tested at ctx=32 where slow bands are useless. At ctx=512+ the multi-rate structure might show genuine timescale separation under global CE.

4. **Pathway 3 (Local Learning)** — paused for the predict-neighbors family. Would need a fundamentally different objective class to revisit. The necessary properties are documented in `research/questions/local-objectives/README.md`.

### Unaddressed items from dictations

- **Pretrained embeddings** (dictation 07): "Did we try using actual pretrained embeddings yet?" Not done. At the character level, "pretrained embeddings" is non-obvious — characters are only 65 classes. Could mean: pretrained byte-level embeddings from a larger model, or a pretrained character-aware word embedding projected back to characters. Worth exploring because richer input representations might change the local-learning story (if band 0 receives features that already encode token-relevant structure, predicting neighbors might become aligned with token prediction). This was not tested.

### What the tested local objectives taught us

- The predict-neighbors family (InfoNCE, hierarchical targets, predict-next-inputs) is structurally misaligned with token prediction on this architecture
- All positive results (detached laterals, multi-rate, width scaling) relied on global CE somewhere
- Detached laterals being BETTER than full backprop suggests weaker coupling + strong global objective may be the right direction
- These are lessons from 3 objectives on 1 architecture — not universal impossibility claims

### What the architecture IS currently competitive at (nothing yet)

16M params, CE 2.69, 220s — vs GRU 820K params, CE 1.35, 8s. The architecture is 20x bigger, 27x slower, and nearly 2x worse. Its value proposition was always "local learning at scale" or "dynamic computation." The tested local objectives didn't work. Dynamic computation is untested.

---

## Suggested ROADMAP.md updates (Max-only edits)

**Pathway 3 (Local Learning):**
- Confidence: **DECREASED significantly**
- New evidence: All tested purely-local objectives (InfoNCE, hierarchical targets, predict-next-inputs, combined) produce CE 3.2-3.5 vs control 2.70. CE worsens with training (3.28→3.52 at 1000 steps). Gradient cosine between local and CE objectives: 0.013. The predict-neighbors family is structurally misaligned.
- Prior results section needs updating with these stronger negative results
- Suggest adding: "Evidence that would increase confidence → A local objective whose gradient is positively correlated with CE (cosine > 0.1)"
- The "Core open question" section still frames this as unsolved — it IS unsolved, but the evidence is now much more negative

**Pathway 1 (Wide Recurrent vs Deep Transformer):**
- Confidence: unchanged (still untested on this architecture)
- Suggest noting: this is now the gating question for the entire project

**Pathway 8 (Multi-Rate):**
- Prior results should note: "At ctx=128, multi-rate gives 20.7% speedup but compute-matched all-rate-1 wins by ~0.018 nats — this is a time/quality tradeoff, not free quality"

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
