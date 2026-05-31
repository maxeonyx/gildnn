# Plan

Working notes. Final day (2026-05-31), project concludes at midnight NZST.

---

## Current state (14:00 NZST, May 31)

### Pathway 1 weight-tying experiment — DONE ✓ (positive result)

Clean test of the core thesis: weight-tied (1 block applied N times) vs untied (N distinct blocks). Pre-LN causal transformer, 8 layers/iterations, 1000 steps, 3 seeds, validation CE.

**Results:**

| Model | Params | Val CE (mean) | Notes |
|-------|--------|---------------|-------|
| Tied d=128 | 230K | 2.070 | Same FLOPs as untied d=128 |
| Tied d=192 | 493K | 1.899 | |
| **Tied d=256** | **854K** | **1.788** | Beats untied with half the params |
| Untied d=128 | 1.6M | 1.866 | 7× more params, WORSE quality |
| Transformer 4L (baseline) | 825K | ~1.84 | Standard pre-LN baseline |
| GRU (baseline) | 820K | ~1.35 | Sequential recurrence wins |

**Conclusions:**
1. At matched FLOPs (same d, same iterations): untied beats tied by 0.20 nats. Extra capacity per depth helps.
2. At matched params (~850K): tied (d=256, 8 iter) beats untied (d=128, 8 layers) AND the 4-layer transformer baseline. Width compensates effectively.
3. The per-param efficiency of weight tying is roughly 3× (tied at ~550K ≈ untied at 1.6M in quality).
4. Weight tying is NOT free at matched compute, but it IS efficient at matched memory. The value proposition: fewer params + dynamic depth potential.

**Still worse than GRU (1.35 vs 1.79).** But GRU has sequential token-to-token recurrence which is a fundamentally different mechanism. The tied transformer's advantage is parallelizability across tokens and dynamic iteration count.

### Iteration scaling — DONE ✓ (logarithmic, stable to N=32)

Same tied d=256 model (854K params), varying iterations:

| N | Val CE | Notes |
|---|--------|-------|
| 2 | 1.845 | Same FLOPs as untied-d128-N8, BETTER quality |
| 4 | 1.802 | |
| 8 | 1.788 | |
| 16 | 1.769 | |
| 32 | 1.740 | Stable, no explosion |

Logarithmic scaling: ~0.02-0.04 improvement per doubling. Pre-LN + grad clip = stable through 32 iterations.

### Dynamic depth — DONE ✓ (practically demonstrated)

CE-based early exit (oracle): 43% compute savings at +0.045 nats. Early exit at small thresholds actually IMPROVES quality (regularization). Entropy-based exit (deployable): works but noisier (~15% savings at +0.025 nats, or 31% at +0.15).

### Learned halting predictor — DONE ✓ (strong positive)

Tiny MLP (261→64→1) trained on frozen model's hidden states predicts remaining CE gain `g_d = L_d - L_8`. Evaluated on held-out 20% of validation set.

| Operating point | Mean Iters | Compute Savings | CE Overhead | Oracle Efficiency |
|----------------|------------|-----------------|-------------|-------------------|
| thr=0.05 | 6.12 | 23.5% | −0.000 | 73% |
| thr=0.10 | 5.53 | 30.9% | +0.023 | 85% |
| thr=0.20 | 4.71 | 41.1% | +0.081 | 96% |

Hidden state Pearson r=0.53 vs entropy-only r=0.42. Per-depth hidden r=0.31–0.40. Dominates all fixed-depth and entropy heuristic baselines.

### Context length robustness — DONE ✓ (mixed: per-param holds, FLOP-matched doesn't)

At ctx=256:
- Per-param: tied d=256 N=8 (887K) val CE 1.841 vs untied d=128 N=8 (1.6M) val CE 1.905 → **tied wins** (0.064 nats)
- FLOP-matched: tied d=256 N=2 (887K) val CE 1.962 vs untied d=128 N=8 (1.6M) val CE 1.905 → **untied wins** (0.057 nats)

Iteration scaling at ctx=256: gains per doubling stay high (0.05-0.07 nats) vs ctx=128 where they taper (0.043→0.014→0.019). The model is "hungry for depth" at longer context — dynamic depth decisions are more impactful.

### What to explore next (remaining runway)

- **Training WITH dynamic depth:** Use ACT/CALM-style training where the model learns to exit early — might front-load computation and improve both quality and efficiency
- **Joint halting + model training:** Train the base model with the halting head simultaneously (should produce clearer halting signals)

### CORRECTION: "CE 2.67 purely-local" was mislabeled

The earlier hierarchical-targets result (CE 2.67) used the standard band-0 CE readout — making it a hybrid (CE on band 0, local on bands 1-7). The truly purely-local version gives CE 3.28 at step 100. No purely-local objective has beaten the control (2.70). See corrected daily report.

### Active experiment — COMPLETED
Hierarchical targets purely-local (detached), 1000 steps. **Result: CE 3.28 → 3.52 (WORSE over training).**

Local prediction learning actively hurts token classification. As modules specialize for inter-band prediction, they move AWAY from representations the attention head can exploit. The ~3.28 at step 100 was incidental token info from initialization; training destroys it.

### Corrected results table (all validation CE)

| Experiment | Val CE | Params | Notes |
|-----------|--------|--------|-------|
| GRU baseline | **1.58** | 820K | Best at step 1000; sequential recurrence |
| **Tied d=256 N=8** | **1.67** | **854K** | **Best at step 2000; beats transformer** |
| Transformer 4L baseline | 1.71 | 825K | Best at step 2000+ |
| Tied d=256 N=8 (@step 1000) | 1.78 | 854K | Still improving |
| Untied d=128 N=8 | 1.87 | 1.6M | 7× more params, WORSE |
| Tied d=256 N=2 | 1.85 | 854K | Same FLOPs as untied, BETTER |
| Control: 192-module arch | ~2.70 | 16M | Communication structure problem |

Previous reports showed misleading "GRU 1.35 vs our 2.70" — that compared GRU train CE to architecture train CE. Fair val-CE comparison: GRU 1.58 vs tied 1.67 (gap: 0.09 nats).

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

1. **Pathway 1 (Wide Recurrent vs Deep Transformer)** — the fundamental thesis. The *assembled 192-module system* was compared against baselines and lost badly (CE 2.69 at 16M params/220s vs GRU 1.74 at 820K/11s). But this doesn't cleanly test the core thesis ("same weights applied N times vs N distinct layers") because the system adds multi-rate, stale laterals, topology, etc. The clean weight-tied-vs-untied comparison at matched FLOPs remains unrun. If resumed, this is the first experiment — with fairness axes (per-param AND per-FLOP) declared up front.

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
