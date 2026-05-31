# Plan

Working notes. Final day (2026-05-31), project concludes at midnight NZST.

---

## Active work (final session, May 31 ~17:30 NZST)

### CUDA graph capture restored — DONE ✓

CPU→CUDA scalar assignments (`prediction_errors[0] = 0.0`, `active_predictions[0] = False`) broke CUDA graph capture when the residual fix was added. Fixed with precomputed `_skip_level0` mask buffer. Committed `aa78d4b`.

**Performance:** 8-level d=128 spt=4 seq=2048 batch=8 → **1.87s/step** with graph vs 11.9s eager (6.3×). Restores the 14× speedup over the original 26s/step baseline.

### 8-level residual convergence — IN PROGRESS (ETA ~18:07 NZST)

Running via systemd: `automaton-8level-residual-full`. Config: n=8, d=128, spt=4, seq=2048, batch=8, noise=0.1, lr=3e-4, 200 steps, eager mode (launched before graph fix).

**Key question:** Does multi-level + residual match/beat the prior 8-level result (CE 2.30 at 500 steps, no residual, seq=2048)? If laterals actually help during autonomous steps, multi-level should outperform the 1-level spt=4 result (CE 2.37).

---

## Completed results (prior sessions)

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

| Context | Operating point | Mean Iters | Compute Savings | CE Overhead | Oracle Efficiency |
|---------|----------------|------------|-----------------|-------------|-------------------|
| 128 | thr=0.05 | 6.12 | 23.5% | −0.000 | 73% |
| 128 | thr=0.10 | 5.53 | 30.9% | +0.023 | 85% |
| 128 | thr=0.20 | 4.71 | 41.1% | +0.081 | 96% |
| 256 | thr=0.05 | 6.23 | 22.1% | +0.006 | >100% at matched overhead |

Ctx=256 confirms the result generalizes: full-depth val CE baseline 1.840, hidden-state Pearson r=0.51 (per-depth 0.30–0.41), and the predictor still gets negative overhead at a smaller threshold (thr=0.02: 16.9% savings, −0.006 CE). The notable change is that the oracle no longer improves CE at longer context (smallest-threshold oracle is already +0.072), while the learned predictor beats oracle at matched overhead for larger thresholds because predicted **total remaining gain** is a better stopping criterion than **next-step gain**.

### Context length robustness — DONE ✓ (mixed: per-param holds, FLOP-matched doesn't)

At ctx=256:
- Per-param: tied d=256 N=8 (887K) val CE 1.841 vs untied d=128 N=8 (1.6M) val CE 1.905 → **tied wins** (0.064 nats)
- FLOP-matched: tied d=256 N=2 (887K) val CE 1.962 vs untied d=128 N=8 (1.6M) val CE 1.905 → **untied wins** (0.057 nats)

Iteration scaling at ctx=256: gains per doubling stay high (0.05-0.07 nats) vs ctx=128 where they taper (0.043→0.014→0.019). The model is "hungry for depth" at longer context — dynamic depth decisions are more impactful.

### Local learning — DONE ✓ (negative for tested family)

All tested purely-local objectives from the "predict neighbor state" family (InfoNCE, hierarchical targets, predict-next-inputs) are structurally misaligned with token prediction. CE 3.2-3.5 vs control 2.70, worsening over training (3.28 → 3.52 at 1000 steps). Gradient cosine 0.013 (orthogonal). Scoped to 3 objectives, 1 architecture, detached laterals.

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

---

## If this project is picked up again

### Answered questions (do NOT re-test)

1. **Pathway 1 — VALIDATED.** Weight tying works. Tied d=256 N=8 (854K params) beats untied d=128 N=8 (1.6M params) at both ctx=128 and ctx=256. Per-param advantage is robust. FLOP-matched advantage (tied N=2 vs untied N=8) holds at ctx=128 but breaks at ctx=256 — longer context needs more iterations.

2. **Pathway 5 — VALIDATED.** Dynamic depth is practical. Learned halting predictor: 22-23% savings at zero/negligible CE overhead, 85-100% oracle efficiency. Hidden state encodes halting signal (r=0.51-0.53). Confirmed at both ctx=128 and ctx=256.

4. **Temporal residual state update — ANSWERED.** State replacement catastrophic at high spt; normalized residual (`normalize(state + output)`) fixes it. The remaining gap to GRU is gating, not attention. Committed `8a6f09a`. See `research/questions/residual-stream-across-time/README.md`.

### What remains open

1. **Gating for the automaton:** The gap between automaton (CE 2.23) and GRU (1.58) is likely gating. A learned gate (`α * state + (1-α) * output` where α is MLP-produced) would test this. Connects to Pathway 1.

2. **Multi-level with residual at convergence:** The normalized residual was only tested at 200 steps for multi-level (still converging). Need controlled comparison vs original.

3. **Joint training with dynamic depth (ACT/CALM-style):** The frozen-model probe shows the information is there. Training model + halting head simultaneously should produce clearer signals. Connects to Pathway 5.

4. **Pathway 8 (Multi-Rate at long context):** Only tested at ctx=32/128 where slow bands are useless. At ctx=512+ the multi-rate structure might show genuine timescale separation.

5. **The 192-module graph:** CE 2.69, fundamentally broken communication. The temporal residual might help, but narrow d_stream=96 and detached laterals are likely the deeper issue.

6. **Local learning (different objective families):** Predict-neighbors failed. Info-theoretic / contrastive / predictive coding with different inductive biases untested.

### Unaddressed items from dictations

- **Pretrained embeddings** (dictation 07): "Did we try using actual pretrained embeddings yet?" Not done. At the character level, "pretrained embeddings" is non-obvious — characters are only 65 classes. Could mean: pretrained byte-level embeddings from a larger model, or a pretrained character-aware word embedding projected back to characters. Worth exploring because richer input representations might change the local-learning story (if band 0 receives features that already encode token-relevant structure, predicting neighbors might become aligned with token prediction). This was not tested.

### What the tested local objectives taught us

- The predict-neighbors family (InfoNCE, hierarchical targets, predict-next-inputs) is structurally misaligned with token prediction on this architecture
- All positive results (detached laterals, multi-rate, width scaling) relied on global CE somewhere
- Detached laterals being BETTER than full backprop suggests weaker coupling + strong global objective may be the right direction
- These are lessons from 3 objectives on 1 architecture — not universal impossibility claims

### Competitiveness summary

**The simpler tied transformer IS competitive:** val CE 1.67 (854K params) vs GRU 1.58 (820K) — gap is only 0.09 nats. Beats standard untied transformer (1.71, 825K) and untied-8L (1.87, 1.6M). Dynamic depth validated: 22% savings at near-zero overhead.

**The 192-module graph is NOT competitive:** CE 2.69, 16M params, 220s. 20x bigger, 27x slower, nearly 2x worse than GRU. Its value proposition (local learning at scale, dynamic computation) was only partly tested — local learning negative, dynamic computation not tested on this topology.

---

## Suggested ROADMAP.md updates (Max-only edits)

**Pathway 3 (Local Learning):**
- Confidence: **DECREASED significantly**
- New evidence: All tested purely-local objectives (InfoNCE, hierarchical targets, predict-next-inputs, combined) produce CE 3.2-3.5 vs control 2.70. CE worsens with training (3.28→3.52 at 1000 steps). Gradient cosine between local and CE objectives: 0.013. The predict-neighbors family is structurally misaligned.
- Prior results section needs updating with these stronger negative results
- Suggest adding: "Evidence that would increase confidence → A local objective whose gradient is positively correlated with CE (cosine > 0.1)"
- The "Core open question" section still frames this as unsolved — it IS unsolved, but the evidence is now much more negative

**Pathway 1 (Wide Recurrent vs Deep Transformer):**
- Confidence: **INCREASED — VALIDATED** on clean tied transformer (not 192-module graph)
- New evidence: tied d=256 N=8 (854K) beats untied d=128 N=8 (1.6M) at both ctx=128 and ctx=256. Per-param advantage robust. FLOP-matched holds at ctx=128, breaks at ctx=256.
- Suggest noting: the 192-module graph (which adds multi-rate, stale laterals, topology) remains untested in a clean comparison

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
