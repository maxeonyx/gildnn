# Plan

Working notes. Current state, what's been done, what's next. Updated every session.

---

## Current state (2026-05-28, evening NZST)

**~1 day remains in timebox.** GPU may still be finishing H=4 sweep (irrelevant to current direction).

**The task is clear: build the cellular automaton model and train it.**

---

## ⚠️ ALIGNMENT SESSION (dictation 16) — what the next agent MUST know

This section distills ALL dictations into clear direction. Read this, then build.

### The architecture (dictation 15, confirmed by dictations 27-2, 27-8, 27-9, 27-10, 24-5)

The model is a **cellular automaton / message-passing graph**:

- **Nodes** = combining functions (tempered PoE / Bayesian-like update on distributions)
- **Vertical edges** (through time, in place) = stateless feedforward blocks. NO GRU. NO internal state. NO carry. A block reads from the stream, computes, writes back one tick later.
- **Lateral edges** (left-right across blocks) = propagation with noise added. The noise is what creates the information hierarchy.

The stream carries **distributions** (not point vectors). Higher blocks have context from further back because lower-block outputs merge onto the stream and propagate rightward with delay. Multiple timesteps per token. Information bounces around over many timesteps.

**Each block is an independent network with its own optimizer and its own loss.** Blocks do NOT share gradients. Laterals are detached (stop gradient). The only coupling is through the stream content itself.

**Embeddings are fixed** (random unit vectors on a sphere). Learned embeddings are an optimization for later, not now.

### What to build

One clean script. Not a framework. Not abstractions.

- Grid: [timesteps × blocks]. Many blocks (8-16). Many timesteps per token (configurable, 4-16).
- Blocks: stateless feedforward (MLP). Each has its own optimizer.
- Combining: at each node, combine the lateral arrival with the block's prediction (tempered PoE or similar validated in piece test 06).
- Laterals: propagation with noise. Detached (stop gradient across blocks).
- Loss: each block predicts the next lateral arrival at its node. Loss = W₂² (= MSE on distribution parameters). Block 0 additionally has CE loss for token prediction (grounding).
- Multi-rate: higher blocks fire less often (configurable rates per level, e.g. [1,1,2,2,4,4,8,8]).
- Training: TinyShakespeare, long sequences (2048+), TBPTT.
- Scale: enough that the mechanism has room to work. Not a 2-block toy.

### What Max has ALWAYS reacted negatively to (avoid these)

1. **GRUs / stock RNN mechanisms.** "I kinda don't like the stock RNN models. They're from the past." (dictation 20-15). "I don't understand what you mean by state here. Stop thinking in terms of RNN." (dictation 24-5). "ParallelDiagonalModel is not a phrase I ever used." (dictation 28-15).

2. **Training models that aren't his model.** "I don't fucking care about training models that don't test the key things that I'm actually interested in." (dictation 27-2). A flat GRU is not his model. A transformer is not his model. Global CE on all blocks is not his model (that's just a transformer with extra steps).

3. **Endless ablations and piece tests without assembly.** "The individual pieces have been poked at for days. Put them together." (dictation 27-2). "Stop toys." (dictation 28-15). "Two blocks doesn't test anything. Sequence lengths of four, of thirty-two don't test anything." (dictation 28-15).

4. **Layer norm / batch norm.** "I find this personally quite disgusting." (dictation 21-1). Use mix-add, normalization after noise if needed, but avoid norms as architectural defaults.

5. **Long runs with no visibility.** "We just had another mechanic go for many hours with no updates at all." (dictation 20-14). Always report before launching, always split long runs into phases.

6. **Following the latest dictation as a stack-bump.** "Following the latest thing I say is not really what I want — I usually want dictations to add to the queue, not bump the stack." (dictation 22-1). HOWEVER: dictation 28-15 and 28-16 are explicit course corrections, not queue additions.

7. **Reports that don't link back to goals or explain the architecture.** "There's completely missing parts of the narrative." (dictation 20-1). Reports need theory, architecture explanation, connection to goals.

8. **Eager mode PyTorch without thought.** torch.compile for stable paths (dictation 22-7). But don't let this block building the model — eager is fine for now, compile later.

9. **Small sequences / hobbled architecture.** "We've been hobbling the RNN to match the transformer's interface." (dictation 27-1). Use LONG sequences. The architecture needs them.

10. **Treating everything as experiments without theory.** "You're treating it all about experiments. Sometimes it's about concept clarification." (dictation 20-15). "I'm concerned we're not doing enough analysis, enough exploration of the theoretical ideas before we commit to experiments." (dictation 20-16).

### What Max responds positively to

- **Multi-rate giving inductive bias.** "Better validation loss for fewer compute steps from multi-rate async is pretty cool. Getting slower results means we've got stuff to work on." (dictation 22-10).
- **Wall-clock speedup proofs.** "The only point of async is to get wall clock time speed up." (dictation 22-3).
- **His architecture actually working.** The local-learning-is-strictly-better result. The combining-works result.
- **Clean small codebase.** "Keep the codebase small."
- **Process improvements that give him visibility.**
- **Theory-first, then experiment.**
- **Fixed embeddings, separate optimizers, stop gradients** — all simplifications that make blocks truly independent.

### Key architectural principles (non-negotiable)

1. Blocks are **stateless feedforward**. No hidden state. No GRU. No LSTM.
2. Blocks are **independent**. Separate optimizers. Detached laterals. No shared gradients.
3. **Bidirectional flow.** Token information flows rightward and NEVER comes back. Only PREDICTIONS flow leftward/backward. This is the predictive coding structure.
4. **Noise on laterals** creates the information hierarchy. Distant blocks can't rely on precise token info.
5. **Multiple timesteps per token.** The mechanism only works with many timesteps. Information bounces around between blocks over time.
6. **Long sequences.** 2048+ tokens. TBPTT.
7. **Multi-rate.** Higher blocks fire less often.
8. **Local predictive loss.** Each block predicts the next lateral arrival.
9. **CE grounding.** Block 0 (output-facing) predicts tokens via dot-product against fixed embeddings.

### Soft preferences (validated but not the core test)

- **Distributional stream** (W₂² loss, diagonal Gaussians). Validated in piece tests. Use it. But the CORE question being tested is "does the cellular automaton grid with stateless blocks and local learning work at scale?" — not "do distributions beat point vectors?" If distributional representations are simpler to implement, use them. If point vectors get us to a running model faster, that's OK too.

### Reporting format (daily/weekly)

Back-chain from the goal:
- Why is the model not yet doing the full architecture?
- If it did the full architecture now, what would happen?
- Why would that result be indeterminate or built on shaky foundations?
- Therefore it's doing this step → therefore that step

At every step, justify why we're not just running the real thing. If you can't justify it, run the real thing.

### What's been validated (use these)

| Component | Status | Evidence |
|---|---|---|
| W₂² = MSE on (μ,σ) | Validated | Piece test 01, 05 |
| Tempered PoE combining | Validated | Piece test 06 (26/26 checks) |
| Stream carries distributions | Validated | Piece test 04 |
| Noise forces prediction advantage | Validated | Piece test 07 |
| Feedforward prediction works | Validated | Horizon sweep H=1, H=2 (recurrence null) |
| Combining helps over passthrough | Validated | Stream combining experiment (42% vs 37%) |
| Local learning > global backprop | Validated | A/B detach test (−0.215 nats) |
| Separate optimizers + fixed embeddings | Validated | 27-May training (genuine learning vs gradient interference) |

### What was DELETED

- ✅ `core/model.py` — ParallelDiagonalModel (deleted)
- ✅ `core/tied_readout.py` — dead code (deleted)
- ✅ 7 dead experiment scripts importing deleted model (deleted, −5200 lines)

---

## Current state (2026-05-30 ~17:00 NZST)

**⚠️ PROJECT ENDS MIDNIGHT SUN 31 MAY NZST. ~31 hours remain.**

### Environment: Linux (Manjaro VM), RTX 3090
- torch 2.12.0, CUDA working, Triton 3.5.1
- System Python 3.14 with venv (`--system-site-packages`)
- einops + jaxtyping in venv
- TinyShakespeare downloaded to `experiments/corpora.ignore/tinyshakespeare_input.txt`
- **GPU power limit set to 185W** (`sudo nvidia-smi -pl 185`, resets on reboot)
- torch.compile HANGS on this model (1024-iteration loop too complex for tracer)
- CUDA graphs fail on Linux (CPU↔CUDA copy during capture)
- **Loop:** `systemctl --user start gildnn-loop` (systemd user service, survives disconnects)

### ✅ Completed this session:
1. ✅ Built 192-module 2D graph automaton (`core/automaton_graph.py`) — 16M params
2. ✅ Adversarial review found/fixed: frozen embeddings, prediction target (predict incoming laterals not combined state)
3. ✅ Sanity check passes: CE 4.24→2.37 in 20 steps (fixed batch overfit)
4. ✅ 200-step training run completed: CE 3.32→2.69 at batch=4, chunk=128 (~5.5 min)
5. ✅ **InfoNCE implemented as default local loss** (commit `e712c9a`)
   - InfoNCE beats MSE on CE: 2.41 vs 2.53 at step 19
   - Band-0 exempt (uses CE for grounding)
   - Stable, no collapse
6. ✅ **Refractory mechanism implemented** (option, default off)
   - Refractory helps CE: 2.46 vs 2.55 baseline
   - Makes prediction harder (expected — signal actually propagates instead of pooling)
7. ✅ **Process documented:** implementation pipeline (implement→review→fix→start) in PROCESS.md
8. ✅ **Gradient clipping** — fixed InfoNCE NaN divergence at step 300 (max_norm=1.0)
9. ✅ **Triton fused forward kernel** — 1.94x speedup, numerically verified (`core/triton_forward.py`)
10. ✅ **1000-step training completed** — CE 4.17→2.69, all bands learning
11. ✅ **Impulse response diagnostic** — confirms diffusion (not waves) with random weights
12. ✅ **Checkpoint saving** added to training script
13. ✅ **5000-step long run launched** (PID 34251, lr=1e-4, checkpoints every 1000 steps, ~3.4h)

### Wave propagation theory (discussed with Max):
- Current model is diffusion-like (symmetric reads, no directional transport)
- Cortical analogy: excitable medium (threshold + refractory → wavefronts)
- Options identified (simplest → most faithful):
  1. Novelty-gated drive (nearly free)
  2. Threshold + refractory scalar (implemented ✅, tested ✅)
  3. Directional edge buffers (moderate cost)
  4. Activator–inhibitor (highest cost, most faithful)
- Diagnostic: impulse response — waves show sharp moving shell, diffusion shows broadening blob

### Architecture spec:

**192-module 2D grid:**
- **8 rate bands × 24 positions** = 192 modules
- **Topology:** 2D grid, horizontal ring (wrap), vertical open boundary
- **Neighbors:** 4-neighborhood (left, right, up, down)
- **Rates by band:** `[1, 2, 4, 8, 16, 32, 64, 128]`
- **Phase stagger:** `phase[b,c] = c mod rate[b]`
- **d_stream=96, d_hidden=384** → ~16M params, ~83K/module
- **batch=16** (tensor core alignment)
- **Communication:** detached noisy neighbor reads from shared buffer
- **Token injection:** band 0 only
- **Output:** mean of all band-0 module logits (tied readout, fixed embeddings, temp=0.07)
- **Loss:** band-0 CE + bands 1-7: InfoNCE local prediction (predict normalized incoming lateral sum)
- **Chunk:** 128 tokens × 8 steps/token = 1024 microsteps

### Triton persistent kernel design:
- One kernel launch, 192 programs (one per module)
- Each program loops at its own rate (rate-1 does 1024 iters, rate-128 does 8)
- Reads neighbor outputs from global memory WITHOUT synchronization barriers
- Writes own output to global memory (visible to neighbors whenever hardware delivers)
- The stale-lateral semantics = GPU's default memory visibility model
- For backward: store activations during forward → custom backward kernel (same structure, reverse)
- Each module's backward is independent (detached laterals → no cross-module gradient)

### NEXT:
1. ✅ **5000-step baseline completed** — lag probe confirms: band 0=33.5%, band 1=23.9%, bands 2-7 random. CE 2.64.
2. ✅ **Multi-scale + streaming 2000-step experiment completed** — Result: FASTER learning (CE 2.49 vs 2.64, band 0=37%, band 1=24.1%) but STILL no band 2-7 specialization.
3. **Root cause is the LOSS, not the input.** Multi-scale input gives better performance but doesn't force different representations. InfoNCE (predict neighbor sum) lets all bands converge to the same "encode recent tokens" strategy because that's the easiest prediction for everyone.
4. **Next experiment needed:** change the loss to incentivize different representations per band. Options:
   - (a) **Residual targets:** band k predicts what band k-1 DOESN'T predict (successive refinement)
   - (b) **Anti-redundancy penalty:** penalize mutual information between adjacent bands' representations
   - (c) **Temporal target per band:** band k explicitly predicts lag=2^k ahead (force timescale matching)
   - Option (c) is closest to Max's design ("predict k steps ahead" + multi-rate). Band with rate R makes predictions evaluated R steps later — this is ALREADY how it works. So why doesn't it specialize? Because the prediction TARGET (neighbor sum) is dominated by band 0 which encodes recent tokens. Higher bands predict "what band 0 will look like" rather than encoding their own temporal view.
   - **Fix:** detach band 0 from the neighbor sum that higher bands predict. Or: each band predicts a DIFFERENT target (e.g., token embedding at lag=rate).
5. **Central attention readout** (Max's dictation idea) — still not implemented, do after fixing loss
6. **Final daily report** needed before end of day

### Key findings from lag probe (definitive):
- **Baseline 5000 steps:** band 0=33.5%, band 1=23.9%, bands 2-7 random. CE 2.64.
- **Multi-scale + streaming 2000 steps:** band 0=37%, band 1=24.1%, bands 2-7 STILL random. CE 2.49 (better!).
- Multi-scale input accelerates learning but does NOT create specialization.
- **The bottleneck is the prediction target**, not the input. All bands predict the same neighbor sum, which is dominated by band 0's recent-token encoding. There's no incentive to encode anything different.

### 1000-step results (anchor run):
- CE: 4.17 → 2.69 (1.48 nats learned)
- **Inverted U pattern in band performance:**
  - Bands 4-5 (rates 16, 32): 1.38 — BEST
  - Bands 1-2 (rates 2, 4): 1.65-1.78
  - Band 6 (rate 64): 2.60
  - Band 7 (rate 128): 3.52
- Interpretation: mid-rate bands have optimal balance of informative targets + sufficient training events

### Key insight from profiling:
- Active-only in pure PyTorch was SLOWER (indexing overhead outweighed savings)
- Bottleneck is NOT any single op — it's 1024 timesteps × many small CUDA kernels
- Only solution: fused Triton kernel (one kernel per timestep doing the whole module computation)
- bmm is only 35% of time; elementwise/indexing is 65%
- **Triton forward kernel achieves 1.94x speedup** (913ms vs 1772ms at batch=16, chunk=128)
- Backward pass not yet fused — training would benefit from ~1.3x (forward-only fusion)

### What's validated:
- 192-module graph learns (CE 4.2→2.4 in 20 steps)
- InfoNCE > MSE for local loss (0.12 nats better CE)
- Refractory helps CE (~0.09 nats) — consistent with wave-like transport being better
- Multi-rate phase stagger working (prediction counts match expected rates)
- Noise forces genuine prediction (prior work, still applies)

---

## Key references

- `dictations/2026-05-28-15.md` — cellular automaton clarification
- `dictations/2026-05-28-16.md` — alignment session instruction
- `dictations/2026-05-28-18.md` through `2026-05-28-25.md` — performance: CUDA graphs, GPU-native, fused kernels
- `dictations/2026-05-28-20.md` — REDACTION: blocks are NOT weight-shared
- `dictations/2026-05-27-2.md` — "stop training models that aren't mine"
- `dictations/2026-05-24-1.md` — Max's original Google Keep architecture note
- `core/automaton_graph.py` — the 192-module graph model (current)
- `core/automaton.py` — the 8-level linear model (reference, superseded)
- `experiments/automaton_graph/` — training + ablation scripts
