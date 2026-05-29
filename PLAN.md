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

## Current state (2026-05-30 05:35 NZST)

### ✅ DONE:
1. ✅ Delete ParallelDiagonalModel
2. ✅ Build cellular automaton model (`core/automaton.py`)
3. ✅ Sanity check — CE drops 4.85→3.79 in 5 steps, model learns
4. ✅ Vectorize across levels with bmm (10x speedup: 55s→5s per step at small scale)
5. ✅ Delete dead automaton-irrelevant helpers from `core/training.py`, `core/run_utils.py`, and `core/dataset.py` (commit `b0f97fa`)
6. ✅ **v1 training complete** (commit `b54615f`). 500 steps, ~3h17m. Results:
   - CE: 3.54 → 2.30 (random baseline 4.17, so 1.87 nats below random)
   - All 8 levels' prediction losses improved (no single-block collapse)
   - Per-level differentiation present but mild: level 1 highest (0.0064), levels 3-4 lowest (0.0033)
   - Model learning: confirmed

### 🏃 IN PROGRESS:
7. **Noise ablation** — PID 6704, lock active. Same config as v1 but `--noise-std 0.0`. Started 05:33 NZST. Expected completion ~08:50 NZST.
   - Log: `experiments/automaton/artifacts/run_v1_no_noise.jsonl`
   - Report: `experiments/automaton/artifacts/report_v1_no_noise.json`
   - Question: Does removing noise change per-level differentiation? If noise doesn't matter, multi-rate timing alone creates the hierarchy.

### Performance notes (dictations 18-25):
- torch.compile with fullgraph=True: TRACING SUCCEEDS (no graph breaks) but Triton not available on Windows. Inductor backend requires Triton.
- Workaround: vectorized bmm gives 10x improvement in eager mode.
- Full GPU-native execution (Triton kernels, CUDA graphs) would need Linux or a Windows Triton build.

### NEXT:
8. **When noise ablation completes**: compare per-level prediction losses between v1 (noise=0.1) and no-noise. Key metric: is differentiation pattern the same or different?
9. **Write daily report** for 2026-05-30 once both results are in (or at 4pm, whichever first).
10. **Optional later cleanup**: consider whether `core/fixed_window_char.py` should move under `base-experiments/` or stay as shared legacy utility. No action needed now.

---

## Key references

- `dictations/2026-05-28-15.md` — cellular automaton clarification
- `dictations/2026-05-28-16.md` — alignment session instruction
- `dictations/2026-05-28-18.md` through `2026-05-28-25.md` — performance: CUDA graphs, GPU-native, fused kernels
- `dictations/2026-05-28-20.md` — REDACTION: blocks are NOT weight-shared
- `dictations/2026-05-27-2.md` — "stop training models that aren't mine"
- `dictations/2026-05-24-1.md` — Max's original Google Keep architecture note
- `core/automaton.py` — the cellular automaton model (vectorized bmm)
- `experiments/automaton/train.py` — training script (TBPTT, per-level optimizers)
