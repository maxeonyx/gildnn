# Plan

Working file. Rewrite it as the state changes.

## Current situation

- The trust foundation for text is now in place: `base_experiments/` has a trustworthy transformer anchor and a trustworthy vanilla RNN anchor on the fixed TinyShakespeare frame.
- The predictive-chain and dynamic-depth reports have been re-framed against that baseline anchor so they no longer overclaim.
- New dictations changed the near-term priority. Early composition of predictive chain + dynamic depth is **not** the main thing Max wants right now.
- The current priority is thorough isolated experiments on the individual performance/mechanism pieces: local learning, stop gradients, asynchronous/asynchronized modules, keeping weights in memory, processing activations in place, and replacing depth with width / propagation across timesteps.
- "Predictive chain" remains a useful experiment label for one simplification, but it is not the project goal. It should not silently drive the plan.
- There is partially completed chain+dynamic-depth work in `experiments/chain_dynamic_depth/`, but that thread is currently paused in planning until it is explicitly re-scoped.
- **Architecture clarification (2026-05-20):** new dictation simplifies the module concept. A node/module is currently a **single residual block on a uniform `d_model` residual stream**, not a recurrent mini-stack with its own hidden width. The interface between modules is the residual stream itself. The interesting mechanisms are at block boundaries: stop gradients, async, attention residual connections. This changes the interpretation of Phase 3 and some already-written docs.

## Execution phases with clear sequencing

### Global rules for every phase

1. Every experiment starts with its report folder/README first: question, simplification, hypotheses, comparison target, and placeholders for evidence.
2. No new architectural thread starts without a baseline for that modality/task to compare against.
3. Base experiments follow the same ladder as everything else: overfit one batch, tiny run, inspect outputs, then scale.
4. Base experiments also get cheap correctness tests where mistakes are easy and expensive: data slicing/targets, embedding lookup shape/content, loss masking/accounting, sampling/generation path, and any modality-specific preprocessing.
5. Keep experiment scope small, but move quickly from one discriminating experiment to the next. Do not spend days polishing an already-answered question.

### Phase 1 — Trust foundation for text baselines (done)

Objective: make text results interpretable enough that later custom-architecture comparisons mean something.

Done:

- Transformer anchor: ~186K params, best val loss 1.632
- Vanilla RNN anchor: ~186K params, best val loss 1.706
- Cheap correctness checks and comparison frame documented in `base_experiments/README.md`

End state required to leave Phase 1:

- At least one text baseline is trustworthy enough to be the comparison anchor.
- The base-experiment path has cheap correctness tests.
- The comparison frame for future reports is fixed: parameter count, loss, compute cost, and important execution constraints.

### Phase 2 — Re-ground the current findings against the new trust anchor (done)

Done:

- Predictive-chain report weakened where the standardized anchors made the old parity framing misleading
- Dynamic-depth report anchored against the standardized frame with honest comparability limits
- Future comparisons now have a fixed text comparison frame

End state required to leave Phase 2:

- The two strongest completed threads are now anchored to the same comparison frame future work will use.

### Phase 3 — Isolated mechanism experiments on the simplified block architecture

Objective: test the individual pieces Max actually cares about, in the simplified frame where a module is a single residual block on a shared `d_model` stream.

#### 3a. Local learning / stop gradients on single-block modules

Each **block boundary** predicts its own next incoming residual stream (A→A), with an auxiliary local head. In the clarified architecture, a module is **one residual block**, not a 3-layer recurrent stack.

The main question is no longer "how thick is a local module?" but:
- Does splitting a model into multiple stop-gradient-separated residual blocks help or hurt?
- Where should the detach boundaries go?
- What local target should be predicted?
- Does a multi-block local-learning system beat a matched single-block or ordinary baseline?

Comparison design: **single-block local-learning control** vs **multi-block local-learning variants**, with `d_model` uniform across all variants.

Note: the first local-learning experiment in `research/questions/local-learning/` tested a different (now-superseded) interpretation — detached recurrent stacks, not single residual blocks. That result narrows one branch of the design tree but does not directly answer this question.

#### 3b. GPU utilization research

What kind of GPU programs actually fit best on the RTX 3090? Can residual block architectures get significantly more FLOPs out of the GPU than transformers? Measure wall-clock, actual utilization, memory bandwidth. This grounds all future performance claims.

#### 3c. Attention residual transformer (core mechanism experiment)

This is now a central mechanism experiment for the clarified architecture, not a side curiosity. The dictation says looped blocks with attention residual connections are essentially a transformer — so this subsection tests the core boundary/residual idea cleanly, without prematurely adding async or richer graph machinery.

Multiple variants:
- Attention over depth only
- Attention over depth AND sequence length in a causal triangle (can't attend to same layer at previous step)

Motivation: this demonstrates the mechanism for time-unrolled / looped-block behavior. Stop gradients across time and across depth are the same kind of mechanism in this framing.

#### 3d. Async execution without synchrony

Can we run residual block boundary updates without full synchrony — with volatile/shared memory or stale reads between updates? Not asynchronous recurrent hidden-state modules, but asynchronous **block boundary updates** on the shared `d_model` stream.

Rules:

- Prefer the cheapest isolated experiment that could produce real evidence about one of those pieces.
- Do not smuggle composition back in by changing several mechanisms at once.
- If a simplification came from an earlier agent rather than the dictations, keep that fact visible in the write-up.

Questions this phase should answer:

- Can stop-gradient-separated **residual blocks** learn useful local objectives at all?
- Do looped / reused residual blocks with attention residual connections show useful behavior before async is added?
- Can we approximate async block updates without changing the architecture into something else?
- What actual GPU utilization do these simplified mechanisms achieve on this hardware?

### Phase 4 — Async/desynchronized execution path on text

Objective: once the individual pieces above are better grounded, test the asynchronous path more directly.

Do this in escalating steps, stopping as soon as the answer is clear:

1. Build the smallest graph/shared-memory prototype that actually targets the async question.
   - Do **not** repeat the earlier "more skip connections on Shakespeare" experiment and call that async.
   - Use sparse local communication plus a clearly limited shared/global memory path.
2. Add a trigger signal for whether a block should update.
   - Prefer the mechanism closest to Max's current thinking: a cheap predictive head that estimates whether further computation will help, or a clearly-defined surprisal proxy if loss-prediction is not yet workable.
3. If selective updates show real signal, run a semi-async approximation:
   - masked/bucketed updates,
   - stale reads allowed,
   - dense GPU-friendly execution where possible.
4. Compare against the synchronous equivalent, not just against standard baselines.

Questions this phase should answer:

- Can the model learn a useful update signal at all?
- Does skipping updates save meaningful compute without collapsing quality?
- Does graph locality still matter once a shared/global path exists, or does the global path bypass the interesting part?

### Phase 5 — Contingent composition work

Objective: only after the isolated pieces are understood well enough, test whether any composition is actually justified.

Candidates may include:

- dynamic depth inside the predictive-chain family
- loss-prediction-driven triggering inside another local-learning family

This phase is contingent, not automatic. If the isolated experiments do not justify composition, skip it.

### Phase 6 — Second modality: arbitrary-order image patches

Objective: open the second modality in the vision, but do it with the same discipline as text.

1. Start with the baseline, not the custom architecture.
   - Create the image-patch question/report first.
   - Build the smallest ordinary recurrent or transformer-style patch baseline that can handle arbitrary patch order honestly.
   - Add modality-specific correctness checks: patch extraction/reassembly, order handling, masking/subset conditioning, and shape-agnostic path where claimed.
2. Once there is a patch baseline, run **one** architectural transfer from the text work.
   - Prefer the single mechanism that looked most promising in Phases 3-4.
   - Default priority: selective/dynamic compute first, full graph complexity second.
3. Aim for a result that narrows the question, not a big image system.

This phase exists because image patches are part of the stated vision, and a text-only finish would leave too much of Max's agenda untouched.

### Phase 7 — Final combination and synthesis

Objective: spend the last part of the window on the strongest surviving combined idea, not on cleanup for its own sake.

1. If Phases 3 and 4 both produced positive signal, run one final combination experiment that uses the best mechanism from each.
   - Likely candidates:
     - dynamic depth inside a predictive-chain / small-graph model
     - loss-prediction-driven triggering for selective updates
2. If the combination does **not** look justified, do not force it. Use the time to tighten the strongest positive or strongest negative result into a clean final report.
3. End the window with a clear decision matrix:
   - what worked,
   - what failed,
   - what remained ambiguous,
   - what would be the first thing worth continuing after this billing window.

## Stop conditions / success criteria

### Hard gates

- Do not leave Phase 1 until at least one text baseline is trustworthy.
- Do not start image-patch architecture work until there is an image-patch baseline.
- Do not keep expanding graph complexity if the comparison needed to interpret it is missing.

### Success floor for this window

This window is successful if all of the following are true:

1. `base-experiments/` contains trustworthy text baselines with correctness checks, enough to anchor honest comparison.
2. Predictive-chain and dynamic-depth are re-stated against that comparison frame.
3. At least two real post-baseline experiments test the individual performance/mechanism pieces Max called out.
4. At least one experiment directly tests selective/dynamic computation rather than only static architecture.

### Strong success for this window

This window is a strong success if, in addition to the floor above:

1. The async/selective-update path yields a clear positive or negative answer that narrows the cortical-column design space.
2. The image-patch modality has at least a baseline plus one first architectural probe.
3. Any composition work that survives is clearly justified by the isolated experiments rather than assumed up front.
4. The final state of the repo answers not just "can we trust the comparisons?" but also "which of Max's architectural ideas are now worth pushing further, and which are not?"

### Explicit cut rules

- Do not spend multiple days polishing baseline quality once the trust question is answered well enough to compare.
- Do not repeat an experiment class that already gave a clear answer unless the new version changes the actual hypothesis being tested.
- If time gets tight, cut contingent composition work before cutting the highest-value isolated mechanism experiments.
- If one late-phase thread is blocked, switch to the next discriminating experiment rather than burning the remaining window on setup/debugging.

## Completed experiments

- **Local learning (residual)** — NEGATIVE. Stop-gradient boundaries + local prediction heads on residual blocks clearly hurt vs matched end-to-end (best val 2.081 vs 1.644, 3-block). Mechanism works mechanically but LM quality tanks. See `research/questions/local-learning-residual/README.md`.
- **Attention-residual (depth-only)** — MARGINAL/INCONCLUSIVE. Content-based attention over earlier boundary states shows no stable improvement (transient 0.013 nat edge at peak, regresses to worse by end of training). 33% slower. Not worth pursuing further at this budget. See `research/questions/attention-residual/README.md`.
- **Async/selective execution** — CUT (NEGATIVE). Learned per-token gating works mechanically (gates learn genuine selectivity, no collapse) but GPU wall-clock is 26-29% WORSE despite 15-32% fewer logical block executions. Per-token conditional execution breaks batched GPU parallelism. Premise falsified at tiny rung; stopped before full standardized. See `research/questions/async-selective/README.md`.
- **GPU utilization study** — CONFIRMED Max's hypothesis, then EXTENDED with full hardware deep dive. GRU 1.8-2.8x faster training, 2.5-3.7x faster decode. Profiler reveals WHY: GRU hits Tensor Core paths (CUTLASS tensorop kernels) even in FP32 while transformer at this scale stays on CUDA cores. GRU spends 69% of time in GPU kernels vs transformer's 51%. Sequence length sweep shows GRU holds 1.2-1.4M tok/s flat from ctx=32 to ctx=1024 while transformer drops from 530K to 276K. See `research/questions/gpu-utilization/README.md`.
- **Causal triangle attention residuals** — NEGATIVE. Extending depth-only attention to attend over depth+sequence in a 2D causal mask. Worse than both baseline (+0.028 nats) and depth-only (+0.049 nats), and 1.8x slower. See `research/questions/causal-triangle-attention/README.md`.
- **Async volatile-memory prototype** — POSITIVE (mechanism viable). Dense execution with stale shared-memory reads: mechanically verified, trains within noise of sync control (best val 1.658 vs 1.657), no wall-clock overhead. Demonstrates the execution model Max asked for. See `research/questions/async-volatile-memory/README.md`.
- **Self-prediction / compute compression** — NEGATIVE. Adding auxiliary KL loss (shallow logits → detached deep logits) to the dynamic-depth GRU. At every fixed depth, the self-prediction variant is slightly worse (Δ +0.011 to +0.018 nats). Halting frontier also worse. See `research/questions/self-prediction-compute-compression/README.md`.

- **Async GRU combination** — QUALIFIED POSITIVE at 1M. GRU modules in the async shared-memory architecture: converges, no measurable throughput overhead (108K vs 104K tok/s), but best-val is +0.006 nats worse than sync in a single-seed run. See `research/questions/async-gru/README.md`.
- **Async GRU scale-up (100k data)** — POSITIVE on saturated data. At 11M params (d_model=512), best-val gap is noise on the 100k/20k slice (mean -0.003 ± 0.009 across 5 seeds). But both variants saturate by step 750-1000 — ceiling masks real differences. Throughput: async ~1.6% slower at 11M. See `research/questions/async-gru-scaleup/README.md`.
- **Async GRU larger-corpus calibration** — MODIFYING. On 900k/100k TinyShakespeare (model still improving at step 5000), async was +0.012 nats worse than sync on one seed. A second run (broadcast experiment control) showed only +0.002. True gap likely in the range 0.002–0.012; multi-seed needed. See `experiments/async_gru_corpus/`.
- **Broadcast channel** — NEGATIVE. Simple mean-pool broadcast (read all module deltas, project, add to shared state) actively hurt: async+broadcast was +0.005 worse than plain async. The naive global channel doesn't compensate for stale reads. See `experiments/broadcast_channel/`.
- **Multi-seed 900k/100k async calibration** — CALIBRATING. 5 seeds on 11M/900k: mean async penalty +0.0055 ± 0.0046 nats (95% CI crosses zero). The gap is real but tiny and not statistically significant. See `experiments/async_gru_corpus/artifacts/multiseed_corpus_report.json`.

## Assessment

Thirteen experiments/investigations completed. Useful data, but **direction correction needed** per dictations 2026-05-20-15 and 2026-05-20-16.

### What the experiments established (still valid as evidence)

1. **RNNs are significantly faster** than transformers on this hardware — Tensor Core paths, kernel consolidation, weight reuse
2. **Stale-read/shared-memory semantics** are mechanically stable and trainable
3. **The async quality cost is small** (~0.005 nats mean, not significant from 5 seeds) — whether on GRU or other architectures
4. **Simple global communication doesn't help** — mean-pool broadcast channel made things worse
5. **Per-token conditional sparsity breaks GPU parallelism**
6. **The hardware story is understood** — arithmetic intensity is the right mental model

### What we got wrong (dictation 2026-05-20-15)

Max does NOT want stock GRU/RNN gating mechanisms. The experiments used the wrong architectural primitive. His actual interest:

- **Residual streams across time** — the same residual concept from transformers, but across timesteps
- **Attention over the past** — as temporal coupling mechanism
- **Learned mix-add operator** — for combining residual stream across time
- **Diagonal residual connections** — block A at time T to next block at time T+1
- **Async as speed** — true parallel execution giving FLOPs/sec improvement, not accuracy
- **Time compression/dilation** — different modules at different effective rates via stale reads
- **Theory first** — concept clarification should be majority of reports; experiments follow understanding

The GRU work tells us stale reads are mechanically stable and GPU hardware is well-understood. But it does NOT address the architecture Max wants to explore.

## Immediate next step (REORIENTED)

**1. Theory-first architecture analysis** — DONE. See `research/questions/residual-stream-across-time/README.md`.

**2. Minimal faithful architecture probe** — DONE (basic). Results:
- Without norm control: broken (stream RMS explodes to 1555)
- With temporal pre-norm (LayerNorm): val loss 1.788 at 481K params
- With learned mix-add (no LN): val loss 1.791 at 479K params — **matches LN, Max's preferred approach**
- Diagonal coupling (2 blocks): val loss 1.788 at 777K — inconclusive (same loss, more params)
- For reference: transformer anchor 1.63 at 186K params

Key finding: temporal norm management is essential. Mix-add works as well as LayerNorm. The architecture IS viable.

**3. Extended training** — DONE. Mix-add probe at 100k chars / 5 epochs: **val loss 1.670** (only 0.038 from transformer anchor 1.632). Still improving at epoch 5. The architecture converges toward transformer quality with more training. One training spike at epoch 4 (recovered). See `experiments/residual_stream_time_mixadd/artifacts/extended-100k-5ep/`.

**4. Stop-gradient across time** — DONE. Detaching gradients between timesteps: val loss **1.894** vs baseline 1.670 — **+0.223 nats cost**. Training was 2.36x faster (no backprop-through-time). The quality hit is too large for "free async" but the throughput gain is real. See `experiments/residual_stream_time_stopgrad/`.

**5. Partial detach sweep** — DONE. Tested N=2, 4, 8, 16. **N=4 is the sweet spot:** val loss 1.711 (+0.041 from baseline) at 2.46× speed. Full stop-grad costs +0.223; partial detach every 4 steps costs only +0.041 at the same speed. Sharp knee in the curve — matches temporal attention window k=4. See `experiments/residual_stream_time_partial_detach/artifacts/sweep/`.

**6. Parameter-matched comparison** — DONE. At d_model=116 (184K params ≈ transformer's 186K): val loss **1.717** vs transformer **1.632** — gap of **+0.085 nats**. The architecture IS less efficient at matched params, but only moderately. The d=192 model (1.670) benefits from excess capacity. See `experiments/residual_stream_time_partial_detach/artifacts/param_matched/`.

**7. Window size ablation** — DONE (NEGATIVE). Both k=8 and k=16 diverge catastrophically. The mix-add architecture has a stability boundary at k≈4 with current LR/clip settings. Larger temporal windows amplify gradients through attention until explosion. k=4 is both the stability limit and the partial-detach sweet spot — likely the same underlying constraint. See `experiments/residual_stream_time_partial_detach/artifacts/window_ablation/`.

**8. Next discriminating experiment:** The architecture is now thoroughly characterized:
- Works at k=4, d=192: val 1.670 (near transformer)
- Moderate efficiency gap: +0.085 at matched params
- Async tradeoff: N=4 partial detach gives 2.46× speed for +0.041 nats
- Stability limit: k>4 diverges with mix-add

Open directions:
- **Stabilize larger windows** — add RMSNorm to attention Q/K, or reduce LR. Could close efficiency gap.
- **Multi-block pipelining** — actual async inference demo with 4-step pipeline stages.
- **Move to a different experiment family** — the residual-stream-across-time thread is well-explored. Consider image patches (Phase 6) or other Phase 3 mechanisms.

**9. Future work (from dictations, not immediate):**
- Mix-add as default over LayerNorm in all experiments
- Async/stale-read execution for throughput (speed demonstration)
- Diagonal coupling at larger scale / more blocks
- Broadcast mechanism (attention-based, async)
- Local learning concept clarification
