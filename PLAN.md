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
- **GPU utilization study** — CONFIRMED Max's hypothesis. At matched params on RTX 3090: GRU is 1.8-2.8x faster training than transformer, LSTM is 1.4-2.5x faster. Decode: GRU 2.5-3.7x faster (constant cost vs transformer's growing KV cache). See `research/questions/gpu-utilization/README.md`.

## Assessment of boundary mechanisms

All three isolated boundary mechanisms from the vision have been tested at ~186K params on TinyShakespeare:
1. Stop-gradient local learning: clearly negative on quality
2. Depth attention residuals: marginal, not worth the compute cost
3. Selective block execution: mechanism works but GPU execution negates compute savings

**None of the proposed boundary mechanisms help at this scale/budget in isolation.**

**However — dictation review reveals important framing mismatch.** Max's primary question was never "do these improve quality at tiny scale?" It was "can we make inference/training FAST through async execution and parallelism?" Quality degradation from stop-gradients is *acceptable* to Max if it unlocks performance. The experiments answered the wrong primary question.

Key untested ideas from dictations:
- **GPU utilization study** (Max: "I can't believe we haven't written that down yet") — RNN vs transformer wall-clock, actual GPU utilization on RTX 3090
- **Attention residual causal triangle** — depth+sequence variant, explicitly requested, not delivered
- **True async/volatile-memory prototype** — even a toy untrained example would satisfy Max
- **Broadcast router** — global communication channel to all modules
- **Dynamic depth in isolation** — previously confounded with predictive chain

## Immediate next step

**Causal triangle attention residuals.** This is the explicitly-requested second variant of depth attention — attending across both depth AND sequence in a causal triangle mask. Max's words: "another one where every attention model is attentive over depth and sequence length in a causal triangle. So it can't attend to the same layer in previous step."

This is interesting because it bridges toward Max's looped-block RNN vision — attention residuals across time are essentially what makes looped transformer blocks equivalent to an RNN with attention over history.

Other candidates after that:
- True async/volatile-memory prototype (toy, possibly untrained)
- Broadcast router experiment
- Dynamic depth in isolation
