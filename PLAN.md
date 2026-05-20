# Plan

Working file. Rewrite it as the state changes.

## Current situation

- The trust foundation for text is now in place: `base_experiments/` has a trustworthy transformer anchor and a trustworthy vanilla RNN anchor on the fixed TinyShakespeare frame.
- The predictive-chain and dynamic-depth reports have been re-framed against that baseline anchor so they no longer overclaim.
- New dictations changed the near-term priority. Early composition of predictive chain + dynamic depth is **not** the main thing Max wants right now.
- The current priority is thorough isolated experiments on the individual performance/mechanism pieces: local learning, stop gradients, asynchronous/asynchronized modules, keeping weights in memory, processing activations in place, and replacing depth with width / propagation across timesteps.
- "Predictive chain" remains a useful experiment label for one simplification, but it is not the project goal. It should not silently drive the plan.
- There is partially completed chain+dynamic-depth work in `experiments/chain_dynamic_depth/`, but that thread is currently paused in planning until it is explicitly re-scoped.

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

### Phase 3 — Isolated mechanism experiments for fast recurrent training

Objective: test the individual pieces Max actually cares about before composing them.

#### 3a. Local learning / stop gradients (first)

Not a "chain" from one end to the other. Each module predicts its own input (A→A), with auxiliary bits hanging off. The boundary/shape of a "local module" is itself experimental — maybe ~3 nodes.

Key questions:
- Does a multi-module model beat a single-module model?
- Does adding more modules improve performance?
- How does it compare at same params / same FLOPs / same wall-clock time?

#### 3b. GPU utilization research

What kind of GPU programs actually fit best on the RTX 3090? Can RNNs get significantly more FLOPs out of the GPU than transformers? Measure wall-clock, actual utilization, memory bandwidth. This grounds all future performance claims.

#### 3c. Attention residual transformer

Multiple variants:
- Attention over depth only
- Attention over depth AND sequence length in a causal triangle (can't attend to same layer at previous step)

Motivation: to parallelize across time by adding only one depth per timestep for the RNN. The causal triangle shows how this could work.

Note: Max doesn't want too much transformer focus. This experiment serves the RNN goal — it demonstrates the mechanism that will later apply to RNN depth.

#### 3d. Stop gradients / overlapping stop gradients

Even just a tiny example of training with stop gradients or overlapping stop gradients. This is separate from the async question — it's about whether local learning works at all.

#### 3e. Async execution without synchrony

The important question: can we run modules in parallel with volatile shared memory between them, so they don't have to synchronize at the GPU level? We want to train without synchrony. Even a tiny working example would be valuable.

Rules:

- Prefer the cheapest isolated experiment that could produce real evidence about one of those pieces.
- Do not smuggle composition back in by changing several mechanisms at once.
- If a simplification came from an earlier agent rather than the dictations, keep that fact visible in the write-up.

Questions this phase should answer:

- Can local learning / stop-gradient style training give useful speed or stability benefits?
- Can a model learn a useful trigger for whether more computation is worth doing?
- Can we approximate asynchronous execution without turning the experiment into a completely different architecture?
- What actual GPU utilization do different architectures achieve on this hardware?

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

## Immediate next step

Choose the first isolated post-baseline mechanism question. The leading candidate is local learning (3a) — multi-module A→A prediction with stop gradients, compared against single-module and baselines at same params/FLOPs/wall-clock. Write the report-first README before touching code.

Also: the baseline reports (`base_experiments/README.md`) need example inputs and outputs at different loss stages from multiple models. Max wants to see what the models actually produce as they train, not just final numbers.
