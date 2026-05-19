# Plan

Working file. Rewrite it as the state changes.

## Current situation

- Process direction is in much better shape: report-first experiments, directory roles, experiment visibility, and background-run rules are now written down.
- Two research threads already produced real findings:
  - **Predictive chain:** local predictive learning works, unhooked gradients work, the small-scale gain looks like regularization, the advantage disappears at larger scale, and the earlier graph-shortcut variant hurt rather than helped on this task.
  - **Dynamic depth:** the mechanism works on character LM, the loss-prediction head becomes useful with enough data, and there is a real compute/quality tradeoff (~43% compute savings for ~1% loss hit at one operating point), but the operating point is seed-dependent.
- The main missing foundation is still the same: `base-experiments/` does not yet contain trustworthy standard baselines with correctness checks, so future claims are not grounded against a reference Max can trust.
- That baseline gap is real, but it is **not** the whole remaining project. Max's current vision work is after the baselines: composition of predictive modules + dynamic depth, selective triggering, async/desynchronized execution, and arbitrary-order image patches.
- The plan should spend the first part earning trust, then spend the rest on the actual architectural questions.

## Execution phases with clear sequencing

### Global rules for every phase

1. Every experiment starts with its report folder/README first: question, simplification, hypotheses, comparison target, and placeholders for evidence.
2. No new architectural thread starts without a baseline for that modality/task to compare against.
3. Base experiments follow the same ladder as everything else: overfit one batch, tiny run, inspect outputs, then scale.
4. Base experiments also get cheap correctness tests where mistakes are easy and expensive: data slicing/targets, embedding lookup shape/content, loss masking/accounting, sampling/generation path, and any modality-specific preprocessing.
5. Keep experiment scope small, but move quickly from one discriminating experiment to the next. Do not spend days polishing an already-answered question.

### Phase 1 — Trust foundation for text baselines (start immediately)

Objective: make text results interpretable enough that later custom-architecture comparisons mean something.

1. Choose and lock the primary text baseline task and target range for the current project dataset.
2. Build the first trustworthy standard baseline in `base-experiments/` with the full ladder and correctness tests.
   - Prefer the cheapest baseline most likely to establish trust quickly.
   - Save evidence that would let a fresh agent answer: why do we believe this implementation is correct?
3. Build the second text reference baseline once the first trust anchor exists.
   - The target baseline set for this window is **ordinary RNN + ordinary transformer** on the same character-level task.
   - Feedforward is optional if it is cheap and clarifies something, but it does not block the rest of the window.
4. Move only genuinely shared, now-verified pieces into `core/`.
5. As soon as there is enough baseline evidence to support honest comparison, move on. Baselines are a gate, not the destination.

End state required to leave Phase 1:

- At least one text baseline is trustworthy enough to be the comparison anchor.
- The base-experiment path has cheap correctness tests.
- The comparison frame for future reports is fixed: parameter count, loss, compute cost, and important execution constraints.

### Phase 2 — Re-ground the current findings against the new trust anchor

Objective: stop carrying around pre-baseline results as if they stand alone.

1. Update the predictive-chain report framing so it is explicit about what is now grounded by the baseline and what still is not.
2. Update the dynamic-depth report framing the same way.
3. Standardize the comparison table shape that future experiment READMEs should use.
4. If any existing claim no longer survives honest comparison framing, rerun or weaken the claim rather than carrying it forward.

End state required to leave Phase 2:

- The two strongest completed threads are now anchored to the same comparison frame future work will use.

### Phase 3 — First composition experiment: dynamic depth inside the predictive-chain family

Objective: test the most obvious combination Max already asked for, using existing positive results rather than starting from scratch.

1. Create a new question/report for the composition experiment before coding.
2. Implement the smallest honest version of **dynamic depth inside the predictive-chain family**.
   - The question is not "can we make it fancy?" It is: does adaptive compute add anything once local predictive modules already exist?
3. Compare against four references where possible:
   - ordinary RNN baseline
   - ordinary transformer baseline
   - predictive chain without dynamic depth
   - dynamic-depth model without predictive chain structure
4. Measure both quality and compute, not just loss.
5. Decide whether the combination is additive, redundant, unstable, or only useful in a narrow regime.

This phase is important because it directly tests whether the two strongest surviving text ideas actually compose, instead of leaving them as separate curiosities.

### Phase 4 — Async/desynchronized execution path on text

Objective: test the part of the cortical-column vision that the predictive-chain work did **not** test: selective updates, stale communication, and the beginning of asynchronous execution.

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

### Phase 5 — Second modality: arbitrary-order image patches

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

### Phase 6 — Final combination and synthesis

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
3. At least **two real post-baseline vision experiments** are completed, with one of them testing a combination rather than a standalone idea.
4. At least one experiment directly tests selective/dynamic computation rather than only static architecture.

### Strong success for this window

This window is a strong success if, in addition to the floor above:

1. The async/selective-update path yields a clear positive or negative answer that narrows the cortical-column design space.
2. The image-patch modality has at least a baseline plus one first architectural probe.
3. The final state of the repo answers not just "can we trust the comparisons?" but also "which of Max's architectural ideas are now worth pushing further, and which are not?"

### Explicit cut rules

- Do not spend multiple days polishing baseline quality once the trust question is answered well enough to compare.
- Do not repeat an experiment class that already gave a clear answer unless the new version changes the actual hypothesis being tested.
- If time gets tight, cut breadth before cutting the highest-value composition work.
- If one late-phase thread is blocked, switch to the next discriminating experiment rather than burning the remaining window on setup/debugging.

## Immediate next step

Start Phase 1 properly: lock the text baseline task and expected target range, define the cheap correctness tests the base experiments need, create the first base-experiment report/README, and begin the overfit→tiny→inspect→scale ladder for the first trustworthy text baseline.
