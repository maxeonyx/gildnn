# Plan

Working file. Rewrite it as the state changes.

## Current situation

- This repo now has the right process direction in `PROCESS.md`: report-first experiments, explicit directory roles, experiment visibility, and background-run rules are written down.
- The trust foundation is still missing. `base-experiments/` is still effectively empty, so there is not yet a verified standard baseline that matches expected published-quality results on the project dataset.
- `core/` is still minimal (`fixed_window_char.py`, `tiny_char_transformer.py`). It is not yet the reusable shared implementation Max expected.
- `experiments/` is still mostly a flat set of scripts rather than per-experiment directories.
- Existing custom-model results may still be interesting, but they are not yet grounded against verified baselines. Reports must stay honest about that.
- Prompt-stated completed work for this phase: predictive-chain report rewrite, experiment/process fixes, and dynamic-depth seed-stability work. Treat those as done enough to move forward unless inspection shows otherwise.

## Goal for the remaining ~10 days

Produce at least one trustworthy baseline foundation and use it to re-ground the most valuable existing experiment reports.

The key question every serious report must be able to answer is: **why can I trust this result?**

Right now the only acceptable answer is some version of: **because our own standard-model implementation reaches the expected loss range on the same dataset, so comparisons against our custom models mean something.**

## Priority order

1. Establish the baseline trust foundation.
2. Integrate verified baseline components into `core/`.
3. Do only the structural cleanup needed to support baseline work and readable reports.
4. Re-state predictive-chain and dynamic-depth results relative to verified baselines.
5. Start no new research thread until the first baseline is trustworthy.

## Execution order

### 0. Every loop start

1. Read this file and any `TASK-*.ignore.md` files.
2. Check whether a daily or weekly report is due.
3. Check `runs/active.lock`.
4. If a long run is active, do analysis / cleanup / reporting work while it runs.

### 1. Baseline foundation — do this next

Objective: define and reproduce a trustworthy standard comparison point.

1. **Choose the primary baseline task and target range.**
   - Confirm the exact dataset/task to use for the trust anchor, likely the current Shakespeare character-level setup unless a better-established dataset is already present in the repo.
   - Research what validation/test loss range is expected from standard models of the sizes we intend to compare against.
   - Record the comparison shape we need in reports: model type, parameter count, loss, wall time, and important training-cost notes such as sequential vs parallel behavior.
2. **Build the first standard baseline in `base-experiments/`.**
   - Prefer the cheapest standard model most likely to reach a believable target quickly.
   - Follow the experiment ladder: overfit one batch, tiny run, inspect outputs, then scale.
   - Save enough evidence to support an honest report.
3. **Verify the baseline honestly.**
   - If it reaches the expected range, it becomes the first trust anchor.
   - If it does not, do not write a comparison report anyway; debug or rerun until the baseline is either trustworthy or clearly blocked.
4. **Integrate reusable pieces into `core/`.**
   - Move only the stable shared dataset/model/training parts needed by the verified baseline.
   - Keep `base-experiments/` as reproducible training entrypoints, not a dumping ground for shared logic.
5. **Only after that, add the second standard baseline.**
   - Prefer the most informative complement to the first one: RNN if transformer came first, or transformer if RNN came first.
   - Feedforward is desirable but should not block the first trust anchor.

### 2. Structural cleanup — only where it helps Phase 1 or Phase 3

1. Restructure only the active threads in `experiments/` into per-experiment directories.
   - Minimum scope: baseline training, predictive chain, dynamic depth.
2. Move standalone artifacts out of `research/questions/` where they should instead be embedded inline in markdown or relocated under `experiments/`.
3. Do not attempt full historical cleanup if it does not help the trust-foundation path.

### 3. Re-ground the existing experimental results

Do this only after at least one verified baseline exists.

1. Rewrite or update the predictive-chain report so its claims are framed relative to the verified baseline(s).
2. Rewrite or update the dynamic-depth report to the same standard.
3. If an existing comparison does not hold up once the proper baseline exists, rerun the experiment instead of shipping a weak report.

## Stop conditions for this plan window

This ~10-day window is successful if all of the following are true:

1. `base-experiments/` contains at least one standard model run that honestly matches the expected quality range on the chosen dataset.
2. The reusable implementation behind that run has been integrated into `core/`.
3. At least one important custom-model report has been rewritten so it can answer Max's trust question honestly.

## Explicit non-goals for this window

- Do not try to finish every baseline before delivering value.
- Do not do repo-wide perfectionist cleanup.
- Do not start a new custom architecture thread before the first trust anchor exists.
- Do not present unverified comparisons as settled conclusions.

## Immediate next step

Start Phase 1: identify the primary baseline dataset/task and the expected published loss range for standard models of the sizes we care about, then set up the first `base-experiments/` baseline around that target.
