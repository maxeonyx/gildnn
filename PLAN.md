# Plan

Working file. Rewrite it as the state changes.

---

## Grounded current state

- Max is broadly happy with progress, but explicitly says he "hasn't got anything out of it yet" because the narrative connecting experiments to his goals is missing.
- The current repo shape does not match Max's expected information architecture:
  - `experiments/` is a flat set of Python scripts, not per-experiment directories.
  - `research/questions/` contains many standalone artifact files (`.txt`, raw prediction dumps, tables) that Max expected to be either embedded inline in markdown or kept with experiment-local artifacts.
  - `base-experiments/` is effectively empty.
  - `core/` is still minimal (`fixed_window_char.py`, `tiny_char_transformer.py`).
- The current predictive-chain write-up contains real results, but the narrative gap Max identified is real: it does not adequately justify how the experiment follows from his dictations, does not clearly explain the architecture, and current planning language is too settled.
- Time horizon is short: about 10 days remain. The plan must optimize for Max getting something meaningful out of the project, not for exhaustive cleanup.

## What counts as success for the remaining time

Minimum meaningful outcome:

1. Max can read one or two question reports and immediately see:
   - which dictation goals they trace back to,
   - what was hypothesized,
   - what architecture was actually built,
   - what result was obtained,
   - what the obvious next steps are and why they were not done yet.
2. There is at least one trustworthy baseline package in `base-experiments/` showing that a standard model from our own implementation can achieve the expected quality range on the chosen dataset.
3. `core/` has started becoming a reusable model library rather than a placeholder.
4. The repo layout stops fighting future work.

Non-goal for the remaining time: finishing every baseline, rewriting every historical artifact, or proving the final architecture direction.

## Priority order across all work

1. Fix experiment/report process so new work stops making the same mistakes.
2. Rewrite the predictive-chain narrative so Max can actually get value from existing results.
3. Establish at least one verified standard baseline in `base-experiments/`, then integrate the reusable parts into `core/`.
4. Do only the structural cleanup needed to support the above; defer nice-to-have reorganization.

---

## Immediate process fixes

### 1. Report-first experiment protocol
- **What:** Before any new experiment, create or update the question README first with: motivating dictation links, hypothesis, architecture sketch, planned evidence, placeholders for results, and explicit non-goals.
- **Why it matters:** Max explicitly said write-up should start at the beginning, not the end, and that the missing narrative is the current bottleneck.
- **Rough size:** quick
- **Dependencies:** none

### 2. Tighten doc/process rules to match Max's layout expectations
- **What:** Update process/documentation guidance so it explicitly says:
  - `experiments/` holds per-experiment directories with grungy local artifacts,
  - `research/questions/` holds Max-readable reports with inline tables/snippets/images,
  - obvious next steps and reasons for deferral must appear in reports,
  - architectures must be explained with code snippets/diagrams aligned to real code.
- **Why it matters:** Max's feedback is partly about missing process, not just messy outputs. Without this, cleanup will regress.
- **Rough size:** quick
- **Dependencies:** action 1

### 3. Stop stating unresolved threads as concluded unless Max can follow the case
- **What:** Remove or soften settled language where the evidence may exist but the explanation is not yet good enough for Max to evaluate, especially for predictive chain.
- **Why it matters:** Current `PLAN.md` says predictive chain is concluded; Max's dictation says he does not yet understand how it follows from his goals and cannot really consume the result.
- **Rough size:** quick
- **Dependencies:** none

### 4. Define the single primary baseline target dataset and "expected quality" reference range
- **What:** Pick one baseline dataset for the remaining time (likely the current character-level text task unless evidence says otherwise), and write down what published/expected validation loss range we are trying to reproduce for feedforward, RNN, and transformer.
- **Why it matters:** Max's request is not merely "run baselines"; it is "achieve what we expect from other people's results on that dataset." Without the target range, baseline success is undefined.
- **Rough size:** medium
- **Dependencies:** action 1

---

## Structural cleanup

### 5. Reorganize `experiments/` into per-experiment directories only for active/high-value threads
- **What:** Convert the flat script layout into subdirectories for the threads that still matter in the final 10-day push (at minimum: predictive chain, dynamic depth, baseline training). Each directory should own its scripts, configs, and disposable artifacts.
- **Why it matters:** Max explicitly expected this layout. It also makes experiment-local evidence easier to map into reports.
- **Rough size:** medium
- **Dependencies:** action 2

### 6. Remove low-value standalone text artifacts from `research/questions/` by embedding or relocating them
- **What:** For predictive chain and dynamic depth first, inline ASCII tables and short examples directly into the README; move grungy prediction dumps / raw text outputs to the relevant experiment directories where needed.
- **Why it matters:** This is one of Max's clearest file-layout complaints. It also improves report readability immediately.
- **Rough size:** medium
- **Dependencies:** actions 2, 5

### 7. Create a clean role boundary between `base-experiments/` and `core/`
- **What:** Make `base-experiments/` the home for reproducible standard-model training entrypoints and result records; make `core/` the shared dataset/model/training components those experiments use.
- **Why it matters:** Max described exactly this split in the dictation. Right now neither directory is fulfilling its role.
- **Rough size:** medium
- **Dependencies:** action 4

### 8. Defer broad historical cleanup outside the critical threads
- **What:** Do not attempt a repo-wide artifact purge unless it directly helps the remaining narrative/baseline work. Limit cleanup to predictive chain, dynamic depth, and the new baseline path.
- **Why it matters:** Ten days is too short for perfectionist reorganization. This protects the real critical path.
- **Rough size:** quick
- **Dependencies:** none

---

## Baseline establishment

### 9. Build one trustworthy end-to-end baseline first: standard RNN or transformer on the primary char dataset
- **What:** Choose the cheapest standard architecture most likely to reach a believable target quickly, train it cleanly in `base-experiments/`, and verify it against the expected validation-loss range.
- **Why it matters:** Max wants verified known-good reference points. One solid baseline is more valuable in 10 days than three half-verified ones.
- **Rough size:** medium
- **Dependencies:** actions 4, 7

### 10. Integrate the reusable training/data/model pieces from that baseline into `core/`
- **What:** Extract the parts needed to reproduce the successful baseline cleanly from `core/`, rather than leaving the logic stranded in one experiment script.
- **Why it matters:** Max wants `core/` to become a flexible model library that can reproduce standard results and then be extended.
- **Rough size:** medium
- **Dependencies:** action 9

### 11. Add the second standard baseline only if the first is already trustworthy
- **What:** After one baseline is reproducible, add the next most informative comparator (probably transformer if the first is RNN, or vice versa). Feedforward can be included if cheap, but should not block the first two.
- **Why it matters:** Max asked for feedforward, RNN, and transformer, but with 10 days left the practical minimum is to establish the pattern with one and preferably two standard baselines.
- **Rough size:** large
- **Dependencies:** actions 9, 10

### 12. Use baseline comparison to recalibrate claims about custom architectures
- **What:** Once at least one baseline is trustworthy, restate predictive-chain and dynamic-depth results relative to that known-good reference instead of relative only to earlier ad hoc baselines.
- **Why it matters:** This is the bridge from "interesting experiments happened" to "Max got something meaningful and interpretable out of them."
- **Rough size:** medium
- **Dependencies:** actions 9 and ideally 11

---

## Narrative improvement

### 13. Rewrite the predictive-chain report from Max's goals outward
- **What:** Replace the current result-heavy README with a narrative that starts from the relevant dictation ideas, states what simplification was chosen and why, shows the architecture clearly (diagram + code snippet), records hypotheses before results, and ends with obvious next steps plus reasons they were deferred.
- **Why it matters:** This is the clearest pain point Max named, and likely the highest-value thing already available to improve.
- **Rough size:** large
- **Dependencies:** actions 1, 3, 6

### 14. Rewrite the dynamic-depth report to the same standard, but only after predictive chain
- **What:** Apply the same structure to `research/questions/dynamic-depth/README.md`: dictation traceability, architecture explanation, inline evidence, hypotheses/placeholders, next steps with reasons.
- **Why it matters:** Dynamic depth already seems promising; a good report could make it one of the first areas where Max feels he has "got something out of it."
- **Rough size:** medium
- **Dependencies:** actions 1, 6

### 15. Add a short "why this experiment exists" preamble to every active question report
- **What:** For each active thread, add a concise opening section that answers: which part of VISION/dictations this serves, why this simplification is legitimate, and what this thread will not answer.
- **Why it matters:** This is the missing connective tissue Max keeps asking for.
- **Rough size:** quick
- **Dependencies:** action 1

### 16. Preserve obvious next steps without pretending they are commitments
- **What:** Every major report should end with: next discriminating experiments, why they matter, and why they were not pursued yet (timebox, missing baseline, too much confounding, etc.).
- **Why it matters:** Max explicitly asked for this, and it prevents the write-ups from feeling arbitrarily cut off.
- **Rough size:** quick
- **Dependencies:** action 1

---

## Recommended sequence for the remaining ~10 days

### Phase 1: fix the process and salvage understanding (days 1-2)
1. Action 1 — report-first experiment protocol
2. Action 2 — tighten doc/process rules
3. Action 3 — remove overstated closure language
4. Action 13 — rewrite predictive-chain report enough that Max can follow it

### Phase 2: create one real baseline foundation (days 3-6)
5. Action 4 — define primary dataset + expected quality target
6. Action 7 — define `base-experiments/` / `core/` boundary
7. Action 9 — build first trustworthy baseline
8. Action 10 — integrate reusable pieces into `core/`

### Phase 3: cleanup only what supports the story (days 5-8, overlapping where possible)
9. Action 5 — restructure active experiment threads into directories
10. Action 6 — move/embed low-value question-folder artifacts
11. Action 14 — rewrite dynamic-depth report

### Phase 4: only if time remains (days 8-10)
12. Action 11 — second baseline
13. Action 12 — restate custom-model claims relative to verified baselines
14. Action 15 + 16 — apply consistent preambles/next-steps framing to active reports

---

## Explicit deferrals

- Do not attempt a full repo-wide migration of every historical experiment into the new structure unless it directly supports the active threads.
- Do not start a new research thread before at least one baseline and one rewritten report exist.
- Do not treat feedforward/RNN/transformer all as mandatory before Max sees value; the first critical milestone is one well-explained report plus one trusted baseline.

## Immediate next step

Rewrite process/state docs so future work starts with report-first planning and so predictive-chain is no longer described as settled in a way Max cannot yet evaluate. Then rewrite the predictive-chain report before doing more model work.
