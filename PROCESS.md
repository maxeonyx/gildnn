# Process

How work gets done in this project.

This is personal hobbyist ML research. The goal is discovery: try ideas, run experiments, look at outputs, and learn what actually happens. Keep the codebase small. Keep conclusions honest. Redo from scratch is always an option.

---

## The autonomous loop

`loop.ps1` relaunches OpenCode if it exits. On relaunch, the agent should reorient quickly and continue.

On session start:

1. Read `PLAN.md` and any `TASK-*.ignore.md` files in the repo root.
2. Check the time and whether a daily or weekly report is due.
3. Check whether a background run is active (`runs/active.lock`).
4. Continue the current task if there is one. Otherwise pick the cheapest useful next step.

Max may speak during a session, but do not assume that means he is back and available. Keep working unless he clearly takes over.

If the last session felt wrong, fix the process first. Bad process compounds.

### How to choose what to do next

Use judgment, not ceremony.

Priority order:

1. Fix broken or misleading process/docs
2. Write any due reports
3. Clean up half-finished or confusing work
4. Integrate finished experimental code into `core/`
5. Continue an existing experiment thread
6. Start a new experiment

When choosing between possible next steps, prefer the one that is:

- cheapest to run honestly
- most likely to produce inspectable evidence
- least likely to bloat the codebase
- most useful for narrowing an open question

If the work feels tangled, restart from a smaller simpler version instead of protecting the tangled version.

---

## Experiment discipline

### The ladder

Every experiment climbs this ladder before scaling:

1. **Overfit one batch.** If it cannot memorize one batch to near-zero loss, nothing else matters.
2. **Tiny model, tiny data.** Run the full pipeline end to end on something trivial.
3. **Inspect actual outputs.** Look at what the model produces, not just loss curves.
4. **Verify on multiple datasets before scaling.** Do not draw broad conclusions from one tiny dataset.
5. **Scale in steps.** Tiny → small → medium. Verify at each step.
6. **Change one thing at a time.** Do not change architecture, data, and training setup all at once.

This ladder is non-negotiable.

### Report-first experiment protocol

Every experiment starts by writing the report first in `research/questions/<question>/README.md`.

Create or update the question README with:

- which dictation or `VISION.md` goal the experiment serves, with a link to the specific dictation when there is one
- the simplification chosen and why it is a legitimate cut
- a clear architecture explanation, using real code snippets and diagrams where that helps
- the hypotheses being tested
- the planned evidence: what artifacts the run will produce and why those artifacts answer the question
- placeholder sections for results, next steps, and deferrals
- explicit non-goals: what this experiment will not settle

Fill in results, next steps, and reasons for deferral as the experiment proceeds, not only at the end.

### Before starting a new experiment

Do not do a formal selection ritual.

Just make sure you can answer, briefly:

- What question am I trying to answer?
- What is the cheapest experiment that could teach me something about it?
- What evidence do I need to save?
- What will this experiment **not** settle?

If those answers are obvious from context, just proceed. If they are not obvious, write 3–6 bullets in the relevant question folder, task file, or commit message and move on.

### Honest scope

Keep experiments narrow enough that the result is interpretable, but not so narrow that you spend all your time selecting instead of running.

Good outcomes:

- a working result
- a negative result
- an ambiguous result with saved artifacts
- discovering that the current framing was wrong and replacing it with a simpler one

Ambiguity is allowed. Failure is allowed. Needing to restart from a simpler version is allowed.

### Starting point for any new capability

Start with the simplest version that could possibly work. For example: tiny context, tiny model, next-token prediction, minimal dataset, inspectable outputs.

Do not smuggle in architectural complexity "for later." Add complexity only after the simpler version genuinely works.

### Reproducibility

- Record git SHA, hyperparameters, and seed for meaningful runs.
- Save enough config that a result can be recreated without guesswork.
- Match the level of rigor to the claim. Tiny exploratory probes can be light; comparison claims need stronger control.

### Checkpointing and artifacts

- Checkpoint during longer runs.
- Keep only what is useful: latest, best, and a few milestones.
- Do not keep old weights forever.
- Never commit model weights or datasets to git.
- Save the artifacts that actually answer the question: sample outputs, tables, plots, example failures, short notes.

A run that "worked" but produced no useful evidence is only half done.

### Dead-end detection

After a few failed attempts at the same idea, stop varying knobs blindly.

Ask:

- Is the implementation broken?
- Is the task badly framed?
- Is the idea itself weak?
- Is there a smaller version that would answer this faster?

Then either simplify, reframe, or drop it.

---

## Integration and code shape

### Keep the codebase small

Small is a success metric.

Delete aggressively. Consolidate shared logic early. Do not let experiments pile up as parallel mini-frameworks.

### Integrate before experimenting

Integrate understood, verified components before starting new experiment threads. But do not promote one-session-old code into `core/` just because it worked once; wait for stability evidence.

Directory roles:

- `experiments/<experiment-name>/` — per-experiment directories containing scripts, configs, and grungy local artifacts such as prediction dumps, raw outputs, and debug files
- `research/questions/<question>/README.md` — Max-readable reports with inline evidence; embed tables, code snippets, and diagrams directly in markdown instead of scattering standalone text artifacts
- `base-experiments/<model>/` — reproducible standard-model training that reaches the expected validation loss on the primary dataset
- `core/` — shared reusable dataset, model, and training components used by both `base-experiments/` and `experiments/`

If two experiments share logic, move that logic into `core/`.

If an experimental branch taught nothing and is only clutter, delete it.

### Architectural options

Open questions stay open.

Keep selectable variants only when they produced an informative comparison or are still actively useful. Do not keep every dead path forever.

---

## Experiment visibility

**Before running any experiment expected to take more than ~30 seconds, the subagent must stop and report back to the orchestrating agent.** The report must include:

- What experiment is about to run
- Expected duration (estimate from corpus size, step count, prior runs)
- What it will produce

The orchestrator then logs a visibility note (so Max can see what's happening if he checks the loop) and resumes the subagent to proceed.

This prevents silent multi-hour waits where Max has no idea what's happening. The loop output should always show what we're doing and how long it's expected to take.

## Background training runs

If a run will take longer than about 5–10 minutes, run it in the background.

Rules:

- Start at most one training run at a time.
- While a run is active, do not wait idly.
- Do cleanup, integration, reporting, or analysis while it runs.
- Check progress from logs without tight polling.

Use `runs/active.lock` to record the active run. Remove it when the run ends or fails.

---

## Literature practice

This is not an academic literature review.

Read papers only when they would genuinely change what you do next, for example:

- someone tried almost the same idea
- someone has evidence against a key assumption
- someone has a clearly better solution to the same problem

Write up relevant findings in the appropriate `research/questions/` folder.

---

## Reporting

### When reports are due

On session start, check whether a report is due.

- If it is after 4pm and there is no daily report for today, write one at the next natural stopping point.
- If it is after 4pm Thursday and there is no weekly report for this week, write that too.

Do not stop mid-experiment just to report. Finish the current small unit cleanly, then write before starting something substantial.

### Quality standard

Daily and weekly reports are for Max. They should be dense, readable, and evidence-backed.

Every factual claim should point to a real artifact: a run log, metric, sample output, saved image, or concrete file.

No free-floating assertions. Open questions should remain open.

Question READMEs should clearly explain the architecture with implementation-aligned code snippets and diagrams, link the work back to Max's goals or dictations, and make the next steps obvious, including why those next steps have not been pursued yet.

Question-folder notes can be rougher. Daily and weekly narratives should be polished.

---

## Information architecture

Files should have clear jobs. Rewrite them when reality changes; do not append stale process sediment forever.

Keep this file about how to work, not about the current state of specific research threads. Current state belongs in `PLAN.md` and `research/questions/`.

**Process changes must propagate to AGENTS.md files.** This file (PROCESS.md) is primarily read by the orchestrating agent. But subagents read `AGENTS.md` files — the root one and the per-directory ones. When the process changes in ways that affect how subagents work in specific directories (file layout conventions, experiment protocols, what goes where), those rules must be reflected in the relevant `AGENTS.md` files, not just here. PROCESS.md defines the process; AGENTS.md files enforce it at the point of work.
