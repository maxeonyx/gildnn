# Process

How work gets done in this project. Covers the autonomous harness, experiment discipline, code standards, literature practice, and reporting.

---

## The autonomous harness

The project runs as an outer loop: `harness.ps1` launches OpenCode, which works until it exits, then relaunches it with a reorientation prompt. The harness lives in the repo and should be refined as the process matures.

**The agent's job when relaunched:**

1. Read `PLAN.md` and any `TASK-*.ignore.md` files in root — orient on the current task and state.
2. Check if a training run is active. If yes — sleep until it completes. Do not exit.
3. Pick up the next task. Continue working.
4. Max may check in during a session. That does not mean he's back. Continue working.

**Priority order when choosing what to work on:**

1. Clean up anything that's broken or half-finished
2. Make the codebase smaller — delete, simplify, consolidate
3. Refactor the process (PROCESS.md, AGENTS.md, harness)
4. Refactor the repo — integrate experimental code into core
5. Continue existing experiments
6. Start new experiments

**Redo from scratch is always an option.** If something is tangled and unclear, starting fresh is cheaper than untangling.

**Common exit failure modes to account for in the prompt:**
- Agent exits to "wait" for a training run — wrong. Sleep instead.
- Agent gets confused mid-task and gives up — wrong. Read PLAN.md and any TASK files, continue.
- Agent hits a wall and starts a new experiment instead of integrating — wrong. Integrate first.
- Agent produces a half-finished thing under context pressure — wrong. Say you can't finish it and leave a clear handoff.

**The best outcome** when a task is infeasible in one session: say so clearly, produce a clean handoff, do not deliver something broken.

---

## Experiment discipline

### The ladder (non-negotiable)

Every experiment climbs this ladder before scaling:

1. **Overfit one batch.** Model memorizes a single batch to near-zero loss. If it can't, nothing else matters.
2. **Tiny model, tiny data.** Full pipeline end-to-end on something trivial.
3. **Inspect actual outputs.** Not just loss curves — look at what the model is producing.
4. **Verify on multiple datasets before scaling.** Gradually build up to more datasets of increasing complexity. Don't scale on a single dataset.
5. **Scale in steps.** Tiny → small → medium. Verify at each step. Never jump.
6. **One variable at a time.** Don't change architecture AND data AND hyperparams simultaneously.

### Starting point for every new capability

The very first experiment is always the most basic version: e.g. 5-character context, feedforward network, next-token prediction. Prove that works. Then build up.

### Reproducibility

- Fixed seeds, git SHA, full hyperparams logged per run.
- Every run locked to a committed config file.
- Results must be reproducible from config alone.

### Checkpointing

- Always checkpoint during runs.
- Keep: last N checkpoints + best + periodic milestones. Delete the rest.
- Never keep weights long-term after an experiment is finished. Retraining in reasonable time is acceptable.
- No model weights committed to git. Ever.

### Monitoring

Principled logging — high quality, not high quantity. When debugging: log many signals, but not much data per signal. When running normally: log what you actually look at.

Always monitor: loss (train + val), gradient norms, NaN/explosion, GPU memory, disk usage, wall-clock time. Set explicit time limits per run — do not let a run go forever.

### Dead-end detection

After a few failed attempts at the same thing: step back and question the approach before trying another variation. Is the idea wrong, or is the implementation wrong?

---

## Code standards

### Small is a success metric

The size and cleanliness of the codebase is an explicit quality metric for this project. A sprawling mess is a failure mode, not a neutral outcome.

### Integration over experimentation

**Integrating finished experimental code is higher priority than starting new experiments.** Working code that is not integrated is a liability — it creates divergence, confusion, and maintenance burden.

### Code structure

```
core/             # integrated, clean, tested code
base-experiments/ # foundational reference experiments
experiments/      # experimental code — not yet integrated or proven
```

New code starts in `experiments/`. Once it works and is understood, it gets refactored into `core/` and the experiment version is either removed or kept as a thin wrapper that uses `core/`.

When two experiments share logic, that logic moves to `core/` immediately.

### Architectural options

If an architectural variant produced results (positive or negative), it stays in the codebase as a selectable option. The codebase should be generic enough to support comparison between options. Dead ends that produced nothing informative can be removed.

### Dependencies and backend

**Language:** Python. **Backend:** open — JAX + XLA is preferred for compiled training loops, but PyTorch or other libraries are acceptable when they make an experiment or integration cleaner.

UV for Python dependency management, local virtualenv. System dependencies (CUDA version, drivers, etc.) should be pinned and documented in `AGENTS.md` or a setup doc as soon as they're confirmed working.

### Tensor readability

Prefer named-dimension / einops-style tensor operations where practical (e.g. `einops.rearrange`, `einops.reduce` with named axes, or equivalent in whatever library is in use). Indexed dimension juggling should be the exception. This is a readability preference; exact library is open.

### OpenCode skills

Load before the relevant work:
- `verifying-work` — before implementing or fixing anything
- `code-principles` — before writing committed code
- `error-handling` — before writing error-handling code
- `information-architecture` — before modifying checked-in docs

---

## Literature practice

Not an academic literature review. The goal is genuine knowledge discovery.

Criteria for a paper being worth reading:
- Someone has tried the exact same idea, or
- Someone has a result that directly refutes a core assumption, or
- Someone has a strictly better solution to the same problem

Do not read papers out of thoroughness or coverage. Read them because they genuinely change what you would do next. Write up what was found in `research/questions/` under the relevant question folder.

---

## Reporting

### Triggers

Reports are written by the agent, not by the harness on a schedule. On each session start:

- Check the current time.
- If after 4pm and no daily report exists for today (`research/daily/YYYY-MM-DD.md`): write it.
- If after 4pm on a Thursday and no weekly report exists for this week (`research/weekly/YYYY-MM-DD.md`): write it too.

Write the report at the **next natural opportunity** — not mid-task, not mid-experiment. Finish the current unit of work cleanly, then write.

### Quality standard

These are not reports. They are narratives Max will read. The standard is distill.pub — dense, precise, readable, zero padding, no unverified claims stated as facts.

Before writing any narrative:
1. Draft it
2. Revise it — cut everything that isn't load-bearing
3. Revise again — is every claim backed by something real?
4. Revise again — is the writing actually good?
5. Revise again — read it as Max would. Is it interesting? Does it waste his time?
6. Final pass — does it meet the standard?

Iterate heavily. The first draft is never the output.

### Inline references — mandatory

Every factual claim in a narrative must link to a real artifact:
- A specific run's output file, or
- An example image embedded inline, or
- A concrete metric from a logged experiment

**Artifacts must be embedded inline, not just linked.** Include the content directly in the markdown — the link is there to prove it's real, but Max should be able to read the file without clicking anything. Link + inline block, not link alone.

No free-floating assertions. If a claim can't be backed by something real, either run the experiment or don't make the claim.

### What goes in a daily

- What ran today
- What it produced (with inline artifacts)
- What it means — one honest sentence per result
- What changed in understanding
- What's next

### What goes in a weekly

- The week's most significant findings, each with evidence
- What open questions narrowed (or opened)
- What the code looks like now vs last week
- Honest assessment: are we making progress?

### Blob policy

**Commit:** small output images, example input/output pairs, embedded in markdown.
**Do not commit:** model weights, datasets, large binary files of any kind.

---

## Information architecture

Files must not grow unbounded. Each file has a job. Rewrite sections when information changes — don't append. Remove stale content. A clean repo is faster to navigate than a comprehensive one.

When something doesn't work and you figure out why: create a question folder in `research/questions/`, document it, add a cheap regression test, integrate the fix into `core/`. Don't just move on.

Question-specific findings and evidence belong in `research/questions/`, not in general process docs. Keep this file about *how* to work, not *what* was found.
