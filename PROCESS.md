# Process

How work gets done in this project.

This is personal hobbyist ML research. The goal is discovery: try ideas, run experiments, look at outputs, and learn what actually happens. Keep the codebase small. Keep conclusions honest. Redo from scratch is always an option.

---

## The autonomous loop

`loop.ps1` relaunches OpenCode if it exits. **The loop uses Windows Task Scheduler** — it is a child of the task scheduler, not a child of the terminal. Do not revert this to a terminal-based loop; closing the terminal must not kill the loop.

This file is what autonomous agents follow when running in that loop.

The process here is not optional ceremony. It exists because an autonomous agent will otherwise drift, overclaim, skip grounding, or leave unreadable outputs. Following it is the job. Skipping it is failure — even if the immediate output looks productive. The goal is research; the process is the only reliable mechanism for getting there.

On session start:

1. If you lack project context, read `VISION.md` first to orient.
2. Read `PLAN.md` and any `TASK-*.ignore.md` files in the repo root.
3. Check the time and whether a daily or weekly report is due.
4. Check whether a background run is active (`runs/active.lock`).
5. Continue the current task if there is one. Otherwise pick the cheapest useful next step.

Max may speak during a session, but do not assume that means he is back and available. Keep working unless he clearly takes over.

If the last session felt wrong, fix the process first. Bad process compounds.

### How to choose what to do next

Use judgment, not ceremony. But **always check that your chosen work connects to a pathway in ROADMAP.md.** If it doesn't, stop and ask why you're doing it. The roadmap is the map of where we want to go — work that doesn't advance any pathway needs strong justification.

Priority order:

1. Fix broken or misleading process/docs
2. Write any due reports
3. Clean up half-finished or confusing work
4. Integrate finished experimental code into `core/`
5. Continue an existing experiment thread **that connects to a roadmap pathway**
6. Start a new experiment **that is the cheapest honest test for a roadmap pathway**

When choosing between possible next steps, prefer the one that is:

- cheapest to run honestly
- most likely to produce inspectable evidence
- least likely to bloat the codebase
- most useful for narrowing an open question **on the roadmap**
- connected to a pathway that hasn't been tested yet (explore breadth before depth)

If the work feels tangled, restart from a smaller simpler version instead of protecting the tangled version.

---

## Grounding and source of truth

`dictations/` is the authoritative source of project intent. Every other file in the repo — including this one — is a derived interpretation and may be wrong: wrong framing, wrong emphasis, wrong level of certainty, things missed entirely.

Before starting or continuing a research avenue, check that the framing still matches the dictations in spirit. If a report, plan, prompt, or code path has drifted from the dictations, correct the derived file — do not treat the drift as truth.

Do not build on unverified assumptions. If a result is not grounded enough that you can explain honestly why it should be trusted, do more validation before scaling it, integrating it, or writing a confident narrative about it. If you cannot write the report honestly without hand-waving or unexplained trust gaps, the work is not ready to be called a result yet.

---

## Experiment discipline

### Theory first

Before running experiments, do conceptual analysis. Often this means multiple rounds of thinker review before any code is written. The theory work should clarify: what exactly is being tested, what the expected outcome is, what alternatives were considered and rejected, and why this specific experiment is the cheapest honest test.

Theory-first is not optional process overhead. It is the primary output for many research questions. Concept clarification should often be the majority of a report — including avenues not pursued and why.

### Incorporating new dictations

Dictations contain two kinds of content that require different responses:

1. **Process corrections** — act immediately. Update PROCESS.md, AGENTS.md, PLAN.md, VISION.md as needed. These take effect now.
2. **New work directions** — add to the queue in PLAN.md. Do NOT start them immediately or abandon current work. The latest dictation is NOT automatically the highest-priority work.

When a new dictation appears:

1. Read it carefully.
2. Identify which parts are process corrections and which are new directions.
3. Apply process corrections to all affected files immediately.
4. Add new directions to PLAN.md as queued future work.
5. Continue with the current task unless the dictation explicitly says to stop.

This is not a one-time task. Every new dictation is a potential course correction. Treat it as authoritative over all derived files.

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

Every experiment starts by writing the report first in `research/questions/<question>/README.md`, even if many sections begin as placeholders.

Create or update the question README with:

- which dictation or `VISION.md` goal the experiment serves, with a link to the specific dictation when there is one
- the simplification chosen and why it is a legitimate cut
- a clear architecture explanation, using real code snippets and diagrams where that helps
- the hypotheses being tested
- the planned evidence: what artifacts the run will produce and why those artifacts answer the question
- placeholder sections for results, next steps, and deferrals
- explicit non-goals: what this experiment will not settle

The point of writing first is to force explicit hypotheses and surface hidden assumptions before the code grows. Blank sections are fine. Hidden assumptions are not.

Fill in results, next steps, and reasons for deferral as the experiment proceeds, not only at the end.

### Before starting a new experiment

Before committing to an experiment, answer these questions:

- **Which ROADMAP pathway does this advance?** If you can't name one, stop.
- What question am I trying to answer?
- What is the cheapest experiment that could teach me something about it?
- What result would INCREASE confidence in this pathway?
- What result would DECREASE confidence — and what would I do then?
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

### The experiment loop (not a linear plan)

Work is structured as **nested loops with adversarial review gates**, not a sequence of phases. Each loop has an explicit exit condition. Failing the exit condition sends you BACK (to an earlier loop or to redesign), never forward with a patch. An adversarial review means delegating to a separate subagent whose job is to find problems — if it can't find significant problems, you pass.

```
┌─────────────────────────────────────────────────────────────────┐
│ PROCESS IMPROVEMENT LOOP (outermost — always active)            │
│                                                                 │
│  Check: is the process working? Did last session feel wrong?    │
│  If yes → fix process/docs BEFORE doing research.              │
│  If no → enter research loop.                                  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ PATHWAY SELECTION LOOP                                    │  │
│  │                                                           │  │
│  │  1. Which roadmap pathway to advance?                     │  │
│  │  2. What's the cheapest honest test for it?               │  │
│  │                                                           │  │
│  │  GATE: Adversarial review (subagent).                     │  │
│  │  - Does this actually connect to the roadmap?             │  │
│  │  - Is this really the cheapest test?                      │  │
│  │  - Are we amplifying a marginal signal?                   │  │
│  │  - Has this been tried before (check prior results)?      │  │
│  │  If review finds problems → go back to step 1.           │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ CONCEPTUAL CLARIFICATION LOOP                       │  │  │
│  │  │                                                     │  │  │
│  │  │  1. What exactly is being tested?                   │  │  │
│  │  │  2. What do we expect and why?                      │  │  │
│  │  │  3. What alternatives were considered?              │  │  │
│  │  │  4. Write hypotheses and planned evidence.          │  │  │
│  │  │                                                     │  │  │
│  │  │  GATE: Adversarial review (subagent).               │  │  │
│  │  │  - Are the hypotheses falsifiable?                  │  │  │
│  │  │  - Is the expected outcome clearly stated?          │  │  │
│  │  │  - Could a simpler experiment answer the same       │  │  │
│  │  │    question?                                        │  │  │
│  │  │  - Are there hidden assumptions?                    │  │  │
│  │  │  If review finds problems → iterate on clarity.    │  │  │
│  │  │                                                     │  │  │
│  │  │  ┌───────────────────────────────────────────────┐  │  │  │
│  │  │  │ EXPERIMENT DESIGN LOOP                        │  │  │  │
│  │  │  │                                               │  │  │  │
│  │  │  │  1. Design the experiment (code, config).     │  │  │  │
│  │  │  │  2. Sanity check (overfit one batch, etc).    │  │  │  │
│  │  │  │                                               │  │  │  │
│  │  │  │  GATE: Adversarial review (subagent).         │  │  │  │
│  │  │  │  - Does the code match the hypothesis?        │  │  │  │
│  │  │  │  - Are there bugs that would invalidate       │  │  │  │
│  │  │  │    results?                                   │  │  │  │
│  │  │  │  - Is the comparison fair (matched compute,   │  │  │  │
│  │  │  │    same data, proper ablation)?               │  │  │  │
│  │  │  │  - Does every component receive learning      │  │  │  │
│  │  │  │    signal at initialization? Are there         │  │  │  │
│  │  │  │    cold-start / chicken-and-egg traps?        │  │  │  │
│  │  │  │  - For mechanism ablations: is this eval-time │  │  │  │
│  │  │  │    (OOD-confounded) or train-time (clean)?    │  │  │  │
│  │  │  │    Eval-time can filter; only train-time is   │  │  │  │
│  │  │  │    definitive.                                │  │  │  │
│  │  │  │  If review finds problems → fix and re-check. │  │  │  │
│  │  │  │                                               │  │  │  │
│  │  │  │  ┌─────────────────────────────────────────┐  │  │  │  │
│  │  │  │  │ RUN & ANALYZE LOOP                      │  │  │  │  │
│  │  │  │  │                                         │  │  │  │  │
│  │  │  │  │  1. Run experiment.                     │  │  │  │  │
│  │  │  │  │  2. Analyze results.                    │  │  │  │  │
│  │  │  │  │  3. Does result match expectations?     │  │  │  │  │
│  │  │  │  │     - If unexpected → investigate why   │  │  │  │  │
│  │  │  │  │       before concluding.                │  │  │  │  │
│  │  │  │  │                                         │  │  │  │  │
│  │  │  │  │  GATE: Adversarial review (subagent).   │  │  │  │  │
│  │  │  │  │  - Is the interpretation honest?        │  │  │  │  │
│  │  │  │  │  - Are we overclaiming?                 │  │  │  │  │
│  │  │  │  │  - Is the pathway still viable?         │  │  │  │  │
│  │  │  │  │  - Should we continue on this pathway   │  │  │  │  │
│  │  │  │  │    or redirect?                         │  │  │  │  │
│  │  │  │  │                                         │  │  │  │  │
│  │  │  │  │  EXIT CONDITIONS:                       │  │  │  │  │
│  │  │  │  │  - Meaningful result → integrate code   │  │  │  │  │
│  │  │  │  │    if applicable, report if due, update  │  │  │  │  │
│  │  │  │  │    PLAN.md, suggest roadmap updates,     │  │  │  │  │
│  │  │  │  │    then pathway selection.               │  │  │  │  │
│  │  │  │  │  - Marginal result → REDIRECT. Record   │  │  │  │  │
│  │  │  │  │    what it teaches, suggest roadmap      │  │  │  │  │
│  │  │  │  │    update in PLAN.md. Do NOT iterate.    │  │  │  │  │
│  │  │  │  │  - Unexpected failure → go back to       │  │  │  │  │
│  │  │  │  │    conceptual clarification.             │  │  │  │  │
│  │  │  │  │  - Design flaw found → go back to        │  │  │  │  │
│  │  │  │  │    experiment design.                    │  │  │  │  │
│  │  │  │  └─────────────────────────────────────────┘  │  │  │  │
│  │  │  └───────────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Adversarial review gates

**Every transition between loops requires an adversarial review.** This is not optional ceremony — it is the mechanism that prevents drift. The reviewing subagent's job is to find problems. If it cannot find significant problems, the gate passes. If it finds problems, you go BACK, not forward.

The reviewer is a DIFFERENT subagent from the one doing the work. It receives the work product and asks:

1. **Does this connect to the roadmap?** Name the pathway. If the connection is vague or requires a stretch, the work doesn't pass.
2. **Is this the cheapest honest test?** Could something simpler answer the same question? If yes, go back and simplify.
3. **Are we amplifying a marginal signal?** If the improvement is small relative to the gap between baseline and target, AND repeating the same class of intervention is unlikely to compound, it's marginal. The reviewer decides — not the experimenter.
4. **Are there hidden assumptions?** Things stated as settled that aren't experimentally confirmed. Things assumed about the architecture that haven't been tested.
5. **Is the interpretation honest?** No overclaiming. No "this proves X" when it merely suggests X. Open things stay open.
6. **Is the result actually meaningful enough to count as evidence?** Would this change what you'd try next on a different pathway? If not, it's not meaningful — it's noise.

### The reviewer cannot be overridden

**If the reviewer identifies ANY problem, the orchestrator must either:**
- (a) Address the problem and re-submit for review, or
- (b) Escalate to Max.

**The orchestrator may NOT:**
- Dismiss review findings as minor
- Rephrase problems as non-significant
- Commission a second review with different framing hoping for a pass
- Interpret "no significant problems" liberally to skip issues

The adversarial reviewer has veto power. This is deliberate — the cost of one blocked experiment is low; the cost of weeks of drift is high.

**Enforcement note:** This rule cannot be mechanically enforced in a single autonomous agent — the orchestrator is ultimately its own judge of compliance. It exists as a strong norm. When Max reviews session transcripts, violation of reviewer findings is a process failure worth calling out. The loop is designed so that violating this rule leaves visible traces (no review record, or review findings that were never addressed).

### Going back to earlier loops

The loop structure is explicitly non-linear. Problems discovered in ANY inner loop can send you back to ANY outer loop:

- **Experiment results reveal the concept was wrong** → go back to conceptual clarification loop (not "try a different hyperparameter")
- **Review finds the pathway selection was wrong** → go back to pathway selection (not "reinterpret the result to fit")
- **Process feels wrong** → go back to process improvement loop (fix the process before continuing)
- **Design review finds the experiment can't answer the question** → go back to conceptual clarification (not "run it anyway and see what happens")

The agent must never move forward through a gate that has identified a problem. Patching forward is the primary failure mode.

### Process re-entry triggers

The Process Improvement Loop is not just a one-shot check on session start. Re-enter it when:

- **Any adversarial gate fails** — after fixing the problem, ask: "should the process itself be updated to prevent this class of problem?"
- **After every daily report** — ask: "was today's work shaped well by the process, or did I work around it?"
- **After any result (meaningful or not)** — update PLAN.md with what was learned. If the result changes pathway viability, note that too.
- **After any unexpected outcome** — if reality surprised you, either your model of the system was wrong or the process let a bad experiment through. Figure out which.

### Updating the roadmap from results

When an experiment produces evidence for or against a pathway, the agent must note this in PLAN.md (which it CAN edit). It must NOT edit ROADMAP.md (which is Max-only), but it must record in PLAN.md: "Pathway X: new evidence [description]. Confidence [increased/decreased/unchanged]. Suggest roadmap update: [specific change]." Max reviews and applies roadmap changes.

### Skill loading at point of use

Load the relevant skills at the point in the loop where they matter, not as general aspirations:

- **Entering Experiment Design Loop** → load `code-principles` and `verifying-work`
- **Before writing any experiment code** → load `code-principles`
- **Before running any experiment** → load `verifying-work` (plan verification BEFORE implementation)
- **Before writing any report or doc update** → load `information-architecture`
- **Before any shell command** → load `tools`

### Code health loop

Part of the Process Improvement Loop (outermost). On every session start, also check:

- Has `core/` grown since last review? Is anything there that shouldn't be?
- Are there dead experiments (taught nothing, just clutter)? Delete them.
- Are there experiment scripts that share logic? Move shared logic to `core/`.
- Is the codebase still small? If not, refactor before adding more.

Code health is not a separate phase — it's a check that can trigger a refactoring sub-loop at any time. If the codebase is unhealthy, fix it BEFORE running more experiments.

### The critical exit condition

A marginal result is one where the improvement is small relative to the gap between current and target, AND where repeating the same class of intervention is unlikely to compound. The 0.05 nats threshold on a 1.5+ baseline is one example, not the only case. The adversarial reviewer decides whether a result is marginal — not the experimenter.

When a result is marginal: record it, ask what it teaches about the pathway, and choose the next cheapest test — which may be on a DIFFERENT pathway. Do NOT iterate on the same mechanism.

Spending multiple sessions amplifying a small signal is the primary failure mode of this project's loop agent. The correct response to a marginal result is "interesting, what does this tell us? what's next?" not "how do I make this number bigger?"

### Starting point for any new capability

Start with the simplest version that could possibly work. For example: tiny context, tiny model, next-token prediction, minimal dataset, inspectable outputs.

Do not smuggle in architectural complexity "for later." Add complexity only after the simpler version genuinely works.

### Isolation before composition

Do not combine multiple speculative mechanisms too early.

When exploring a new architectural idea, run thorough experiments on the individual pieces in isolation before composing them into a larger system. Composition is a later experiment — once the pieces are at least somewhat understood in their own right.

If a simplified architecture or framing was introduced by a prior agent rather than the dictations, treat it as a hypothesis to justify or replace, not as the project goal.

### Reproducibility

- Record git SHA, hyperparameters, and seed for meaningful runs.
- Save enough config that a result can be recreated without guesswork.
- Match the level of rigor to the claim. Tiny exploratory probes can be light; comparison claims need stronger control.
- **Do not call a stochastic result decisive from a single seed.** Single-seed runs are exploratory only; any comparison or conclusion claim requires multiple seeds (default: 3) or must be explicitly labelled provisional.

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

### Baselines before bold claims

Trustworthy baseline experiments are part of the foundation, not optional side work.

Every comparison claim requires TWO controls per [dictation 2026-05-23-4](dictations/2026-05-23-4.md):

1. **Ablation control** — the most similar possible network WITHOUT the modification being tested. Same architecture family, same compute budget, one thing removed.
2. **Standard baseline** — a transformer (the accepted state-of-the-art architecture) on the same dataset and task, achieving results consistent with published expectations.

Before making strong claims about a custom architecture, establish what a standard model should achieve on the same dataset and reproduce that result closely enough to trust the training stack, data pipeline, and evaluation. Standard baselines belong in `base-experiments/`, and the reusable parts of those implementations belong in `core/`.

A custom result without both controls is weak evidence. "Wins versus our own ablation" is interesting but insufficient. "Wins versus transformer baseline at matched compute" is a real claim.

### Architectural options

Open questions stay open. Do not collapse real uncertainty into a single clean story just to simplify the docs or code. When multiple architectural options are genuinely still alive, record them as open and compare them deliberately rather than quietly picking one.

Keep selectable variants only when they produced an informative comparison or are still actively useful. Do not keep every dead path forever.

---

## Experiment visibility

**Before running any experiment expected to take more than ~30 seconds, the subagent must stop and report back to the orchestrating agent.** The report must include:

- What experiment is about to run
- Expected duration (estimate from corpus size, step count, prior runs)
- What it will produce

The orchestrator then logs a visibility note with the current time (so Max can see what's happening if he checks the loop) and resumes the subagent to proceed.

This prevents silent multi-hour waits where Max has no idea what's happening. The loop output should always show what we're doing and how long it's expected to take.

**Orchestrators must instruct subagents to report back before each new run.** When delegating experiment work, explicitly tell the subagent: "Report back to me with results before launching any new training run — including retries of failed runs, even if the fix seems obvious. Do not chain runs without returning first. Each run that exceeds ~30 seconds is a separate report-back cycle."

## Background training runs and multiple workstreams

Long training runs must not block the agent. This is a significant process requirement per [dictation 2026-05-21-3](dictations/2026-05-21-3.md).

Rules:

- **Never block on a training run expected to take >5 minutes.** Start it in the background (e.g. `Start-Process` on Windows), capture the PID and log path, then continue with other work.
- **Use unbuffered Python output** for background runs: set `$env:PYTHONUNBUFFERED = '1'` before `Start-Process`, or pass `-u` to the Python interpreter. Without this, stdout is buffered and logs appear empty until the process ends.
- At most one large training run (>10 min) at a time.
- One small/fast experiment can run alongside a large run.
- **NEVER be idle while a run is active.** This is not optional. GPU time is the project's most expensive resource — the agent's job is to maximize the value produced per GPU-hour by doing high-quality intellectual work in parallel. See the "Quality loops while GPU is busy" section below.
- The orchestrator checks progress when it decides to — subagents do not poll.

**Subagent instruction requirement — this is mandatory, not optional:**

When delegating experiment work that may involve a GPU training run longer than ~5 minutes, the delegation MUST be split into two separate phases:

**Phase 1 — Launch only.** Delegate to a subagent with instructions that include words to this effect:

> "Your job in this session is only to set up and start the experiment. Write the script, start the background process, and return to me with the PID and log path. Do NOT wait for the run to finish. Do NOT poll. Do NOT check results. Stop as soon as the process is running."

The subagent must reply with the PID and log path and stop. If it does anything else after starting the run, it has failed this instruction.

**Phase 2 — Check and analyse.** Only after doing other useful work (docs, theory, next experiment design), the orchestrator resumes the same subagent (or delegates a fresh one) with:

> "The run has been going for a while. Check the log at [path] and report what you see."

The orchestrator is responsible for deciding when to check back — not the subagent. The subagent must never poll or self-resume.

**If you (the orchestrator) delegate an experiment without splitting it this way, you have failed the process.** There is always other useful work to do while the GPU runs — theory, reports, doc updates, small non-GPU experiments. Name that work before checking on the run.

Use `runs/active.lock` to record the active large run. **The experiment scripts manage this file automatically** — they create it on startup (with PID and experiment name) and remove it via `atexit` on normal exit. If the process crashes, the lock persists as a stale indicator; agents should check whether the PID is still alive before trusting it. Small runs (<5 min) do not need lock files.

**Windows launch reliability:**

- Use a single documented `Start-Process` pattern. Do not improvise argument passing — use comma-separated array for simple args, or a wrapper script for complex commands.
- **Before launching a GPU run, preflight GPU availability** — confirm `nvidia-smi` shows the GPU idle and CUDA will initialize. On this desktop, gaming can hold the GPU exclusively; CUDA init will block indefinitely if the GPU is busy.
- If launch infrastructure is flaky, fix and document the launch mechanism before spending more time on experiments.
- **Do NOT use `-RedirectStandardOutput`/`-RedirectStandardError` with `Start-Process -WindowStyle Hidden`.** This causes the child process to hang when stdout buffers fill (the hidden window has no console to flush to). Instead: let stdout go to the hidden window's console (effectively discarded) and rely on the experiment's own JSONL log file for monitoring.
- **Corpus loading takes ~90 seconds** (WikiText-103 raw is 538M chars). The log file won't appear until after loading completes. Don't assume the process is dead during this period — check CPU/memory via `Get-Process`.

**torch.compile vs CUDA graphs:**

- **Default: use `--no-compile`** (which activates the `ClosedLoopPredictionGraphTrainer` CUDA graph path). This starts fast and trains fast.
- **`torch.compile` (`--compile`)**: uses `aot_eager` backend. Can take 5-10+ minutes for initial trace on complex models (E_grounded's extra LM head makes it worse). Once traced, training is fast. Use only when CUDA graphs aren't available or the model has incompatible dynamic control flow.
- The CUDA graph path requires fixed batch sizes and static tensor shapes. Models with dynamic shapes can't use it.
- If adding new model variants, always test with `--sanity-check --no-compile` first to verify the CUDA graph path works before attempting compiled runs.

**Experiment log files (JSONL):**

- Experiment scripts write progress to JSONL log files using `append_log()`.
- On startup, scripts must NOT truncate existing logs. If the log file exists and has content, append a `run_restarted` marker. This preserves partial results from crashed runs.
- The `write_text("")` pattern that was previously used to clear logs on startup is a bug — it destroys evidence from prior runs.

---

## Quality loops while GPU is busy

**This is a first-class obligation, not a fallback.** Per [dictation 2026-05-23-6](dictations/2026-05-23-6.md): "Right now, go through the process and make it extremely heavy and obvious that while there are experiments running, you should never be idle."

The GPU produces evidence. The agent's parallel job is to produce *understanding* — operationalize concepts, challenge assumptions, improve communication quality, and prepare for what comes next. Idle time during a GPU run is process failure.

### What to do (priority order)

1. **Deep theory work** — Send the same design question to a thinker 5-10 times with different framings. Collect alternatives. Do the maths to operationalize concepts into concrete architecture variants. Don't accept the first answer — iterate until real alternatives emerge.

2. **Improve past reports** — Model the reader (Max). Reread past daily/weekly narratives. Are they actually good? Do they tell a story? Are they dense and interesting? Rewrite the weak ones. Every report should be worth reading, not just technically present.

3. **First-principles re-derivation** — Go back to the dictations. Re-derive the project direction from scratch. Does the current framing match Max's actual intent? Has something drifted? What open questions have been quietly assumed closed?

4. **Skill and process consistency** — Check the repo against Max's global skills (code-principles, information-architecture, etc.). Is the codebase consistent with them? Are the AGENTS.md files accurate? Is PROCESS.md up to date?

5. **Get critique and iterate** — Delegate a review to an agent. Have it challenge the current approach. Then respond to the challenges substantively. Repeat. The goal is to surface blind spots, not to confirm the status quo.

6. **Design next experiments** — Write the question doc, hypotheses, and planned evidence for the next thing to run. Don't just pick the next item off the list — think about what would be most discriminating.

### Quality standard for theory work

Theory work during GPU time is not "thinking about it." It must produce durable artifacts:

- A written analysis in `research/questions/<topic>/README.md`
- Multiple concrete alternative formulations (not just one)
- Mathematical operationalization where applicable
- Explicit comparison to the current approach
- Clear statement of what experiment would discriminate between alternatives

If the theory work doesn't produce a written artifact, it didn't happen.

### Anti-patterns (process failures)

- Polling the log file every 5 minutes → set a timer and do real work in between
- Doing "quick cleanups" that take 2 minutes each → batch them or do substantial work instead
- Writing a single thinker prompt and accepting the answer → iterate 5-10 times minimum for design questions
- Waiting for "a natural stopping point" → the stopping point is now; start theory work immediately after launch

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

A good report lets Max answer: what was tried, why it was a legitimate step relative to the dictations, what happened, and why the result should or should not be trusted. It explains the path the experiment took, the simplifications made, the hypotheses tested, the evidence gathered, and the limits of the conclusion. If you cannot write it honestly without hand-waving or unexplained trust gaps, either weaken the claim or do the additional validation first.

**Include process improvements.** When the development process was improved (gitignore fixes, doc restructuring, new conventions, friction removed), report it. Max wants to see the process getting better, not just experimental results. A day where the only real work was making the project more runnable is still a real day — say exactly what process risk was reduced and why it was the highest-leverage step.

Question READMEs should clearly explain the architecture with implementation-aligned code snippets and diagrams, link the work back to Max's goals or dictations, and make the next steps obvious, including why those next steps have not been pursued yet.

Question-folder notes can be rougher. Daily and weekly narratives should be polished.

---

## Delegating experiment work to subagents

This section is for the orchestrating agent. It codifies the pattern that prevents wasted context and ensures visibility.

### Durable question docs come first

Before delegating an experiment, write `research/questions/<question>/README.md` with the context, question, hypotheses, and planned approach. This doc persists across sessions. If the subagent dies or hands over, a new agent can read the same doc without re-prompting.

Prompt content is ephemeral. Durable docs are infrastructure.

### What goes in the prompt vs. what's already in docs

| Belongs in prompt | Belongs in AGENTS.md / conventions | Belongs in question doc |
|---|---|---|
| Which rung of the ladder we're on | File naming, directory structure | Architecture, context, prior results |
| Process instructions (report back, don't poll) | How to run experiments, environment | Hypotheses, planned evidence |
| What the goal is (1 sentence) | Hyperparameter defaults | Implementation details worth preserving |

Don't repeat what's already documented. Reference it: "Read `research/questions/fixed-multi-rate/README.md` for the question, hypotheses, and architecture."

### The report-back cycle

Each experiment delegation follows an up-down-up-down pattern:

1. **Setup prompt** — subagent reads the question doc, implements the experiment, reports back what it built and what it's about to run. It does NOT run yet.
2. **Launch prompt** — orchestrator reviews the plan, says go. Subagent launches in background, returns PID and log path, stops immediately.
3. **Check prompt** — orchestrator does other work, then checks back. Subagent reads log, reports results or failure.

Each "report back" is a checkpoint where the orchestrator can course-correct, do other work, or maintain visibility. Without this: 30 minutes of idle context, no visibility, no ability to redirect.

The exact number of prompts varies. Fast experiments might skip the background phase. Complex ones might need debugging rounds. The non-negotiable parts:

- Subagent does not autonomously run long experiments without reporting what it's about to do
- Setup is separate from execution
- Long runs (>5 min) go to background; orchestrator does other work

### What NOT to put in prompts

- Environment details already in AGENTS.md (runtime, GPU, how to invoke scripts)
- Full architecture explanations (put those in the question doc)
- Exact file names (let the subagent decide based on conventions and existing patterns)
- Step-by-step implementation instructions (that's the subagent's job)
- All hyperparameter defaults (only mention what's specific to THIS experiment)

---

## Information architecture

Files should have clear jobs. Rewrite them when reality changes; do not append stale process sediment forever.

Keep this file about how to work, not about the current state of specific research threads. Current state belongs in `PLAN.md` and `research/questions/`.

**Process changes must propagate to AGENTS.md files.** This file (PROCESS.md) is primarily read by the orchestrating agent. But subagents read `AGENTS.md` files — the root one and the per-directory ones. When the process changes in ways that affect how subagents work in specific directories (file layout conventions, experiment protocols, what goes where), those rules must be reflected in the relevant `AGENTS.md` files, not just here. PROCESS.md defines the process; AGENTS.md files enforce it at the point of work.
