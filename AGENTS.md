# AGENTS.md — gildnn repo contract

## Git hygiene

**Commit and push early and often.** Keeping work preserved remotely is part of the normal workflow in this repo. Test first when there is something real to test.

**PowerShell git commit quirk:** Multi-line commit messages passed via `-m` in PowerShell sometimes silently fail (exit code 0 but nothing committed). Always use single-line `-m` messages. If you need more detail, put it in a second `-m` flag or just keep it short. Always check `git log --oneline -1` after committing if the message was complex.

## ⚠ Epistemic status of all files in this repo

**The authoritative source of truth for this project is [`dictations/`](dictations/).** Everything else — this file, VISION.md, PROCESS.md, PLAN.md — is a derived interpretation, written quickly, and likely imperfect. Do not treat them as authoritative.

The dictations are Max's unedited words. They contain his actual intent, including hedges, uncertainties, and contradictions. All other files are summaries of those words, and summaries can be wrong — wrong framing, wrong emphasis, wrong level of certainty, things missed entirely.

If you find a conflict between the dictations and any other file: the dictations win. Rewrite the other file.

Before starting a new research avenue, review the relevant dictations and the derived files against each other. Review against the *spirit* of what Max said, not the literal instructions. Things that were stated with uncertainty should remain uncertain. Things that were left open should remain open.

Periodically — especially after completing a significant task or before starting a new phase — read the full dictations directory and check the other files against them. Is the framing right? Are open questions correctly marked open? Is anything stated as settled that isn't?

**Do not modify the dictations.** They are a record of what Max said, not a working document.

---

## Project constraints

**Timebox:** this project runs until the end of the month when GitHub Copilot changes its billing. Make the most of the remaining time.

**Work sizing:** before starting any significant task, estimate whether it fits in one agent context window. If not, split it deliberately and plan the handovers in advance. Don't drift into a task that can't be finished in one session.

**Writing guidance:** before writing the first daily or weekly narrative, produce a short document on how to write really good explanations for this project — dense, readable, tuned for Max. Then strictly follow it. This doc belongs in `research/AGENTS.md` or a linked file.

---

## Relevant OpenCode skills

Load these before doing the relevant work:

| Task | Skill to load |
|---|---|
| Writing or modifying any checked-in docs/markdown | `information-architecture` |
| Before implementing or fixing anything | `verifying-work` |
| Before writing code that will be committed | `code-principles` |
| Before writing error-handling code | `error-handling` |

---

## What this project is

Personal hobbyist ML research. **Not academic. Not for publication.** The goal is discovery — finding out what ideas actually do when you run them. Redoing work others have done is fine. Rigor matters; novelty does not.

Read [VISION.md](VISION.md) for what we're exploring. Read [PROCESS.md](PROCESS.md) for how.

## Technical preferences

- **Language:** Python for ML experimentation. Max has a personal preference for Rust but recognises it's probably not the right choice here.
- **Backend:** PyTorch deliberately. `torch.compile` for stable core/ paths, custom CUDA/Triton for async research. Decision per [dictation 2026-05-22-12](dictations/2026-05-22-12.md). See `research/questions/backend-choice/README.md`.
- **⚠ torch.compile on Windows:** Triton is not available on Windows. The Inductor backend requires Triton, so `torch.compile` with any optimization mode fails with `TritonMissing`. The cellular automaton model traces successfully (no graph breaks with `fullgraph=True`), so compilation WOULD work on Linux. Current workaround: vectorized bmm across levels in eager mode (~5s/step at small scale, ~22s/step at full scale). For true GPU-native execution, would need Linux or a Windows Triton build.
- **Tensor readability:** prefer named-dimension / einops-style operations where practical (e.g. `einops.rearrange`, `einops.reduce` with named axes, or equivalent). Use `jaxtyping` annotations for tensor shape documentation (e.g. `Float[Tensor, "batch seq dim"]`). Both are installed and torch.compile-compatible.
- Dependencies managed via UV, local virtualenv.
- **Unified architecture:** an off-the-shelf framework that provides abstractions over backbones, embeddings, data pipelines, and prediction heads is acceptable — probably one already exists. If used, it must be explained very well, not assumed. Max wants to understand what it's doing, not just use it as a black box.

## Verified local runtime facts

- Verified on 2026-05-08: one minimal local backend path works on this Windows machine with UV + local `.venv`, CPython 3.12.12, and `torch==2.11.0+cu128`.
- CUDA proof for that path: tiny tensor multiply succeeded on `cuda:0` / `NVIDIA GeForce RTX 3090`.
- Machine evidence captured during the probe: NVIDIA driver 591.86, `nvidia-smi` reports CUDA 13.1.
- Probe artifact and exact output: see `research/questions/backend-validation/README.md`.
- This is an operational probe only. It does **not** settle the project-wide backend choice.

---

## Directory conventions

Each directory has its own AGENTS.md explaining its role. The key boundaries:

| Directory | Contains | Does NOT contain |
|---|---|---|
| `experiments/<name>/` | Per-experiment scripts, configs, raw outputs, debug artifacts | Polished reports |
| `research/questions/<name>/` | Max-readable reports (README.md) with inline evidence | Standalone artifact files (no .txt dumps, no raw JSON tables) |
| `base-experiments/<model>/` | Standard model training that reproduces expected published results | Custom/experimental architectures |
| `core/` | Shared reusable components (datasets, models, training loops) | Experiment-specific scripts |

Evidence in question READMEs must be **embedded inline** in the markdown (tables, code blocks, small images). The link proves it's real; the inline content means Max doesn't have to click anything. Standalone .txt or .json files that exist only to be referenced belong in `experiments/`, not in `research/questions/`.

## File map

| File/Dir | Purpose | Read when |
|---|---|---|
| `VISION.md` | What the final model should DO and BE — stakeholder requirements. **Do not edit without Max asking.** | Starting fresh, orienting on goals |
| `ROADMAP.md` | Research pathways toward the vision — directions to explore, hypotheses, connections. **Do not edit without Max asking.** | Choosing what to work on, understanding the bigger picture |
| `PROCESS.md` | How work is done — experiment discipline, loop, reporting | Deciding how to proceed |
| `PLAN.md` | Working notes — current state, what's been done, what's next. Edit freely, keep up to date. | Picking up after a handover |
| `loop.ps1` | Outer restart loop — uses Task Scheduler (NOT terminal child). Do not revert to terminal-based loop. **To stop: `.\loop.ps1 stop` — this is the ONLY way to stop the loop. NEVER manually kill opencode processes by PID; you will kill unrelated sessions. To restart: `.\loop.ps1`** | Understanding how the loop works |
| `research/daily/` | Daily output narratives for Max | Reviewing recent progress |
| `research/weekly/` | Weekly synthesis narratives for Max | Weekly review |
| `research/questions/` | Per-question reports — Max-readable, inline evidence only | Investigating a specific open question |
| `dictations/` | Raw unedited capture of Max's words | Recovering original intent |
| `core/` | Integrated, clean, tested Python code | Writing or reading production code |
| `base-experiments/` | Standard model baselines achieving expected published results | Understanding baseline results |
| `experiments/` | Per-experiment directories with scripts and grungy artifacts | Running or reviewing an experiment |
| `rust-archive/` | Prior Rust implementation — reference only, not active | Historical reference |
| `runs/` | Training run logs and lock file | Checking on active/recent runs |

## Core rules

**Only train Max's architecture.** Do not train standard models (flat GRU, vanilla transformer, multi-block with global CE only). Those tell us nothing about whether Max's design works. Max's architecture is: multi-block grid, stale laterals, per-block local predictive loss (higher blocks predict further ahead), weight-tied normalized readout, noise on laterals for information hierarchy, multi-rate blocks, long sequences with TBPTT. If an experiment doesn't test a faithful piece of this design, don't run it.

**Sanity check = 30 seconds.** "Does this piece learn at all? Is the loss going down? Is the architecture not completely broken?" That's all. NOT training to convergence and analyzing the result. NOT comparing against a standard baseline.

**Per-block loss is the metric.** Stop looking at global cross-entropy. The questions that matter: is each block's local loss going down? Is the higher block learning different representations than the lower block? Does the multi-rate structure create timescale separation? Good global CE is meaningless if one block is doing all the work.

**Proven pieces are the default baseline.** CUDA graph training, dynamic depth/halting, weight-tied readout, normalized embeddings, recurrent depth, stale laterals — these are ALL validated. Every experiment starts from this stack. A new experiment ADDS a piece; it doesn't start from scratch.

**Build up with faithful pieces, not broken intermediate models.** Bad: test flat GRU → multi-block with global CE → add local loss later. Good: "single block with tied readout learns? (30s) → second block predicts further ahead? (30s) → different representations? → wire together with stale laterals → add noise → train for real." Each step tests something faithful to the final design.

**Valid comparisons change ONE thing.** Take something that works, change one thing, see what happens. Noise on laterals. Observer block. CUDA graphs. Global vs local backprop. Different loss variants. All faithful to the final design; none involves training a broken model.

**Cumulative progress, not regression.** Don't regress across eight dimensions to advance one. Build more sophisticated things into the core. Proven pieces stay in. New experiments add; they don't subtract.

**Every experiment must connect to a ROADMAP pathway.** Before running any experiment, name which pathway it advances and what the exit condition is. If you can't, stop and redirect. Do NOT amplify marginal signals — the correct response to a 0.01 nat improvement is "interesting, what does this teach us?" not "how do I make this bigger?" See PROCESS.md for the full experiment loop with exit conditions.

**Check the time on every session start.** If it's after 4pm and no daily report exists for today in `research/daily/`, write it before starting new work. If it's after 4pm Thursday and no weekly report exists for this week in `research/weekly/`, write that too. See `research/AGENTS.md` for the report process.

**Improve the process before delivering results.** If the last session felt wrong — docs were misleading, something was hard to find, the process was awkward — fix it first (update `AGENTS.md`, `PROCESS.md`, `loop.ps1`). Then deliver the result. This includes small niggles: if `__pycache__` is showing up in `git status`, add it to `.gitignore` immediately rather than working around it forever. Don't let friction accumulate.

**Deliver readable outputs before declaring done.** Before calling any experiment complete, produce a clean self-contained narrative with inline artifacts. Max reads these, not the code. See `research/AGENTS.md`.

**Integrate before experimenting.** Finishing integration of working experimental code is higher priority than starting new experiments. Code that works and is not integrated is a liability.

**Keep the codebase small.** This is an explicit quality metric. If the code is growing without producing clarity, stop and refactor. Prefer deleting to keeping.

**No large blobs.** Do not commit model weights or datasets. Do commit small output artifacts: example images, input/output samples, embedded in markdown.

**No status.md.** Use `PLAN.md` for current state and transient `TASK-*.ignore.md` files for handover notes. Delete them when done.

**Review after doing things.** After completing any significant task, pause and review it against the spirit of what was asked — not the literal instructions. Did the framing come out right? Are open things still open? Is anything stated with more certainty than it deserves?

**Rewrite, don't append.** Every file has a job. When information becomes stale, remove it. Files should shrink over time as things become clear, not grow.

**Open questions stay open.** Do not state architectural decisions as settled unless they have been experimentally verified. If it's not confirmed, mark it as an open question.

**Don't interrupt the desktop.** No popups, notifications, or focus-stealing windows. Max may be doing other things.

**Experiment visibility.** Before running any experiment expected to take more than ~30 seconds, subagents must stop and report back with what they're about to run, how long it's expected to take, and what it will produce. This applies to every long run, including retries after obvious fixes. The orchestrator logs a fresh visibility note with the current time before resuming each run. See `PROCESS.md` for details. This keeps the loop output informative for Max.

**Background execution.** Experiments expected to take more than ~5 minutes must use a **two-phase delegation**: Phase 1 (launch only) — subagent starts the process, returns PID and log path, and stops. It does NOT wait, poll, or check results. Phase 2 (check) — orchestrator decides when to check back and resumes the subagent to analyse results. Failing to split this way is a process failure. See `PROCESS.md` for the full rule.

**⚠️ NEVER run experiment scripts against the real artifacts directory while a long experiment is active.** The `runs/active.lock` mechanism is advisory — it can be overwritten, and the atexit handler of a short test run will DELETE the lock even if a real long-running experiment is still using the GPU. Before running ANY experiment script (even for testing a code change), ALWAYS use `--sanity-check-only` which routes to a temp directory. If you need to test non-sanity-check behavior, use `--no-lock` AND redirect `--report-path` and `--log-path` to a separate temp directory. Violating this can corrupt in-progress experiment checkpoints and leave the GPU unprotected.

## Dictation notification

New dictations may appear at any time — Max dictates via a separate support session that runs alongside the loop. The dictation-notifier plugin injects a `[DICTATION NOTIFICATION]` after every tool call when new files appear in `dictations/`. Dictations get picked up:
1. **Immediately** via the plugin notification (fires on NEW files only)
2. On loop restart (opencode process ends for any reason and loop.ps1 relaunches)
3. On handover cycle start (agent picks up fresh context)

**⚠️ Edits to existing dictation files do NOT trigger the notification.** Only new file creation does. If the support session edits an existing dictation, it must create a NEW dictation file referencing the edit — otherwise you won't see the changes until next restart/handover.

Loop restart is NOT the same as handover. Loop restart can happen at any time due to external factors (crash, timeout, compaction). Handover is a deliberate agent action at a natural stopping point. They are completely unrelated mechanisms.

When you notice a new dictation, read it immediately — it contains Max's instructions and takes priority over current work.

## Support session

**Checking loop status:** Use the `opencode-history` skill FIRST to read the loop agent's actual conversation history. Do NOT infer what the loop is doing from git logs or process lists — those are indirect and often misleading. OpenCode history shows you exactly what the agent is thinking and doing.

A separate OpenCode session (the "support session") runs alongside the loop agent. It monitors the loop, writes dictations on Max's behalf, and may modify files in the repo independently (especially `dictations/` and `AGENTS.md`). If you notice unexpected file changes, this is likely the cause — not a conflict.

**Dictation drafting process:** The support session drafts dictation content using Max's exact words, then shows Max the draft before committing. Do NOT create/commit the file until Max confirms. One creation, one notification, complete content.

**Don't rewrite dictations.** Capture Max's exact words. Only correct obvious speech-to-text errors. Do not paraphrase, add em-dashes, restructure sentences, or polish the language. The dictations are a record of what Max said, not a cleaned-up version.

**Don't add analysis or expansion into dictations.** If the support session wants to add interpretation, implications, or connections — draft it and get Max to say it back in his own words. The support session does NOT write new content into dictations. If Max doesn't echo it back, it doesn't go in.

**Redactions:** If a dictation was published with incorrect content, publish a NEW dictation explicitly retracting the wrong parts. Edits to existing files don't trigger notifications.

## Handover protocol

When picking up after a handover:
1. Read `VISION.md` — understand where the current work sits in the bigger picture
2. Read `PLAN.md` — immediate checklist and next steps
3. Check for `TASK-*.ignore.md` in root — read any that exist
5. Check `runs/active.lock` — if a background run is active, do other work (don't start a second long run)
6. **Check the time** — if after 4pm, write the daily report at the next natural stopping point; if Thursday, weekly too

When handing over:
1. Update `PLAN.md` with current state and clear next step
2. Write `TASK-current.ignore.md` with any mid-task context that doesn't belong in PLAN
3. Commit everything
4. **Include a time-check reminder** in any handoff note — tell the incoming agent to check whether a report is due

A handoff note must include: current question being investigated, last concrete change made, exact next step, and any active run/log/PID paths. When a `PLAN.md` item is completed or invalidated, rewrite it immediately — do not leave completed items lingering across sessions.

See `PROCESS.md` for loop behavior, experiment workflow, and reporting standards.

## Report quality

Daily and weekly narratives in `research/` must meet a high bar. See [PROCESS.md](PROCESS.md#reporting) for the full standard. The short version: distill.pub quality, every claim linked to a real artifact, no free-floating assertions.

## Ad hoc report format (support session → Max)

When Max asks "how's the loop going?" or similar, back-chain from the goal:
- Why is the model not yet doing the full architecture?
- If it did the full architecture now, what would happen?
- Why would that result be indeterminate or built on shaky foundations?
- Therefore it's doing this step → therefore it's doing that step → therefore it's doing that step

At every step, justify why we're not just running the real thing. If you can't justify it, the loop should be running the real thing.

The reporting agent needs serious context to write this report — understand the architecture, the current state, and the reasoning chain BEFORE reporting. Don't just relay status; explain the WHY.

## Subdirectory AGENTS.md files

Each subdirectory has its own AGENTS.md explaining its purpose. Read them when entering a new directory.
