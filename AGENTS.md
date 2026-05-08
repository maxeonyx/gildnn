# AGENTS.md — gildnn repo contract

## Git hygiene

**Commit and push early and often.** Keeping work preserved remotely is part of the normal workflow in this repo. Test first when there is something real to test.

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
- **Backend:** open. PyTorch, JAX, or other options are acceptable when they make an experiment or integration cleaner.
- **Tensor readability:** prefer named-dimension / einops-style operations where practical (e.g. `einops.rearrange`, `einops.reduce` with named axes, or equivalent). This is a readability preference, not a mandatory dependency. Indexed dimension juggling should be the exception, not the default.
- Dependencies managed via UV, local virtualenv.
- **Unified architecture:** an off-the-shelf framework that provides abstractions over backbones, embeddings, data pipelines, and prediction heads is acceptable — probably one already exists. If used, it must be explained very well, not assumed. Max wants to understand what it's doing, not just use it as a black box.

## Verified local runtime facts

- Verified on 2026-05-08: one minimal local backend path works on this Windows machine with UV + local `.venv`, CPython 3.12.12, and `torch==2.11.0+cu128`.
- CUDA proof for that path: tiny tensor multiply succeeded on `cuda:0` / `NVIDIA GeForce RTX 3090`.
- Machine evidence captured during the probe: NVIDIA driver 591.86, `nvidia-smi` reports CUDA 13.1.
- Probe artifact and exact output: see `research/questions/backend-validation/README.md`.
- This is an operational probe only. It does **not** settle the project-wide backend choice.

---

## File map

| File/Dir | Purpose | Read when |
|---|---|---|
| `VISION.md` | Motivating ideas, open questions, research threads | Starting fresh, orienting on goals |
| `PROCESS.md` | How work is done — experiment discipline, loop, reporting | Deciding how to proceed |
| `PLAN.md` | Prioritized next work — rewritten not appended | Picking up after a handover |
| `loop.ps1` | Outer restart loop — relaunches OpenCode on exit, reuses session ID | Understanding how the loop works |
| `research/daily/` | Daily output narratives for Max | Reviewing recent progress |
| `research/weekly/` | Weekly synthesis narratives for Max | Weekly review |
| `research/questions/` | Per-question investigation folders | Investigating a specific open question |
| `dictations/` | Raw unedited capture of Max's words | Recovering original intent |
| `core/` | Integrated, clean, tested Python code | Writing or reading production code |
| `base-experiments/` | Foundational reference experiments | Understanding baseline results |
| `experiments/` | Experimental code — not yet integrated | Running or reviewing an experiment |
| `rust-archive/` | Prior Rust implementation — reference only, not active | Historical reference |
| `runs/` | Training run logs and lock file | Checking on active/recent runs |

## Core rules

**Check the time on every session start.** If it's after 4pm and no daily report exists for today in `research/daily/`, write it before starting new work. If it's after 4pm Thursday and no weekly report exists for this week in `research/weekly/`, write that too. See `research/AGENTS.md` for the report process.

**Improve the process before delivering results.** If the last session felt wrong — docs were misleading, something was hard to find, the process was awkward — fix it first (update `AGENTS.md`, `PROCESS.md`, `loop.ps1`). Then deliver the result.

**Deliver readable outputs before declaring done.** Before calling any experiment complete, produce a clean self-contained narrative with inline artifacts. Max reads these, not the code. See `research/AGENTS.md`.

**Integrate before experimenting.** Finishing integration of working experimental code is higher priority than starting new experiments. Code that works and is not integrated is a liability.

**Keep the codebase small.** This is an explicit quality metric. If the code is growing without producing clarity, stop and refactor. Prefer deleting to keeping.

**No large blobs.** Do not commit model weights or datasets. Do commit small output artifacts: example images, input/output samples, embedded in markdown.

**No status.md.** Use `PLAN.md` for current state and transient `TASK-*.ignore.md` files for handover notes. Delete them when done.

**Review after doing things.** After completing any significant task, pause and review it against the spirit of what was asked — not the literal instructions. Did the framing come out right? Are open things still open? Is anything stated with more certainty than it deserves?

**Rewrite, don't append.** Every file has a job. When information becomes stale, remove it. Files should shrink over time as things become clear, not grow.

**Open questions stay open.** Do not state architectural decisions as settled unless they have been experimentally verified. If it's not confirmed, mark it as an open question.

**Don't interrupt the desktop.** No popups, notifications, or focus-stealing windows. Max may be doing other things.

## Handover protocol

When picking up after a handover:
1. Read `PLAN.md` — current task and next steps
2. Check for `TASK-*.ignore.md` in root — read any that exist
3. Read `VISION.md` briefly if the direction is unclear
4. **Check the time** — if after 4pm, write the daily report at the next natural stopping point; if Thursday, weekly too

When handing over:
1. Update `PLAN.md` with current state and clear next step
2. Write `TASK-current.ignore.md` with any mid-task context that doesn't belong in PLAN
3. Commit everything
4. **Include a time-check reminder** in any handoff note — tell the incoming agent to check whether a report is due

See `PROCESS.md` for loop behavior, experiment workflow, and reporting standards.

## Report quality

Daily and weekly narratives in `research/` must meet a high bar. See [PROCESS.md](PROCESS.md#reporting) for the full standard. The short version: distill.pub quality, every claim linked to a real artifact, no free-floating assertions.

## Subdirectory AGENTS.md files

Each subdirectory has its own AGENTS.md explaining its purpose. Read them when entering a new directory.
