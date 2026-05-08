# research/questions/

One folder per open question, investigation thread, or specific research goal.

## Purpose

Each folder documents one question or goal: what we think we know, what we tried, what it showed, what remains open. When a question is resolved, the folder stays — it becomes a record of how the question was closed.

Folders are not only for open questions. A folder can be for a specific research goal (e.g. "make Mix-Add work on image patches") or for a debugging investigation that turned into a documented finding.

## Add new folders freely

If you encounter something that doesn't work and you figure out why — create a folder, document it, add a cheap regression test, integrate the fix. Don't just move on.

If an experiment raises a new question that isn't in an existing folder — create one. The folder name should be a terse description of the question (e.g. `layer-norm-instability`, `patch-order-bias`).

## Folder structure

Each question folder should contain at minimum:

- `README.md` — what the question is, current best understanding, open sub-questions, and what the next discriminating experiment would be
- Any relevant experiment configs, output artifacts, or inline result images

Each folder should contain only **that question's** material. Do not restate general project vision or process; those live in the root docs. Keep question folders lean.

Unresolved conclusions must stay explicitly unresolved until backed by real artifacts. Do not state a question as closed without evidence.

When a question is closed: update `README.md` to state the conclusion clearly, with evidence linked and embedded inline.

## Docs reminder

Load the `information-architecture` OpenCode skill before restructuring or reorganising question write-ups.
