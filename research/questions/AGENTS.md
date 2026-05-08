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

- `README.md` — what the question is, current state of knowledge, open sub-questions
- Any relevant experiment configs, output artifacts, or inline result images

When a question is closed: update `README.md` to state the conclusion clearly, with evidence linked.

## Current questions

- `mix-add/` — does Mix-Add actually preserve norms? when does it help vs hurt?
- `broadcast-router/` — what should the global communication channel look like?
- `async-execution/` — how real can async column updates be on GPU?
- `gradient-unhooked/` — what should and shouldn't be unhooked? what does unhooking do?
- `dynamic-depth/` — does adaptive computation depth work on a standard model?
- `column-residual-stream/` — what is the right definition of a column's internal state?
- `update-trigger/` — what triggers a column update? surprisal semantics?
- `io-contract/` — what is the I/O contract for the architecture?
- `bidirectional-propagation/` — what does bidirectional propagation look like?
- `graph-vs-global-channel/` — does the graph matter or does the global channel dominate?
