# research/questions/

One folder per open question, investigation thread, or specific research goal.

## Purpose

Each folder documents one question or goal: what we think we know, what we tried, what it showed, what remains open. When a question is resolved, the folder stays — it becomes a record of how the question was closed.

Folders are not only for open questions. A folder can be for a specific research goal (e.g. "make Mix-Add work on image patches") or for a debugging investigation that turned into a documented finding.

## Add new folders freely

If you encounter something that doesn't work and you figure out why — create a folder, document it, add a cheap regression test, integrate the fix. Don't just move on.

If an experiment raises a new question that isn't in an existing folder — create one. The folder name should be a terse description of the question (e.g. `layer-norm-instability`, `patch-order-bias`).

## Folder structure

Each question folder contains:

- `README.md` — the Max-readable report. This is the primary artifact.

That's it. The README is the output. It should be self-contained.

## What goes IN the README

- Which dictation/VISION goal this question serves (link to the specific dictation)
- The simplification chosen and why it's legitimate
- Architecture explanation with code snippets and/or diagrams aligned to real code
- Hypotheses being tested
- Results with **inline evidence** — tables, code blocks, small images embedded directly in the markdown
- What this does not settle (explicit non-goals)
- Obvious next steps with reasons they haven't been pursued yet

The link proves the evidence is real; the inline content means Max doesn't have to click anything.

## What does NOT go here

- Standalone .txt files, raw JSON dumps, prediction logs, debug outputs
- Any artifact that exists only to be referenced by a link

Those belong in `experiments/<experiment-name>/` as grungy experiment artifacts. If an artifact is worth showing Max, embed it inline in the README. If it's not worth showing inline, it doesn't belong in this directory.

## Quality standard

Unresolved conclusions must stay explicitly unresolved until backed by real artifacts. Do not state a question as closed without evidence.

Reports should link back to Max's goals, explain the path the experiment took, and end with discriminating next steps. See `PROCESS.md` for the report-first experiment protocol.

If a framing, term, or architecture name came from a prior agent rather than the dictations (for example, "predictive chain"), present it as a simplification or hypothesis, not as Max's core idea.

## Docs reminder

Load the `information-architecture` OpenCode skill before restructuring or reorganising question write-ups.
