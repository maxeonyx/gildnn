# research/

Distilled verified discovery — not reports, not logs. Everything in here that Max reads should earn its place.

## Structure

- `daily/YYYY-MM-DD.md` — daily narrative. What ran, what it produced, what changed, what's next.
- `weekly/YYYY-MM-DD.md` — weekly synthesis. Significant findings, evidence, honest assessment.
- `questions/` — one folder per open question or investigation thread.

Curated, Max-readable output only. Transient notes, debug logs, and scratch work do not belong here. Detailed investigation records belong in `questions/<name>/README.md`; daily/weekly narratives should synthesize and point to evidence, not restate the full investigation.

---

## Report process

### When to write

Check the time on every session start. See `AGENTS.md` for the rule. Write at the next natural stopping point — not mid-task.

### Before writing

Each experiment's question folder (`research/questions/<name>/`) should have been keeping a record as work proceeded: run configs, key metrics, output artifacts (images, example outputs), and a note of what was learned. The daily/weekly narrative draws from these — it does not reconstruct from memory.

If artifacts are missing, run the experiment again or note the gap explicitly. Do not write claims that aren't backed by a saved artifact.

### How to write

1. Draft
2. Revise — cut everything that isn't load-bearing
3. Revise — is every claim backed by something real and embedded inline?
4. Revise — is the writing good? Is it interesting?
5. Revise — read it as Max would. Does it waste his time?
6. Final pass — does it meet the standard?

Iterate heavily. The first draft is never the output.

### Quality standard

distill.pub quality. Scott Alexander / Slate Star Codex level prose. Dense, precise, readable. Zero padding. No unverified claims stated as facts.

Every factual claim must be backed by a real artifact:
- embedded inline in the markdown (not just linked — Max should be able to read without clicking),
- with the link present to prove it's real (make the link a word or image, not a bare URL).

### What goes in a daily

- What ran
- What it produced — with inline artifacts
- One honest sentence per result: what it means
- What changed in understanding
- What's next

### What goes in a weekly

- The week's most significant findings, each with inline evidence
- What open questions narrowed (or opened)
- What the code looks like now vs last week
- Honest assessment: are we making progress?

---

## Artifact policy

**Commit:** small output images, example input/output pairs, embedded in markdown files.
**Do not commit:** model weights, datasets, large binary files of any kind.

Artifacts live in the question folder they belong to. Reference them from daily/weekly by relative path or embed them inline.
