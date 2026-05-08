# Question: Async Execution

## What we're asking

How real can asynchronous column updates be on a GPU? Can columns genuinely run at different rates, triggered by surprisal, or does GPU architecture force a synchronous approximation? The aspiration is full asynchrony — even at the GPU level. Whether that's achievable is unknown.

## Current state of knowledge

- True per-column event queues with irregular triggering are likely hostile to GPU efficiency (small kernels, branching, scatter/gather, poor tensor-core utilisation) — but this is an engineering concern, not a settled fact
- A semi-async masked approximation (fixed-size macrosteps where only active columns update, represented as masked dense tensors) is one possible practical path — not necessarily the only one
- Whether "real" async is achievable, or whether the semi-async approximation is meaningfully different, is unknown
- The surprisal trigger is intuitive but not designed — see `update-trigger/` for details on that sub-question

## Open sub-questions

- Can GPU-level async kernels (CUDA streams, custom ops) get close to the ideal?
- Is the semi-async approximation distinguishable from true async in practice?
- What is the performance/expressivity tradeoff at different levels of async approximation?
- Can cascading updates happen within a macrostep, or do we need a refractory period?
- Does the answer change at different column-count scales?

## Status

Not yet investigated experimentally in this repo.

