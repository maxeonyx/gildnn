# Plan

Immediate checklist. What's next, what I'll do based on each outcome. For the bigger picture, read VISION.md.

## Now

- [ ] **Check 5k-step multi-rate results** — background run active (PID 5512 in `runs/active.lock`). When done: if quality-neutral + speedup holds → scale to more aggressive rates. If quality degrades → investigate why.
- [ ] **VISION.md rewrite** — one-time task. Rewrite from the full dictations, keeping Max's language, rearranging for flow. Multiple review rounds asking: "does this capture the spirit of my vision?" Then tick off.

## Next (after current items)

- [ ] **Scale multi-rate** — try rates [1, 2, 4, 8], try 8 blocks. Goal: >20% speedup without quality loss. If speedup scales → this is the path. If quality degrades at aggressive rates → find the boundary.
- [ ] **Diagonal + multi-rate** — the diagonal residual (block1's output at time t feeds block2 at time t+1) naturally pairs with multi-rate. If block2 runs every 2nd step, the diagonal connection IS the stale-read mechanism. Design experiment, write question doc, run.
- [ ] **Backend decision** — JAX recommended (`research/questions/backend-choice/README.md`). Awaiting Max's input.
- [ ] **Core reintegration** — once backend is decided, rewrite core/ to be clean, compiled, and reusable.

## Queue (lower priority)

- Arbitrary-order sampling — deprioritized per [dictation 2026-05-22-6](dictations/2026-05-22-6.md). Prototype works (MSE 0.0195).
- Self-prediction — NEGATIVE (+0.011-0.018 nats)
- Dynamic token count — not yet explored
- Complex-valued / orthogonal parameterization — exp-map was quality-negative
- Volume-preserving nonlinearities — highly speculative
- Declining batch size — easy bolt-on for any run

## Done

- Fixed multi-rate: **POSITIVE** — 14.8% speedup, no quality loss. See `research/questions/fixed-multi-rate/README.md`.
- Arbitrary-order MNIST: **WORKING** — per-pixel MSE 0.0195. See `research/questions/arbitrary-order-sampling/README.md`.
- Dynamic depth: **POSITIVE** — 43% compute savings for 1% quality loss. See `research/questions/dynamic-depth/README.md`.
- Backend research: **DONE** — See `research/questions/backend-choice/README.md`.
- Thread 1 (18 experiments): architecture viable but dominated by transformers. See `research/SYNTHESIS.md`.
