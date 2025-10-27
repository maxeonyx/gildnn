# Dynamic Depth Output Scheduling

## Objective
Study mechanisms that let the network emit predictions from multiple depths based on estimated confidence or anticipated loss.

## Key Ideas
- Attach lightweight readout heads at several layers that can produce outputs when local loss estimates fall below a threshold.
- Train auxiliary modules to forecast downstream loss so the system can decide whether to continue propagating information.
- Allow adaptive halting policies that choose between shallow, fast predictions and deeper, more informed passes.

## Open Questions
- What training signals best calibrate per-depth loss estimators without destabilizing the main objective?
- How should gradients be shared between early-exit heads and deeper layers to avoid mode collapse on the shallow paths?
- Can dynamic depth decisions be coordinated with dynamic chunking to balance latency and accuracy jointly?
