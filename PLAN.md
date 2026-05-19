# Plan

Working file. Rewrite it as the state changes.

---

## Current state

- Working Python path: UV + `.venv` + CPython 3.12.12 + PyTorch CUDA on RTX 3090.
- **Predictive chain architecture shows real generalization advantage.** On Shakespeare (7k chars, context=5, 80/20 split), the 8-node detached chain gets val loss 2.56-2.59, beating transformer (2.80), RNN (3.05), feedforward (4.87).
- **Why it generalizes:** ablations show it's the multi-hop recurrent structure itself — not aux pressure (aux weight doesn't matter) and not message bottleneck (removing it doesn't hurt). The architecture's structural constraint provides implicit regularization.
- **Unhooked gradients work** — detached messages don't harm performance.
- All experiments run in minutes on the RTX 3090.

## Next experiment

**Attention between neighbors.** Max's vision specifically describes attention as the aggregation mechanism: "how do inputs get aggregated by a node? They use attention." Currently each node receives a single message vector from its predecessor. The next step is to let each node attend over its neighbor's recent output history (last K messages).

This is a meaningful structural change:
- Nodes would maintain a buffer of recent messages
- Each node attends over its neighbor's buffer rather than just receiving the latest single message
- This enables the "predict attention for the next timestep" idea from Max's dictation

Start small: try K=3-5 message history, single-head attention, on the Shakespeare comparison.

## Possible future directions

- Increase context size (5 → 20-50) to test longer-range dependency handling
- Graph topology (grid, tree) instead of line
- Loss-prediction heads for dynamic halting
- Larger model / more training to see if advantage grows or shrinks

## Live constraints

- Integrate before starting broad new branches.
- Keep the codebase small.
- Experiment ladder: overfit one batch, tiny end-to-end, inspect outputs, scale.
- Open questions stay open.
