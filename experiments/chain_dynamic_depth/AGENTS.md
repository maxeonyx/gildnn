# experiments/chain_dynamic_depth/

This experiment directory implements a **now-superseded architecture interpretation** — a chain of blocks with their own hidden structures, not the current single-residual-block design.

At the time this code was written, the "chain" meant a sequence of nodes where each node had its own hidden dimension and internal structure. The [dictation from 2026-05-20-10](../../dictations/2026-05-20-10.md) later clarified that modules should instead be **single residual blocks on a shared `d_model` residual stream**.

This code is kept for historical reference. The results it produced are documented in [research/questions/predictive-chain/README.md](../../research/questions/predictive-chain/README.md) and [research/questions/chain-dynamic-depth/README.md](../../research/questions/chain-dynamic-depth/README.md).

Do not use this as a base for new experiments. New chain or dynamic-depth experiments should implement the clarified architecture (single residual blocks, uniform `d_model`).
