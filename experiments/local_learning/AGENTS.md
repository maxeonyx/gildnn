# experiments/local_learning/

This experiment directory implements a **now-superseded architecture interpretation** — detached recurrent stacks as local modules, not the current single-residual-block design.

At the time this code was written, "module" meant a small stack of recurrent layers (3 layers per module) with its own persistent hidden state and stop-gradient boundaries between modules. The [dictation from 2026-05-20-10](../../dictations/2026-05-20-10.md) later clarified that modules should instead be **single residual blocks on a shared `d_model` residual stream**.

This code is kept for historical reference. The results it produced are documented in [research/questions/local-learning/README.md](../../research/questions/local-learning/README.md).

Do not use this as a base for new experiments. New local-learning code should implement the clarified architecture (single residual blocks, uniform `d_model`, stop-gradient boundaries at block interfaces).
