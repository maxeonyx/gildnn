# research/questions/local-learning/

This directory contains results from an **early local-learning experiment that used a now-superseded architecture interpretation**.

At the time of this experiment, "module" was interpreted as a small stack of recurrent layers (3 layers per module) with its own persistent hidden state. The [dictation from 2026-05-20-10](../../../dictations/2026-05-20-10.md) later clarified that modules should instead be **single residual blocks on a shared `d_model` residual stream** — not thick recurrent stacks with their own hidden dimension.

The results here are valid evidence about the **detached recurrent-stack architecture family**. They are historical record, not a model for the current design direction. Keep this context when reading README.md.

Do not build new experiments in this directory on top of the old architecture. New local-learning experiments should be in a separate question folder under the clarified single-residual-block framing.
