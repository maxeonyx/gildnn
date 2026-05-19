# core/

Shared reusable components: datasets, models, training loops, utilities.

## Purpose

This is the project's flexible model library. Code here must be:
- **Tested** — it should work reliably
- **Reusable** — used by both `base-experiments/` and `experiments/`
- **Understood** — well-commented, clear interfaces

Code moves here from `experiments/` once it's verified stable and reused. Do not promote one-session-old experiment code into core/ just because it worked once — wait for stability evidence (e.g., seed stability, reuse across experiments).

## What belongs here

- Dataset classes (e.g., fixed-window character dataset)
- Model components (embeddings, attention, recurrent cells, prediction heads)
- Training utilities (training loops, evaluation, device/seed setup)
- Anything shared between two or more experiments

## What does NOT belong here

- Experiment-specific scripts or hyperparameter configs
- One-off analysis code
- Code that's still evolving rapidly

## Running

Components are imported as `from core.module_name import ClassName`.
