# gildnn
Ultra flexible recurrent transformer

## Development Approach
- Build one focused experiment crate at a time and keep it runnable on tiny datasets.
- Push back when scope creeps—break multi-idea requests into sequential steps.
- Each experiment (`base-experiments/hello-world-mnist-classifier`, `base-experiments/autoregressive-mnist`, etc.) ships with its own `config.json`, `benchmark.json`, and deterministic `report.md`.
- Every crate must support both `--mode full` and `--mode test` so we can validate changes quickly before running full training.
- Preserve prior artefacts so new runs can be compared against old baselines directly.
