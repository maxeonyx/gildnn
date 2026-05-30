# gildnn
Ultra flexible recurrent transformer

## Development Approach
- Build one focused experiment crate at a time and keep it runnable on tiny datasets.
- Each experiment (`base-experiments/*/`, `experiments/*/`) focuses on creating a deterministic, self-contained `report.md` file, explaining the full experiment step-by-step with high-quality visualizations.
- Every crate must support both `--mode full` and `--mode test` so that experiments can also be used as regression tests when refactoring out shared code into `core/`.
