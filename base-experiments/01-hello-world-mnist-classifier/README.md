# Hello World MNIST Classifier

This stage now hosts the executable base crate (see `Cargo.toml` and `src/`) that exercises the MNIST pipeline. Use it as the canonical training loop before layering on research aspects.

Objective:
- Stand up the core training harness with the smallest possible supervised task.

Key Requirements:
- Download MNIST, normalize inputs, and create batched dataloaders.
- Implement a minimal CNN or MLP classifier using the shared model interface.
- Have the experiment produce a single self-contained `report.md`.
- Surface the primary knobs (seed, hidden width, train steps for `full` and `test` modes) through `config.json`.
- Have a test mode that writes `actual.ignore.json`; the shared harness compares it against `expected.json` for regression checks.

Scaffolding Expectations:
- Define the base project layout (data module, model module, trainer module).
- Introduce a registry or factory pattern that later stages can extend without modifying call sites.
- Ensure command-line or config-driven entrypoint can be reused for subsequent demos.
