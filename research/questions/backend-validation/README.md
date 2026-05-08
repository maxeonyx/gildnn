# Question: Backend Validation

## What we're asking

What is the smallest local Python backend path that actually works on this Windows machine for this repo?

The bar for "works" here is operational, not research-facing: a UV-managed local virtualenv, one backend import, one visible CUDA device, and one tiny tensor operation that actually executes on the GPU.

## Probe run — 2026-05-08

Project files created for the probe:

- [`.python-version`](../../../.python-version)
- [`pyproject.toml`](../../../pyproject.toml)

Pinned backend source for this probe:

```toml
[project]
requires-python = "==3.12.*"
dependencies = [
    "torch==2.11.0+cu128",
]
```

Observed proof artifact from the probe:

```json
{
  "python_version": "3.12.12",
  "torch_version": "2.11.0+cu128",
  "torch_cuda_runtime": "12.8",
  "cuda_is_available": true,
  "cuda_device_count": 1,
  "tensor_device": "cuda:0",
  "result_device": "cuda:0",
  "result_values": [2.0, 4.0, 6.0],
  "current_device_name": "NVIDIA GeForce RTX 3090"
}
```

Machine context captured during the same probe:

- NVIDIA driver `591.86`
- `nvidia-smi` reported CUDA `13.1`

## What this shows

One minimal local backend path works in this repo on this machine: UV + local `.venv` + CPython 3.12.12 + `torch==2.11.0+cu128`, with a tiny tensor multiply succeeding on `cuda:0`.

## What this does **not** show

- It does **not** settle the project backend/framework choice.
- It does **not** prove the experiment harness, sanity-check pipeline, or first model.
- It does **not** prove `loop.ps1` end-to-end in production.

## Next discriminating step

Use this proven PyTorch path for the smallest sanity-check experiment, unless a separate bounded probe of another backend path becomes the more discriminating next step. Keep the backend choice explicitly open.
