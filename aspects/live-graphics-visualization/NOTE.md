# Live Graphics Stack Integration

## Objective
Create a training pipeline that shares buffers between neural network computation and a graphics renderer so that internal state can be displayed with minimal latency.

## Key Ideas
- Maintain GPU-resident tensors for activations, gradients, and weights that double as renderable buffers.
- Design a modular visualization layer capable of mapping tensor slices to color channels, meshes, or particle systems.
- Stream updates through a ring-buffer mechanism to allow continuous rendering without stalling training steps.

## Open Questions
- Which graphics API (e.g., Vulkan, OpenGL, DirectX, WebGPU) best aligns with the compute framework in use?
- How can synchronization primitives be minimized to avoid blocking either training or rendering pipelines?
- What level of spatial or temporal aggregation keeps the visualization intelligible without overwhelming bandwidth?
