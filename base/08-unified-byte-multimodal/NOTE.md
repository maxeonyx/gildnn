# Unified Byte Multi-Modal Model

Objective:
- Demonstrate a single architecture that seamlessly alternates between byte-text and byte-image tasks.

Key Requirements:
- Implement modality tags or routing tokens so the model can distinguish input domains.
- Provide training curricula that mix datasets while preserving balanced sampling and logging per-modality metrics.
- Enable evaluation scripts to select the appropriate decoding/rendering path based on recorded modality metadata.

Integration Notes:
- Centralize configuration to toggle between single-task and multi-task schedules without altering the training entrypoint.
- Ensure checkpoint formats capture modality-specific heads and any shared parameters needed for joint training.
- Prepare for future research augmentations (dynamic chunking, depth scheduling, etc.) by keeping hooks for auxiliary heads and
  visualization streams in the shared infrastructure.
