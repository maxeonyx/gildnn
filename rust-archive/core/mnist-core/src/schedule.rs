/// Generate a logarithmically spaced prediction schedule limited to `max_images` snapshots.
pub fn prediction_schedule(total_steps: usize, max_images: usize) -> Vec<usize> {
    if total_steps == 0 || max_images == 0 {
        return Vec::new();
    }

    let mut steps = Vec::new();
    let mut value = 1usize;
    while value < total_steps && steps.len() + 1 < max_images {
        steps.push(value);
        value *= 2;
    }
    if steps.last().copied() != Some(total_steps) {
        steps.push(total_steps);
    }
    steps
}
