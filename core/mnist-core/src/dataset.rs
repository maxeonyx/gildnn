use anyhow::{anyhow, Result};
use burn_dataset::{
    vision::{MnistDataset, MnistItem},
    Dataset,
};
use rand::{rngs::StdRng, seq::index::sample};

use crate::{MNIST_IMAGE_SIDE, MNIST_PIXEL_COUNT};

/// Collect the first `count` items from the dataset.
pub fn collect_first_items(dataset: &MnistDataset, count: usize) -> Result<Vec<MnistItem>> {
    let available = count.min(dataset.len());
    let mut items = Vec::with_capacity(available);
    for index in 0..available {
        let item = dataset
            .get(index)
            .ok_or_else(|| anyhow!("dataset index {} out of bounds", index))?;
        items.push(item);
    }
    Ok(items)
}

/// Sample `count` random items from the dataset with a deterministic RNG.
pub fn sample_random_items(
    dataset: &MnistDataset,
    count: usize,
    rng: &mut StdRng,
) -> Result<Vec<MnistItem>> {
    let available = count.min(dataset.len());
    if available == 0 {
        return Ok(Vec::new());
    }

    let indices = sample(rng, dataset.len(), available).into_vec();
    let mut items = Vec::with_capacity(available);
    for index in indices {
        let item = dataset
            .get(index)
            .ok_or_else(|| anyhow!("dataset index {} out of bounds", index))?;
        items.push(item);
    }
    Ok(items)
}

/// Convert a dataset item into a flattened vector of `[0, 1]` pixel intensities.
pub fn item_to_normalized_pixels(item: &MnistItem) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(MNIST_PIXEL_COUNT);
    for row in item.image.iter() {
        for &pixel in row.iter() {
            pixels.push(pixel as f32 / 255.0);
        }
    }
    pixels
}

/// Convert a dataset item into a flattened vector of raw pixel tokens `[0, 255]`.
pub fn item_to_token_sequence(item: &MnistItem) -> Vec<u8> {
    let mut tokens = Vec::with_capacity(MNIST_PIXEL_COUNT);
    for row in item.image.iter() {
        for &pixel in row.iter() {
            tokens.push(pixel as u8);
        }
    }
    tokens
}

/// Downsample an MNIST digit into average-valued patch tokens.
pub fn item_to_patch_tokens(item: &MnistItem, patch_size: usize) -> Vec<u8> {
    let patches_per_side = MNIST_IMAGE_SIDE / patch_size;
    let mut tokens = Vec::with_capacity(patches_per_side * patches_per_side);

    for py in 0..patches_per_side {
        for px in 0..patches_per_side {
            let mut sum = 0u32;
            for dy in 0..patch_size {
                for dx in 0..patch_size {
                    let y = py * patch_size + dy;
                    let x = px * patch_size + dx;
                    sum += item.image[y][x] as u32;
                }
            }
            let area = (patch_size * patch_size) as u32;
            let value = (sum / area) as u8;
            tokens.push(value);
        }
    }

    tokens
}

/// Extract the label index for the dataset item.
pub fn item_label(item: &MnistItem) -> usize {
    item.label as usize
}

/// Re-export the dataset item type for convenience.
pub use burn_dataset::vision::MnistItem as Item;
