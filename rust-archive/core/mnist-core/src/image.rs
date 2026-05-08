use anyhow::{anyhow, Result};
use burn_dataset::vision::MnistItem;

use crate::{dataset::item_to_normalized_pixels, MNIST_IMAGE_SIDE};
use gildnn_core::{encode_luma_png, encode_rgb_png};

/// Bundles metadata for an evaluation panel image.
pub struct EvaluationPanel {
    pub index: usize,
    pub correct: usize,
    pub total: usize,
    pub image_bytes: Vec<u8>,
}

/// Render a contiguous row of MNIST digits as a grayscale PNG.
pub fn build_dataset_row_image(items: &[MnistItem]) -> Result<Vec<u8>> {
    if items.is_empty() {
        return Err(anyhow!("cannot build dataset row for empty slice"));
    }

    let width = (MNIST_IMAGE_SIDE * items.len()) as u32;
    let height = MNIST_IMAGE_SIDE as u32;
    let mut pixels = Vec::with_capacity((width * height) as usize);

    for y in 0..MNIST_IMAGE_SIDE {
        for item in items {
            for &pixel in item.image[y].iter() {
                pixels.push(pixel as f32 / 255.0);
            }
        }
    }

    encode_luma_png(width, height, &pixels)
}

/// Render digit + label rows used in the hypothesis panels.
pub fn build_hypothesis_panel(items: &[MnistItem]) -> Result<Vec<u8>> {
    if items.is_empty() {
        return Err(anyhow!("cannot build hypothesis panel for empty slice"));
    }

    let width = (MNIST_IMAGE_SIDE * items.len()) as u32;
    let height = (MNIST_IMAGE_SIDE * 2) as u32;
    let mut pixels = vec![0.0; (width * height) as usize];

    for (column, item) in items.iter().enumerate() {
        let digit_pixels = item_to_normalized_pixels(item);
        for y in 0..MNIST_IMAGE_SIDE {
            for x in 0..MNIST_IMAGE_SIDE {
                let dest_x = column * MNIST_IMAGE_SIDE + x;
                let dest_y = y;
                let dest_index = dest_y * (width as usize) + dest_x;
                pixels[dest_index] = digit_pixels[y * MNIST_IMAGE_SIDE + x];
            }
        }

        let label_mask = render_digit_mask(item.label as usize);
        for y in 0..MNIST_IMAGE_SIDE {
            for x in 0..MNIST_IMAGE_SIDE {
                let dest_x = column * MNIST_IMAGE_SIDE + x;
                let dest_y = MNIST_IMAGE_SIDE + y;
                let dest_index = dest_y * (width as usize) + dest_x;
                let mask = label_mask[y * MNIST_IMAGE_SIDE + x];
                pixels[dest_index] = mask * 0.85f32;
            }
        }
    }

    encode_luma_png(width, height, &pixels)
}

/// Render prediction panels stacking input, expected label, and predicted label rows.
pub fn build_prediction_panel(
    inputs: &[Vec<f32>],
    expected: &[usize],
    predicted: &[usize],
) -> Result<Vec<u8>> {
    if inputs.len() != expected.len() || expected.len() != predicted.len() {
        return Err(anyhow!(
            "prediction panel requires equal lengths (inputs {}, expected {}, predicted {})",
            inputs.len(),
            expected.len(),
            predicted.len()
        ));
    }
    if inputs.is_empty() {
        return Err(anyhow!("cannot build prediction panel for empty inputs"));
    }

    let columns = inputs.len();
    let width = (MNIST_IMAGE_SIDE * columns) as u32;
    let height = (MNIST_IMAGE_SIDE * 3) as u32;
    let mut pixels = vec![0.0; (width * height * 3) as usize];

    for column in 0..columns {
        let is_correct = expected[column] == predicted[column];
        let background = if is_correct {
            [0.08, 0.24, 0.08]
        } else {
            [0.24, 0.08, 0.08]
        };

        let input_pixels = &inputs[column];
        for y in 0..MNIST_IMAGE_SIDE {
            for x in 0..MNIST_IMAGE_SIDE {
                let dest_x = column * MNIST_IMAGE_SIDE + x;
                let dest_y = y;
                let dest_index = (dest_y * (width as usize) + dest_x) * 3;
                let value = input_pixels[y * MNIST_IMAGE_SIDE + x];
                let color = mix_grayscale_with_background(value, background);
                pixels[dest_index] = color[0];
                pixels[dest_index + 1] = color[1];
                pixels[dest_index + 2] = color[2];
            }
        }

        let expected_mask = render_digit_mask(expected[column]);
        for y in 0..MNIST_IMAGE_SIDE {
            for x in 0..MNIST_IMAGE_SIDE {
                let dest_x = column * MNIST_IMAGE_SIDE + x;
                let dest_y = MNIST_IMAGE_SIDE + y;
                let dest_index = (dest_y * (width as usize) + dest_x) * 3;
                let mask = expected_mask[y * MNIST_IMAGE_SIDE + x];
                let color = render_label_pixel(mask, background);
                pixels[dest_index] = color[0];
                pixels[dest_index + 1] = color[1];
                pixels[dest_index + 2] = color[2];
            }
        }

        let predicted_mask = render_digit_mask(predicted[column]);
        for y in 0..MNIST_IMAGE_SIDE {
            for x in 0..MNIST_IMAGE_SIDE {
                let dest_x = column * MNIST_IMAGE_SIDE + x;
                let dest_y = MNIST_IMAGE_SIDE * 2 + y;
                let dest_index = (dest_y * (width as usize) + dest_x) * 3;
                let mask = predicted_mask[y * MNIST_IMAGE_SIDE + x];
                let color = render_label_pixel(mask, background);
                pixels[dest_index] = color[0];
                pixels[dest_index + 1] = color[1];
                pixels[dest_index + 2] = color[2];
            }
        }
    }

    encode_rgb_png(width, height, &pixels)
}

/// Construct a set of evaluation panels segmented into chunks of `panel_size`.
pub fn build_final_evaluation_panels(
    inputs: &[Vec<f32>],
    labels: &[usize],
    predictions: &[usize],
    panel_size: usize,
) -> Result<Vec<EvaluationPanel>> {
    if inputs.len() != labels.len() || labels.len() != predictions.len() {
        return Err(anyhow!(
            "final evaluation requires equal lengths (inputs {}, labels {}, predictions {})",
            inputs.len(),
            labels.len(),
            predictions.len()
        ));
    }
    if inputs.is_empty() {
        return Ok(Vec::new());
    }

    let mut panels = Vec::new();
    for (panel_idx, start) in (0..inputs.len()).step_by(panel_size).enumerate() {
        let end = (start + panel_size).min(inputs.len());
        if start >= end {
            break;
        }

        let panel_inputs = &inputs[start..end];
        let panel_labels = &labels[start..end];
        let panel_predictions = &predictions[start..end];
        let image = build_prediction_panel(panel_inputs, panel_labels, panel_predictions)?;
        let correct = panel_predictions
            .iter()
            .zip(panel_labels)
            .filter(|(pred, label)| pred == label)
            .count();
        panels.push(EvaluationPanel {
            index: panel_idx + 1,
            correct,
            total: panel_labels.len(),
            image_bytes: image,
        });
    }

    Ok(panels)
}

/// Render an autoregressive completion panel showing masked inputs and model outputs.
pub fn build_autoregressive_panel(
    context_tokens: usize,
    seeds: &[Vec<u8>],
    completions: &[Vec<u8>],
    patch_size: usize,
) -> Result<Vec<u8>> {
    if seeds.len() != completions.len() {
        return Err(anyhow!(
            "seed/completion mismatch ({} vs {})",
            seeds.len(),
            completions.len()
        ));
    }
    if seeds.is_empty() {
        return Err(anyhow!("cannot build autoregressive panel for empty inputs"));
    }

    let patches_per_side = MNIST_IMAGE_SIDE / patch_size;
    let token_count = patches_per_side * patches_per_side;

    let columns = seeds.len();
    let width = (MNIST_IMAGE_SIDE * columns) as u32;
    let height = (MNIST_IMAGE_SIDE * 2) as u32;
    let mut pixels = vec![0.0f32; (width * height * 3) as usize];

    for column in 0..columns {
        let observed = &seeds[column];
        let generated = &completions[column];
        if observed.len() < token_count || generated.len() < token_count {
            return Err(anyhow!("token sequence too short for MNIST patch reconstruction"));
        }

        for py in 0..patches_per_side {
            for px in 0..patches_per_side {
                let idx = py * patches_per_side + px;
                let obs_color = if idx < context_tokens {
                    let value = observed[idx] as f32 / 255.0;
                    [value, value, value]
                } else {
                    [1.0, 0.6, 0.8]
                };
                let gen_value = generated[idx] as f32 / 255.0;

                for dy in 0..patch_size {
                    for dx in 0..patch_size {
                        let base_x = px * patch_size + dx;
                        let base_y = py * patch_size + dy;

                        // Observed row (top)
                        let dest_x_obs = column * MNIST_IMAGE_SIDE + base_x;
                        let dest_y_obs = base_y;
                        let idx_obs = (dest_y_obs * (width as usize) + dest_x_obs) * 3;
                        pixels[idx_obs] = obs_color[0];
                        pixels[idx_obs + 1] = obs_color[1];
                        pixels[idx_obs + 2] = obs_color[2];

                        // Completion row (bottom)
                        let dest_y_out = MNIST_IMAGE_SIDE + base_y;
                        let idx_out = (dest_y_out * (width as usize) + dest_x_obs) * 3;
                        pixels[idx_out] = gen_value;
                        pixels[idx_out + 1] = gen_value;
                        pixels[idx_out + 2] = gen_value;
                    }
                }
            }
        }
    }

    encode_rgb_png(width, height, &pixels)
}

fn render_digit_mask(digit: usize) -> Vec<f32> {
    const FONT: [[&str; 7]; 10] = [
        [
            "  ###  ",
            " #   # ",
            "#     #",
            "#     #",
            "#     #",
            " #   # ",
            "  ###  ",
        ],
        [
            "   #   ",
            "  ##   ",
            " # #   ",
            "   #   ",
            "   #   ",
            "   #   ",
            " ##### ",
        ],
        [
            " ##### ",
            "#     #",
            "      #",
            "   ### ",
            "  #    ",
            " #     ",
            "#######",
        ],
        [
            " ##### ",
            "#     #",
            "      #",
            " ##### ",
            "      #",
            "#     #",
            " ##### ",
        ],
        [
            "    ## ",
            "   # # ",
            "  #  # ",
            " #   # ",
            "#######",
            "     # ",
            "     # ",
        ],
        [
            "#######",
            "#      ",
            "#      ",
            "###### ",
            "      #",
            "#     #",
            " ##### ",
        ],
        [
            "  #### ",
            " #     ",
            "#      ",
            "###### ",
            "#     #",
            "#     #",
            " ##### ",
        ],
        [
            "#######",
            "     # ",
            "    #  ",
            "   #   ",
            "  #    ",
            " #     ",
            "#      ",
        ],
        [
            " ##### ",
            "#     #",
            "#     #",
            " ##### ",
            "#     #",
            "#     #",
            " ##### ",
        ],
        [
            " ##### ",
            "#     #",
            "#     #",
            " ######",
            "      #",
            "     # ",
            " ####  ",
        ],
    ];

    let digit = digit % 10;
    let glyph = &FONT[digit];
    let mut mask = vec![0.0; MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE];
    let x_offset = (MNIST_IMAGE_SIDE - glyph[0].len()) / 2;
    let y_offset = (MNIST_IMAGE_SIDE - glyph.len()) / 2;

    for (y, row) in glyph.iter().enumerate() {
        for (x, ch) in row.chars().enumerate() {
            if ch != ' ' {
                let dest_x = x_offset + x;
                let dest_y = y_offset + y;
                if dest_x < MNIST_IMAGE_SIDE && dest_y < MNIST_IMAGE_SIDE {
                    mask[dest_y * MNIST_IMAGE_SIDE + dest_x] = 1.0;
                }
            }
        }
    }

    mask
}

fn mix_grayscale_with_background(value: f32, background: [f32; 3]) -> [f32; 3] {
    [
        background[0] * (1.0 - value) + value,
        background[1] * (1.0 - value) + value,
        background[2] * (1.0 - value) + value,
    ]
}

fn render_label_pixel(mask: f32, background: [f32; 3]) -> [f32; 3] {
    if mask > 0.0 {
        [1.0, 1.0, 1.0]
    } else {
        background
    }
}
