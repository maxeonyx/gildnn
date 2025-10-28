use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};

/// Encode a grayscale image (values in [0, 1]) as PNG bytes.
pub fn encode_luma_png(width: u32, height: u32, pixels: &[f32]) -> Result<Vec<u8>> {
    if pixels.len() != (width * height) as usize {
        anyhow::bail!(
            "pixel buffer length {} does not match image size {}x{}",
            pixels.len(),
            width,
            height
        );
    }

    let mut encoded = Vec::with_capacity((width * height) as usize);
    for &value in pixels {
        let clamped = value.clamp(0.0, 1.0);
        encoded.push((clamped * 255.0).round() as u8);
    }

    let mut buffer = Vec::new();
    let encoder = PngEncoder::new(&mut buffer);
    encoder
        .write_image(&encoded, width, height, ColorType::L8)
        .context("failed to encode PNG data")?;

    Ok(buffer)
}

/// Encode an RGB image (values in [0, 1]) as PNG bytes.
pub fn encode_rgb_png(width: u32, height: u32, pixels: &[f32]) -> Result<Vec<u8>> {
    let expected_len = (width * height * 3) as usize;
    if pixels.len() != expected_len {
        anyhow::bail!(
            "pixel buffer length {} does not match RGB image size {}x{}",
            pixels.len(),
            width,
            height
        );
    }

    let mut encoded = Vec::with_capacity(expected_len);
    for chunk in pixels.chunks_exact(3) {
        encoded.push((chunk[0].clamp(0.0, 1.0) * 255.0).round() as u8);
        encoded.push((chunk[1].clamp(0.0, 1.0) * 255.0).round() as u8);
        encoded.push((chunk[2].clamp(0.0, 1.0) * 255.0).round() as u8);
    }

    let mut buffer = Vec::new();
    let encoder = PngEncoder::new(&mut buffer);
    encoder
        .write_image(&encoded, width, height, ColorType::Rgb8)
        .context("failed to encode RGB PNG data")?;

    Ok(buffer)
}

/// Convert pre-encoded PNG bytes into a data URI suitable for embedding in markdown.
pub fn png_data_uri(bytes: &[u8]) -> String {
    let encoded = STANDARD.encode(bytes);
    format!("data:image/png;base64,{encoded}")
}
