use anyhow::{Context, Result};
use base64::Engine;
use image::{codecs::png::PngEncoder, ColorType, ImageEncoder};

/// Encode a grayscale image (values in [0, 1]) as a PNG data URL.
pub fn encode_luma_png_data_url(width: u32, height: u32, pixels: &[f32]) -> Result<String> {
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

    let base64 = base64::engine::general_purpose::STANDARD.encode(&buffer);
    Ok(format!("data:image/png;base64,{base64}"))
}
