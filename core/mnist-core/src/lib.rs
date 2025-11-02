//! Shared MNIST utilities for gildnn experiments.

pub mod dataset;
pub mod image;
pub mod schedule;

/// Side length of MNIST images (28 pixels).
pub const MNIST_IMAGE_SIDE: usize = 28;

/// Total pixel count for a single MNIST image.
pub const MNIST_PIXEL_COUNT: usize = MNIST_IMAGE_SIDE * MNIST_IMAGE_SIDE;

/// Default number of digits rendered per panel in our reports.
pub const DEFAULT_PANEL_DIGITS: usize = 10;
