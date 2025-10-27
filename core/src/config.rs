use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::{de::DeserializeOwned, Serialize};

/// Load a JSON configuration from disk, creating it with the provided initializer if missing.
pub fn load_or_init<T, F>(path: &Path, initializer: F) -> Result<T>
where
    T: Serialize + DeserializeOwned,
    F: FnOnce() -> T,
{
    if path.exists() {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("failed to read config from {}", path.display()))?;
        let value = serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse config from {}", path.display()))?;
        Ok(value)
    } else {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
        let value = initializer();
        let serialized = serde_json::to_string_pretty(&value)?;
        fs::write(path, serialized)
            .with_context(|| format!("failed to write config to {}", path.display()))?;
        Ok(value)
    }
}
