use std::{fs, path::Path};

use anyhow::{anyhow, Context, Result};

#[derive(Clone, Debug)]
pub struct ReportSection {
    id: String,
    content: String,
}

impl ReportSection {
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
        }
    }

    fn start_marker(&self) -> String {
        format!("<!-- OUTPUTSLOT:{} start -->", self.id)
    }

    fn end_marker(&self) -> String {
        format!("<!-- OUTPUTSLOT:{} end -->", self.id)
    }
}

pub fn ensure_report_file(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }

    if !path.exists() {
        return Err(anyhow!(
            "expected report template at {} but none was found",
            path.display()
        ));
    }

    Ok(())
}

pub fn update_sections(path: &Path, sections: &[ReportSection]) -> Result<()> {
    let mut content = fs::read_to_string(path)
        .with_context(|| format!("failed to read report at {}", path.display()))?;

    for section in sections {
        content = replace_section(&content, section)?;
    }

    fs::write(path, content)
        .with_context(|| format!("failed to write updated report to {}", path.display()))?;
    Ok(())
}

fn replace_section(content: &str, section: &ReportSection) -> Result<String> {
    let start_marker = section.start_marker();
    let end_marker = section.end_marker();

    let start_idx = content
        .find(&start_marker)
        .ok_or_else(|| anyhow!("missing start marker: {}", start_marker))?;
    let after_start = start_idx + start_marker.len();
    let end_relative = content[after_start..]
        .find(&end_marker)
        .ok_or_else(|| anyhow!("missing end marker: {}", end_marker))?;
    let end_idx = after_start + end_relative;

    let mut updated = String::with_capacity(content.len() + section.content.len());
    updated.push_str(&content[..start_idx]);
    updated.push_str(&start_marker);

    let trimmed = section.content.trim_matches('\n');
    updated.push('\n');
    if !trimmed.is_empty() {
        updated.push_str(trimmed);
        updated.push('\n');
    }

    updated.push_str(&content[end_idx..]);
    Ok(updated)
}
