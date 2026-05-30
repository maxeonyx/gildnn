#[cfg(test)]
mod tests {
    use std::{fs, path::Path, process::Command};

    use anyhow::{Context, Result};
    use serde::Deserialize;
    use serde_json::Value;

    #[test]
    fn experiments_conform_and_pass_test_mode() {
        run_smoke_checks().expect("experiment smoke tests failed");
    }

    fn run_smoke_checks() -> Result<()> {
        let output = Command::new("cargo")
            .args(["metadata", "--no-deps", "--format-version", "1"])
            .output()
            .context("failed to execute `cargo metadata`")?;

        anyhow::ensure!(
            output.status.success(),
            "`cargo metadata` exited with status {}",
            output.status
        );

        let metadata: Metadata =
            serde_json::from_slice(&output.stdout).context("failed to parse cargo metadata")?;

        let workspace_root = Path::new(&metadata.workspace_root);
        let experiments_root = workspace_root.join("base-experiments");

        for package in metadata.packages.into_iter().filter(|pkg| {
            let manifest_path = Path::new(&pkg.manifest_path);
            manifest_path.starts_with(&experiments_root)
        }) {
            check_package_metadata(&package)?;
            run_package_test_mode(&package)?;
        }

        Ok(())
    }

    fn check_package_metadata(package: &Package) -> Result<()> {
        let description = package
            .description
            .as_ref()
            .map(|value| value.trim())
            .unwrap_or_default();
        anyhow::ensure!(
            !description.is_empty(),
            "package `{}` is missing a manifest description; every experiment must document itself in Cargo.toml",
            package.name
        );

        let has_bin = package
            .targets
            .iter()
            .any(|target| target.kind.iter().any(|kind| kind == "bin"));
        anyhow::ensure!(
            has_bin,
            "package `{}` under base-experiments lacks a binary target; experiments must expose a CLI with `--mode test`",
            package.name
        );

        Ok(())
    }

    fn run_package_test_mode(package: &Package) -> Result<()> {
        let manifest_dir = Path::new(&package.manifest_path)
            .parent()
            .context("package manifest lacks parent directory")?;
        let actual_path = manifest_dir.join("actual.ignore.json");
        let expected_path = manifest_dir.join("expected.json");

        let status = Command::new("cargo")
            .args(["run", "-p", &package.name, "--", "--mode", "test"])
            .status()
            .with_context(|| format!("failed to invoke cargo run for {}", package.name))?;

        anyhow::ensure!(
            status.success(),
            "experiment `{}` failed `cargo run -- --mode test` (status {status}). All experiments must implement the shared CLI interface with `--mode test`.",
            package.name
        );

        anyhow::ensure!(
            actual_path.exists(),
            "no `{}` found for experiment `{}`; every experiment must have an `expected.json` so the harness can compare outputs. Generate it by running that experiment in --mode test --generate-expected",
            expected_path.display(),
            package.name,
        );

        anyhow::ensure!(
            actual_path.exists(),
            "experiment `{}` did not produce `{}`; every experiment must write `actual.ignore.json` in test mode so the harness can compare outputs.",
            package.name,
            actual_path.display()
        );

        anyhow::ensure!(
            expected_path.exists(),
            "experiment `{}` is missing `{}`; add an expected snapshot so the harness can validate regression outputs.",
            package.name,
            expected_path.display()
        );

        let actual = load_json(&actual_path)
            .with_context(|| format!("failed to parse {}", actual_path.display()))?;
        let expected = load_json(&expected_path)
            .with_context(|| format!("failed to parse {}", expected_path.display()))?;

        anyhow::ensure!(
            actual == expected,
            "experiment `{}` produced results that differ from `expected.json`. All experiments must emit deterministic outputs via the shared CLI `--mode test`; inspect `{}` for details.",
            package.name,
            actual_path.display()
        );

        Ok(())
    }

    fn load_json(path: &Path) -> Result<Value> {
        let contents = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let value = serde_json::from_str(&contents)?;
        Ok(value)
    }

    #[derive(Deserialize)]
    struct Metadata {
        workspace_root: String,
        packages: Vec<Package>,
    }

    #[derive(Deserialize)]
    struct Package {
        name: String,
        description: Option<String>,
        manifest_path: String,
        targets: Vec<Target>,
    }

    #[derive(Deserialize)]
    struct Target {
        kind: Vec<String>,
    }
}
