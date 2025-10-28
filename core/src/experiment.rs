use std::env;

use anyhow::{anyhow, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExperimentMode {
    Full,
    Test,
}

impl ExperimentMode {
    pub fn from_str(value: &str) -> Result<Self> {
        match value {
            "full" => Ok(Self::Full),
            "test" => Ok(Self::Test),
            other => Err(anyhow!("invalid mode: {}", other)),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Test => "test",
        }
    }

    pub fn select<T>(&self, full: T, test: T) -> T {
        match self {
            Self::Full => full,
            Self::Test => test,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExperimentModeArgs {
    mode: ExperimentMode,
    help_requested: bool,
}

impl ExperimentModeArgs {
    pub fn parse_from_env() -> Result<Self> {
        Self::parse(env::args().skip(1))
    }

    pub fn parse<I>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = String>,
    {
        let mut mode: Option<ExperimentMode> = None;
        let mut help_requested = false;
        let mut iter = args.into_iter();

        while let Some(arg) = iter.next() {
            if arg == "--mode" || arg == "-m" {
                let value = iter
                    .next()
                    .ok_or_else(|| anyhow!("expected value after {}", arg))?;
                mode = Some(ExperimentMode::from_str(&value)?);
            } else if arg == "--help" || arg == "-h" {
                help_requested = true;
            } else if let Some(mode_value) = arg.strip_prefix("--mode=") {
                mode = Some(ExperimentMode::from_str(mode_value)?);
            } else {
                return Err(anyhow!("unexpected argument: {}", arg));
            }
        }

        Ok(Self {
            mode: mode.unwrap_or(ExperimentMode::Full),
            help_requested,
        })
    }

    pub fn help_requested(&self) -> bool {
        self.help_requested
    }

    pub fn mode(&self) -> ExperimentMode {
        self.mode
    }
}
