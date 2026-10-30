//! `pimd-rs <config.toml> --out <dir> [--restart <checkpoint>]` (MILESTONE_4).

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use pimd_cli::{CliError, Config, run, run_from_checkpoint};

/// Path-integral MD run driver.
#[derive(Parser)]
#[command(name = "pimd-rs", version)]
struct Args {
    /// TOML config file.
    config: PathBuf,
    /// Output directory.
    #[arg(long)]
    out: PathBuf,
    /// Continue from this checkpoint file.
    #[arg(long)]
    restart: Option<PathBuf>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let result = (|| -> Result<(), CliError> {
        let text = std::fs::read_to_string(&args.config)?;
        let config = Config::from_toml_str(&text)?;
        match &args.restart {
            Some(ckpt) => run_from_checkpoint(&config, ckpt, &args.out)?,
            None => run(&config, &args.out)?,
        };
        Ok(())
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
