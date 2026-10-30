//! The run driver: initial state, equilibration, production with recording, checkpoint/restart
//! (MILESTONE_4; I-1, T-1, E-1 … E-4, E-7, D-2, D-3).

use std::path::Path;

use crate::config::Config;
use crate::error::CliError;

/// Results of a run (or of the run continued by `run_from_checkpoint`).
#[derive(Debug, Clone, PartialEq)]
pub struct RunSummary {
    /// Production steps completed (the config's `n_steps`).
    pub n_steps: usize,
    /// ⟨E_prim⟩ (E-4) per atom, Eh/atom, over every recorded production step.
    pub mean_e_prim: f64,
    /// ⟨E_vir⟩ (E-4) per atom, Eh/atom, over every recorded production step.
    pub mean_e_vir: f64,
    /// (bin centres in bohr, g) when the config has an RDF (E-7), else `None`.
    pub rdf: Option<(Vec<f64>, Vec<f64>)>,
}

/// Runs `config` from the initial lattice: `n_equil` equilibration steps, then `n_steps`
/// production steps. Writes `observables.csv`, `rdf.csv` (with an RDF), `checkpoint.bin` and
/// `run.log` into `out_dir` (created if missing).
///
/// Errors: `CliError::Io` for file-system errors; errors of the core crates.
pub fn run(config: &Config, out_dir: &Path) -> Result<RunSummary, CliError> {
    let _ = (config, out_dir);
    unimplemented!()
}

/// Continues the run saved in `checkpoint` until production step `config.sampling.n_steps`,
/// writing the outputs of the continued part into `out_dir`. The result is bitwise identical to
/// an uninterrupted `run` (D-3).
///
/// Errors: `CliError::Checkpoint` for a corrupt file or a config whose hash differs from the
/// checkpoint's; otherwise as `run`.
pub fn run_from_checkpoint(
    config: &Config,
    checkpoint: &Path,
    out_dir: &Path,
) -> Result<RunSummary, CliError> {
    let _ = (config, checkpoint, out_dir);
    unimplemented!()
}
