//! Shared by the `kind = "mace"` tests of `pimd-cli` (MILESTONE_6 "Config"): the brief's config
//! example (gas-phase water with symbols) on the committed tiny test model of `pimd-torch`.
#![allow(dead_code)]

use std::path::PathBuf;

/// The committed float64 TorchScript export of the tiny MACE test model
/// (`crates/pimd-torch/tests/data/gen_tiny_mace.py`; elements H, C, N, O; r_max 5 Å).
pub fn tiny_export() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../pimd-torch/tests/data/tiny_mace_f64.pt")
}

/// Gas-phase water (M5 start geometry, Å) with `kind = "mace"` on `model` and only the required
/// potential keys (every optional key at its default). P = 4, 300 K, 0.5 fs, PILE-L τ_0 = 20 fs,
/// `n_steps` production steps, stride 2, checkpoint every 10 steps.
pub fn mace_water_toml(model: &std::path::Path, n_steps: usize) -> String {
    format!(
        r#"
[system]
n_atoms = 3
masses_amu = [15.999, 1.008, 1.008]
symbols = ["O", "H", "H"]
positions_angstrom = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]

[potential]
kind = "mace"
model = "{}"

[sampling]
temperature_kelvin = 300.0
n_beads = 4
dt_fs = 0.5
tau0_fs = 20.0
n_equil = 4
n_steps = {n_steps}
stride = 2
seed = 4242

[output]
checkpoint_every = 10
trajectory_every = 5
"#,
        model.display()
    )
}
