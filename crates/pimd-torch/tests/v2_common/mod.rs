//! Test-only helpers for the M9a tests (MILESTONE_9 "M9a: float64 reference energies"; SPEC P-6,
//! v2 sentence), used next to `common` (the M6 helpers, locked).
//!
//! Test exports (all from `data/gen_tiny_mace_v2.py`, which reads the committed
//! `data/tiny_mace.model` and writes its own v1 and v2 wrappers, independent of
//! `scripts/export_mace.py`):
//! - `tiny_mace_v2_f64.pt`: the v2 export of the tiny model (the locked v1 export is
//!   `tiny_mace_f64.pt`, `common::export_path`);
//! - `tiny_mace_bige0_{v1,v2}_f64.pt`: v1 and v2 exports of the tiny model with its E0 table
//!   replaced by `big_e0_ev` of `data/tiny_mace_v2.json` (about ±1e6 eV, mixed signs);
//! - `stub_v2_*.pt`: TorchScript stubs (no MACE) with the export interface, whose `forward`
//!   returns the interaction energy `stub.interaction_energy_ev` (0.375 eV) for every graph and
//!   zero forces; `stub_v2_ok` has a valid E0 table, the others one defect each.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use crate::common::{Json, data_dir};
use pimd_potentials::PotentialError;
use pimd_torch::{Device, Dtype, MacePotential};

pub fn v2_export() -> PathBuf {
    data_dir().join("tiny_mace_v2_f64.pt")
}

pub fn bige0_v1() -> PathBuf {
    data_dir().join("tiny_mace_bige0_v1_f64.pt")
}

pub fn bige0_v2() -> PathBuf {
    data_dir().join("tiny_mace_bige0_v2_f64.pt")
}

/// `name` = "ok", "len", "nan", "inf", "ver3", "noe0".
pub fn stub(name: &str) -> PathBuf {
    data_dir().join(format!("stub_v2_{name}.pt"))
}

/// The parsed `data/tiny_mace_v2.json`.
pub fn v2_json() -> Json {
    let text =
        std::fs::read_to_string(data_dir().join("tiny_mace_v2.json")).expect("tiny_mace_v2.json");
    Json::parse(&text)
}

/// `MacePotential::load` with deterministic = true.
pub fn try_load(
    path: &Path,
    z: &[u8],
    dtype: Dtype,
    threads: usize,
) -> Result<MacePotential, PotentialError> {
    MacePotential::load(path, z, Device::Cpu, dtype, threads, true)
}

/// `try_load` on CPU, panicking on error.
pub fn load_path(path: &Path, z: &[u8], dtype: Dtype, threads: usize) -> MacePotential {
    try_load(path, z, dtype, threads)
        .unwrap_or_else(|e| panic!("load {} failed: {e:?}", path.display()))
}

/// 64-bit FNV-1a of the little-endian bytes of the f64 bit patterns of `x`.
pub fn fnv1a64_bits(x: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in x {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Whether two f64 slices are bitwise equal.
pub fn bits_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
