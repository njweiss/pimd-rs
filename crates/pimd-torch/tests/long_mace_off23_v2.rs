//! MACE-OFF23 small through a v2 export against Python MACE (MILESTONE_9 "M9a: Required tests" 6;
//! SPEC P-6 v2 sentence, U-3, D-1). Features `torch` and `long-validation` (CHECK_LONG=1
//! CHECK_TORCH=1).
//!
//! Same system, positions, reference and model check as the M8a test 7 (`long_mace_off23_pbc.rs`):
//! NMA + 216 waters (660 atoms, `nma_w216` of `crates/pimd-cli/tests/data/m8_boxes.json`), cubic
//! L = 18.760371262 Å, P = 4 beads from `common::displaced(.., 4, 0.1, 20261105)`, unwrapped;
//! reference `data/mace_off23_nma.json` (`gen_mace_off23_nma.py`: mace's ASE `MACECalculator`,
//! float64, CPU, periodic, total energies). Needs `PIMD_MACE_MODEL` (MACE-OFF23 small; never
//! committed), else the test prints a message and returns; its FNV-1a must match the reference's.
//! The model is exported into a temporary directory with `scripts/export_mace.py` and its default
//! options (float64), which must now give interface v2 (`interface_version()` = 2, checked here
//! through `tch`).
//!
//! Rows and tolerances (brief):
//! - float64 CPU, chunks of 2: |ΔE| ≤ 1e-12·|E| per bead (forces as in M8a test 7,
//!   max|ΔF| ≤ 1e-10·max|F|);
//! - float32 CUDA, chunks of 2: |ΔE| ≤ 5e-3 eV per bead (the M6 tolerance; v1 measured 0.106 eV,
//!   Q34) and max|ΔF| ≤ 1e-4·max|F|. Skips without CUDA.
//!
//! Measured with the throwaway reference implementation (2026-11-12, RTX 4060, PyTorch 2.13.0,
//! mace-torch 0.3.17, release): float64 CPU c = 2: max|ΔE| 9.9e-10 eV (2.2e-15 relative), max|ΔF|
//! 5.8e-16 max|F|; float32 CUDA c = 2: max|ΔE| 2.68e-4 eV (v1: 0.106 eV), max|ΔF| 1.75e-5 max|F|.
//! About 15 s in release.
//!
//! Unit literals (U-2, CODATA 2018) are written out here.
#![cfg(all(feature = "torch", feature = "long-validation"))]

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use common::{BOHR_A, EV_PER_EH, Json, bead, displaced, max_abs};
use pimd_core::Cell;
use pimd_potentials::{Evaluation, Potential};
use pimd_torch::{Device, Dtype, MacePotential};
use tch::{CModule, IValue};

const P: usize = 4;
const SEED: u64 = 20261105;
const AMP_BOHR: f64 = 0.1;

/// float32 energy tolerance per bead, eV (the M6 value; brief test 6).
const F32_ENERGY_TOL_EV: f64 = 5e-3;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The `ml` environment's Python (direct when present, else through pixi).
fn ml_python() -> Command {
    let py = root().join(".pixi/envs/ml/bin/python");
    if py.exists() {
        Command::new(py)
    } else {
        let mut c = Command::new("pixi");
        c.args(["run", "-e", "ml", "python"]);
        c
    }
}

/// The model file from `PIMD_MACE_MODEL`, or `None` (with a message) when unset.
fn model_file() -> Option<PathBuf> {
    match std::env::var_os("PIMD_MACE_MODEL") {
        Some(p) => {
            let p = PathBuf::from(p);
            assert!(p.is_file(), "PIMD_MACE_MODEL={} is not a file", p.display());
            Some(p)
        }
        None => {
            eprintln!(
                "SKIP MACE-OFF23 v2 validation: set PIMD_MACE_MODEL to the MACE-OFF23 small \
                 model file (e.g. ff_models/mace_off23/MACE-OFF23_small.model)"
            );
            None
        }
    }
}

/// 64-bit FNV-1a (as the generator's).
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Exports `model` with export_mace.py's defaults (float64, interface v2) into `dir`; returns the
/// export's path.
fn export(model: &Path, dir: &Path) -> PathBuf {
    let out = dir.join("export.pt");
    let st = ml_python()
        .current_dir(root())
        .arg(root().join("crates/pimd-torch/scripts/export_mace.py"))
        .arg(model)
        .arg("--out")
        .arg(&out)
        .stdin(Stdio::null())
        .status()
        .expect("cannot run export_mace.py (pixi ml env)");
    assert!(st.success(), "export_mace.py failed");
    out
}

fn read_json(path: &Path) -> Json {
    Json::parse(
        &std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
}

fn string(j: &Json) -> String {
    match j {
        Json::Str(s) => s.clone(),
        other => panic!("expected a string, got {other:?}"),
    }
}

/// Atomic numbers of the box's elements (periodic table; independent of `atomic_number`).
fn z_of(symbol: &str) -> u8 {
    match symbol {
        "H" => 1,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        other => panic!("unexpected element {other}"),
    }
}

/// Worst per-bead errors against the reference: (max |ΔE| eV, max |ΔE|/|E|, max max|ΔF|/max|F|).
fn errors_ev(got: &Evaluation, e_ref_ev: &[f64], f_ref: &[f64], n: usize) -> (f64, f64, f64) {
    assert_eq!(got.energies.len(), P, "one energy per bead");
    assert_eq!(got.forces.n_beads(), P);
    assert_eq!(got.forces.n_atoms(), n);
    let (mut ea, mut er, mut fr) = (0.0f64, 0.0f64, 0.0f64);
    for j in 0..P {
        // Eh → eV and Eh/bohr → eV/Å with the U-2 literals.
        let e_ev = got.energies[j] * EV_PER_EH;
        let de = (e_ev - e_ref_ev[j]).abs();
        ea = ea.max(de);
        er = er.max(de / e_ref_ev[j].abs());
        let fj = &f_ref[j * n * 3..(j + 1) * n * 3];
        let df = max_abs(
            bead(&got.forces, j)
                .iter()
                .zip(fj)
                .map(|(g, r)| g * EV_PER_EH / BOHR_A - r),
        );
        fr = fr.max(df / max_abs(fj.iter().copied()));
    }
    (ea, er, fr)
}

/// P-6 (v2 sentence), U-3, D-1 (MILESTONE_9 test 6): the default export is v2; in-process
/// float64 CPU (chunks of 2) against Python MACE to 1e-12·|E| (energy) and 1e-10·max|F|
/// (forces); float32 CUDA (chunks of 2) to 5e-3 eV per bead and 1e-4·max|F|.
#[test]
fn mace_off23_v2_export_matches_python_mace() {
    let Some(model) = model_file() else {
        return;
    };
    let r = read_json(&common::data_dir().join("mace_off23_nma.json"));
    let want = string(r.at(&["model", "fnv1a64"]));
    let got = format!("{:#018x}", fnv1a64(&std::fs::read(&model).unwrap()));
    assert_eq!(
        got,
        want,
        "PIMD_MACE_MODEL={} is not the model of mace_off23_nma.json ({})",
        model.display(),
        string(r.at(&["model", "file"]))
    );
    assert_eq!(r.get("n_beads").num() as usize, P);
    assert_eq!(r.get("seed").num() as u64, SEED);
    assert_eq!(r.get("amp_bohr").num(), AMP_BOHR);

    let boxes = read_json(&root().join("crates/pimd-cli/tests/data/m8_boxes.json"));
    let b = boxes.at(&["boxes", "nma_w216"]);
    let n = b.get("n_atoms").num() as usize;
    let symbols: Vec<String> = b.get("symbols").arr().iter().map(string).collect();
    let z: Vec<u8> = symbols.iter().map(|s| z_of(s)).collect();
    let flat = b.get("positions_angstrom").arr();
    assert_eq!(flat.len(), n);
    let base: Vec<[f64; 3]> = flat
        .iter()
        .map(|x| {
            let v = x.nums();
            [v[0] / BOHR_A, v[1] / BOHR_A, v[2] / BOHR_A]
        })
        .collect();
    let l = b.get("box_angstrom").num() / BOHR_A;
    let cell = Cell::new([l; 3]).unwrap();
    let q = displaced(&base, P, AMP_BOHR, SEED);
    let e_ref = r.get("energies_ev").nums();
    let f_ref = r.get("forces_ev_per_angstrom").nums();
    assert_eq!(f_ref.len(), P * n * 3);

    let tmp = tempfile::tempdir().unwrap();
    let exported = export(&model, tmp.path());
    match CModule::load(&exported)
        .expect("load export")
        .method_is::<IValue>("interface_version", &[])
    {
        Ok(IValue::Int(2)) => {}
        other => panic!("export_mace.py default must write interface v2, got {other:?}"),
    }

    // float64 CPU, chunks of 2.
    let mut pot = MacePotential::load(&exported, &z, Device::Cpu, Dtype::F64, 4, true)
        .unwrap()
        .with_beads_per_graph(2)
        .unwrap();
    let out = pot.evaluate(&q, Some(&cell), None).unwrap();
    drop(pot);
    let (ea, er, fr) = errors_ev(&out, &e_ref, &f_ref, n);
    eprintln!("f64 cpu c = 2: |dE| {ea:.2e} eV ({er:.2e} rel), |dF|/max|F| {fr:.2e}");
    assert!(er <= 1e-12, "f64 CPU: energy {er:e} relative");
    assert!(fr <= 1e-10, "f64 CPU: forces {fr:e}");

    // float32 CUDA, chunks of 2.
    if !common::cuda_available() {
        eprintln!("SKIP float32 CUDA row: libtorch sees no CUDA device");
        return;
    }
    let mut pot = MacePotential::load(&exported, &z, Device::Cuda(0), Dtype::F32, 1, true)
        .unwrap()
        .with_beads_per_graph(2)
        .unwrap();
    let out = pot.evaluate(&q, Some(&cell), None).unwrap();
    drop(pot);
    let (ea, er, fr) = errors_ev(&out, &e_ref, &f_ref, n);
    eprintln!("f32 cuda c = 2: |dE| {ea:.2e} eV ({er:.2e} rel), |dF|/max|F| {fr:.2e}");
    assert!(ea <= F32_ENERGY_TOL_EV, "f32 CUDA c = 2: energy {ea:e} eV");
    assert!(fr <= 1e-4, "f32 CUDA c = 2: forces {fr:e}");
}
