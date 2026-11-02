//! Cases added after the M6b review (MILESTONE_6 "After review": N5, N6; SPEC P-6).
//!
//! - P-6: r_max < L_min/2 is strict, so a box with L_min = 2·r_max exactly is rejected, and the
//!   next representable length above it is accepted.
//! - A CUDA device index at or above the device count is `Backend` (skips without CUDA).
//! - Q27: a float32 export (`export_mace.py --dtype float32`) is refused for `Dtype::F64`
//!   (`Backend`; it would run with float32-rounded weights), and accepted for `Dtype::F32`.
#![cfg(feature = "torch")]

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use common::{cuda_available, eval, load, pickled_model_path, system};
use pimd_core::Cell;
use pimd_potentials::{Potential, PotentialError};
use pimd_torch::{Device, Dtype, MacePotential};

/// P-6: L_min = 2·r_max exactly (all axes, and one axis) → `InvalidParameter("cutoff", r_max)`;
/// the next f64 above 2·r_max evaluates.
#[test]
fn cutoff_exactly_half_box_is_rejected() {
    let s = system("water_single");
    let mut pot = load(&s.z, Device::Cpu, Dtype::F64, 1);
    let r = pot.cutoff();
    let l = 2.0 * r;
    for lengths in [[l; 3], [4.0 * r, 4.0 * r, l]] {
        let cell = Cell::new(lengths).expect("cell");
        match pot.evaluate(&s.q, Some(&cell), None) {
            Err(PotentialError::InvalidParameter { what, value }) => {
                assert_eq!(what, "cutoff");
                assert_eq!(value, r, "value must be r_max in bohr");
            }
            other => panic!("L_min = 2 r_max: expected InvalidParameter(cutoff), got {other:?}"),
        }
    }
    let ok = Cell::new([l.next_up(); 3]).expect("cell");
    pot.evaluate(&s.q, Some(&ok), None)
        .expect("L_min just above 2 r_max must evaluate");
}

/// A CUDA device index equal to the device count → `Backend`.
#[test]
fn cuda_index_out_of_range_is_backend_error() {
    if !cuda_available() {
        eprintln!("SKIP cuda_index_out_of_range_is_backend_error: libtorch sees no CUDA device");
        return;
    }
    let n = usize::try_from(tch::Cuda::device_count()).expect("device count");
    match MacePotential::load(
        &common::export_path(),
        &[8, 1, 1],
        Device::Cuda(n),
        Dtype::F64,
        1,
        true,
    ) {
        Err(PotentialError::Backend(_)) => {}
        other => panic!(
            "cuda:{n}: expected Backend, got {:?}",
            other.map(|_| "a potential")
        ),
    }
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The `ml` environment's Python (direct when present, else through pixi), as long_mace_off23.rs.
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

/// Q27: a float32 export of the committed tiny model is `Backend` for `Dtype::F64` and loads and
/// evaluates for `Dtype::F32`.
#[test]
fn float32_export_refused_for_float64() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("tiny_f32.pt");
    let st = ml_python()
        .current_dir(root())
        .arg(root().join("crates/pimd-torch/scripts/export_mace.py"))
        .arg(pickled_model_path())
        .args(["--dtype", "float32", "--out"])
        .arg(&out)
        .stdin(Stdio::null())
        .status()
        .expect("run export_mace.py");
    assert!(st.success(), "export_mace.py --dtype float32 failed");

    let s = system("water");
    match MacePotential::load(&out, &s.z, Device::Cpu, Dtype::F64, 1, true) {
        Err(PotentialError::Backend(_)) => {}
        other => panic!(
            "float32 export, Dtype::F64: expected Backend, got {:?}",
            other.map(|_| "a potential")
        ),
    }
    let mut pot = MacePotential::load(&out, &s.z, Device::Cpu, Dtype::F32, 1, true)
        .expect("float32 export, Dtype::F32");
    let e = eval(&mut pot, &s.q, None);
    assert!(e.energies.iter().all(|v| v.is_finite()));
}
