//! `scripts/export_mace.py` writes interface v2 by default (MILESTONE_9 "M9a: In scope" row
//! "export", "Export interface v2"; SPEC P-6 v2 sentence).
//!
//! Each test runs the export script from the pixi `ml` environment (as the M6/M8a long tests do;
//! `CHECK_TORCH=1` runs inside `pixi run -e ml`) on the committed pickled tiny model
//! `data/tiny_mace.model`, into a temporary directory, then reads the export back through `tch`
//! and the adapter. Expected values: the E0 table of the tiny model (`tiny_mace.json`, `e0_ev`:
//! −13.6, −1029, −1485, −2042 eV for Z = 1, 6, 7, 8, as built by gen_tiny_mace.py), the locked v1
//! export `tiny_mace_f64.pt` (same script at v1, M6) and Python MACE (`mace_ref.json`).
//!
//! Brief: `forward` returns the interaction energy; forces unchanged; `atomic_energies()` →
//! float64 `[S]` in the order of `atomic_numbers()`, float64 in both the float64 and float32
//! exports; `interface_version()` = 2; a model without `atomic_energies_fn` is refused with a clear
//! message; `--interface 1` keeps the old output.
#![cfg(feature = "torch")]

mod common;
mod v2_common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use common::{Json, data_dir, eval, export_path, pickled_model_path, system};
use pimd_torch::Dtype;
use tch::{CModule, IValue, Kind, Tensor};
use v2_common::{bits_eq, load_path};

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

/// Runs export_mace.py on `model` with `args`, writing `out`.
fn run_export(model: &Path, out: &Path, args: &[&str]) -> Output {
    ml_python()
        .current_dir(root())
        .env("OMP_NUM_THREADS", "1")
        .arg(root().join("crates/pimd-torch/scripts/export_mace.py"))
        .arg(model)
        .arg("--out")
        .arg(out)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("cannot run export_mace.py (pixi ml env)")
}

fn export_ok(dir: &Path, file: &str, args: &[&str]) -> PathBuf {
    let out = dir.join(file);
    let o = run_export(&pickled_model_path(), &out, args);
    assert!(
        o.status.success(),
        "export_mace.py {args:?} failed: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    out
}

/// The E0 table of the tiny model in the order of its atomic numbers (from tiny_mace.json).
fn tiny_e0() -> (Vec<i64>, Vec<f64>) {
    let j = Json::parse(&std::fs::read_to_string(data_dir().join("tiny_mace.json")).unwrap());
    let z = j
        .get("atomic_numbers")
        .nums()
        .iter()
        .map(|&z| z as i64)
        .collect();
    (z, j.get("e0_ev").nums())
}

/// interface_version() if present, atomic_numbers(), atomic_energies() with its kind if present.
type Methods = (Option<i64>, Vec<i64>, Option<(Kind, Vec<f64>)>);

fn read_methods(path: &Path) -> Methods {
    let m = CModule::load(path).expect("load export");
    let version = match m.method_is::<IValue>("interface_version", &[]) {
        Ok(IValue::Int(v)) => Some(v),
        Ok(other) => panic!("interface_version() returned {other:?}"),
        Err(_) => None,
    };
    let z = Vec::<i64>::try_from(&m.method_ts::<Tensor>("atomic_numbers", &[]).unwrap()).unwrap();
    let e0 = m.method_ts::<Tensor>("atomic_energies", &[]).ok().map(|t| {
        let kind = t.kind();
        assert_eq!(t.dim(), 1, "atomic_energies() is one-dimensional [S]");
        (
            kind,
            Vec::<f64>::try_from(&t.to_kind(Kind::Double)).unwrap(),
        )
    });
    (version, z, e0)
}

/// Export row, P-6: the default export (float64) is v2: `interface_version()` = 2,
/// `atomic_energies()` float64 and equal to the tiny model's E0 table in the order of
/// `atomic_numbers()`; through the adapter (float64, threads 1) its energies equal the locked v1
/// export's to 1e-12 relative and its forces are bitwise equal (water, 4-water box, P = 8), so its
/// `forward` returns the interaction energy (with the total, E0 would count twice).
#[test]
fn default_export_is_v2() {
    let d = tempfile::tempdir().unwrap();
    let out = export_ok(d.path(), "v2.pt", &[]);
    let (version, z, e0) = read_methods(&out);
    assert_eq!(version, Some(2), "interface_version()");
    let (z_ref, e0_ref) = tiny_e0();
    assert_eq!(z, z_ref, "atomic_numbers()");
    let (kind, e0) = e0.expect("atomic_energies() missing");
    assert_eq!(kind, Kind::Double, "atomic_energies() dtype");
    assert!(
        bits_eq(&e0, &e0_ref),
        "atomic_energies() {e0:?} vs {e0_ref:?}"
    );
    for name in ["water", "waterbox"] {
        let s = system(name);
        let v2 = eval(
            &mut load_path(&out, &s.z, Dtype::F64, 1),
            &s.q,
            s.cell.as_ref(),
        );
        let v1 = eval(
            &mut load_path(&export_path(), &s.z, Dtype::F64, 1),
            &s.q,
            s.cell.as_ref(),
        );
        for j in 0..s.p {
            let rel = (v2.energies[j] - v1.energies[j]).abs() / v1.energies[j].abs();
            assert!(rel <= 1e-12, "{name} bead {j}: v2 vs v1 {rel:e} relative");
        }
        assert!(
            bits_eq(v2.forces.as_slice(), v1.forces.as_slice()),
            "{name}: forces differ from the v1 export"
        );
    }
}

/// Export row: `--dtype float32` is v2 as well, and its `atomic_energies()` is still float64 with
/// the exact E0 values; it evaluates with `Dtype::F32` within the M6 float32 tolerances of Python
/// MACE (|ΔE| ≤ 5e-3 eV, max|ΔF| ≤ 1e-4·max|F| per bead; water, P = 8).
#[test]
fn float32_export_keeps_float64_e0() {
    let d = tempfile::tempdir().unwrap();
    let out = export_ok(d.path(), "v2_f32.pt", &["--dtype", "float32"]);
    let (version, _, e0) = read_methods(&out);
    assert_eq!(version, Some(2), "interface_version()");
    let (kind, e0) = e0.expect("atomic_energies() missing");
    assert_eq!(
        kind,
        Kind::Double,
        "atomic_energies() dtype in a float32 export"
    );
    assert!(bits_eq(&e0, &tiny_e0().1), "atomic_energies() {e0:?}");
    let s = system("water");
    let got = eval(&mut load_path(&out, &s.z, Dtype::F32, 1), &s.q, None);
    let (_, ea, fr) = common::errors(&got, &s.energies, &s.forces);
    assert!(
        ea * common::EV_PER_EH <= 5e-3,
        "energy error {} eV",
        ea * common::EV_PER_EH
    );
    assert!(fr <= 1e-4, "force error {fr:e}");
}

/// Export row, decision 2: `--interface 1` writes the M6 export: no `interface_version()`, and
/// through the adapter energies and forces bitwise equal to the locked `tiny_mace_f64.pt` (made by
/// the same script at v1 from the same model file) for water and the 4-water box, float64.
#[test]
fn interface_1_keeps_the_old_output() {
    let d = tempfile::tempdir().unwrap();
    let out = export_ok(d.path(), "v1.pt", &["--interface", "1"]);
    let (version, _, _) = read_methods(&out);
    assert_eq!(version, None, "a v1 export has no interface_version()");
    for name in ["water", "waterbox"] {
        let s = system(name);
        let a = eval(
            &mut load_path(&out, &s.z, Dtype::F64, 1),
            &s.q,
            s.cell.as_ref(),
        );
        let b = eval(
            &mut load_path(&export_path(), &s.z, Dtype::F64, 1),
            &s.q,
            s.cell.as_ref(),
        );
        assert!(
            bits_eq(&a.energies, &b.energies),
            "{name}: energies differ from the M6 export"
        );
        assert!(
            bits_eq(a.forces.as_slice(), b.forces.as_slice()),
            "{name}: forces differ from the M6 export"
        );
    }
}

/// Export row: a model without `atomic_energies_fn` (the tiny model with that submodule deleted)
/// is refused for the default (v2) export: non-zero exit, no export written, and a message on
/// stderr that names `atomic_energies_fn`, without a Python traceback.
#[test]
fn model_without_atomic_energies_fn_is_refused() {
    let d = tempfile::tempdir().unwrap();
    let broken = d.path().join("no_e0.model");
    let st = ml_python()
        .current_dir(root())
        .arg("-c")
        .arg(
            "import sys, mace, torch\n\
             m = torch.load(sys.argv[1], map_location='cpu', weights_only=False)\n\
             del m.atomic_energies_fn\n\
             assert not hasattr(m, 'atomic_energies_fn')\n\
             torch.save(m, sys.argv[2])\n",
        )
        .arg(pickled_model_path())
        .arg(&broken)
        .stdin(Stdio::null())
        .status()
        .expect("python (pixi ml env)");
    assert!(st.success(), "test setup: writing the broken model failed");
    let out = d.path().join("refused.pt");
    let o = run_export(&broken, &out, &[]);
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(
        !o.status.success(),
        "export of a model without atomic_energies_fn succeeded"
    );
    assert!(!out.exists(), "an export was written");
    assert!(
        stderr.contains("atomic_energies_fn"),
        "the message does not name atomic_energies_fn: {stderr}"
    );
    // A refusal, not a crash: an unchecked script also fails here, with an AttributeError
    // traceback that happens to name the attribute.
    assert!(
        !stderr.contains("Traceback"),
        "a crash, not a refusal with a clear message: {stderr}"
    );
}
