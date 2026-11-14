//! The supported MACE socket client on the production system (MILESTONE_10 "M10a: Required
//! tests" 7; SPEC K-7, D-1, D-3, K-4). Features `torch` and `long-validation` (CHECK_TORCH=1
//! CHECK_LONG=1, release). Needs CUDA and `ff_models/mace_off23/MACE-OFF23_small.model` (never
//! committed; README); skips with a message without either. The cueq part runs only with
//! `CHECK_CUEQ=1` and `CUEQ_PYTHON` (as `mace_client_cueq.rs`).
//!
//! System: the M8b start box, NMA + 216 waters (660 atoms, L = 18.760371262 Å), box `nma_w216` of
//! the committed `data/m8_boxes.json` (gen_m8_boxes.py; the study's
//! studies/m8-nma-water/out/start/nma_w216.xyz is written from it). P = 4: bead 0 the box, beads
//! 1–3 displaced by a uniform ±0.1 bohr per component (SplitMix64, seeded). The clients read the
//! species from a config whose `positions_xyz` is that box (as the M8b configs).
//!
//! Expectation: the in-process adapter on the v2 float64 export of the same model file
//! (export_mace.py, into a temporary directory), at `Dtype::F32` on CUDA, deterministic, one graph
//! per bead (`beads_per_graph = 1`, as the clients evaluate). Clients: K = 2, `--dtype float32
//! --device cuda`, e3nn; then cueq when enabled. At most two clients are alive (host RAM).
#![cfg(all(feature = "torch", feature = "long-validation"))]

#[path = "../../pimd-potentials/tests/common/lammps.rs"]
mod lammps;
mod mace_client_common;

use std::path::{Path, PathBuf};
use std::process::Stdio;

use lammps::Json;
use mace_client_common::{
    ClientOpts, Session, beads, lock, max_abs, max_diffs, ml_python, root, write_config_xyz,
};
use pimd_core::Cell;
use pimd_core::units::{BOHR_IN_ANGSTROM, HARTREE_IN_EV};
use pimd_potentials::{Evaluation, Potential};
use pimd_torch::{Device, Dtype, MacePotential};

const P: usize = 4;

fn model_file() -> Option<PathBuf> {
    let m = root().join("ff_models/mace_off23/MACE-OFF23_small.model");
    if !m.is_file() {
        eprintln!(
            "SKIP MACE client long test: {} is absent (README: download MACE-OFF23 small)",
            m.display()
        );
        return None;
    }
    Some(m)
}

fn string(j: &Json) -> String {
    match j {
        Json::Str(s) => s.clone(),
        other => panic!("expected a string, got {other:?}"),
    }
}

/// (symbols, positions Å, L Å) of `nma_w216`.
fn nma_box() -> (Vec<String>, Vec<[f64; 3]>, f64) {
    let path = root().join("crates/pimd-cli/tests/data/m8_boxes.json");
    let j = Json::parse(&std::fs::read_to_string(&path).unwrap());
    let b = j.at(&["boxes", "nma_w216"]);
    let symbols: Vec<String> = b.get("symbols").arr().iter().map(string).collect();
    let pos = b
        .get("positions_angstrom")
        .arr()
        .iter()
        .map(|x| {
            let v = x.nums();
            [v[0], v[1], v[2]]
        })
        .collect();
    (symbols, pos, b.get("box_angstrom").num())
}

/// The v2 float64 export of `model` in `dir`.
fn export(model: &Path, dir: &Path) -> PathBuf {
    let out = dir.join("off23_small_v2_f64.pt");
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

/// E0 of MACE-OFF23 small (eV; its `atomic_energies_fn`, read from the model file with the `ml`
/// env, 2026-11-13) for the elements of the box; only used to scale the energy tolerance.
fn e0_off23(symbol: &str) -> f64 {
    match symbol {
        "H" => -13.571964772646918,
        "C" => -1030.5671648271828,
        "N" => -1486.3750255780376,
        "O" => -2043.933693071156,
        other => panic!("unexpected element {other}"),
    }
}

/// Per bead |ΔE| (Eh) and max |ΔF| (Eh/bohr) within float32 tolerances.
///
/// Both sides are float32 with Σ E0 in float64 (K-7, P-6 v2); they differ in kernels and in the
/// summation order of messages and of the 660 node energies, so the energies differ by a few
/// float32 ulp of the interaction energy E_int = E − Σ E0 (≈ −2281 eV here: ulp 2.4e-4 eV) and
/// the forces by a few ulp of max|F|. Tolerance: the M9a v2 float32 bound (Q38), per bead
/// |ΔE| ≤ 2^-24 · 2^8 · max(|E_int|, N eV) (≈ 0.035 eV = 1.3e-3 Eh) and |ΔF| ≤ 2^-24 · 2^8 ·
/// (max|F| + 1 eV/Å) (≈ 1.5e-4 eV/Å at max|F| = 9.0 eV/Å).
///
/// Measured with the reference client (2026-11-13, RTX 4060, beads as below): client vs adapter
/// max|ΔE| 2.4e-4 – 9.8e-4 eV (e3nn), 7.3e-4 – 1.5e-3 eV (cueq) over two runs (CUDA is not
/// bitwise repeatable, D-3), max|ΔF| 3.0e-6 – 3.8e-6 eV/Å; against the float64 adapter: adapter
/// float32 3.3e-4 eV, e3nn 1.2e-3 eV, cueq 1.1e-3 eV, forces 1.3e-4 eV/Å (spike Q3 (ii): 1.9e-4
/// eV and 8.7e-5 eV/Å at P = 1). A float32 E0 sum (Σ E0 ≈ −4.5e5 eV, float32 ulp 0.03 eV) or an
/// E0 table rounded to float32 is the large-E0 test's job (`mace_client.rs`).
fn check(what: &str, symbols: &[&str], got: &Evaluation, want: &Evaluation) {
    let e0_sum: f64 = symbols.iter().map(|s| e0_off23(s)).sum();
    let n = symbols.len() as f64;
    let tol32 = 2f64.powi(-24) * 2f64.powi(8);
    for (j, (g, w)) in got.energies.iter().zip(&want.energies).enumerate() {
        let e_int = w * HARTREE_IN_EV - e0_sum;
        let e_tol = tol32 * e_int.abs().max(n) / HARTREE_IN_EV;
        let de = (g - w).abs();
        eprintln!(
            "{what} bead {j}: |dE| {:.3e} eV (tol {:.3e}), E_int {e_int:.4} eV",
            de * HARTREE_IN_EV,
            e_tol * HARTREE_IN_EV
        );
        assert!(de <= e_tol, "{what} bead {j}: |dE| {de:e} Eh > {e_tol:e}");
    }
    let (_, df) = max_diffs(got, want);
    let f_tol = tol32 * (max_abs(want) + BOHR_IN_ANGSTROM / HARTREE_IN_EV);
    eprintln!(
        "{what}: max|dF| {:.3e} eV/Å (tol {:.3e})",
        df * HARTREE_IN_EV / BOHR_IN_ANGSTROM,
        f_tol * HARTREE_IN_EV / BOHR_IN_ANGSTROM
    );
    assert!(df <= f_tol, "{what}: |dF| {df:e} Eh/bohr > {f_tol:e}");
}

/// K-7, D-1, K-4: MACE-OFF23 small on the M8b start box, P = 4, K = 2 clients, e3nn float32 CUDA
/// (and cueq float32 CUDA with CHECK_CUEQ=1), equal to the in-process float32 adapter on CUDA within
/// the float32 tolerances of `check`.
#[test]
fn mace_off23_small_m8b_box_clients_equal_adapter() {
    let Some(model) = model_file() else { return };
    let cuda = MacePotential::load(
        &mace_client_common::tiny_export(),
        &[8, 1, 1],
        Device::Cuda(0),
        Dtype::F32,
        1,
        true,
    );
    if cuda.is_err() {
        eprintln!("SKIP MACE client long test: libtorch sees no CUDA device");
        return;
    }
    let _g = lock();
    let (symbols, pos, l) = nma_box();
    assert_eq!(symbols.len(), 660);
    assert_eq!(l, 18.760371262, "M8b box length (configs/pilot/*.toml)");
    let sym: Vec<&str> = symbols.iter().map(String::as_str).collect();
    let base: Vec<[f64; 3]> = pos
        .iter()
        .map(|x| x.map(|v| v / BOHR_IN_ANGSTROM))
        .collect();
    let q = beads(&base, P, 0.1, 70);
    let lb = l / BOHR_IN_ANGSTROM;
    let cell = Cell::new([lb, lb, lb]).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let export = export(&model, dir.path());
    let z: Vec<u8> = sym
        .iter()
        .map(|s| pimd_torch::atomic_number(s).unwrap())
        .collect();
    let mut pot = MacePotential::load(&export, &z, Device::Cuda(0), Dtype::F32, 1, true)
        .unwrap()
        .with_beads_per_graph(1)
        .unwrap();
    let want = pot.evaluate(&q, Some(&cell), None).unwrap();
    drop(pot);
    let mut runs = vec![("e3nn", None)];
    if std::env::var("CHECK_CUEQ").as_deref() == Ok("1") {
        match std::env::var_os("CUEQ_PYTHON") {
            Some(py) => runs.push(("cueq", Some(PathBuf::from(py)))),
            None => eprintln!("SKIP cueq part: CHECK_CUEQ=1 but CUEQ_PYTHON is not set"),
        }
    } else {
        eprintln!("SKIP cueq part: set CHECK_CUEQ=1 and CUEQ_PYTHON");
    }
    for (backend, py) in runs {
        let mut opts = ClientOpts {
            backend: Some(backend),
            dtype: Some("float32"),
            device: Some("cuda"),
            ..ClientOpts::defaults(model.clone())
        };
        if let Some(py) = py {
            opts.python = py;
        }
        let run = tempfile::tempdir().unwrap();
        let sock = run.path().join("k7.sock");
        let config = write_config_xyz(run.path(), &sym, &pos, &sock, 2);
        let mut sess = Session::start_with(&opts, &config, &sock, sym.len(), 2, run.path());
        let got = sess.eval(&q, Some(&cell));
        sess.finish();
        check(backend, &sym, &got, &want);
    }
}
