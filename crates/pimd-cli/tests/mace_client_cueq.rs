//! The supported MACE socket client with cuEquivariance kernels (MILESTONE_10 "M10a: Required
//! tests" 6; SPEC K-7, D-1, D-3).
//!
//! Runs only with `CHECK_CUEQ=1` and `CUEQ_PYTHON` naming the Python of a venv with
//! cuequivariance installed out of the repository (README; `CHECK_CUEQ=1 ./scripts/check.sh`
//! sets it to `.venv-cueq/bin/python` unless given); otherwise it prints SKIP and passes. Needs
//! CUDA. The client runs with `--backend cueq --device cuda` on `tiny_mace_cueq.model` (the committed
//! tiny model with its `use_reduced_cg` flag corrected so mace's converter accepts it; Q41; same
//! weights as `tiny_mace_v2_f64.pt`) (test system 1, gas phase and box, P = 4, K = 1) inside the activated `ml` environment (check.sh
//! runs cargo under `pixi run -e ml`; Triton needs that environment's `ptxas`).
//!
//! Expectations: the in-process adapter on the v2 export of the same model, float64 on the CPU
//! and float32 on CUDA (one graph of P beads, deterministic). D-3: cueq on CUDA is not bitwise
//! reproducible, so only tolerances are claimed.
#![cfg(feature = "torch")]

mod mace_client_common;

use std::path::PathBuf;

use mace_client_common::{
    ClientOpts, Session, adapter, beads, cell, cueq_model, eval_adapter, lock, max_abs, max_diffs,
    system1, tiny_export,
};
use pimd_torch::{Device, Dtype};

const P: usize = 4;

/// The cueq venv's Python, or `None` (with a message) unless CHECK_CUEQ=1 and CUEQ_PYTHON is set.
fn cueq_python() -> Option<PathBuf> {
    if std::env::var("CHECK_CUEQ").as_deref() != Ok("1") {
        eprintln!("SKIP cueq client tests: set CHECK_CUEQ=1 and CUEQ_PYTHON (README)");
        return None;
    }
    let Some(py) = std::env::var_os("CUEQ_PYTHON").map(PathBuf::from) else {
        eprintln!("SKIP cueq client tests: CHECK_CUEQ=1 but CUEQ_PYTHON is not set");
        return None;
    };
    assert!(py.exists(), "CUEQ_PYTHON={} does not exist", py.display());
    Some(py)
}

/// K-7, D-3: `--backend cueq --dtype float64 --device cuda` equals the adapter at float64 (CPU)
/// to near round-off, gas phase and box.
///
/// Bound: same model and edges in float64; cueq's fused kernels sum the tensor-product and
/// contraction terms in another order than e3nn (spike Q3 (ii): cueq float64 forces equal e3nn's
/// to 1e-15 eV/Å on MACE-OFF23 small), and CUDA atomics reorder the per-node sums. Measured with
/// the reference client (2026-11-13, RTX 4060): max|ΔE| 5.7e-14 Eh (1 ulp), max|ΔF| 2.0e-17 Eh/bohr. Tolerance per bead |ΔE| ≤ 1e-11
/// Eh (|E| ≈ 457 Eh: 2e-14 relative, ≈ 180 ulp) and |ΔF| ≤ 1e-11 · max|F| + 1e-14 Eh/bohr.
#[test]
fn cueq_float64_cuda_equals_adapter_float64() {
    let Some(py) = cueq_python() else { return };
    let _g = lock();
    let s = system1();
    let c = cell();
    let gas = beads(&s.cluster, P, 0.2, 61);
    let boxed = beads(&s.boxed, P, 0.2, 62);
    let mut pot = adapter(&tiny_export(), &s.symbols, Device::Cpu, Dtype::F64);
    let want = [
        eval_adapter(&mut pot, &gas, None),
        eval_adapter(&mut pot, &boxed, Some(&c)),
    ];
    drop(pot);
    let opts = ClientOpts {
        python: py,
        model: cueq_model(),
        backend: Some("cueq"),
        dtype: Some("float64"),
        device: Some("cuda"),
    };
    let dir = tempfile::tempdir().unwrap();
    let mut sess = Session::start(&opts, &s.symbols, 1, dir.path());
    let got = [sess.eval(&gas, None), sess.eval(&boxed, Some(&c))];
    sess.finish();
    for (what, g, w) in [("gas", &got[0], &want[0]), ("box", &got[1], &want[1])] {
        let (de, df) = max_diffs(g, w);
        let f_tol = 1e-11 * max_abs(w) + 1e-14;
        eprintln!(
            "cueq f64 {what}: max|dE| {de:.3e} Eh, max|dF| {df:.3e} Eh/bohr (tol {f_tol:.1e})"
        );
        assert!(de <= 1e-11, "{what}: |dE| {de:e} Eh");
        assert!(df <= f_tol, "{what}: |dF| {df:e} Eh/bohr > {f_tol:e}");
    }
}

/// K-7, D-1: `--backend cueq --dtype float32 --device cuda` equals the adapter at float32 on CUDA
/// to float32 tolerance, gas phase and box.
///
/// Tolerance as the float32 test of `mace_client.rs` (M9a bound, Q38): per bead |ΔE| ≤ 2^-16 ·
/// max(|E_int|, N eV) and |ΔF| ≤ 2^-16 · (max|F| + 1 eV/Å), with E_int = E − Σ E0 (tiny E0 table
/// −13.6, −1029, −1485, −2042 eV for H, C, N, O; gen_tiny_mace.py). Both sides round the same
/// float64 inputs to float32 and differ in kernels and summation order only. Measured with the
/// reference client (2026-11-13, RTX 4060): max|ΔE| 1.8e-8 Eh (tolerance 1.1e-5),
/// max|ΔF| 5.5e-9 Eh/bohr (tolerance 4.6e-7).
#[test]
fn cueq_float32_cuda_equals_adapter_float32_cuda() {
    let Some(py) = cueq_python() else { return };
    let _g = lock();
    let s = system1();
    let c = cell();
    let gas = beads(&s.cluster, P, 0.2, 63);
    let boxed = beads(&s.boxed, P, 0.2, 64);
    let mut pot = adapter(&tiny_export(), &s.symbols, Device::Cuda(0), Dtype::F32);
    let want = [
        eval_adapter(&mut pot, &gas, None),
        eval_adapter(&mut pot, &boxed, Some(&c)),
    ];
    drop(pot);
    let opts = ClientOpts {
        python: py,
        model: cueq_model(),
        backend: Some("cueq"),
        dtype: Some("float32"),
        device: Some("cuda"),
    };
    let dir = tempfile::tempdir().unwrap();
    let mut sess = Session::start(&opts, &s.symbols, 1, dir.path());
    let got = [sess.eval(&gas, None), sess.eval(&boxed, Some(&c))];
    sess.finish();
    let e0 = |sym: &str| match sym {
        "H" => -13.6,
        "C" => -1029.0,
        "N" => -1485.0,
        "O" => -2042.0,
        _ => unreachable!(),
    };
    let e0_sum: f64 = s.symbols.iter().map(|x| e0(x)).sum();
    let n = s.symbols.len() as f64;
    let ev = pimd_core::units::HARTREE_IN_EV;
    let tol32 = 2f64.powi(-16);
    for (what, g, w) in [("gas", &got[0], &want[0]), ("box", &got[1], &want[1])] {
        for j in 0..P {
            let e_int = w.energies[j] * ev - e0_sum;
            let e_tol = tol32 * e_int.abs().max(n) / ev;
            let de = (g.energies[j] - w.energies[j]).abs();
            eprintln!("cueq f32 {what} bead {j}: |dE| {de:.3e} Eh (tol {e_tol:.2e})");
            assert!(de <= e_tol, "{what} bead {j}: |dE| {de:e} Eh > {e_tol:e}");
        }
        let f_tol = tol32 * (max_abs(w) + pimd_core::units::BOHR_IN_ANGSTROM / ev);
        let (_, df) = max_diffs(g, w);
        eprintln!("cueq f32 {what}: max|dF| {df:.3e} Eh/bohr (tol {f_tol:.2e})");
        assert!(df <= f_tol, "{what}: |dF| {df:e} Eh/bohr > {f_tol:e}");
    }
}
