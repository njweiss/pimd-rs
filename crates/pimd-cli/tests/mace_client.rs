//! The supported MACE socket client against the in-process adapter (MILESTONE_10 "M10a: Required
//! tests" 1, 2 and 5; SPEC K-7, U-2, D-1, D-3, K-4).
//!
//! Each test starts `crates/pimd-torch/scripts/mace_client.py` from the pixi `ml` environment on a
//! committed pickled tiny MACE model (e3nn backend, CPU) and drives it through `SocketPotential`
//! (K = 1 or 2 clients, P = 4 beads); the expectation is `MacePotential` (P-6) on the committed
//! v2 float64 export of the same model, in-process, one libtorch thread. Systems: see
//! `mace_client_common` (a gas-phase cluster and a non-cubic orthorhombic box with contacts only
//! through periodic images along x, y and z).
#![cfg(feature = "torch")]

mod mace_client_common;

use mace_client_common::{
    BIG_E0, BOX, ClientOpts, Session, System, adapter, beads, bige0_export, bige0_model,
    bits_equal, cell, eval_adapter, lock, max_abs, max_diffs, min_image_distance, system1, system2,
    tiny_export, tiny_model, write_config_xyz,
};
use pimd_core::units::{BOHR_IN_ANGSTROM, HARTREE_IN_EV};
use pimd_torch::{Device, Dtype};

const P: usize = 4;
/// Per-component bead displacement amplitude, bohr.
const SPREAD: f64 = 0.2;
/// r_max of the tiny models, bohr (5 Å; gen_tiny_mace.py).
fn r_max() -> f64 {
    5.0 / BOHR_IN_ANGSTROM
}

/// Premise of the box geometry: for each axis there is a pair of atoms of one bead within r_max
/// only through a periodic image along that axis (direct distance ≥ r_max), so dropping the
/// minimum image, or using the wrong length on any axis, removes or changes edges.
fn assert_image_contacts(s: &System) {
    let q = beads(&s.boxed, 1, 0.0, 0);
    for a in 0..3 {
        let found = (0..s.symbols.len()).any(|i| {
            (0..s.symbols.len()).any(|j| {
                let (x, y) = (q.get(0, i), q.get(0, j));
                let d = y[a] - x[a];
                let crosses = (d / BOX[a] + 0.5).floor() != 0.0;
                let direct = (0..3).map(|k| (y[k] - x[k]).powi(2)).sum::<f64>().sqrt();
                i != j && crosses && min_image_distance(x, y, BOX) < r_max() && direct >= r_max()
            })
        });
        assert!(found, "premise: no image contact along axis {a}");
    }
}

/// K-7, U-2, K-4: e3nn backend, float64 (the default), CPU, tiny model, test system 1, P = 4, with
/// K = 1 and K = 2 clients: a gas-phase cluster and the orthorhombic box (both through the same
/// clients: h = 0 → gas phase, h diagonal → box) give the adapter's energies and forces to near
/// round-off.
///
/// Bound (both float64, same weights, same edge set): the client evaluates one bead per graph
/// with its own edge order; the adapter one batched graph of all P beads, edges in (i, j) row
/// order. Only the summation order of the per-edge messages and of the node energies differs, and
/// Σ E0 (≈ 1.6e4 eV here) and the eV → Eh conversion may be grouped differently. Measured with
/// the throwaway reference client (2026-11-13): |ΔE| ≤ 5.7e-14 Eh (1 ulp), |ΔF| ≤ 1.7e-18 Eh/bohr.
/// Tolerance: |ΔE| ≤ 64 ulp(|E|) per bead (|E| ≈ 457 Eh, ulp 5.7e-14 Eh: 3.6e-12 Eh) and
/// |ΔF| ≤ 1e-12 · max|F| + 1e-14 Eh/bohr. A unit-constant change of CODATA 2014 (ase.units: Hartree
/// −8.1e-9 relative, bohr −6.4e-10 relative) moves E by ≈ 5e-6 Eh and F by ≈ 1e-9 relative: far
/// outside.
#[test]
fn e3nn_float64_equals_adapter() {
    let _g = lock();
    let s = system1();
    assert_image_contacts(&s);
    let gas = beads(&s.cluster, P, SPREAD, 11);
    let boxed = beads(&s.boxed, P, SPREAD, 12);
    let c = cell();
    let mut pot = adapter(&tiny_export(), &s.symbols, Device::Cpu, Dtype::F64);
    let want_gas = eval_adapter(&mut pot, &gas, None);
    let want_box = eval_adapter(&mut pot, &boxed, Some(&c));
    drop(pot);
    for k in [1, 2] {
        let dir = tempfile::tempdir().unwrap();
        let mut sess = Session::start(
            &ClientOpts::defaults(tiny_model()),
            &s.symbols,
            k,
            dir.path(),
        );
        for (what, q, cl, want) in [
            ("gas", &gas, None, &want_gas),
            ("box", &boxed, Some(&c), &want_box),
        ] {
            let got = sess.eval(q, cl);
            let (de, df) = max_diffs(&got, want);
            let e_tol = want
                .energies
                .iter()
                .map(|e| 64.0 * ulp(*e))
                .fold(f64::INFINITY, f64::min);
            let f_tol = 1e-12 * max_abs(want) + 1e-14;
            eprintln!(
                "K = {k}, {what}: max|dE| {de:.3e} Eh (tol {e_tol:.1e}), max|dF| {df:.3e} Eh/bohr \
                 (tol {f_tol:.1e}), E[0] {:.10} Eh",
                want.energies[0]
            );
            assert!(de <= e_tol, "K = {k}, {what}: |dE| {de:e} Eh > {e_tol:e}");
            assert!(
                df <= f_tol,
                "K = {k}, {what}: |dF| {df:e} Eh/bohr > {f_tol:e}"
            );
        }
        sess.finish();
    }
}

/// Spacing of f64 values at |x|.
fn ulp(x: f64) -> f64 {
    let x = x.abs();
    f64::from_bits(x.to_bits() + 1) - x
}

/// K-7, D-1: e3nn, `--dtype float32`, CPU, K = 2, the big-E0 tiny model (E0 ≈ ±1e6 eV per atom,
/// pickled `tiny_mace_bige0.model`), test system 2 (H, C, N, O), gas phase and box, P = 4: equal
/// to the adapter at `Dtype::F32` on the v2 export of the same model (`tiny_mace_bige0_v2_f64.pt`),
/// to float32 tolerance; the energy is not offset (Σ E0 in float64).
///
/// Tolerance: both sides run the same float32 network on the same float32 inputs (positions and
/// shifts rounded from the same float64 values) and differ only in summation order (one bead per
/// graph vs P beads batched, edge order), so the interaction energies differ by a few float32 ulp
/// of the node-energy sum; Σ E0 is exact float64 on both sides. Measured with the reference client
/// (2026-11-13): |ΔE| 0 (E_int ≈ −2.6 eV), max|ΔF| 1.2e-9 Eh/bohr (tolerance 3.7e-7).
/// Tolerance per bead |ΔE| ≤ 2^-24 · 2^8 · max(|E_int|, N eV) / H (the M9a v2 float32 bound, Q38:
/// 256 float32 half-ulp of the interaction energy or of N eV) and |ΔF| ≤ 2^-24 · 2^8 · (max|F| +
/// 1 eV/Å). Premise (checked): Σ E0 summed in float32 in atom order, or the table rounded to
/// float32 first, misses the float64 sum by more than 100 times the energy tolerance; dropping E0
/// misses it by ≈ 1e6 eV. And the client really runs float32 (D-1): its forces differ from the
/// float64 adapter's by more than float64 round-off (> 1e-12 · max|F|; float32 rounding gives
/// ≈ 1e-7 relative, a client ignoring `--dtype` ≈ 1e-16).
#[test]
fn e3nn_float32_equals_adapter_without_e0_offset() {
    let _g = lock();
    let s = system2();
    assert_image_contacts(&s);
    let gas = beads(&s.cluster, P, SPREAD, 21);
    let boxed = beads(&s.boxed, P, SPREAD, 22);
    let c = cell();
    let n = s.symbols.len() as f64;
    // Premise: float32 E0 sums vs the float64 sum (eV).
    let e0 = |sym: &str| {
        let z = pimd_torch::atomic_number(sym).unwrap();
        BIG_E0.iter().find(|(zz, _)| *zz == z).unwrap().1
    };
    let sum64: f64 = s.symbols.iter().map(|x| e0(x)).sum();
    let sum32 = s.symbols.iter().fold(0f32, |acc, x| acc + e0(x) as f32);
    let sum_rounded: f64 = s.symbols.iter().map(|x| f64::from(e0(x) as f32)).sum();
    let mut pot = adapter(&bige0_export(), &s.symbols, Device::Cpu, Dtype::F32);
    let want_gas = eval_adapter(&mut pot, &gas, None);
    let want_box = eval_adapter(&mut pot, &boxed, Some(&c));
    drop(pot);
    let mut pot64 = adapter(&bige0_export(), &s.symbols, Device::Cpu, Dtype::F64);
    let f64_gas = eval_adapter(&mut pot64, &gas, None);
    let f64_box = eval_adapter(&mut pot64, &boxed, Some(&c));
    drop(pot64);
    let opts = ClientOpts {
        dtype: Some("float32"),
        ..ClientOpts::defaults(bige0_model())
    };
    let dir = tempfile::tempdir().unwrap();
    let mut sess = Session::start(&opts, &s.symbols, 2, dir.path());
    for (what, q, cl, want, want64) in [
        ("gas", &gas, None, &want_gas, &f64_gas),
        ("box", &boxed, Some(&c), &want_box, &f64_box),
    ] {
        let got = sess.eval(q, cl);
        let (_, df64) = max_diffs(&got, want64);
        eprintln!("{what}: max|dF| vs the float64 adapter {df64:.3e} Eh/bohr");
        assert!(
            df64 > 1e-12 * max_abs(want64),
            "{what}: forces equal the float64 adapter's to {df64:e}: not float32"
        );
        let tol32 = 2f64.powi(-24) * 2f64.powi(8);
        for j in 0..P {
            let e_int = want.energies[j] * HARTREE_IN_EV - sum64;
            let e_tol = tol32 * e_int.abs().max(n) / HARTREE_IN_EV;
            let de = (got.energies[j] - want.energies[j]).abs();
            eprintln!(
                "{what} bead {j}: |dE| {de:.3e} Eh (tol {e_tol:.2e}), E_int {e_int:.6} eV, \
                 float32 E0 sum off by {:.3e} eV, rounded table by {:.3e} eV",
                f64::from(sum32) - sum64,
                sum_rounded - sum64
            );
            assert!(
                (f64::from(sum32) - sum64).abs() / HARTREE_IN_EV > 100.0 * e_tol
                    && (sum_rounded - sum64).abs() / HARTREE_IN_EV > 100.0 * e_tol,
                "premise: a float32 E0 sum must be detectable"
            );
            assert!(de <= e_tol, "{what} bead {j}: |dE| {de:e} Eh > {e_tol:e}");
        }
        let f_scale = max_abs(want) + BOHR_IN_ANGSTROM / HARTREE_IN_EV;
        let f_tol = tol32 * f_scale;
        let (_, df) = max_diffs(&got, want);
        eprintln!("{what}: max|dF| {df:.3e} Eh/bohr (tol {f_tol:.2e})");
        assert!(df <= f_tol, "{what}: |dF| {df:e} Eh/bohr > {f_tol:e}");
    }
    sess.finish();
}

/// K-7, D-3, K-4: statelessness (CPU, float64, tiny model, test system 1, P = 4). With K = 2
/// clients the box beads A, then different (gas-phase) beads B, then A again give bitwise the same
/// energies and forces both times; and a fresh K = 1 client, which evaluates bead j after beads
/// 0 … j − 1 (with K = 2 after beads j − 2, j − 4, …), gives bitwise the same results for A too;
/// that client reads its species from the config's `positions_xyz` file instead of `symbols`
/// (K-7), so this also checks the xyz route.
/// So no result depends on earlier requests or on which client serves which bead (D-3 note: the
/// guarantee holds for deterministic clients that keep no state; a cached edge list, a cached
/// result or TorchScript's profiling executor, which re-optimises after the first calls, break
/// it).
#[test]
fn same_positions_give_bitwise_same_result() {
    let _g = lock();
    let s = system1();
    let a = beads(&s.boxed, P, SPREAD, 51);
    let b = beads(&s.cluster, P, SPREAD, 52);
    let c = cell();
    let dir = tempfile::tempdir().unwrap();
    let mut sess = Session::start(
        &ClientOpts::defaults(tiny_model()),
        &s.symbols,
        2,
        dir.path(),
    );
    let first = sess.eval(&a, Some(&c));
    let other = sess.eval(&b, None);
    let again = sess.eval(&a, Some(&c));
    sess.finish();
    assert!(!bits_equal(&first, &other), "premise: A and B must differ");
    assert!(
        bits_equal(&first, &again),
        "A after B differs from A: {:?}",
        max_diffs(&first, &again)
    );
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("k7.sock");
    let ang: Vec<[f64; 3]> = s
        .boxed
        .iter()
        .map(|x| x.map(|v| v * BOHR_IN_ANGSTROM))
        .collect();
    let config = write_config_xyz(dir.path(), &s.symbols, &ang, &sock, 1);
    let opts = ClientOpts::defaults(tiny_model());
    let mut one = Session::start_with(&opts, &config, &sock, s.symbols.len(), 1, dir.path());
    let single = one.eval(&a, Some(&c));
    one.finish();
    assert!(
        bits_equal(&first, &single),
        "K = 1 differs from K = 2: {:?}",
        max_diffs(&first, &single)
    );
}
