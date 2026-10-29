//! Tests for `kinetic_classical` and `pressure_classical` (SPEC E-8, P-5): hand-computed examples,
//! the ideal-gas limit and LAMMPS `compute pressure` on the LJ snapshot.

#[path = "../../pimd-potentials/tests/common/lammps.rs"]
mod lammps;

use std::panic::{AssertUnwindSafe, catch_unwind};

use lammps::{Snapshot, snapshot_json};
use pimd_core::{Beads, Cell};
use pimd_estimators::{EstimatorError, kinetic_classical, pressure_classical};
use pimd_potentials::{LennardJones, LjParams, Potential};

const ZERO: [[f64; 3]; 3] = [[0.0; 3]; 3];

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1e-300)
}

/// Masses (2, 3) m_e; momenta p0 = (1, 0, −2), p1 = (0, 3, 0).
/// K = |p0|²/(2·2) + |p1|²/(2·3) = 5/4 + 9/6 = 2.75 Eh.
fn two_body() -> (Vec<f64>, Beads) {
    let p = Beads::from_vec(1, 2, vec![1.0, 0.0, -2.0, 0.0, 3.0, 0.0]).unwrap();
    (vec![2.0, 3.0], p)
}

/// E-8: K = Σ |p_i|²/(2 m_i) = 2.75 for the two-body example (exact in binary).
#[test]
fn kinetic_hand_computed() {
    let (m, p) = two_body();
    assert_eq!(kinetic_classical(&m, &p), 2.75);
}

/// Runs `f` and returns true if it panicked with something other than a stub's
/// `unimplemented!` ("not implemented"), so the test fails until the estimator exists (same helper
/// as `energy.rs`).
fn panics_for_real(f: impl FnOnce()) -> bool {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(()) => false,
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            !msg.contains("not implemented")
        }
    }
}

/// E-8: `kinetic_classical` is an energy estimator returning `f64`; masses whose count differs
/// from N are a programming error and panic (SPEC §8, Q4), for fewer and for more masses.
#[test]
fn kinetic_panics_on_mass_mismatch() {
    let (_, p) = two_body();
    assert!(panics_for_real(|| {
        kinetic_classical(&[1.0], &p);
    }));
    assert!(panics_for_real(|| {
        kinetic_classical(&[1.0, 2.0, 3.0], &p);
    }));
}

/// E-8, P-5: hand-computed pressure p = (2K + tr W)/(3V) in the cell (10, 10, 12), V = 1200, with
/// the two-body momenta (K = 2.75). LJ with ε = σ = 1, r_c = 2.5, two atoms a distance r apart
/// along x: W = r F(r) e_x ⊗ e_x with F(r) = (24/r)[2 r⁻¹² − r⁻⁶] (E-8: r_01 = −r e_x, f_01 =
/// −F e_x).
///   r = 1:  F = 24, tr W = 24:            p = (5.5 + 24)/3600 = 29.5/3600;
///   r = 2:  F = 12(2/4096 − 1/64) = −93/512, tr W = −93/256: p = (5.5 − 93/256)/3600.
/// Checked with the hand virial (1e-15 relative) and with the virial returned by `LennardJones`
/// for the same two atoms (1e-13), so the sign conventions of P-5 and E-8 agree.
#[test]
fn two_body_hand_computed() {
    let cell = Cell::new([10.0, 10.0, 12.0]).unwrap();
    let (m, p) = two_body();
    for (r, tr_w) in [(1.0, 24.0), (2.0, -93.0 / 256.0)] {
        let want = (5.5 + tr_w) / 3600.0;
        let mut w = ZERO;
        w[0][0] = tr_w;
        let got = pressure_classical(&m, &p, w, &cell).unwrap();
        assert!(close(got, want, 1e-15), "r = {r}: p = {got}, want {want}");
        let mut lj = LennardJones::new(LjParams {
            epsilon: 1.0,
            sigma: 1.0,
            cutoff: 2.5,
            skin: 0.3,
            shift: false,
        })
        .unwrap();
        let q = Beads::from_vec(1, 2, vec![1.0, 1.0, 1.0, 1.0 + r, 1.0, 1.0]).unwrap();
        let w = lj.evaluate(&q, Some(&cell), None).unwrap().virial.unwrap();
        let got = pressure_classical(&m, &p, w, &cell).unwrap();
        assert!(
            close(got, want, 1e-13),
            "r = {r} via LennardJones: p = {got}, want {want}"
        );
    }
}

/// E-8: ideal-gas limit. With no interactions (W = 0) p = 2K/(3V), K computed in the test, for 50
/// atoms with masses 1 … 50 and seeded momenta in a cell (7, 8, 9); also with W from
/// `LennardJones` for atoms placed on a lattice with spacing 3 > r_c = 2.5 (no pair interacts,
/// so W = 0 exactly).
#[test]
fn ideal_gas_limit() {
    let cell = Cell::new([7.0, 8.0, 9.0]).unwrap();
    let n = 50;
    let masses: Vec<f64> = (1..=n).map(|k| k as f64).collect();
    let mut s = 0x9E37_79B9u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let p = Beads::from_vec(1, n, (0..3 * n).map(|_| 4.0 * next()).collect()).unwrap();
    let k: f64 = (0..n)
        .map(|i| p.get(0, i).iter().map(|x| x * x).sum::<f64>() / (2.0 * masses[i]))
        .sum();
    let want = 2.0 * k / (3.0 * 504.0);
    assert!(close(
        pressure_classical(&masses, &p, ZERO, &cell).unwrap(),
        want,
        1e-14
    ));

    let big = Cell::new([30.0; 3]).unwrap();
    let mut q = Beads::zeros(1, n);
    for i in 0..n {
        q.set(
            0,
            i,
            [
                3.0 * (i % 5) as f64,
                3.0 * ((i / 5) % 5) as f64,
                3.0 * (i / 25) as f64,
            ],
        );
    }
    let mut lj = LennardJones::new(LjParams {
        epsilon: 1.0,
        sigma: 1.0,
        cutoff: 2.5,
        skin: 0.3,
        shift: true,
    })
    .unwrap();
    let w = lj.evaluate(&q, Some(&big), None).unwrap().virial.unwrap();
    assert_eq!(w, ZERO);
    let want = 2.0 * k / (3.0 * 27000.0);
    assert!(close(
        pressure_classical(&masses, &p, w, &big).unwrap(),
        want,
        1e-14
    ));
}

/// E-8 against LAMMPS on the 256-atom snapshot (`data/lj_snapshot.json`, `compute pressure`
/// thermo_press, which is (dof·k_B·T + tr W)/(3V) with dof·k_B·T = Σ m v² = 2K, i.e. E-8 exactly;
/// lj units = atomic units). K matches LAMMPS `compute ke` to 1e-12; the pressure from LAMMPS's own
/// virial matches to 1e-10 relative; and the full pipeline, the virial from `LennardJones` (P-5)
/// fed to `pressure_classical`, matches to 1e-10 relative (shift yes and no give the same
/// pressure: the shift does not change forces).
#[test]
fn snapshot_matches_lammps() {
    let s = Snapshot::load();
    let json = snapshot_json();
    let cell = s.cell();
    let masses = s.masses();
    for key in ["shift_yes", "shift_no"] {
        let r = json.get(key);
        let p_ref = r.get("pressure").num();
        let k = kinetic_classical(&masses, &s.p);
        assert!(close(k, r.get("kinetic_energy").num(), 1e-12), "K = {k}");
        let rows = r.get("virial").arr();
        let mut w = ZERO;
        for a in 0..3 {
            let row = rows[a].nums();
            w[a].copy_from_slice(&row);
        }
        let p = pressure_classical(&masses, &s.p, w, &cell).unwrap();
        assert!(
            close(p, p_ref, 1e-10),
            "{key}: p(LAMMPS W) = {p}, LAMMPS {p_ref}"
        );
        let mut lj = LennardJones::new(LjParams {
            epsilon: s.epsilon,
            sigma: s.sigma,
            cutoff: s.cutoff,
            skin: s.skin,
            shift: key == "shift_yes",
        })
        .unwrap();
        let w = lj
            .evaluate(&s.q, Some(&cell), None)
            .unwrap()
            .virial
            .unwrap();
        let p = pressure_classical(&masses, &s.p, w, &cell).unwrap();
        assert!(
            close(p, p_ref, 1e-10),
            "{key}: p(engine W) = {p}, LAMMPS {p_ref}"
        );
    }
}

/// E-8 (P = 1): momenta with P = 2 give `ShapeMismatch`; masses whose count differs from N give
/// `ShapeMismatch`.
#[test]
fn errors() {
    let cell = Cell::new([10.0; 3]).unwrap();
    let r = pressure_classical(&[1.0, 2.0], &Beads::zeros(2, 2), ZERO, &cell);
    assert!(
        matches!(r, Err(EstimatorError::ShapeMismatch { .. })),
        "P = 2: {r:?}"
    );
    for m in [vec![1.0], vec![1.0, 2.0, 3.0]] {
        let r = pressure_classical(&m, &Beads::zeros(1, 2), ZERO, &cell);
        assert!(
            matches!(r, Err(EstimatorError::ShapeMismatch { .. })),
            "{} masses: {r:?}",
            m.len()
        );
    }
}
