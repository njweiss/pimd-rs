//! LAMMPS single point (SPEC P-3, P-5, C-2): `LennardJones` on the 256-atom snapshot of
//! `data/lj_snapshot.json` against LAMMPS `pair_style lj/cut` with `pair_modify shift yes` and
//! `no` (generator `data/gen_lj_snapshot.py`; LAMMPS version, inputs and the unit mapping
//! LJ units = atomic units with ε = σ = m = 1 are recorded in the file).
//!
//! Tolerance 1e-10 relative (MILESTONE_3). Both codes evaluate the same doubles in f64; they differ
//! only in summation order (LAMMPS: half list with Newton's third law; engine: full lists), so the
//! expected disagreement is a few ulp of the largest pair terms (|F| ≤ 108, |E| ≈ 1200), ~1e-13
//! relative; measured with the reference implementation: 4e-15 (E) and 3e-14 (F). Floors: force
//! components relative to max(|F_ref|, 1 ε/σ); virial entries relative to the largest |W| entry
//! (off-diagonal entries are sums with heavy cancellation).

mod common;

use common::lammps::{Snapshot, max_rel, snapshot_json};
use pimd_potentials::{LennardJones, LjParams, Potential};

fn check(shift: bool) {
    let s = Snapshot::load();
    let json = snapshot_json();
    let reference = json.get(if shift { "shift_yes" } else { "shift_no" });
    let mut pot = LennardJones::new(LjParams {
        epsilon: s.epsilon,
        sigma: s.sigma,
        cutoff: s.cutoff,
        skin: s.skin,
        shift,
    })
    .unwrap();
    let cell = s.cell();
    let ev = pot.evaluate(&s.q, Some(&cell), None).unwrap();

    let e_ref = reference.get("potential_energy").num();
    let de = (ev.energies[0] - e_ref).abs() / e_ref.abs();
    eprintln!(
        "shift {shift}: E = {} vs LAMMPS {e_ref}, rel {de:.2e}",
        ev.energies[0]
    );
    assert!(de <= 1e-10, "energy rel diff {de:e}");

    let f_ref = reference.get("forces").nums();
    let df = max_rel(ev.forces.as_slice(), &f_ref, 1.0);
    eprintln!("shift {shift}: forces max rel diff {df:.2e}");
    assert!(df <= 1e-10, "forces rel diff {df:e}");

    let w = ev.virial.expect("P = 1 virial (P-5)");
    let w_ref: Vec<Vec<f64>> = reference
        .get("virial")
        .arr()
        .iter()
        .map(|row| row.nums())
        .collect();
    let wmax = w_ref.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
    for a in 0..3 {
        for b in 0..3 {
            let d = (w[a][b] - w_ref[a][b]).abs() / wmax;
            assert!(
                d <= 1e-10,
                "W[{a}][{b}] = {} vs LAMMPS {} (rel to max {d:e})",
                w[a][b],
                w_ref[a][b]
            );
        }
    }
}

/// P-3, P-5: `pair_modify shift yes`: total energy, all 768 force components and the virial tensor
/// (LAMMPS `compute pressure NULL virial` × V) match LAMMPS to 1e-10 relative.
#[test]
fn snapshot_matches_lammps_shifted() {
    check(true);
}

/// P-3, P-5: `pair_modify shift no` (plain truncation): as above.
#[test]
fn snapshot_matches_lammps_unshifted() {
    check(false);
}
