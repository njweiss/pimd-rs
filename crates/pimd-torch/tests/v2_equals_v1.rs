//! v2 export = v1 export in float64 (MILESTONE_9 "M9a: Required tests" 1; SPEC P-6 v2 sentence,
//! U-3, D-1).
//!
//! The same tiny model exported as v1 (`tiny_mace_f64.pt`, locked, energy = MACE's total
//! e0 + inter_e) and as v2 (`tiny_mace_v2_f64.pt`, energy = inter_e, E0 table from
//! `atomic_energies()`), loaded with `Dtype::F64`, libtorch threads 1:
//! - energies agree to 1e-12 relative per bead: v1 sums e0 inside the model in float64, the
//!   adapter sums the same E0 values in f64, so they differ by f64 rounding only (a few 1e-16
//!   relative at |E| ≈ 2e3 to 8e3 eV);
//! - forces are bitwise equal: MACE takes forces from the gradient of inter_e in both (models.py,
//!   `get_outputs(energy=inter_e, ..)`), and v2 changes nothing on the force path.
//!
//! Cases (brief: "gas phase, the 4-water box, P = 3, and c = 2"): water and ethanol in the gas
//! phase and the periodic 4-water box of `data/mace_ref.json`, each with all 8 beads and with the
//! first 3 beads, each unchunked and with c = 2 (chunks 2+2+2+2, and 2+1 for P = 3). The 8-bead
//! v2 energies are also checked against Python MACE (`mace_ref.json`) to the M6 1e-12 relative.
#![cfg(feature = "torch")]

mod common;
mod v2_common;

use common::{eval, export_path, system};
use pimd_core::Beads;
use pimd_potentials::Evaluation;
use pimd_torch::Dtype;
use v2_common::{bits_eq, load_path, v2_export};

fn first_beads(q: &Beads, p: usize) -> Beads {
    let n3 = q.n_atoms() * 3;
    Beads::from_vec(p, q.n_atoms(), q.as_slice()[..p * n3].to_vec()).unwrap()
}

fn run(
    path: &std::path::Path,
    z: &[u8],
    q: &Beads,
    cell: Option<&pimd_core::Cell>,
    c: Option<usize>,
) -> Evaluation {
    let mut pot = load_path(path, z, Dtype::F64, 1);
    if let Some(c) = c {
        pot = pot.with_beads_per_graph(c).unwrap();
    }
    eval(&mut pot, q, cell)
}

/// P-6 (v2 sentence), U-3: for water, ethanol (gas phase) and the 4-water box, P = 8 and P = 3,
/// unchunked and c = 2: v2 energies equal v1 energies to 1e-12 relative per bead and v2 forces
/// equal v1 forces bitwise.
#[test]
fn v2_matches_v1_in_float64() {
    for name in ["water", "ethanol", "waterbox"] {
        let s = system(name);
        for p in [8, 3] {
            let q = first_beads(&s.q, p);
            for c in [None, Some(2)] {
                let what = format!("{name}, P = {p}, c = {c:?}");
                let v1 = run(&export_path(), &s.z, &q, s.cell.as_ref(), c);
                let v2 = run(&v2_export(), &s.z, &q, s.cell.as_ref(), c);
                assert_eq!(v2.energies.len(), p, "{what}: one energy per bead");
                for j in 0..p {
                    let rel = (v2.energies[j] - v1.energies[j]).abs() / v1.energies[j].abs();
                    assert!(
                        rel <= 1e-12,
                        "{what}: bead {j}: v2 E {} vs v1 E {} Eh ({rel:e} relative)",
                        v2.energies[j],
                        v1.energies[j]
                    );
                }
                assert!(
                    bits_eq(v2.forces.as_slice(), v1.forces.as_slice()),
                    "{what}: v2 forces differ from v1 forces"
                );
                assert!(v2.virial.is_none(), "{what}: virial is None (P-5)");
                if p == 8 {
                    for j in 0..p {
                        let rel = (v2.energies[j] - s.energies[j]).abs() / s.energies[j].abs();
                        assert!(
                            rel <= 1e-12,
                            "{what}: bead {j}: v2 vs Python MACE {rel:e} relative"
                        );
                    }
                }
            }
        }
    }
}
