//! Batching over beads (MILESTONE_6 "Batching"; SPEC P-6: one `evaluate` call evaluates all P
//! beads as one graph, bead j holding atoms jN … jN+N−1; D-3 note: batched and per-bead
//! evaluation agree only to stated tolerances).
//!
//! Tolerances (brief): batched vs per-bead 1e-12 relative (|ΔE| ≤ 1e-12·|E|, max|ΔF| ≤
//! 1e-12·max|F| per bead; not bitwise: spike measured ≤ 5e-15); bead independence 1e-14 relative.
//! float64, CPU, 1 libtorch thread.
#![cfg(feature = "torch")]

mod common;

use common::{SYSTEMS, bead, errors, eval, load, max_abs, single, system};
use pimd_torch::{Device, Dtype};

/// P-6, S-1: the 8-bead batch equals 8 separate one-bead calls to 1e-12 relative per bead
/// (water, ethanol, 4-water box).
#[test]
fn batched_equals_per_bead() {
    for name in SYSTEMS {
        let s = system(name);
        let mut pot = load(&s.z, Device::Cpu, Dtype::F64, 1);
        let batch = eval(&mut pot, &s.q, s.cell.as_ref());
        let mut e = Vec::new();
        let mut f = Vec::new();
        for j in 0..s.p {
            let one = eval(&mut pot, &single(&s.q, j), s.cell.as_ref());
            assert_eq!(one.energies.len(), 1);
            e.push(one.energies[0]);
            f.extend_from_slice(one.forces.as_slice());
        }
        let f = pimd_core::Beads::from_vec(s.p, s.n, f).unwrap();
        let (er, _, fr) = errors(&batch, &e, &f);
        eprintln!("{name}: batched vs per-bead |dE|/|E| {er:.2e}, |dF|/max|F| {fr:.2e}");
        assert!(er <= 1e-12, "{name}: energy {er:e}");
        assert!(fr <= 1e-12, "{name}: forces {fr:e}");
    }
}

/// P-6: beads are independent graphs: displacing every atom of bead k (by up to 0.02 bohr) changes
/// bead k's result and leaves every other bead's energy and forces unchanged to 1e-14 relative.
/// Checked for k = 0, 3 and 7 (first, middle, last) in each system.
#[test]
fn moving_one_bead_leaves_the_others_unchanged() {
    for name in SYSTEMS {
        let s = system(name);
        let mut pot = load(&s.z, Device::Cpu, Dtype::F64, 1);
        let before = eval(&mut pot, &s.q, s.cell.as_ref());
        for k in [0, 3, 7] {
            let mut q = s.q.clone();
            let mut rng = common::SplitMix64(100 + k as u64);
            for i in 0..s.n {
                let x = q.get(k, i);
                q.set(k, i, x.map(|c| c + rng.sym(0.02)));
            }
            let after = eval(&mut pot, &q, s.cell.as_ref());
            for j in 0..s.p {
                let de = (after.energies[j] - before.energies[j]).abs();
                let fa = bead(&after.forces, j);
                let fb = bead(&before.forces, j);
                let df = max_abs(fa.iter().zip(fb).map(|(a, b)| a - b));
                let fmax = max_abs(fb.iter().copied());
                if j == k {
                    assert!(de > 0.0, "{name}: moving bead {k} must change its energy");
                } else {
                    assert!(
                        de <= 1e-14 * before.energies[j].abs(),
                        "{name}: moving bead {k} changed bead {j}'s energy by {de:e} Eh"
                    );
                    assert!(
                        df <= 1e-14 * fmax,
                        "{name}: moving bead {k} changed bead {j}'s forces by {df:e}"
                    );
                }
            }
        }
    }
}
