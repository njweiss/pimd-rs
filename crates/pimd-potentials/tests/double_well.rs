//! Tests for `DoubleWell` (SPEC P-1, P-2, P-4, S-3, S-4).

use pimd_core::{Beads, Cell};
use pimd_potentials::{DoubleWell, Evaluation, Potential, PotentialError};

fn eval(dw: &mut DoubleWell, q: &Beads) -> Evaluation {
    dw.evaluate(q, None, None).unwrap()
}

/// P-2: a, b must be positive and finite; otherwise `PotentialError::InvalidParameter`.
#[test]
fn rejects_bad_parameters() {
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                DoubleWell::new(bad, 1.0),
                Err(PotentialError::InvalidParameter { .. })
            ),
            "a = {bad}"
        );
        assert!(
            matches!(
                DoubleWell::new(1.0, bad),
                Err(PotentialError::InvalidParameter { .. })
            ),
            "b = {bad}"
        );
    }
    assert!(DoubleWell::new(0.125, 1.0).is_ok());
}

/// P-2, S-3, S-4: hand-computed values, a = 0.5, b = 2, separable over components:
/// V(x) = 0.5 x⁴ − 2 x², F(x) = −(2 x³ − 4 x).
/// V(1) = −1.5, F(1) = 2; V(−2) = 8 − 8 = 0, F(−2) = −(−16 + 8) = 8; V(0) = 0, F(0) = 0;
/// V(0.5) = 0.03125 − 0.5 = −0.46875, F(0.5) = −(0.25 − 2) = 1.75; V(3) = 40.5 − 18 = 22.5,
/// F(3) = −(54 − 12) = −42.
/// Bead 0: atom 0 at (1, −2, 0), atom 1 at (0.5, 3, 1): V = −1.5 + 0 + 0 + (−0.46875 + 22.5 − 1.5)
/// = 19.03125. Bead 1: atom 0 at (−2, 0, 0.5), atom 1 at (0, 0, 0): V = 0 − 0.46875 = −0.46875.
#[test]
fn hand_computed_energies_and_forces() {
    let mut dw = DoubleWell::new(0.5, 2.0).unwrap();
    let mut q = Beads::zeros(2, 2);
    q.set(0, 0, [1.0, -2.0, 0.0]);
    q.set(0, 1, [0.5, 3.0, 1.0]);
    q.set(1, 0, [-2.0, 0.0, 0.5]);
    q.set(1, 1, [0.0, 0.0, 0.0]);
    let e = eval(&mut dw, &q);
    assert_eq!(e.energies.len(), 2);
    assert!(
        (e.energies[0] - 19.03125).abs() <= 1e-13,
        "{:?}",
        e.energies
    );
    assert!((e.energies[1] + 0.46875).abs() <= 1e-13, "{:?}", e.energies);
    let want = [
        ((0, 0), [2.0, 8.0, 0.0]),
        ((0, 1), [1.75, -42.0, 2.0]),
        ((1, 0), [8.0, 0.0, 1.75]),
        ((1, 1), [0.0, 0.0, 0.0]),
    ];
    for ((j, i), f) in want {
        let got = e.forces.get(j, i);
        for d in 0..3 {
            assert!(
                (got[d] - f[d]).abs() <= 1e-13,
                "bead {j} atom {i}: {got:?} vs {f:?}"
            );
        }
    }
}

/// S-3: forces are −∇V. Central finite differences (h = 1e-5) of the bead energy at seeded
/// pseudo-random points, P = 3, N = 2, a = 0.3, b = 1.7, agree to 1e-6 relative (floor 1);
/// the O(h²) truncation error is about h²·|V'''|/6 ≈ 1e-9 here.
#[test]
fn forces_match_finite_differences() {
    let mut dw = DoubleWell::new(0.3, 1.7).unwrap();
    let mut s: u64 = 12345;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 5.0 - 2.5
    };
    let data: Vec<f64> = (0..3 * 2 * 3).map(|_| next()).collect();
    let q = Beads::from_vec(3, 2, data).unwrap();
    let e = eval(&mut dw, &q);
    let h = 1e-5;
    for idx in 0..q.as_slice().len() {
        let j = idx / 6;
        let mut qp = q.clone();
        qp.as_mut_slice()[idx] += h;
        let mut qm = q.clone();
        qm.as_mut_slice()[idx] -= h;
        let fd = -(eval(&mut dw, &qp).energies[j] - eval(&mut dw, &qm).energies[j]) / (2.0 * h);
        let f = e.forces.as_slice()[idx];
        assert!(
            (f - fd).abs() <= 1e-6 * fd.abs().max(1.0),
            "index {idx}: {f} vs {fd}"
        );
    }
}

/// P-4: native potentials return `Unsupported` for `subset = Some`.
#[test]
fn subset_is_unsupported() {
    let mut dw = DoubleWell::new(1.0, 1.0).unwrap();
    let q = Beads::zeros(2, 3);
    let r = dw.evaluate(&q, None, Some(&[0, 2]));
    assert!(matches!(r, Err(PotentialError::Unsupported)), "{r:?}");
}

/// P-2 (Q3): the double well is an external field and ignores the cell: `Some(cell)` gives a
/// bitwise identical result to `None`, including for positions outside the cell.
#[test]
fn cell_is_ignored() {
    let mut dw = DoubleWell::new(0.2, 0.9).unwrap();
    let q = Beads::from_vec(2, 1, vec![0.3, -7.0, 12.0, 1.5, 0.0, -2.2]).unwrap();
    let cell = Cell::new([4.0, 5.0, 6.0]).unwrap();
    let a = eval(&mut dw, &q);
    let b = dw.evaluate(&q, Some(&cell), None).unwrap();
    assert_eq!(a, b);
}

/// P-1, S-4: one energy per bead; beads are independent (each bead's energy equals a P = 1
/// evaluation of that bead alone); `DoubleWell` is not built for an atom count, so it accepts
/// any N; `virial` is `None`.
#[test]
fn shapes_and_bead_independence() {
    let mut dw = DoubleWell::new(0.125, 1.0).unwrap();
    for n in [1usize, 4] {
        let data: Vec<f64> = (0..3 * n * 3)
            .map(|k| (k as f64 * 0.37).sin() * 2.5)
            .collect();
        let q = Beads::from_vec(3, n, data).unwrap();
        let e = eval(&mut dw, &q);
        assert_eq!(e.energies.len(), 3);
        assert_eq!((e.forces.n_beads(), e.forces.n_atoms()), (3, n));
        assert!(e.virial.is_none());
        for j in 0..3 {
            let one =
                Beads::from_vec(1, n, q.as_slice()[j * n * 3..(j + 1) * n * 3].to_vec()).unwrap();
            let e1 = eval(&mut dw, &one);
            assert_eq!(e1.energies[0], e.energies[j], "bead {j}");
            assert_eq!(
                e1.forces.as_slice(),
                &e.forces.as_slice()[j * n * 3..(j + 1) * n * 3]
            );
        }
    }
}
