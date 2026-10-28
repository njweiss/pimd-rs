//! Tests for `pimd_core::Beads` (SPEC S-1).

use pimd_core::Beads;

/// S-1: element (j, i, d) is at flat index (j*N + i)*3 + d (Phase 0 accessors).
#[test]
fn layout_is_bead_major_row_major() {
    let (p, n) = (3, 4);
    let data: Vec<f64> = (0..p * n * 3).map(|x| x as f64).collect();
    let b = Beads::from_vec(p, n, data).unwrap();
    for j in 0..p {
        for i in 0..n {
            let v = b.get(j, i);
            for (d, &x) in v.iter().enumerate() {
                assert_eq!(x, ((j * n + i) * 3 + d) as f64);
            }
        }
    }
    assert_eq!(b.n_beads(), p);
    assert_eq!(b.n_atoms(), n);
    assert!(Beads::from_vec(p, n, vec![0.0; p * n * 3 - 1]).is_err());
}

/// S-1: `centroid` is the per-atom mean over beads; hand-computed example, P = 2, N = 2.
#[test]
fn centroid_hand_example_two_beads() {
    let mut b = Beads::zeros(2, 2);
    b.set(0, 0, [1.0, 2.0, 3.0]);
    b.set(0, 1, [-1.0, 0.0, 4.0]);
    b.set(1, 0, [3.0, 2.0, 1.0]);
    b.set(1, 1, [1.0, 0.0, -2.0]);
    // Atom 0: ((1+3)/2, (2+2)/2, (3+1)/2) = (2, 2, 2).
    // Atom 1: ((-1+1)/2, (0+0)/2, (4-2)/2) = (0, 0, 1).
    let c = b.centroid();
    assert_eq!(c.len(), 2);
    assert_eq!(c[0], [2.0, 2.0, 2.0]);
    assert_eq!(c[1], [0.0, 0.0, 1.0]);
}

/// S-1: `centroid` hand-computed example, P = 3, N = 1 (mean, not sum or first bead).
#[test]
fn centroid_hand_example_three_beads() {
    let mut b = Beads::zeros(3, 1);
    b.set(0, 0, [0.0, -3.0, 1.5]);
    b.set(1, 0, [1.0, 0.0, 1.5]);
    b.set(2, 0, [5.0, 6.0, 1.5]);
    // x: (0+1+5)/3 = 2; y: (-3+0+6)/3 = 1; z: 4.5/3 = 1.5.
    let c = b.centroid();
    assert_eq!(c.len(), 1);
    for (got, want) in c[0].iter().zip([2.0, 1.0, 1.5]) {
        assert!((got - want).abs() <= 1e-15, "{:?}", c[0]);
    }
}

/// S-1, RP-6: with P = 1 the centroid is the single bead.
#[test]
fn centroid_single_bead_is_identity() {
    let mut b = Beads::zeros(1, 2);
    b.set(0, 0, [0.25, -7.0, 3.0]);
    b.set(0, 1, [1e-3, 2e3, -0.5]);
    let c = b.centroid();
    assert_eq!(c, vec![[0.25, -7.0, 3.0], [1e-3, 2e3, -0.5]]);
}
