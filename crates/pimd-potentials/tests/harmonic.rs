//! Tests for the `Potential` trait and `Harmonic` (SPEC P-1, P-2, P-4, S-2, S-3, S-4).

use pimd_core::Beads;
use pimd_potentials::{Evaluation, Harmonic, Potential, PotentialError};

/// SplitMix64, test-only seeded generator independent of `pimd_core::Rng`.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

fn random_beads(rng: &mut SplitMix64, p: usize, n: usize) -> Beads {
    let data = (0..p * n * 3).map(|_| rng.range(-2.0, 2.0)).collect();
    Beads::from_vec(p, n, data).unwrap()
}

fn eval(h: &mut Harmonic, q: &Beads) -> Evaluation {
    h.evaluate(q, None, None).unwrap()
}

/// P-1, P-2, S-3, S-4: energies and forces at hand-computed points, P = 2, N = 2,
/// masses (1, 4), ω = 1/2 (so m ω² = 1/4 and 1), open boundaries. Every value is exact in binary.
#[test]
fn hand_computed_energies_and_forces() {
    let mut h = Harmonic::new(vec![1.0, 4.0], 0.5).unwrap();
    let mut q = Beads::zeros(2, 2);
    q.set(0, 0, [1.0, 2.0, -2.0]);
    q.set(0, 1, [0.5, 0.0, 0.0]);
    q.set(1, 0, [0.0, 3.0, 0.0]);
    q.set(1, 1, [0.0, 0.0, -1.0]);
    // Bead 0: atom 0 |q|² = 1 + 4 + 4 = 9, V = ½·1·¼·9 = 1.125;
    //         atom 1 |q|² = 0.25,        V = ½·4·¼·0.25 = 0.125;  total 1.25.
    // Bead 1: atom 0 |q|² = 9, V = 1.125; atom 1 |q|² = 1, V = ½·4·¼·1 = 0.5;  total 1.625.
    // Forces F = −m ω² q: bead 0 atom 0 = −¼(1, 2, −2) = (−0.25, −0.5, 0.5);
    //   bead 0 atom 1 = −1·(0.5, 0, 0) = (−0.5, 0, 0); bead 1 atom 0 = −¼(0, 3, 0) = (0, −0.75, 0);
    //   bead 1 atom 1 = −1·(0, 0, −1) = (0, 0, 1).
    let e = eval(&mut h, &q);
    assert_eq!(e.energies.len(), 2);
    assert_eq!(e.forces.n_beads(), 2);
    assert_eq!(e.forces.n_atoms(), 2);
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-14;
    assert!(close(e.energies[0], 1.25), "{:?}", e.energies);
    assert!(close(e.energies[1], 1.625), "{:?}", e.energies);
    let want = [
        ((0, 0), [-0.25, -0.5, 0.5]),
        ((0, 1), [-0.5, 0.0, 0.0]),
        ((1, 0), [0.0, -0.75, 0.0]),
        ((1, 1), [0.0, 0.0, 1.0]),
    ];
    for ((j, i), f) in want {
        let got = e.forces.get(j, i);
        for d in 0..3 {
            assert!(
                close(got[d], f[d]),
                "bead {j} atom {i}: {got:?}, want {f:?}"
            );
        }
    }
}

/// P-2: a second hand-computed point, P = 1, N = 1, m = 2, ω = 3 (m ω² = 18), q = (0.5, −1, 0):
/// V = ½·18·1.25 = 11.25; F = −18 q = (−9, 18, 0).
#[test]
fn hand_computed_single_particle() {
    let mut h = Harmonic::new(vec![2.0], 3.0).unwrap();
    let q = Beads::from_vec(1, 1, vec![0.5, -1.0, 0.0]).unwrap();
    let e = eval(&mut h, &q);
    assert_eq!(e.energies.len(), 1);
    assert!((e.energies[0] - 11.25).abs() <= 1e-13, "{:?}", e.energies);
    let f = e.forces.get(0, 0);
    for (got, want) in f.iter().zip([-9.0, 18.0, 0.0]) {
        assert!((got - want).abs() <= 1e-13, "{f:?}");
    }
}

/// P-2: ω = 0 is allowed and gives zero energies and forces.
#[test]
fn zero_frequency_is_free() {
    let mut h = Harmonic::new(vec![1.0, 2.0, 3.0], 0.0).unwrap();
    let mut rng = SplitMix64(3);
    let q = random_beads(&mut rng, 4, 3);
    let e = eval(&mut h, &q);
    assert_eq!(e.energies, vec![0.0; 4]);
    assert!(e.forces.as_slice().iter().all(|&f| f == 0.0));
}

/// P-2, S-3: forces are linear, F(a x + b y) = a F(x) + b F(y), and energies are quadratic,
/// V(a x) = a² V(x), to 1e-12 relative; seeded random beads, P = 3, N = 4.
#[test]
fn forces_are_linear_and_energy_quadratic() {
    let masses = vec![1.0, 2.5, 10.0, 0.3];
    let mut h = Harmonic::new(masses, 0.8).unwrap();
    let mut rng = SplitMix64(11);
    let x = random_beads(&mut rng, 3, 4);
    let y = random_beads(&mut rng, 3, 4);
    let (a, b) = (1.75, -0.6);
    let z_data = x
        .as_slice()
        .iter()
        .zip(y.as_slice())
        .map(|(u, v)| a * u + b * v)
        .collect();
    let z = Beads::from_vec(3, 4, z_data).unwrap();
    let (ex, ey, ez) = (eval(&mut h, &x), eval(&mut h, &y), eval(&mut h, &z));
    for (idx, ((fx, fy), fz)) in ex
        .forces
        .as_slice()
        .iter()
        .zip(ey.forces.as_slice())
        .zip(ez.forces.as_slice())
        .enumerate()
    {
        let want = a * fx + b * fy;
        assert!(
            (fz - want).abs() <= 1e-12 * want.abs().max(1.0),
            "index {idx}: {fz} vs {want}"
        );
    }
    let ax_data = x.as_slice().iter().map(|u| a * u).collect();
    let ax = Beads::from_vec(3, 4, ax_data).unwrap();
    let eax = eval(&mut h, &ax);
    for j in 0..3 {
        let want = a * a * ex.energies[j];
        assert!((eax.energies[j] - want).abs() <= 1e-12 * want.abs().max(1.0));
    }
}

/// P-2, S-4: beads are independent. Changing bead 1 leaves bead 0 and 2 energies and forces
/// bitwise unchanged, and each bead's result equals a P = 1 evaluation of that bead alone.
#[test]
fn beads_are_independent() {
    let mut h = Harmonic::new(vec![1.0, 3.0], 1.3).unwrap();
    let mut rng = SplitMix64(17);
    let q = random_beads(&mut rng, 3, 2);
    let e = eval(&mut h, &q);
    let mut q2 = q.clone();
    q2.set(1, 0, [5.0, -4.0, 3.0]);
    q2.set(1, 1, [0.1, 0.2, 0.3]);
    let e2 = eval(&mut h, &q2);
    assert_ne!(e.energies[1], e2.energies[1]);
    for j in [0, 2] {
        assert_eq!(
            e.energies[j].to_bits(),
            e2.energies[j].to_bits(),
            "bead {j}"
        );
        for i in 0..2 {
            assert_eq!(e.forces.get(j, i), e2.forces.get(j, i), "bead {j} atom {i}");
        }
    }
    for j in 0..3 {
        let mut single = Beads::zeros(1, 2);
        for i in 0..2 {
            single.set(0, i, q.get(j, i));
        }
        let es = eval(&mut h, &single);
        assert!((es.energies[0] - e.energies[j]).abs() <= 1e-14 * e.energies[j].max(1.0));
        for i in 0..2 {
            let (a, b) = (es.forces.get(0, i), e.forces.get(j, i));
            for d in 0..3 {
                assert!((a[d] - b[d]).abs() <= 1e-14 * b[d].abs().max(1.0));
            }
        }
    }
}

/// P-1, P-2, S-3: forces equal −∂E_j/∂q_{j,i,d} by central finite differences (h = 1e-5) to 1e-6,
/// on seeded random beads, P = 4, N = 5, masses in [1, 50), ω = 0.3; also through `dyn Potential`.
/// For a quadratic V the central difference is exact up to rounding (~1e-10 here).
#[test]
fn forces_match_finite_difference_gradient() {
    let mut rng = SplitMix64(0xFD);
    let (p, n) = (4, 5);
    let masses: Vec<f64> = (0..n).map(|_| rng.range(1.0, 50.0)).collect();
    let mut pot: Box<dyn Potential> = Box::new(Harmonic::new(masses, 0.3).unwrap());
    let q = random_beads(&mut rng, p, n);
    let e = pot.evaluate(&q, None, None).unwrap();
    let h = 1e-5;
    for j in 0..p {
        for i in 0..n {
            for d in 0..3 {
                let idx = (j * n + i) * 3 + d;
                let mut qp = q.clone();
                qp.as_mut_slice()[idx] += h;
                let mut qm = q.clone();
                qm.as_mut_slice()[idx] -= h;
                let ep = pot.evaluate(&qp, None, None).unwrap().energies[j];
                let em = pot.evaluate(&qm, None, None).unwrap().energies[j];
                let grad = (ep - em) / (2.0 * h);
                let f = e.forces.get(j, i)[d];
                assert!(
                    (f + grad).abs() <= 1e-6 * f.abs().max(1.0),
                    "bead {j} atom {i} d {d}: F = {f}, -dV/dq = {}",
                    -grad
                );
            }
        }
    }
}

/// P-4: `subset = Some(..)` returns `PotentialError::Unsupported`, for empty and non-empty subsets.
#[test]
fn subset_is_unsupported() {
    let mut h = Harmonic::new(vec![1.0, 2.0], 1.0).unwrap();
    let q = Beads::zeros(2, 2);
    for subset in [&[][..], &[0][..], &[0, 1][..]] {
        let r = h.evaluate(&q, None, Some(subset));
        assert!(
            matches!(r, Err(PotentialError::Unsupported)),
            "subset {subset:?}: {r:?}"
        );
    }
    assert!(h.evaluate(&q, None, None).is_ok());
}

/// P-1, S-2: positions whose atom count differs from the number of masses give
/// `ShapeMismatch { expected: masses, actual: positions }`, for more and for fewer atoms.
#[test]
fn atom_count_mismatch_is_shape_mismatch() {
    let mut h = Harmonic::new(vec![1.0, 2.0], 0.5).unwrap();
    for n in [1, 3] {
        assert_eq!(
            h.evaluate(&Beads::zeros(2, n), None, None).unwrap_err(),
            PotentialError::ShapeMismatch {
                expected: 2,
                actual: n
            }
        );
    }
}

/// P-2, S-2: non-positive or non-finite masses, and negative or non-finite ω, are
/// `InvalidParameter`; ω = 0 and positive finite masses are accepted.
#[test]
fn new_validates_parameters() {
    for m in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let r = Harmonic::new(vec![1.0, m, 2.0], 1.0);
        assert!(
            matches!(r, Err(PotentialError::InvalidParameter { .. })),
            "mass {m}: {r:?}"
        );
    }
    for w in [-1.0, -1e-12, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let r = Harmonic::new(vec![1.0, 2.0], w);
        assert!(
            matches!(r, Err(PotentialError::InvalidParameter { .. })),
            "omega {w}: {r:?}"
        );
    }
    assert!(Harmonic::new(vec![1.0, 1837.0], 0.0).is_ok());
    assert!(Harmonic::new(vec![1e-3], 1e3).is_ok());
}
