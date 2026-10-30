//! Tests for the optional FFT normal-mode transform (SPEC RP-4, D-3; MILESTONE_4 "Normal-mode
//! transform"): `NormalModes::with_transform(.., Transform::Fft)` must produce exactly the RP-4
//! columns (centroid, cos/sin pairs, alternating mode for even P) with the RP-4 normalisation, for
//! every P including odd P, agreeing with the matrix transform to 1e-13 relative.
//!
//! The reference C is built here from the RP-4 formula and applied by a plain double loop,
//! independently of the code under test. "Relative" is measured against the largest magnitude of
//! the reference output (a norm-wise relative error, so near-zero entries do not blow up):
//! |x − x_ref| ≤ 1e-13 · max|x_ref|. A length-P FFT has rounding error O(ε log P) ≈ 1e-15 here,
//! so 1e-13 leaves two orders of margin and still catches any wrong column, sign or factor.

mod common;

use std::f64::consts::PI;
use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{SplitMix64, random_vec};
use pimd_core::normal_modes::Transform;
use pimd_core::{Beads, CoreError, NormalModes};

/// The bead counts the brief requires: odd, even, powers of two and P = 1.
const PS: [usize; 7] = [1, 2, 3, 4, 8, 16, 32];
const N_ATOMS: usize = 5;
const TOL: f64 = 1e-13;

/// C from RP-4, row-major P × P (C[j][k] at j*P + k).
fn reference_matrix(p: usize) -> Vec<f64> {
    let pf = p as f64;
    let mut c = vec![0.0; p * p];
    for j in 0..p {
        let jf = j as f64;
        c[j * p] = 1.0 / pf.sqrt();
        for m in 1..=(p - 1) / 2 {
            let mf = m as f64;
            c[j * p + 2 * m - 1] = (2.0 / pf).sqrt() * (2.0 * PI * mf * jf / pf).cos();
            c[j * p + 2 * m] = (2.0 / pf).sqrt() * (2.0 * PI * mf * jf / pf).sin();
        }
        if p.is_multiple_of(2) {
            let sign = if j.is_multiple_of(2) { 1.0 } else { -1.0 };
            c[j * p + p - 1] = sign / pf.sqrt();
        }
    }
    c
}

/// q̃_k = Σ_j C_jk q_j (forward) or q_j = Σ_k C_jk q̃_k (inverse), plain loops over S-1 layout.
fn reference_apply(c: &[f64], x: &Beads, inverse: bool) -> Vec<f64> {
    let (p, n) = (x.n_beads(), x.n_atoms());
    let src = x.as_slice();
    let mut out = vec![0.0; p * n * 3];
    for a in 0..p {
        for b in 0..p {
            let coef = if inverse { c[a * p + b] } else { c[b * p + a] };
            for e in 0..n * 3 {
                out[a * n * 3 + e] += coef * src[b * n * 3 + e];
            }
        }
    }
    out
}

/// max |a − b| / max |b| (norm-wise relative error).
fn rel_err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let diff = a
        .iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    diff / scale
}

fn random_beads(rng: &mut SplitMix64, p: usize) -> Beads {
    Beads::from_vec(p, N_ATOMS, random_vec(rng, p * N_ATOMS * 3, -2.0, 2.0)).unwrap()
}

/// RP-4: `Transform::default()` is `Matrix`; `new` reports `Matrix`; `with_transform` reports
/// what it was given.
#[test]
fn transform_setting_reported() {
    assert_eq!(Transform::default(), Transform::Matrix);
    assert_eq!(
        NormalModes::new(4, 2.0).unwrap().transform(),
        Transform::Matrix
    );
    for t in [Transform::Matrix, Transform::Fft] {
        for p in PS {
            assert_eq!(
                NormalModes::with_transform(p, 2.0, t).unwrap().transform(),
                t
            );
        }
    }
}

/// RP-4: the FFT path, forward, equals Σ_j C_jk q_j with C from the RP-4 formula to 1e-13
/// (norm-wise relative), for P ∈ {1, 2, 3, 4, 8, 16, 32} and random inputs.
#[test]
fn fft_forward_matches_rp4_matrix() {
    let mut rng = SplitMix64::new(0xFF7);
    for p in PS {
        let c = reference_matrix(p);
        let nm = NormalModes::with_transform(p, 1.7, Transform::Fft).unwrap();
        for _ in 0..3 {
            let x = random_beads(&mut rng, p);
            let got = nm.to_normal(&x);
            assert_eq!((got.n_beads(), got.n_atoms()), (p, N_ATOMS));
            let err = rel_err(got.as_slice(), &reference_apply(&c, &x, false));
            assert!(err <= TOL, "P = {p}: forward relative error {err:e}");
        }
    }
}

/// RP-4: the FFT path, inverse, equals Σ_k C_jk q̃_k to 1e-13, for every required P.
#[test]
fn fft_inverse_matches_rp4_matrix() {
    let mut rng = SplitMix64::new(0x1FF7);
    for p in PS {
        let c = reference_matrix(p);
        let nm = NormalModes::with_transform(p, 0.3, Transform::Fft).unwrap();
        for _ in 0..3 {
            let x = random_beads(&mut rng, p);
            let got = nm.to_beads(&x);
            assert_eq!((got.n_beads(), got.n_atoms()), (p, N_ATOMS));
            let err = rel_err(got.as_slice(), &reference_apply(&c, &x, true));
            assert!(err <= TOL, "P = {p}: inverse relative error {err:e}");
        }
    }
}

/// RP-4 column order: a bead delta (q_j = 1 at bead j0, one component) maps to row j0 of C, so
/// mode k receives C[j0][k] (centroid first, then cos/sin pairs m = 1, 2, …, then the alternating
/// mode for even P); a mode delta maps back to column k. Absolute tolerance 1e-13 (entries are
/// O(1/√P)).
#[test]
fn fft_column_order_by_basis_vectors() {
    for p in PS {
        let c = reference_matrix(p);
        let nm = NormalModes::with_transform(p, 1.0, Transform::Fft).unwrap();
        for j0 in 0..p {
            let mut x = Beads::zeros(p, 2);
            x.set(j0, 1, [0.0, 1.0, 0.0]);
            let q = nm.to_normal(&x);
            for k in 0..p {
                let got = q.get(k, 1);
                let want = c[j0 * p + k];
                assert!(
                    (got[1] - want).abs() <= TOL && got[0] == 0.0 && got[2] == 0.0,
                    "P = {p}: bead {j0} -> mode {k}: {got:?}, want C = {want}"
                );
                assert_eq!(q.get(k, 0), [0.0; 3], "atom 0 must stay zero");
            }
        }
        for k0 in 0..p {
            let mut x = Beads::zeros(p, 1);
            x.set(k0, 0, [0.0, 0.0, 1.0]);
            let b = nm.to_beads(&x);
            for j in 0..p {
                let got = b.get(j, 0)[2];
                let want = c[j * p + k0];
                assert!(
                    (got - want).abs() <= TOL,
                    "P = {p}: mode {k0} -> bead {j}: {got}, want C = {want}"
                );
            }
        }
    }
}

/// RP-4: FFT and matrix settings agree with each other to 1e-13, forward and inverse; the FFT
/// round trip `to_beads(to_normal(x))` returns x to 1e-13.
#[test]
fn fft_agrees_with_matrix_setting_and_round_trips() {
    let mut rng = SplitMix64::new(0xA6);
    for p in PS {
        let fft = NormalModes::with_transform(p, 4.0, Transform::Fft).unwrap();
        let mat = NormalModes::new(p, 4.0).unwrap();
        let x = random_beads(&mut rng, p);
        let e1 = rel_err(fft.to_normal(&x).as_slice(), mat.to_normal(&x).as_slice());
        let e2 = rel_err(fft.to_beads(&x).as_slice(), mat.to_beads(&x).as_slice());
        let e3 = rel_err(fft.to_beads(&fft.to_normal(&x)).as_slice(), x.as_slice());
        assert!(
            e1 <= TOL && e2 <= TOL && e3 <= TOL,
            "P = {p}: {e1:e} {e2:e} {e3:e}"
        );
    }
}

/// RP-4: `frequencies()`, `matrix()`, `omega_p()` and `n_beads()` do not depend on the transform
/// setting (bitwise equal to `new`'s); `with_transform(.., Matrix)` transforms bitwise like `new`.
#[test]
fn setting_does_not_change_frequencies_or_matrix() {
    let mut rng = SplitMix64::new(0x5E7);
    for p in PS {
        let base = NormalModes::new(p, 0.9).unwrap();
        for t in [Transform::Matrix, Transform::Fft] {
            let nm = NormalModes::with_transform(p, 0.9, t).unwrap();
            assert_eq!(nm.n_beads(), p);
            assert_eq!(nm.omega_p().to_bits(), base.omega_p().to_bits());
            assert_eq!(nm.frequencies(), base.frequencies(), "P = {p}, {t:?}");
            assert_eq!(nm.matrix(), base.matrix(), "P = {p}, {t:?}");
        }
        let m = NormalModes::with_transform(p, 0.9, Transform::Matrix).unwrap();
        let x = random_beads(&mut rng, p);
        assert_eq!(m.to_normal(&x), base.to_normal(&x));
        assert_eq!(m.to_beads(&x), base.to_beads(&x));
    }
}

/// D-3 (per setting): the FFT transform is deterministic: repeated calls on the same input give
/// bitwise-identical output, also from a second, independently built object.
#[test]
fn fft_is_deterministic() {
    let mut rng = SplitMix64::new(0xD3);
    for p in PS {
        let a = NormalModes::with_transform(p, 2.0, Transform::Fft).unwrap();
        let b = NormalModes::with_transform(p, 2.0, Transform::Fft).unwrap();
        let x = random_beads(&mut rng, p);
        assert_eq!(a.to_normal(&x), a.to_normal(&x));
        assert_eq!(a.to_normal(&x), b.to_normal(&x));
        assert_eq!(a.to_beads(&x), b.to_beads(&x));
    }
}

/// RP-1, U-4: `with_transform` validates like `new`, for both settings: P = 0 →
/// `InvalidBeadCount(0)`; non-positive or non-finite β → `InvalidValue`.
#[test]
fn with_transform_errors() {
    for t in [Transform::Matrix, Transform::Fft] {
        assert_eq!(
            NormalModes::with_transform(0, 1.0, t).unwrap_err(),
            CoreError::InvalidBeadCount(0)
        );
        for beta in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                matches!(
                    NormalModes::with_transform(4, beta, t),
                    Err(CoreError::InvalidValue { .. })
                ),
                "{t:?}, beta = {beta}"
            );
        }
    }
}

/// RP-4: the FFT path panics, like the matrix path, when the input bead count differs from P,
/// in `to_normal` and in `to_beads`. (The object is built outside `catch_unwind`, so a panicking
/// constructor fails the test instead of satisfying it.)
#[test]
fn fft_panics_on_bead_mismatch() {
    let nm4 = NormalModes::with_transform(4, 1.0, Transform::Fft).unwrap();
    let nm3 = NormalModes::with_transform(3, 1.0, Transform::Fft).unwrap();
    let r1 = catch_unwind(AssertUnwindSafe(|| nm4.to_normal(&Beads::zeros(3, 2))));
    let r2 = catch_unwind(AssertUnwindSafe(|| nm3.to_beads(&Beads::zeros(4, 2))));
    assert!(r1.is_err(), "to_normal accepted 3 beads for P = 4");
    assert!(r2.is_err(), "to_beads accepted 4 beads for P = 3");
    // Sanity: the matching shapes do not panic.
    let _ = nm4.to_normal(&Beads::zeros(4, 2));
    let _ = nm3.to_beads(&Beads::zeros(3, 2));
}
