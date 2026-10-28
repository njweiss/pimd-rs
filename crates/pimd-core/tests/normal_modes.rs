//! Tests for `pimd_core::NormalModes` (SPEC RP-1, RP-2, RP-4, RP-5, RP-6, S-1).
//!
//! The reference matrix C, the frequencies and the spring matrix S are built here directly from
//! the SPEC formulas, independently of the code under test.

mod common;

use std::f64::consts::PI;

use common::{SplitMix64, random_vec};
use pimd_core::{Beads, CoreError, NormalModes};

const PS: [usize; 8] = [1, 2, 3, 4, 5, 8, 9, 16];
const BETAS: [f64; 3] = [0.5, 2.5, 1000.0];

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

/// ω_P = P/β (RP-2).
fn reference_omega_p(p: usize, beta: f64) -> f64 {
    p as f64 / beta
}

/// ω_k from RP-4, aligned with the columns of `reference_matrix`.
fn reference_frequencies(p: usize, beta: f64) -> Vec<f64> {
    let wp = reference_omega_p(p, beta);
    let mut w = vec![0.0; p];
    for m in 1..=(p - 1) / 2 {
        let wk = 2.0 * wp * (m as f64 * PI / p as f64).sin();
        w[2 * m - 1] = wk;
        w[2 * m] = wk;
    }
    if p.is_multiple_of(2) {
        w[p - 1] = 2.0 * wp;
    }
    w
}

/// S = ω_P² Σ_j (e_j − e_{j+1})(e_j − e_{j+1})ᵀ with cyclic j + 1 (RP-5), row-major P × P.
fn reference_spring_matrix(p: usize, beta: f64) -> Vec<f64> {
    let wp2 = reference_omega_p(p, beta).powi(2);
    let mut s = vec![0.0; p * p];
    for j in 0..p {
        let jn = (j + 1) % p;
        let mut v = vec![0.0; p];
        v[j] += 1.0;
        v[jn] -= 1.0;
        for a in 0..p {
            for b in 0..p {
                s[a * p + b] += wp2 * v[a] * v[b];
            }
        }
    }
    s
}

fn random_beads(rng: &mut SplitMix64, p: usize, n: usize) -> Beads {
    Beads::from_vec(p, n, random_vec(rng, p * n * 3, -2.0, 2.0)).unwrap()
}

fn assert_beads_close(a: &Beads, b: &Beads, tol: f64, what: &str) {
    assert_eq!(a.n_beads(), b.n_beads(), "{what}: n_beads");
    assert_eq!(a.n_atoms(), b.n_atoms(), "{what}: n_atoms");
    for (idx, (x, y)) in a.as_slice().iter().zip(b.as_slice()).enumerate() {
        assert!((x - y).abs() <= tol, "{what}: index {idx}: {x} vs {y}");
    }
}

/// RP-4: `matrix()` equals the RP-4 formula elementwise to 1e-14, in the fixed column order
/// (centroid, cos/sin pairs, alternating column for even P), for P ∈ {1,2,3,4,5,8,9,16}.
#[test]
fn matrix_matches_rp4_formula() {
    for p in PS {
        let nm = NormalModes::new(p, 2.5).unwrap();
        let want = reference_matrix(p);
        let got = nm.matrix();
        assert_eq!(got.len(), p * p, "P = {p}");
        for j in 0..p {
            for k in 0..p {
                let (g, w) = (got[j * p + k], want[j * p + k]);
                assert!(
                    (g - w).abs() <= 1e-14,
                    "P = {p}, C[{j}][{k}] = {g}, want {w}"
                );
            }
        }
    }
}

/// RP-4: C is orthogonal, CᵀC = I and CCᵀ = I to 1e-13.
#[test]
fn matrix_is_orthogonal() {
    for p in PS {
        let nm = NormalModes::new(p, 2.5).unwrap();
        let c = nm.matrix();
        for a in 0..p {
            for b in 0..p {
                let ctc: f64 = (0..p).map(|j| c[j * p + a] * c[j * p + b]).sum();
                let cct: f64 = (0..p).map(|k| c[a * p + k] * c[b * p + k]).sum();
                let id = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (ctc - id).abs() <= 1e-13,
                    "P = {p}: (CᵀC)[{a}][{b}] = {ctc}"
                );
                assert!(
                    (cct - id).abs() <= 1e-13,
                    "P = {p}: (CCᵀ)[{a}][{b}] = {cct}"
                );
            }
        }
    }
}

/// RP-2, RP-4: `omega_p()` = P/β, `n_beads()` = P, and `frequencies()` (length P) equal the RP-4
/// formula ω_k = 2 ω_P sin(mπ/P) (and 2 ω_P for the last even-P column) to 1e-14 relative to ω_P.
#[test]
fn frequencies_and_omega_p_match_formula() {
    for beta in BETAS {
        for p in PS {
            let nm = NormalModes::new(p, beta).unwrap();
            let wp = reference_omega_p(p, beta);
            assert_eq!(nm.n_beads(), p);
            assert!(
                ((nm.omega_p() - wp) / wp).abs() <= 1e-15,
                "P = {p}, beta = {beta}: omega_p = {}, want {wp}",
                nm.omega_p()
            );
            let want = reference_frequencies(p, beta);
            let got = nm.frequencies();
            assert_eq!(got.len(), p);
            for k in 0..p {
                assert!(
                    (got[k] - want[k]).abs() <= 1e-14 * wp,
                    "P = {p}, beta = {beta}: omega_{k} = {}, want {}",
                    got[k],
                    want[k]
                );
            }
            assert_eq!(got[0], 0.0, "centroid frequency must be exactly 0");
        }
    }
}

/// RP-5: Cᵀ S C = diag(ω_k²) with S built from its definition and ω_k from `frequencies()`,
/// to 1e-11 relative to ω_P².
#[test]
fn matrix_diagonalises_spring_matrix() {
    for beta in BETAS {
        for p in PS {
            let nm = NormalModes::new(p, beta).unwrap();
            let c = nm.matrix();
            let w = nm.frequencies();
            let s = reference_spring_matrix(p, beta);
            let wp2 = reference_omega_p(p, beta).powi(2);
            for a in 0..p {
                for b in 0..p {
                    let mut v = 0.0;
                    for i in 0..p {
                        for j in 0..p {
                            v += c[i * p + a] * s[i * p + j] * c[j * p + b];
                        }
                    }
                    let want = if a == b { w[a] * w[a] } else { 0.0 };
                    assert!(
                        (v - want).abs() <= 1e-11 * wp2,
                        "P = {p}, beta = {beta}: (CᵀSC)[{a}][{b}] = {v}, want {want}"
                    );
                }
            }
        }
    }
}

/// RP-6: P = 1 gives C = [1.0] and ω = [0.0], and both transforms are the identity.
#[test]
fn single_bead_is_classical() {
    let nm = NormalModes::new(1, 3.0).unwrap();
    assert_eq!(nm.matrix(), &[1.0]);
    assert_eq!(nm.frequencies(), &[0.0]);
    assert_eq!(nm.n_beads(), 1);
    let mut rng = SplitMix64::new(61);
    let q = random_beads(&mut rng, 1, 4);
    assert_beads_close(&nm.to_normal(&q), &q, 1e-15, "to_normal, P = 1");
    assert_beads_close(&nm.to_beads(&q), &q, 1e-15, "to_beads, P = 1");
}

/// RP-1: any P ≥ 1 with positive finite β is accepted.
#[test]
fn new_accepts_valid_arguments() {
    for p in [1, 2, 3, 7, 32, 64] {
        for beta in [1e-6, 1.0, 1e6] {
            assert!(NormalModes::new(p, beta).is_ok(), "P = {p}, beta = {beta}");
        }
    }
}

/// RP-1: P = 0 is `InvalidBeadCount`; β ≤ 0 or non-finite is `InvalidValue`.
#[test]
fn new_rejects_invalid_arguments() {
    let r = NormalModes::new(0, 1.0);
    assert!(
        matches!(r, Err(CoreError::InvalidBeadCount(0))),
        "P = 0: {:?}",
        r.err()
    );
    for beta in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let r = NormalModes::new(4, beta);
        assert!(
            matches!(r, Err(CoreError::InvalidValue { .. })),
            "beta = {beta}: {:?}",
            r.err()
        );
    }
}

/// RP-4, S-1: `to_normal` computes q̃_k = Σ_j C_jk q_j per atom and component (C from the RP-4
/// formula), stored mode-major in the S-1 layout; seeded random beads, N = 3.
#[test]
fn to_normal_matches_explicit_sum() {
    let mut rng = SplitMix64::new(0x4E4D_0001);
    for p in PS {
        let nm = NormalModes::new(p, 2.5).unwrap();
        let c = reference_matrix(p);
        let q = random_beads(&mut rng, p, 3);
        let mut want = Beads::zeros(p, 3);
        for k in 0..p {
            for i in 0..3 {
                let mut v = [0.0; 3];
                for j in 0..p {
                    let qj = q.get(j, i);
                    for d in 0..3 {
                        v[d] += c[j * p + k] * qj[d];
                    }
                }
                want.set(k, i, v);
            }
        }
        assert_beads_close(&nm.to_normal(&q), &want, 1e-13, &format!("P = {p}"));
    }
}

/// RP-4, S-1: `to_beads` computes q_j = Σ_k C_jk q̃_k (the transpose of `to_normal`), C from the
/// RP-4 formula; seeded random modes, N = 3.
#[test]
fn to_beads_matches_explicit_sum() {
    let mut rng = SplitMix64::new(0x4E4D_0002);
    for p in PS {
        let nm = NormalModes::new(p, 2.5).unwrap();
        let c = reference_matrix(p);
        let qt = random_beads(&mut rng, p, 3);
        let mut want = Beads::zeros(p, 3);
        for j in 0..p {
            for i in 0..3 {
                let mut v = [0.0; 3];
                for k in 0..p {
                    let qk = qt.get(k, i);
                    for d in 0..3 {
                        v[d] += c[j * p + k] * qk[d];
                    }
                }
                want.set(j, i, v);
            }
        }
        assert_beads_close(&nm.to_beads(&qt), &want, 1e-13, &format!("P = {p}"));
    }
}

/// RP-4: round trips `to_beads(to_normal(x)) = x` and `to_normal(to_beads(x)) = x` to 1e-13 on
/// seeded random beads, P ∈ {1, 2, 3, 8}, N = 3.
#[test]
fn round_trip_is_identity() {
    let mut rng = SplitMix64::new(0x4E4D_0003);
    for p in [1, 2, 3, 8] {
        let nm = NormalModes::new(p, 0.7).unwrap();
        let x = random_beads(&mut rng, p, 3);
        let back = nm.to_beads(&nm.to_normal(&x));
        assert_beads_close(&back, &x, 1e-13, &format!("beads round trip, P = {p}"));
        let back = nm.to_normal(&nm.to_beads(&x));
        assert_beads_close(&back, &x, 1e-13, &format!("modes round trip, P = {p}"));
    }
}

/// RP-4, S-1: the k = 0 mode is q̃_0 = √P · q̄, with q̄ from `Beads::centroid` and from a mean
/// computed here, to 1e-13.
#[test]
fn centroid_mode_is_sqrt_p_times_centroid() {
    let mut rng = SplitMix64::new(0x4E4D_0004);
    for p in PS {
        let n = 3;
        let nm = NormalModes::new(p, 2.5).unwrap();
        let q = random_beads(&mut rng, p, n);
        let qt = nm.to_normal(&q);
        let cen = q.centroid();
        assert_eq!(cen.len(), n);
        for (i, ci) in cen.iter().enumerate() {
            let mut mean = [0.0; 3];
            for j in 0..p {
                let qj = q.get(j, i);
                for d in 0..3 {
                    mean[d] += qj[d] / p as f64;
                }
            }
            let q0 = qt.get(0, i);
            for d in 0..3 {
                assert!((ci[d] - mean[d]).abs() <= 1e-14, "P = {p}: centroid");
                let want = (p as f64).sqrt() * mean[d];
                assert!(
                    (q0[d] - want).abs() <= 1e-13,
                    "P = {p}, atom {i}, d {d}: q~_0 = {}, want {want}",
                    q0[d]
                );
            }
        }
    }
}

/// RP-5: Σ_j Σ_i ½ m_i ω_P² |q_j − q_{j+1}|² (cyclic, ω_P = P/β computed here) equals
/// Σ_k Σ_i ½ m_i ω_k² |q̃_k|² to 1e-12 relative, seeded random beads and masses, N = 3.
#[test]
fn spring_energy_is_invariant() {
    let mut rng = SplitMix64::new(0x4E4D_0005);
    let masses = [1.0, 1837.0, 7.5];
    for beta in BETAS {
        for p in PS {
            let nm = NormalModes::new(p, beta).unwrap();
            let q = random_beads(&mut rng, p, 3);
            let wp = reference_omega_p(p, beta);
            let mut lhs = 0.0;
            for j in 0..p {
                let jn = (j + 1) % p;
                for (i, &m) in masses.iter().enumerate() {
                    let (a, b) = (q.get(j, i), q.get(jn, i));
                    let d2: f64 = (0..3).map(|d| (a[d] - b[d]).powi(2)).sum();
                    lhs += 0.5 * m * wp * wp * d2;
                }
            }
            let qt = nm.to_normal(&q);
            let w = nm.frequencies();
            let mut rhs = 0.0;
            for (k, &wk) in w.iter().enumerate() {
                for (i, &m) in masses.iter().enumerate() {
                    let v = qt.get(k, i);
                    rhs += 0.5 * m * wk * wk * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                }
            }
            let scale = lhs.abs().max(1e-300);
            assert!(
                (lhs - rhs).abs() <= 1e-12 * scale || (p == 1 && lhs == 0.0 && rhs == 0.0),
                "P = {p}, beta = {beta}: springs {lhs} vs modes {rhs}"
            );
        }
    }
}
