//! Tests for `CentroidCorrelation`, `centroid_velocity` and `green_kubo` (SPEC Y-3, Y-4, Y-5;
//! MILESTONE_7 required tests 6 and 7).
//!
//! Every expected number is a hand computation written out next to the test (small integer
//! sequences, so the sums are exact), or the closed-form integral of cos(ωt) with the trapezoid
//! error bound, or the Y-4 formula evaluated by brute force in the test.
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

use pimd_core::Beads;
use pimd_estimators::{CentroidCorrelation, EstimatorError, centroid_velocity, green_kubo};

type Sample = Vec<[f64; 3]>;

/// Segment A (N = 2, L = 4). Hand computation of Y-4 with n_lags = 3, x_i(s):
///   s = 0: (1,0,0), (0,2,0);  s = 1: (2,1,0), (0,1,1);
///   s = 2: (0,0,3), (1,1,1);  s = 3: (1,1,1), (2,0,0).
/// τ = 0: Σ_s Σ_i |x_i(s)|² = (1+4) + (5+2) + (9+3) + (3+4) = 31, /(N·L) = /8 → 31/8.
/// τ = 1: (2+2) + (0+2) + (3+2) = 11, /(N·(L−1)) = /6 → 11/6.
/// τ = 2: (0+2) + (3+0) = 5, /(N·(L−2)) = /4 → 5/4.
fn seg_a() -> Vec<Sample> {
    vec![
        vec![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        vec![[2.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        vec![[0.0, 0.0, 3.0], [1.0, 1.0, 1.0]],
        vec![[1.0, 1.0, 1.0], [2.0, 0.0, 0.0]],
    ]
}
const C_A: [f64; 3] = [31.0 / 8.0, 11.0 / 6.0, 5.0 / 4.0];

/// Segment B (N = 2, L = 3 = n_lags):
///   s = 0: (1,1,0), (0,0,1);  s = 1: (0,1,0), (1,0,1);  s = 2: (2,0,0), (0,0,−1).
/// τ = 0: (2+1) + (1+2) + (4+1) = 11, /6 → 11/6.
/// τ = 1: (1+1) + (0−1) = 1, /4 → 1/4.
/// τ = 2: (2−1) = 1, /2 → 1/2.
fn seg_b() -> Vec<Sample> {
    vec![
        vec![[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        vec![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
        vec![[2.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
    ]
}
const C_B: [f64; 3] = [11.0 / 6.0, 1.0 / 4.0, 1.0 / 2.0];

/// Segment C (N = 2, L = 4): atom 0 at (s, 0, 0), atom 1 at (0, 1, 0) for s = 0 … 3.
/// τ = 0: Σ (s² + 1) = 14 + 4 = 18, /8 → 9/4.
/// τ = 1: Σ_{s<3} (s(s+1) + 1) = 8 + 3 = 11, /6 → 11/6.
/// τ = 2: Σ_{s<2} (s(s+2) + 1) = 3 + 2 = 5, /4 → 5/4.
fn seg_c() -> Vec<Sample> {
    (0..4)
        .map(|s| vec![[s as f64, 0.0, 0.0], [0.0, 1.0, 0.0]])
        .collect()
}
const C_C: [f64; 3] = [9.0 / 4.0, 11.0 / 6.0, 5.0 / 4.0];

fn push_all(c: &mut CentroidCorrelation, seg: &[Sample]) {
    for x in seg {
        c.push(x).unwrap();
    }
}

fn close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * y.abs().max(1.0),
            "{what}[{k}]: {x} vs {y}"
        );
    }
}

/// Sample mean and σ of the mean (n − 1 denominator, /√n) of `rows`, per column (Y-4).
fn mean_sem(rows: &[[f64; 3]]) -> (Vec<f64>, Vec<f64>) {
    let n = rows.len() as f64;
    let mean: Vec<f64> = (0..3)
        .map(|k| rows.iter().map(|r| r[k]).sum::<f64>() / n)
        .collect();
    let sem = (0..3)
        .map(|k| {
            let ss: f64 = rows.iter().map(|r| (r[k] - mean[k]).powi(2)).sum();
            (ss / (n - 1.0)).sqrt() / n.sqrt()
        })
        .collect();
    (mean, sem)
}

/// Y-4: C_seg of one segment equals the hand values (including the 1/(N·(L − τ)) normalisation
/// and every s as a time origin); `mean` is C_seg after one segment, `sem` is NaN with fewer than
/// two segments, both have length n_lags; `n_segments` counts closed segments.
#[test]
fn one_segment_hand_values() {
    let mut c = CentroidCorrelation::new(3, 2).unwrap();
    assert_eq!(c.n_segments(), 0);
    push_all(&mut c, &seg_a());
    c.end_segment().unwrap();
    assert_eq!(c.n_segments(), 1);
    close(&c.mean(), &C_A, 1e-14, "mean after segment A");
    let sem = c.sem();
    assert_eq!(sem.len(), 3);
    assert!(
        sem.iter().all(|v| v.is_nan()),
        "sem with one segment: {sem:?}"
    );
}

/// Y-4: before any segment the mean and sem are NaN (length n_lags); pending samples do not
/// count as a segment.
#[test]
fn empty_accumulator_is_nan() {
    let mut c = CentroidCorrelation::new(3, 2).unwrap();
    let (m, s) = (c.mean(), c.sem());
    assert_eq!((m.len(), s.len()), (3, 3));
    assert!(m.iter().chain(&s).all(|v| v.is_nan()));
    c.push(&seg_a()[0]).unwrap();
    assert_eq!(c.n_segments(), 0);
    assert!(c.mean().iter().all(|v| v.is_nan()));
}

/// Y-4: after segments A, B, C (L = 4, 3, 4) the mean is the equal-weight mean of the three hand
/// C_seg, (191/72, 47/36, 1), and the sem is their sample standard deviation (n − 1) over √3,
/// computed here from the hand values.
#[test]
fn three_segments_mean_and_sem() {
    let mut c = CentroidCorrelation::new(3, 2).unwrap();
    for seg in [seg_a(), seg_b(), seg_c()] {
        push_all(&mut c, &seg);
        c.end_segment().unwrap();
    }
    assert_eq!(c.n_segments(), 3);
    close(&c.mean(), &[191.0 / 72.0, 47.0 / 36.0, 1.0], 1e-14, "mean");
    let (m, s) = mean_sem(&[C_A, C_B, C_C]);
    close(&c.mean(), &m, 1e-14, "mean vs hand mean");
    close(&c.sem(), &s, 1e-12, "sem");
}

/// Y-4 on random data: N = 5, L = 41, n_lags = 17, two segments of different random sequences;
/// the mean equals the brute-force Y-4 formula written here (1e-12 relative).
#[test]
fn random_segments_match_brute_force() {
    let mut state = 12345u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let (n, l, n_lags) = (5, 41, 17);
    let mut c = CentroidCorrelation::new(n_lags, n).unwrap();
    let mut want = vec![0.0; n_lags];
    for _ in 0..2 {
        let seg: Vec<Sample> = (0..l)
            .map(|_| (0..n).map(|_| [next(), next(), next()]).collect())
            .collect();
        for tau in 0..n_lags {
            let mut s = 0.0;
            for t in 0..l - tau {
                for i in 0..n {
                    for d in 0..3 {
                        s += seg[t][i][d] * seg[t + tau][i][d];
                    }
                }
            }
            want[tau] += 0.5 * s / (n * (l - tau)) as f64;
        }
        push_all(&mut c, &seg);
        c.end_segment().unwrap();
    }
    close(&c.mean(), &want, 1e-12, "random mean");
}

/// Y-4 errors: `new` with n_lags = 0 or n_atoms = 0 is `InvalidValue`; `push` with the wrong
/// atom count is `ShapeMismatch` and changes nothing; `end_segment` with fewer than n_lags
/// samples is `InvalidValue` and changes nothing (the pending samples are kept: completing the
/// segment afterwards gives the hand values of segment A).
#[test]
fn errors_leave_state_unchanged() {
    assert!(matches!(
        CentroidCorrelation::new(0, 2),
        Err(EstimatorError::InvalidValue { .. })
    ));
    assert!(matches!(
        CentroidCorrelation::new(3, 0),
        Err(EstimatorError::InvalidValue { .. })
    ));
    let mut c = CentroidCorrelation::new(3, 2).unwrap();
    let a = seg_a();
    c.push(&a[0]).unwrap();
    let before = c.clone();
    assert!(matches!(
        c.push(&[[1.0, 2.0, 3.0]]),
        Err(EstimatorError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        c.push(&[[1.0; 3], [2.0; 3], [3.0; 3]]),
        Err(EstimatorError::ShapeMismatch { .. })
    ));
    assert_eq!(c, before, "a rejected push changed the accumulator");
    c.push(&a[1]).unwrap();
    let before = c.clone();
    assert!(matches!(
        c.end_segment(),
        Err(EstimatorError::InvalidValue { .. })
    ));
    assert_eq!(c, before, "a rejected end_segment changed the accumulator");
    assert_eq!(c.n_segments(), 0);
    c.push(&a[2]).unwrap();
    c.push(&a[3]).unwrap();
    c.end_segment().unwrap();
    close(&c.mean(), &C_A, 1e-14, "segment A after the rejected calls");
    // An empty segment is also too short.
    assert!(matches!(
        c.end_segment(),
        Err(EstimatorError::InvalidValue { .. })
    ));
    assert_eq!(c.n_segments(), 1);
}

/// Y-4 (accumulator serialisation): `to_bytes` is refused with samples pending (`InvalidValue`);
/// between segments `from_bytes(to_bytes())` restores an equal accumulator (same mean and sem
/// bitwise), and continuing both with segment C gives bitwise identical results.
#[test]
fn bytes_round_trip() {
    let mut c = CentroidCorrelation::new(3, 2).unwrap();
    push_all(&mut c, &seg_a());
    c.end_segment().unwrap();
    push_all(&mut c, &seg_b());
    c.end_segment().unwrap();
    let bytes = c.to_bytes().unwrap();
    let mut r = CentroidCorrelation::from_bytes(&bytes).unwrap();
    assert_eq!(r, c);
    assert_eq!(r.n_segments(), 2);
    let bits = |v: Vec<f64>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(r.mean()), bits(c.mean()));
    assert_eq!(bits(r.sem()), bits(c.sem()));
    for x in seg_c() {
        c.push(&x).unwrap();
        r.push(&x).unwrap();
    }
    assert!(matches!(
        c.to_bytes(),
        Err(EstimatorError::InvalidValue { .. })
    ));
    c.end_segment().unwrap();
    r.end_segment().unwrap();
    assert_eq!(bits(r.mean()), bits(c.mean()));
    assert_eq!(bits(r.sem()), bits(c.sem()));
    // An empty accumulator round-trips too.
    let e = CentroidCorrelation::new(5, 7).unwrap();
    assert_eq!(
        CentroidCorrelation::from_bytes(&e.to_bytes().unwrap()).unwrap(),
        e
    );
}

/// Y-4 (malformed buffers): every proper prefix of a valid buffer (including the empty one) and
/// the buffer with one extra trailing byte are `InvalidValue`.
#[test]
fn truncated_or_padded_bytes_rejected() {
    let mut c = CentroidCorrelation::new(3, 2).unwrap();
    push_all(&mut c, &seg_a());
    c.end_segment().unwrap();
    let bytes = c.to_bytes().unwrap();
    for k in 0..bytes.len() {
        assert!(
            matches!(
                CentroidCorrelation::from_bytes(&bytes[..k]),
                Err(EstimatorError::InvalidValue { .. })
            ),
            "prefix of {k} bytes accepted"
        );
    }
    let mut long = bytes.clone();
    long.push(0);
    assert!(matches!(
        CentroidCorrelation::from_bytes(&long),
        Err(EstimatorError::InvalidValue { .. })
    ));
}

/// Y-3: v̄_i = (1/P) Σ_j p_{j,i}/m_i. Hand values, P = 2, N = 2, m = (2, 4):
/// p_{0,0} = (2,4,6), p_{1,0} = (6,0,2) → v̄_0 = ((8, 4, 8)/2)/2 = (2, 1, 2);
/// p_{0,1} = (4,0,8), p_{1,1} = (0,8,−4) → v̄_1 = ((4, 8, 4)/4)/2 = (0.5, 1, 0.5).
/// (Bead 0 alone would give (1, 2, 3); no 1/m would give (4, 2, 4).)
#[test]
fn centroid_velocity_hand_values() {
    let mut p = Beads::zeros(2, 2);
    p.set(0, 0, [2.0, 4.0, 6.0]);
    p.set(1, 0, [6.0, 0.0, 2.0]);
    p.set(0, 1, [4.0, 0.0, 8.0]);
    p.set(1, 1, [0.0, 8.0, -4.0]);
    let v = centroid_velocity(&p, &[2.0, 4.0]);
    assert_eq!(v, vec![[2.0, 1.0, 2.0], [0.5, 1.0, 0.5]]);
    // P = 1: v = p/m.
    let mut p1 = Beads::zeros(1, 1);
    p1.set(0, 0, [3.0, -6.0, 9.0]);
    assert_eq!(centroid_velocity(&p1, &[3.0]), vec![[1.0, -2.0, 3.0]]);
}

/// Y-3: a mass list whose length differs from the atom count panics.
#[test]
#[should_panic]
fn centroid_velocity_mass_count_panics() {
    let p = Beads::zeros(2, 2);
    let _ = centroid_velocity(&p, &[1.0]);
}

/// Y-5 hand values: C = (3, 1, −1, 2), Δ = 0.5:
/// D(0) = 0; D(1) = (1/3)·0.5·(3 + 1)/2 = 1/3; D(2) = 1/3 + (1/3)·0.5·(1 − 1)/2 = 1/3;
/// D(3) = 1/3 + (1/3)·0.5·(−1 + 2)/2 = 5/12. (Rectangle rule: D(1) = 1/2; no 1/3: D(1) = 1.)
/// Lengths: one lag gives [0], none gives [].
#[test]
fn green_kubo_hand_values() {
    let d = green_kubo(&[3.0, 1.0, -1.0, 2.0], 0.5);
    close(&d, &[0.0, 1.0 / 3.0, 1.0 / 3.0, 5.0 / 12.0], 1e-15, "D");
    assert_eq!(d[0], 0.0);
    assert_eq!(green_kubo(&[7.0], 0.1), vec![0.0]);
    assert!(green_kubo(&[], 0.1).is_empty());
}

/// Y-5: for C(t) = cos(ωt) (ω = 2, Δ = 0.01, 500 lags) the running integral is
/// D(t) = sin(ωt)/(3ω); the composite trapezoid rule's error is bounded by
/// t·Δ²·max|C''|/12 /3 = t Δ² ω²/36 (standard bound), checked at every lag, and D(0) = 0.
#[test]
fn green_kubo_trapezoid_on_cosine() {
    let (w, h, n) = (2.0f64, 0.01f64, 500);
    let c: Vec<f64> = (0..n).map(|k| (w * k as f64 * h).cos()).collect();
    let d = green_kubo(&c, h);
    assert_eq!(d.len(), n);
    assert_eq!(d[0], 0.0);
    for (k, &dk) in d.iter().enumerate() {
        let t = k as f64 * h;
        let exact = (w * t).sin() / (3.0 * w);
        let bound = t * h * h * w * w / 36.0 + 1e-15;
        assert!(
            (dk - exact).abs() <= bound,
            "lag {k}: {dk} vs {exact} (bound {bound:e})"
        );
    }
    // The rectangle rule would be off by ~Δ/6 per unit of C: far outside the bound.
    let rect: f64 = c[..n - 1].iter().sum::<f64>() * h / 3.0;
    let exact = (w * (n - 1) as f64 * h).sin() / (3.0 * w);
    assert!((rect - exact).abs() > 10.0 * (d[n - 1] - exact).abs());
}
