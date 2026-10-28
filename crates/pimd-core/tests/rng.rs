//! Tests for `pimd_core::Rng` (SPEC D-2). Statistical checks use a fixed seed and 5σ bounds
//! from the exact moments of the target distribution; draws are independent, so the naive
//! standard error is the right one here (no autocorrelation, unlike MD samples).

use pimd_core::Rng;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

const N: usize = 200_000;

/// D-2: the same seed gives bitwise-identical interleaved normal/uniform sequences.
#[test]
fn same_seed_same_sequence() {
    let mut a = Rng::from_seed(12_345);
    let mut b = Rng::from_seed(12_345);
    for n in 0..2_000 {
        if n % 3 == 0 {
            assert_eq!(a.uniform().to_bits(), b.uniform().to_bits(), "draw {n}");
        } else {
            assert_eq!(a.normal().to_bits(), b.normal().to_bits(), "draw {n}");
        }
    }
}

/// D-2: different seeds give different sequences (no two of these share their first 100 normals).
#[test]
fn different_seeds_differ() {
    let seeds = [0u64, 1, 2, 42, u64::MAX];
    let seqs: Vec<Vec<u64>> = seeds
        .iter()
        .map(|&s| {
            let mut r = Rng::from_seed(s);
            (0..100).map(|_| r.normal().to_bits()).collect()
        })
        .collect();
    for a in 0..seqs.len() {
        for b in a + 1..seqs.len() {
            assert_ne!(seqs[a], seqs[b], "seeds {} and {}", seeds[a], seeds[b]);
        }
    }
    let mut u1 = Rng::from_seed(7);
    let mut u2 = Rng::from_seed(8);
    let s1: Vec<u64> = (0..100).map(|_| u1.uniform().to_bits()).collect();
    let s2: Vec<u64> = (0..100).map(|_| u2.uniform().to_bits()).collect();
    assert_ne!(s1, s2);
}

/// D-2: `normal()` is `rand_distr::StandardNormal` sampled from `ChaCha8Rng::seed_from_u64(seed)`,
/// draw for draw. The reference stream is built here directly from the generators SPEC D-2 names
/// (not golden values), so the draw-order contract (T-1) is reproducible.
#[test]
fn normal_is_standard_normal_on_chacha8_seed_from_u64() {
    for seed in [0u64, 1, 42, 20_261_028, u64::MAX] {
        let mut ours = Rng::from_seed(seed);
        let mut reference = ChaCha8Rng::seed_from_u64(seed);
        for n in 0..1_000 {
            let want: f64 = StandardNormal.sample(&mut reference);
            assert_eq!(
                ours.normal().to_bits(),
                want.to_bits(),
                "seed {seed}, draw {n}"
            );
        }
    }
}

/// D-2: 200 000 normals (seed 20261028) have mean within 5/√n of 0, variance within 5√(2/n)
/// of 1, and P(|x| < 1) within 5σ of erf(1/√2) = 0.682689492137086 (σ = √(p(1−p)/n)).
#[test]
fn normal_moments() {
    let mut r = Rng::from_seed(20_261_028);
    let x: Vec<f64> = (0..N).map(|_| r.normal()).collect();
    let n = N as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    assert!(x.iter().all(|v| v.is_finite()));
    assert!(mean.abs() <= 5.0 / n.sqrt(), "mean = {mean}");
    assert!(
        (var - 1.0).abs() <= 5.0 * (2.0 / n).sqrt(),
        "variance = {var}"
    );
    // P(|X| < 1) for X ~ N(0,1) is erf(1/√2) (standard table value).
    let p = 0.682_689_492_137_086;
    let frac = x.iter().filter(|v| v.abs() < 1.0).count() as f64 / n;
    assert!(
        (frac - p).abs() <= 5.0 * (p * (1.0 - p) / n).sqrt(),
        "P(|x|<1) = {frac}"
    );
}

/// D-2: 200 000 uniforms (seed 20261028) lie in [0, 1); mean within 5σ of 1/2 (σ² = 1/(12n));
/// variance within 5σ of 1/12 (σ² = (1/80 − 1/144)/n = 1/(180n), exact moments of U(0,1)).
#[test]
fn uniform_range_and_moments() {
    let mut r = Rng::from_seed(20_261_028);
    let x: Vec<f64> = (0..N).map(|_| r.uniform()).collect();
    assert!(
        x.iter().all(|&v| (0.0..1.0).contains(&v)),
        "value outside [0, 1)"
    );
    let n = N as f64;
    let mean = x.iter().sum::<f64>() / n;
    assert!(
        (mean - 0.5).abs() <= 5.0 * (1.0 / (12.0 * n)).sqrt(),
        "mean = {mean}"
    );
    let var = x.iter().map(|v| (v - 0.5) * (v - 0.5)).sum::<f64>() / n;
    assert!(
        (var - 1.0 / 12.0).abs() <= 5.0 * (1.0 / (180.0 * n)).sqrt(),
        "variance = {var}"
    );
}
