//! Tests for `RingPolymerIntegrator::thermostat_step` (SPEC T-1, T-2, T-3, RP-3, D-2, D-4).
//!
//! Expectations come from the T-1 formula with c_k from T-2, evaluated in the test's own RP-4
//! basis (`common::ref_o_step`), and drawing from a second `pimd_core::Rng` with the same seed.

mod common;

use common::{SplitMix64, max_rel_diff, random_beads, ref_c, ref_matrix, ref_o_step, to_modes};
use pimd_core::{Beads, Rng};
use pimd_integrators::{RingPolymerIntegrator, RingState, Thermostat};

const MASSES: [f64; 3] = [1.0, 4.0, 1836.0];

fn random_state(rng: &mut SplitMix64, p: usize) -> RingState {
    RingState {
        q: random_beads(rng, p, MASSES.len(), -1.0, 1.0),
        p: random_beads(rng, p, MASSES.len(), -3.0, 3.0),
        masses: MASSES.to_vec(),
    }
}

/// T-3, D-4: `Thermostat::Off` is the identity (bitwise) and draws no random numbers, so the
/// caller's stream is untouched (the next draw equals the first draw of a fresh same-seed Rng).
#[test]
fn off_is_identity_and_draws_nothing() {
    let mut srng = SplitMix64(1);
    for p in [1, 3, 4] {
        let ig = RingPolymerIntegrator::new(0.2, 1.5, p, Thermostat::Off).unwrap();
        let s0 = random_state(&mut srng, p);
        let mut s = s0.clone();
        let mut rng = Rng::from_seed(77);
        ig.thermostat_step(&mut s, &mut rng);
        assert_eq!(s, s0, "P = {p}: Off must not change the state");
        let mut fresh = Rng::from_seed(77);
        assert_eq!(
            rng.normal(),
            fresh.normal(),
            "P = {p}: Off consumed random numbers"
        );
    }
}

/// T-1, T-2, D-2, D-4: draw-order contract. One O step equals the test's own T-1 formula with
/// c_k = exp(−γ_k dt), γ_0 = 1/τ_0 (or 0), γ_k = 2ω_k, noise variance (1 − c_k²) m_i P/β, drawing
/// one normal per (k, i, d) from a same-seed `Rng` with k outermost (RP-4 column order), then
/// atom i, then dimension d. Agreement to 1e-12 (relative, floor 1), and both streams are at the
/// same position afterwards (exactly P·N·3 draws). Covers P = 1 (Langevin on the single bead),
/// odd and even P, and τ_0 = None (γ_0 = 0, a draw is still consumed for k = 0).
#[test]
fn draw_order_matches_t1() {
    let mut srng = SplitMix64(2);
    let (beta, dt) = (1.3, 0.3);
    for p in [1, 3, 4, 6] {
        for tau0 in [Some(2.0), Some(0.05), None] {
            let ig = RingPolymerIntegrator::new(dt, beta, p, Thermostat::PileL { tau0 }).unwrap();
            let s0 = random_state(&mut srng, p);
            let seed = 1000 + p as u64;
            let mut rng = Rng::from_seed(seed);
            let mut s = s0.clone();
            ig.thermostat_step(&mut s, &mut rng);
            let mut rng_ref = Rng::from_seed(seed);
            let p_want = ref_o_step(&s0.p, &MASSES, beta, dt, tau0, &mut rng_ref);
            let d = max_rel_diff(&s.p, &p_want, 1.0);
            assert!(d <= 1e-12, "P = {p}, tau0 = {tau0:?}: rel diff {d:e}");
            assert_eq!(s.q, s0.q, "the O step must not touch positions");
            assert_eq!(s.masses, s0.masses);
            assert_eq!(
                rng.normal(),
                rng_ref.normal(),
                "P = {p}: wrong number of draws"
            );
        }
    }
}

/// T-2: with τ_0 = None the centroid momentum Σ_j p_j is unchanged (γ_0 = 0, c_0 = 1, zero
/// noise amplitude), to rounding (1e-12 relative, floor 1), while the internal modes do change.
#[test]
fn no_tau0_leaves_centroid_momentum() {
    let mut srng = SplitMix64(3);
    for p in [1, 2, 5, 8] {
        let ig = RingPolymerIntegrator::new(0.1, 2.0, p, Thermostat::PileL { tau0: None }).unwrap();
        let s0 = random_state(&mut srng, p);
        let mut s = s0.clone();
        let mut rng = Rng::from_seed(9);
        ig.thermostat_step(&mut s, &mut rng);
        for i in 0..MASSES.len() {
            for d in 0..3 {
                let a: f64 = (0..p).map(|j| s0.p.get(j, i)[d]).sum();
                let b: f64 = (0..p).map(|j| s.p.get(j, i)[d]).sum();
                assert!(
                    (a - b).abs() <= 1e-12 * a.abs().max(1.0),
                    "P = {p}: {a} -> {b}"
                );
            }
        }
        if p > 1 {
            assert!(
                max_rel_diff(&s.p, &s0.p, 1.0) > 1e-3,
                "P = {p}: internal modes unchanged"
            );
        }
    }
}

/// T-1, T-2, RP-3: one O step from p = 0, N = 4000 atoms (masses 1..5 m_e), P = 4, β = 4,
/// dt = 0.05, for τ_0 ∈ {0.5, 5, 50, None}. In mode k the new p̃_k/√m_i are iid
/// N(0, (1 − c_k²) P/β) over the n = 3N = 12000 (atom, dimension) samples, c_k from T-2:
/// ω_P = 1, ω_1 = ω_2 = √2, ω_3 = 2 (RP-4), so γ_k dt = 0.141, 0.141, 0.2 for k ≥ 1, and
/// γ_0 dt = 0.1, 0.01, 0.001, 0. The sample variance s² (mean subtracted) is compared with
/// that value within 5σ, σ = var·√(2/(n−1)) ≈ 1.3 % of var (normal sampling theory).
/// For τ_0 = None the centroid stays exactly 0 up to rounding (< 1e-12).
#[test]
fn one_step_variance_from_rest() {
    let (p, n, beta, dt) = (4usize, 4000usize, 4.0, 0.05);
    let masses: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64).collect();
    let c = ref_matrix(p);
    for (seed, tau0) in [
        (11u64, Some(0.5)),
        (12, Some(5.0)),
        (13, Some(50.0)),
        (14, None),
    ] {
        let ig = RingPolymerIntegrator::new(dt, beta, p, Thermostat::PileL { tau0 }).unwrap();
        let mut s = RingState {
            q: Beads::zeros(p, n),
            p: Beads::zeros(p, n),
            masses: masses.clone(),
        };
        let mut rng = Rng::from_seed(seed);
        ig.thermostat_step(&mut s, &mut rng);
        let pt = to_modes(&c, &s.p);
        let ck = ref_c(p, beta, dt, tau0);
        for (k, &ckk) in ck.iter().enumerate() {
            let want = (1.0 - ckk * ckk) * p as f64 / beta;
            let xs: Vec<f64> = (0..n)
                .flat_map(|i| {
                    let m = masses[i];
                    pt.get(k, i).map(|x| x / m.sqrt())
                })
                .collect();
            let ns = xs.len() as f64;
            let mean = xs.iter().sum::<f64>() / ns;
            let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (ns - 1.0);
            if want == 0.0 {
                assert!(var < 1e-24, "tau0 None: centroid variance {var:e}");
                continue;
            }
            let sigma = want * (2.0 / (ns - 1.0)).sqrt();
            assert!(
                (var - want).abs() <= 5.0 * sigma,
                "tau0 = {tau0:?}, mode {k}: variance {var} vs {want} (sigma {sigma})"
            );
        }
    }
}
