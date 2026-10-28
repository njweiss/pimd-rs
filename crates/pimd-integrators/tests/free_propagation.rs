//! Tests for `RingPolymerIntegrator::free_propagate` (SPEC I-2, RP-4, RP-5, RP-6).
//!
//! Expectations come from the I-2 per-mode formula applied in the test's own RP-4 basis
//! (`common::ref_free_propagate`), and from exact properties of the free ring polymer.
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

mod common;

use common::{SplitMix64, max_rel_diff, random_beads, ref_free_propagate};
use pimd_core::Beads;
use pimd_integrators::{RingPolymerIntegrator, RingState, Thermostat};

const PS: [usize; 5] = [1, 2, 3, 4, 8];
const BETA: f64 = 1.7;
const MASSES: [f64; 3] = [1.0, 7.5, 1836.0];

fn state(rng: &mut SplitMix64, p: usize) -> RingState {
    let q = random_beads(rng, p, MASSES.len(), -1.5, 1.5);
    // Momenta of order sqrt(m P / β) per component, so q and p/(m ω) are comparable.
    let mut pm = random_beads(rng, p, MASSES.len(), -1.0, 1.0);
    for j in 0..p {
        for (i, &m) in MASSES.iter().enumerate() {
            let s = (m * p as f64 / BETA).sqrt();
            pm.set(j, i, pm.get(j, i).map(|x| s * x));
        }
    }
    RingState {
        q,
        p: pm,
        masses: MASSES.to_vec(),
    }
}

fn integrator(p: usize) -> RingPolymerIntegrator {
    RingPolymerIntegrator::new(0.1, BETA, p, Thermostat::Off).unwrap()
}

/// Free ring Hamiltonian Σ p²/(2m) + Σ_j ½ m ω_P² |q_j − q_{j+1}|² (RP-3 with V = 0).
fn free_h(s: &RingState) -> f64 {
    let v = vec![0.0; s.q.n_beads()];
    common::ring_hamiltonian(&s.q, &s.p, &s.masses, BETA, &v)
}

/// I-2, RP-4: for P ∈ {1, 2, 3, 4, 8} and t of both signs (including large t, several periods of
/// the stiffest mode), positions and momenta equal the per-mode I-2 formula evaluated in the
/// test's own RP-4 basis, to 1e-12 relative (floor 1 for q, floor sqrt(mP/β) scale ~1 for p).
#[test]
fn matches_per_mode_formula() {
    let mut rng = SplitMix64(101);
    for p in PS {
        let ig = integrator(p);
        for t in [0.37, -1.2, 5.0, 0.0] {
            let s0 = state(&mut rng, p);
            let (q_want, p_want) = ref_free_propagate(&s0.q, &s0.p, &s0.masses, BETA, t);
            let mut s = s0.clone();
            ig.free_propagate(&mut s, t);
            assert_eq!(s.masses, s0.masses, "masses must not change");
            let dq = max_rel_diff(&s.q, &q_want, 1.0);
            let dp = max_rel_diff(&s.p, &p_want, 1.0);
            assert!(dq <= 1e-12, "P = {p}, t = {t}: q rel diff {dq:e}");
            assert!(dp <= 1e-12, "P = {p}, t = {t}: p rel diff {dp:e}");
        }
    }
}

/// I-2: the free propagator is a flow: propagating t₁ then t₂ equals t₁ + t₂, to 1e-12.
#[test]
fn composition() {
    let mut rng = SplitMix64(202);
    for p in PS {
        let ig = integrator(p);
        for (t1, t2) in [(0.3, 0.45), (1.1, -2.5), (-0.2, -0.7)] {
            let s0 = state(&mut rng, p);
            let mut a = s0.clone();
            ig.free_propagate(&mut a, t1);
            ig.free_propagate(&mut a, t2);
            let mut b = s0.clone();
            ig.free_propagate(&mut b, t1 + t2);
            assert!(max_rel_diff(&a.q, &b.q, 1.0) <= 1e-12, "P = {p}, q");
            assert!(max_rel_diff(&a.p, &b.p, 1.0) <= 1e-12, "P = {p}, p");
        }
    }
}

/// I-2: time reversal: t then −t returns to the start, to 1e-12.
#[test]
fn reversal() {
    let mut rng = SplitMix64(303);
    for p in PS {
        let ig = integrator(p);
        for t in [0.25, 3.7, -1.9] {
            let s0 = state(&mut rng, p);
            let mut s = s0.clone();
            ig.free_propagate(&mut s, t);
            ig.free_propagate(&mut s, -t);
            assert!(
                max_rel_diff(&s.q, &s0.q, 1.0) <= 1e-12,
                "P = {p}, t = {t}, q"
            );
            assert!(
                max_rel_diff(&s.p, &s0.p, 1.0) <= 1e-12,
                "P = {p}, t = {t}, p"
            );
        }
    }
}

/// I-2, RP-3, RP-5: the free ring Hamiltonian Σ p²/2m + springs is conserved exactly by the
/// exact propagator (1e-12 relative), for short and long times. The test also checks that the
/// state actually moved, so an identity propagator cannot pass.
#[test]
fn conserves_free_ring_hamiltonian() {
    let mut rng = SplitMix64(404);
    for p in PS {
        let ig = integrator(p);
        for t in [0.1, 2.3, 40.0] {
            let s0 = state(&mut rng, p);
            let h0 = free_h(&s0);
            let mut s = s0.clone();
            ig.free_propagate(&mut s, t);
            let h = free_h(&s);
            assert!(
                (h - h0).abs() <= 1e-12 * h0.abs(),
                "P = {p}, t = {t}: H {h0} -> {h}"
            );
            assert!(
                max_rel_diff(&s.q, &s0.q, 1.0) > 1e-6,
                "P = {p}, t = {t}: q did not move"
            );
        }
    }
}

/// I-2 (ω_0 = 0 branch), RP-4: the centroid moves uniformly. With q̄ = (1/P) Σ_j q_j and
/// Π = Σ_j p_j, q̄(t) = q̄(0) + t Π/(P m) and Π(t) = Π(0), to 1e-12 (relative, floor 1).
/// (q̃_0 = √P q̄ and p̃_0 = Π/√P, so q̃_0 ← q̃_0 + t p̃_0/m is exactly this.)
#[test]
fn centroid_moves_uniformly() {
    let mut rng = SplitMix64(505);
    for p in PS {
        let ig = integrator(p);
        let pf = p as f64;
        for t in [0.6, -4.0, 25.0] {
            let s0 = state(&mut rng, p);
            let mut s = s0.clone();
            ig.free_propagate(&mut s, t);
            let (c0, c1) = (s0.q.centroid(), s.q.centroid());
            for (i, &m) in MASSES.iter().enumerate() {
                let mut pi0 = [0.0; 3];
                let mut pi1 = [0.0; 3];
                for j in 0..p {
                    for d in 0..3 {
                        pi0[d] += s0.p.get(j, i)[d];
                        pi1[d] += s.p.get(j, i)[d];
                    }
                }
                for d in 0..3 {
                    let want = c0[i][d] + t * pi0[d] / (pf * m);
                    assert!(
                        (c1[i][d] - want).abs() <= 1e-12 * want.abs().max(1.0),
                        "P = {p}, atom {i}, d {d}: centroid {} vs {want}",
                        c1[i][d]
                    );
                    assert!(
                        (pi1[d] - pi0[d]).abs() <= 1e-12 * pi0[d].abs().max(1.0),
                        "P = {p}: centroid momentum changed"
                    );
                }
            }
        }
    }
}

/// I-2, RP-6: at P = 1 the only mode is ω_0 = 0, so free propagation is a pure drift
/// q ← q + t p/m with p unchanged (to 1e-14: C = [1] makes the transform exact up to one
/// rounding per operation). Also checks nothing becomes NaN (no division by ω_0 = 0).
#[test]
fn p1_is_free_drift() {
    let ig = integrator(1);
    let q = Beads::from_vec(1, 3, vec![0.5, -1.0, 2.0, 0.0, 0.25, -0.75, 3.0, 1.0, -2.0]).unwrap();
    let pm = Beads::from_vec(
        1,
        3,
        vec![1.0, 2.0, -3.0, 15.0, -7.5, 0.0, 1836.0, -918.0, 459.0],
    )
    .unwrap();
    let s0 = RingState {
        q,
        p: pm,
        masses: MASSES.to_vec(),
    };
    for t in [0.5, -2.0] {
        let mut s = s0.clone();
        ig.free_propagate(&mut s, t);
        for i in 0..3 {
            let (q0, p0, q1, p1) = (s0.q.get(0, i), s0.p.get(0, i), s.q.get(0, i), s.p.get(0, i));
            for d in 0..3 {
                let want = q0[d] + t * p0[d] / MASSES[i];
                assert!(q1[d].is_finite() && p1[d].is_finite());
                assert!(
                    (q1[d] - want).abs() <= 1e-14 * want.abs().max(1.0),
                    "{q1:?}"
                );
                assert!(
                    (p1[d] - p0[d]).abs() <= 1e-14 * p0[d].abs().max(1.0),
                    "{p1:?}"
                );
            }
        }
    }
}
