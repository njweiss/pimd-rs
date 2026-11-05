//! `resample_momenta` with PA-CMD dynamical masses (MILESTONE_7 M7b required test 3; SPEC Y-2 as
//! amended, Y-6, D-2, D-4).
//!
//! Expectations: the amended Y-2 draw written out in the test (`m7b_common::ref_redraw_dyn`):
//! z_{j,i,d} = √(m_i·P/β)·ξ from a forked RNG, atom i outermost, then bead j, then d; the test's
//! RP-4 matrix; mode k scaled by √s_k with s_k by Y-6; back to beads. Evaluation counts from the
//! counting `Potential` of `m7_common`.
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

mod common;
mod m7_common;
mod m7b_common;

use std::cell::Cell as StdCell;
use std::rc::Rc;

use common::{max_rel_diff, ref_matrix, to_modes};
use m7_common::{CLUSTER_BETA, CLUSTER_DT, CLUSTER_MASSES, Counting, cluster_state, fork, lj};
use m7b_common::{ref_mass_factors, ref_redraw_dyn};
use pimd_core::{Beads, Rng};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat};

/// A counted cluster sim (P beads) with PILE-L τ_0 = 0.7 and, if given, Ω.
fn counted(p: usize, omega: Option<f64>) -> (RingPolymerSim, Rc<StdCell<usize>>) {
    let calls = Rc::new(StdCell::new(0));
    let pot = Counting {
        inner: lj(),
        calls: Rc::clone(&calls),
    };
    let mut ig = RingPolymerIntegrator::new(
        CLUSTER_DT,
        CLUSTER_BETA,
        p,
        Thermostat::PileL { tau0: Some(0.7) },
    )
    .unwrap();
    if let Some(w) = omega {
        ig = ig.with_adiabatic_frequency(w).unwrap();
    }
    let sim = RingPolymerSim::new(cluster_state(p, 25), None, ig, Box::new(pot)).unwrap();
    (sim, calls)
}

/// Y-2 (amended), Y-6, D-2, D-4: with Ω, `resample_momenta` sets the bead momenta to the scaled
/// Y-2 draw (from a forked RNG, through the test's RP-4 matrix, mode k × √s_k, back) to 1e-12
/// relative (floor 1e-3 of the momentum scale √(m P/β)); exactly P·N·3 draws (the streams agree
/// afterwards); positions, masses, forces and energies bitwise unchanged; no evaluation.
/// Distinct masses (8-atom cluster), P = 1, 3, 4, 8, two Ω.
#[test]
fn redraw_with_omega_is_the_scaled_y2_draw() {
    for p in [1, 3, 4, 8] {
        for omega in [3.0, 40.0] {
            let (mut sim, calls) = counted(p, Some(omega));
            let mut rng = Rng::from_seed(41 + p as u64);
            for _ in 0..3 {
                sim.step(&mut rng).unwrap();
            }
            let (q0, f0, e0) = (
                sim.state().q.clone(),
                sim.forces().clone(),
                sim.energies().to_vec(),
            );
            let n0 = calls.get();
            let mut xi = fork(&rng);
            sim.resample_momenta(&mut rng);
            let s = ref_mass_factors(p, CLUSTER_BETA, omega);
            let want = ref_redraw_dyn(&CLUSTER_MASSES, p, CLUSTER_BETA, &s, &mut xi);
            let floor = 1e-3 * (p as f64 / CLUSTER_BETA).sqrt();
            let d = max_rel_diff(&sim.state().p, &want, floor);
            assert!(d <= 1e-12, "P = {p}, Ω = {omega}: rel diff {d:e}");
            assert_eq!(rng.snapshot(), xi.snapshot(), "P = {p}: not P·N·3 draws");
            assert_eq!(sim.state().q, q0);
            assert_eq!(sim.state().masses, CLUSTER_MASSES.to_vec());
            assert_eq!(sim.forces(), &f0);
            assert_eq!(sim.energies(), e0.as_slice());
            assert_eq!(calls.get(), n0, "resample_momenta evaluated");
        }
    }
}

/// Y-2 (MILESTONE_7 M7b decision 6: "RPMD and TRPMD keep the M7a code path"): without Ω the
/// redraw is the unscaled Y-2 draw, bitwise (√(m_i·P/β)·ξ evaluated as `((m_i·P)/β).sqrt()·ξ`, the
/// M7a form), with P·N·3 draws.
#[test]
fn redraw_without_omega_is_bitwise_y2() {
    for p in [3, 4] {
        let (mut sim, _) = counted(p, None);
        let mut rng = Rng::from_seed(51);
        sim.step(&mut rng).unwrap();
        let mut xi = fork(&rng);
        sim.resample_momenta(&mut rng);
        let mut want = Beads::zeros(p, CLUSTER_MASSES.len());
        for (i, &m) in CLUSTER_MASSES.iter().enumerate() {
            let sd = (m * p as f64 / CLUSTER_BETA).sqrt();
            for j in 0..p {
                want.set(j, i, [sd * xi.normal(), sd * xi.normal(), sd * xi.normal()]);
            }
        }
        assert_eq!(sim.state().p, want);
        assert_eq!(rng.snapshot(), xi.snapshot());
    }
}

/// Y-2 (amended), Y-6 (statistics of the draw, independent of its order): over 4 000 redraws of
/// one atom (m = 3, P = 4, β = 2, Ω = 6: s = [1, 2/9, 2/9, 4/9]) the variance of every
/// normal-mode momentum component (test's RP-4 basis) is m·s_k·P/β within 5σ (σ of a sample
/// variance of n normals: var·√(2/(n − 1))), and the covariance of modes 0 and 3 is zero within
/// 5σ (σ = √(var_0 var_3/n)). Catches a redraw without the √s_k scaling (variance m P/β = 6 for
/// every mode) or with s_k instead of √s_k.
#[test]
fn redraw_mode_variance_is_m_s_p_over_beta() {
    let (p, m, beta, omega) = (4usize, 3.0, 2.0, 6.0);
    let ig = RingPolymerIntegrator::new(0.01, beta, p, Thermostat::Off)
        .unwrap()
        .with_adiabatic_frequency(omega)
        .unwrap();
    let state = RingState {
        q: Beads::zeros(p, 1),
        p: Beads::zeros(p, 1),
        masses: vec![m],
    };
    let pot = pimd_potentials::Harmonic::new(vec![m], 1.0).unwrap();
    let mut sim = RingPolymerSim::new(state, None, ig, Box::new(pot)).unwrap();
    let mat = ref_matrix(p);
    let s = [1.0, 2.0 / 9.0, 2.0 / 9.0, 4.0 / 9.0];
    let mut rng = Rng::from_seed(78);
    let n = 4000;
    let mut s2 = vec![0.0; p * 3];
    let mut cross = 0.0;
    for _ in 0..n {
        sim.resample_momenta(&mut rng);
        let pt = to_modes(&mat, &sim.state().p);
        for k in 0..p {
            let v = pt.get(k, 0);
            for d in 0..3 {
                s2[3 * k + d] += v[d] * v[d];
            }
        }
        cross += pt.get(0, 0)[0] * pt.get(3, 0)[0];
    }
    let nf = n as f64;
    for k in 0..p {
        let var = m * s[k] * p as f64 / beta;
        let sig = var * (2.0 / (nf - 1.0)).sqrt();
        for d in 0..3 {
            let got = s2[3 * k + d] / nf;
            assert!(
                (got - var).abs() <= 5.0 * sig,
                "mode {k}, d {d}: variance {got} vs {var} ± {sig}"
            );
        }
    }
    let v0 = m * p as f64 / beta;
    let v3 = m * s[3] * p as f64 / beta;
    let c = cross / nf;
    assert!(
        c.abs() <= 5.0 * (v0 * v3 / nf).sqrt(),
        "mode covariance {c}"
    );
}
