//! Test-only helpers for the M7 integrator tests (MILESTONE_7; SPEC T-4, Y-1, Y-2).
//!
//! Built on the locked M2 helpers in `common` (RP-4 matrix and frequencies written from the SPEC).
//! Nothing here calls the code under test to compute an expectation.
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)] // index loops mirror the SPEC sums

use std::cell::Cell as StdCell;
use std::rc::Rc;

use pimd_core::{Beads, Cell, Rng};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat};
use pimd_potentials::{Evaluation, LennardJones, LjParams, Potential, PotentialError};

use crate::common::{SplitMix64, ref_freqs, ref_matrix, thermal_momenta, to_beads, to_modes};

/// A `Potential` that forwards to `inner` and counts `evaluate` calls in a shared counter.
pub struct Counting<T: Potential> {
    pub inner: T,
    pub calls: Rc<StdCell<usize>>,
}

impl<T: Potential> Potential for Counting<T> {
    fn evaluate(
        &mut self,
        positions: &Beads,
        cell: Option<&Cell>,
        subset: Option<&[usize]>,
    ) -> Result<Evaluation, PotentialError> {
        self.calls.set(self.calls.get() + 1);
        self.inner.evaluate(positions, cell, subset)
    }
}

/// Reduced Lennard-Jones (ε = σ = 1, r_c = 3σ, skin 0.3σ, shifted): atomic units with ε = 1 Eh,
/// σ = 1 bohr. Gas phase.
pub fn lj() -> LennardJones {
    LennardJones::new(LjParams {
        epsilon: 1.0,
        sigma: 1.0,
        cutoff: 3.0,
        skin: 0.3,
        shift: true,
    })
    .unwrap()
}

/// Masses of the 8-atom cluster (m_e): distinct, so a wrong atom index in a draw shows.
pub const CLUSTER_MASSES: [f64; 8] = [1.0, 2.0, 1.5, 3.0, 1.0, 2.5, 4.0, 1.25];

/// β of the cluster runs (k_B T = 0.25 ε: a bound, gently vibrating cluster).
pub const CLUSTER_BETA: f64 = 4.0;

/// Time step of the cluster runs (ħ/Eh); ω_P dt = P dt/β ≤ 0.02 for P ≤ 16.
pub const CLUSTER_DT: f64 = 0.005;

/// 8 LJ atoms on a 2 × 2 × 2 cube of edge 1.12σ (near the pair minimum 2^(1/6)σ), each displaced
/// by up to ±0.05σ; every bead of atom i within ±0.02σ of it (gas phase); thermal momenta
/// N(0, m_i P/β) from `Rng::from_seed(seed)` (RP-3).
pub fn cluster_state(p: usize, seed: u64) -> RingState {
    let mut sm = SplitMix64(seed);
    let n = CLUSTER_MASSES.len();
    let mut q = Beads::zeros(p, n);
    for i in 0..n {
        let site = [
            1.12 * (i % 2) as f64 + sm.range(-0.05, 0.05),
            1.12 * ((i / 2) % 2) as f64 + sm.range(-0.05, 0.05),
            1.12 * (i / 4) as f64 + sm.range(-0.05, 0.05),
        ];
        for j in 0..p {
            q.set(j, i, site.map(|x| x + sm.range(-0.02, 0.02)));
        }
    }
    let mut rng = Rng::from_seed(seed);
    let mom = thermal_momenta(&mut rng, p, &CLUSTER_MASSES, CLUSTER_BETA);
    RingState {
        q,
        p: mom,
        masses: CLUSTER_MASSES.to_vec(),
    }
}

/// A cluster simulation with thermostat `th` (BAOAB, `Transform::Matrix`).
pub fn cluster_sim(p: usize, seed: u64, th: Thermostat) -> RingPolymerSim {
    let ig = RingPolymerIntegrator::new(CLUSTER_DT, CLUSTER_BETA, p, th).unwrap();
    RingPolymerSim::new(cluster_state(p, seed), None, ig, Box::new(lj())).unwrap()
}

/// As `cluster_sim`, with an evaluation counter.
pub fn counted_cluster_sim(
    p: usize,
    seed: u64,
    th: Thermostat,
) -> (RingPolymerSim, Rc<StdCell<usize>>) {
    let calls = Rc::new(StdCell::new(0));
    let pot = Counting {
        inner: lj(),
        calls: Rc::clone(&calls),
    };
    let ig = RingPolymerIntegrator::new(CLUSTER_DT, CLUSTER_BETA, p, th).unwrap();
    let sim = RingPolymerSim::new(cluster_state(p, seed), None, ig, Box::new(pot)).unwrap();
    (sim, calls)
}

/// An independent copy of `rng` at the same stream position (D-2: snapshot/restore).
pub fn fork(rng: &Rng) -> Rng {
    Rng::restore(&rng.snapshot())
}

/// T-4 coefficients written from the SPEC: γ_0 = 0, γ_k = 2 λ ω_k (ω_k of RP-2/RP-4, aligned
/// with the RP-4 columns), c_k = exp(−γ_k dt) (T-1).
pub fn ref_c_trpmd(p: usize, beta: f64, dt: f64, lambda: f64) -> Vec<f64> {
    ref_freqs(p, beta)
        .iter()
        .enumerate()
        .map(|(k, &w)| {
            let g = if k == 0 { 0.0 } else { 2.0 * lambda * w };
            (-g * dt).exp()
        })
        .collect()
}

/// The T-1 O step with coefficients `c` (one per RP-4 column) on bead momenta `p`:
/// p̃_k ← c_k p̃_k + sqrt((1 − c_k²) m_i P/β) ξ, ξ drawn k outermost, then atom i, then d.
pub fn ref_o_step_with(p: &Beads, masses: &[f64], beta: f64, c: &[f64], rng: &mut Rng) -> Beads {
    let nb = p.n_beads();
    let pf = nb as f64;
    let mat = ref_matrix(nb);
    let mut pt = to_modes(&mat, p);
    for k in 0..nb {
        for (i, &m) in masses.iter().enumerate() {
            let mut v = pt.get(k, i);
            for x in &mut v {
                let xi = rng.normal();
                *x = c[k] * *x + ((1.0 - c[k] * c[k]) * m * pf / beta).sqrt() * xi;
            }
            pt.set(k, i, v);
        }
    }
    to_beads(&mat, &pt)
}

/// Centroid of bead momenta (1/P) Σ_j p_j per atom (as `Beads::centroid` on momenta).
pub fn centroid_of(b: &Beads) -> Vec<[f64; 3]> {
    let (p, n) = (b.n_beads(), b.n_atoms());
    (0..n)
        .map(|i| {
            let mut s = [0.0; 3];
            for j in 0..p {
                let x = b.get(j, i);
                for d in 0..3 {
                    s[d] += x[d];
                }
            }
            s.map(|v| v / p as f64)
        })
        .collect()
}

/// Asserts that two simulations hold bitwise identical state, forces and energies.
pub fn assert_same_sim(a: &RingPolymerSim, b: &RingPolymerSim, what: &str) {
    assert_eq!(a.state(), b.state(), "{what}: states differ");
    assert_eq!(a.forces(), b.forces(), "{what}: forces differ");
    assert_eq!(a.energies(), b.energies(), "{what}: energies differ");
}
