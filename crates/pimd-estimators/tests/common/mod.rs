//! Test-only helpers for the long sampling tests of `pimd-estimators`.
#![allow(dead_code)]

use pimd_core::{Beads, Rng};
use pimd_estimators::{total_primitive, total_virial};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat};
use pimd_potentials::Potential;

/// Mean of `x` and its block-averaged standard error with blocks of `len` samples
/// (std of the block means / √n_blocks; a trailing partial block is dropped).
pub fn block_mean_sigma(x: &[f64], len: usize) -> (f64, f64) {
    let nb = x.len() / len;
    assert!(nb >= 20, "too few blocks: {nb}");
    let means: Vec<f64> = (0..nb)
        .map(|b| x[b * len..(b + 1) * len].iter().sum::<f64>() / len as f64)
        .collect();
    let m = means.iter().sum::<f64>() / nb as f64;
    let var = means.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (nb - 1) as f64;
    (m, (var / nb as f64).sqrt())
}

/// Block-averaged mean and σ (WORKFLOW tolerance policy): σ from blocks of `len` and `4·len`
/// samples; asserts σ_4len ≤ 1.5 σ_len (blocks of `len` already uncorrelated, up to the
/// sampling noise of σ itself) and returns the larger of the two.
pub fn mean_and_sigma(x: &[f64], len: usize, what: &str) -> (f64, f64) {
    let (m, s1) = block_mean_sigma(x, len);
    let (_, s4) = block_mean_sigma(x, 4 * len);
    assert!(
        s4 <= 1.5 * s1,
        "{what}: block length {len} too short (sigma {s1:e} vs {s4:e} at 4x)"
    );
    (m, s1.max(s4))
}

/// Sample variance (mean subtracted, n − 1).
pub fn sample_variance(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0)
}

/// Time series of `total_primitive` and `total_virial` from a PILE-L run.
pub struct Series {
    pub prim: Vec<f64>,
    pub vir: Vec<f64>,
}

/// Runs PILE-L(τ_0) BAOAB from positions `q0` with RP-3 thermal momenta N(0, m P/β) drawn from
/// a `pimd_core::Rng` seeded with `seed` (the same stream then drives the thermostat),
/// discards `n_eq` steps and records E_prim and E_vir (E-4) after each of `n_steps` steps.
#[allow(clippy::too_many_arguments)]
pub fn run_pile(
    q0: Beads,
    masses: &[f64],
    potential: Box<dyn Potential>,
    beta: f64,
    dt: f64,
    tau0: f64,
    n_eq: usize,
    n_steps: usize,
    seed: u64,
) -> Series {
    let p = q0.n_beads();
    let mut rng = Rng::from_seed(seed);
    let mut pm = Beads::zeros(p, masses.len());
    for j in 0..p {
        for (i, &m) in masses.iter().enumerate() {
            let s = (m * p as f64 / beta).sqrt();
            pm.set(j, i, [s * rng.normal(), s * rng.normal(), s * rng.normal()]);
        }
    }
    let state = RingState {
        q: q0,
        p: pm,
        masses: masses.to_vec(),
    };
    let ig =
        RingPolymerIntegrator::new(dt, beta, p, Thermostat::PileL { tau0: Some(tau0) }).unwrap();
    let mut sim = RingPolymerSim::new(state, None, ig, potential).unwrap();
    for _ in 0..n_eq {
        sim.step(&mut rng).unwrap();
    }
    let mut out = Series {
        prim: Vec::with_capacity(n_steps),
        vir: Vec::with_capacity(n_steps),
    };
    for _ in 0..n_steps {
        sim.step(&mut rng).unwrap();
        let s = sim.state();
        out.prim
            .push(total_primitive(&s.q, &s.masses, beta, sim.energies()));
        out.vir
            .push(total_virial(&s.q, sim.forces(), beta, sim.energies()));
    }
    out
}
