//! The PA-CMD cycle of SPEC Y-7 for the M7b long tests, driven through the library (MILESTONE_7
//! M7b required tests 8 and 9; the CLI has no `Harmonic` or `DoubleWell` potential). A copy of the
//! locked M7a driver `m7_dynamics::run_cycle` with the Y-6 dynamical masses for the whole run.
//!
//! One `Rng` drives every draw in the order they occur (D-2): the initial momenta (the amended Y-2
//! draw, `resample_momenta` with Ω, so the internal modes start at the temperature of their
//! dynamical masses), `n_equil` PILE-L steps, then per cycle (a) `decorr` PILE-L(τ_0) steps, (b)
//! the amended Y-2 redraw, (c) `seg` steps with the segment thermostat; centroid samples (Y-3) at
//! segment steps 0, stride, 2·stride, … into `CentroidCorrelation` (Y-4). The integrator carries
//! Ω from construction (`with_adiabatic_frequency`), through every `set_thermostat`. H̃_P
//! (`hamiltonian_dynamical`, the segments.csv definition under PA-CMD) right after the redraw and
//! at the segment end.
//!
//! Error bars (WORKFLOW tolerance policy, σ from segments, Y-4), as `m7_dynamics`: σ_1 = Y-4 σ over
//! single segments, σ_4 from blocks of 4 consecutive segments; asserted σ_4 ≤ 1.5 σ_1 at every lag;
//! σ = max(σ_1, σ_4).
#![allow(dead_code)]

use pimd_core::{Beads, Rng};
use pimd_estimators::{CentroidCorrelation, centroid_velocity, hamiltonian_dynamical};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat};
use pimd_potentials::Potential;

pub struct Protocol {
    pub mass: f64,
    pub beta: f64,
    pub n_beads: usize,
    pub dt: f64,
    pub tau0: f64,
    /// Adiabatic frequency Ω (Eh/ħ), Y-6.
    pub omega: f64,
    pub n_equil: usize,
    pub decorr: usize,
    pub seg: usize,
    pub stride: usize,
    pub n_lags: usize,
    pub n_segments: usize,
    pub seed: u64,
    /// Initial position of the (single) atom, all beads there, bohr.
    pub x0: [f64; 3],
}

/// Mean and error bars of one correlation function.
pub struct Stats {
    pub mean: Vec<f64>,
    /// max(σ_1, σ_4) per lag.
    pub sigma: Vec<f64>,
    pub sigma_1: Vec<f64>,
    pub sigma_4: Vec<f64>,
}

pub struct Result {
    pub c_vv: Stats,
    pub c_xx: Stats,
    /// |H̃_P(end) − H̃_P(start)| per segment, Eh (one atom, so also per atom).
    pub energy_error: Vec<f64>,
    pub n_segments: usize,
}

fn block_sigma(blocks: &[Vec<f64>]) -> Vec<f64> {
    let n = blocks.len() as f64;
    (0..blocks[0].len())
        .map(|k| {
            let m = blocks.iter().map(|b| b[k]).sum::<f64>() / n;
            let v = blocks.iter().map(|b| (b[k] - m).powi(2)).sum::<f64>() / (n - 1.0);
            (v / n).sqrt()
        })
        .collect()
}

fn stats(all: &CentroidCorrelation, blocks: &[Vec<f64>], what: &str) -> Stats {
    let mean = all.mean();
    let s1 = all.sem();
    let s4 = block_sigma(blocks);
    for k in 0..mean.len() {
        assert!(
            s4[k] <= 1.5 * s1[k],
            "{what} lag {k}: σ_4 = {:e} > 1.5 σ_1 = {:e} (segments correlated)",
            s4[k],
            s1[k]
        );
    }
    let sigma = s1.iter().zip(&s4).map(|(a, b)| a.max(*b)).collect();
    Stats {
        mean,
        sigma,
        sigma_1: s1,
        sigma_4: s4,
    }
}

/// Mean of `x` and its σ = SD(ddof = 1)/√n, with the 4-block check (σ_4 ≤ 1.5 σ_1); returns
/// (mean, max(σ_1, σ_4)).
pub fn mean_sigma(x: &[f64], what: &str) -> (f64, f64) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let s1 = (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
    let blocks: Vec<Vec<f64>> = x
        .chunks_exact(4)
        .map(|c| vec![c.iter().sum::<f64>() / 4.0])
        .collect();
    let s4 = block_sigma(&blocks)[0];
    assert!(s4 <= 1.5 * s1, "{what}: σ_4 = {s4:e} > 1.5 σ_1 = {s1:e}");
    (m, s1.max(s4))
}

/// Runs the PA-CMD protocol for one atom in `potential` with segment thermostat `dynamics`.
pub fn run_cycle(pr: &Protocol, potential: Box<dyn Potential>, dynamics: Thermostat) -> Result {
    let mut rng = Rng::from_seed(pr.seed);
    let p = pr.n_beads;
    let mut q = Beads::zeros(p, 1);
    for j in 0..p {
        q.set(j, 0, pr.x0);
    }
    let state = RingState {
        q,
        p: Beads::zeros(p, 1),
        masses: vec![pr.mass],
    };
    let sampling = Thermostat::PileL {
        tau0: Some(pr.tau0),
    };
    let ig = RingPolymerIntegrator::new(pr.dt, pr.beta, p, sampling)
        .unwrap()
        .with_adiabatic_frequency(pr.omega)
        .unwrap();
    let mut sim = RingPolymerSim::new(state, None, ig, potential).unwrap();
    sim.resample_momenta(&mut rng);
    for _ in 0..pr.n_equil {
        sim.step(&mut rng).unwrap();
    }
    let h = |sim: &RingPolymerSim| {
        let ig = sim.integrator();
        hamiltonian_dynamical(
            sim.state(),
            pr.beta,
            sim.energies(),
            ig.modes(),
            ig.mass_factors(),
        )
    };
    let mut cvv = CentroidCorrelation::new(pr.n_lags, 1).unwrap();
    let mut cxx = CentroidCorrelation::new(pr.n_lags, 1).unwrap();
    let (mut bvv, mut bxx) = (cvv.clone(), cxx.clone());
    let (mut blocks_vv, mut blocks_xx) = (Vec::new(), Vec::new());
    let mut energy_error = Vec::with_capacity(pr.n_segments);
    let push = |sim: &RingPolymerSim, a: &mut [&mut CentroidCorrelation; 4]| {
        let st = sim.state();
        let v = centroid_velocity(&st.p, &st.masses);
        let x = st.q.centroid();
        a[0].push(&v).unwrap();
        a[1].push(&x).unwrap();
        a[2].push(&v).unwrap();
        a[3].push(&x).unwrap();
    };
    for s in 0..pr.n_segments {
        sim.set_thermostat(sampling).unwrap();
        for _ in 0..pr.decorr {
            sim.step(&mut rng).unwrap();
        }
        sim.resample_momenta(&mut rng);
        let h0 = h(&sim);
        sim.set_thermostat(dynamics).unwrap();
        push(&sim, &mut [&mut cvv, &mut cxx, &mut bvv, &mut bxx]);
        for k in 1..=pr.seg {
            sim.step(&mut rng).unwrap();
            if k % pr.stride == 0 {
                push(&sim, &mut [&mut cvv, &mut cxx, &mut bvv, &mut bxx]);
            }
        }
        energy_error.push((h(&sim) - h0).abs());
        for c in [&mut cvv, &mut cxx, &mut bvv, &mut bxx] {
            c.end_segment().unwrap();
        }
        if (s + 1) % 4 == 0 {
            blocks_vv.push(bvv.mean());
            blocks_xx.push(bxx.mean());
            bvv = CentroidCorrelation::new(pr.n_lags, 1).unwrap();
            bxx = CentroidCorrelation::new(pr.n_lags, 1).unwrap();
        }
    }
    assert_eq!(cvv.n_segments(), pr.n_segments);
    assert_eq!(sim.integrator().adiabatic_frequency(), Some(pr.omega));
    Result {
        c_vv: stats(&cvv, &blocks_vv, "C_vv"),
        c_xx: stats(&cxx, &blocks_xx, "C_xx"),
        energy_error,
        n_segments: pr.n_segments,
    }
}
