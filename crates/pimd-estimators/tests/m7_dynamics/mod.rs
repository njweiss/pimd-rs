//! The segment cycle of SPEC Y-1 for the M7 long tests, driven through the library (MILESTONE_7
//! required tests 10 and 11; the CLI has no `Harmonic` or `DoubleWell` potential).
//!
//! One `Rng` drives every draw in the order they occur (D-2): the initial thermal momenta (RP-3),
//! `n_equil` PILE-L steps, then per cycle (a) `decorr` PILE-L(τ_0) steps, (b) the Y-2 redraw
//! (`resample_momenta`), (c) `seg` steps with the segment thermostat; centroid samples (Y-3) at
//! segment steps 0, stride, 2·stride, … into `CentroidCorrelation` (Y-4). H_P per atom
//! (`hamiltonian`/N) right after the redraw and at the segment end, as MILESTONE_7's
//! segments.csv defines it.
//!
//! Error bars (WORKFLOW tolerance policy, σ from segments, Y-4): σ_1 is the Y-4 σ of the mean over
//! single segments (`CentroidCorrelation::sem`); σ_4 is the σ of the means of blocks of 4
//! consecutive segments (a fresh `CentroidCorrelation` per block); the tests assert
//! σ_4 ≤ 1.5 σ_1 at every lag (segments are close to independent) and use σ = max(σ_1, σ_4).
#![allow(dead_code)]

use pimd_core::{Beads, Rng};
use pimd_estimators::{CentroidCorrelation, centroid_velocity, hamiltonian};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat};
use pimd_potentials::Potential;

pub struct Protocol {
    pub mass: f64,
    pub beta: f64,
    pub n_beads: usize,
    pub dt: f64,
    pub tau0: f64,
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
    /// |H_P/N(end) − H_P/N(start)| per segment, Eh.
    pub energy_error: Vec<f64>,
    pub n_segments: usize,
}

/// The block statistics of `blocks` (a list of block-mean vectors).
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

/// Runs the protocol for one atom in `potential` with segment thermostat `dynamics`.
pub fn run_cycle(pr: &Protocol, potential: Box<dyn Potential>, dynamics: Thermostat) -> Result {
    let mut rng = Rng::from_seed(pr.seed);
    let p = pr.n_beads;
    let mut q = Beads::zeros(p, 1);
    let mut mom = Beads::zeros(p, 1);
    let sd = (pr.mass * p as f64 / pr.beta).sqrt();
    for j in 0..p {
        q.set(j, 0, pr.x0);
        mom.set(
            j,
            0,
            [sd * rng.normal(), sd * rng.normal(), sd * rng.normal()],
        );
    }
    let state = RingState {
        q,
        p: mom,
        masses: vec![pr.mass],
    };
    let sampling = Thermostat::PileL {
        tau0: Some(pr.tau0),
    };
    let ig = RingPolymerIntegrator::new(pr.dt, pr.beta, p, sampling).unwrap();
    let mut sim = RingPolymerSim::new(state, None, ig, potential).unwrap();
    for _ in 0..pr.n_equil {
        sim.step(&mut rng).unwrap();
    }
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
        let h0 = hamiltonian(sim.state(), pr.beta, sim.energies());
        sim.set_thermostat(dynamics).unwrap();
        push(&sim, &mut [&mut cvv, &mut cxx, &mut bvv, &mut bxx]);
        for k in 1..=pr.seg {
            sim.step(&mut rng).unwrap();
            if k % pr.stride == 0 {
                push(&sim, &mut [&mut cvv, &mut cxx, &mut bvv, &mut bxx]);
            }
        }
        let h1 = hamiltonian(sim.state(), pr.beta, sim.energies());
        energy_error.push((h1 - h0).abs());
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
    Result {
        c_vv: stats(&cvv, &blocks_vv, "C_vv"),
        c_xx: stats(&cxx, &blocks_xx, "C_xx"),
        energy_error,
        n_segments: pr.n_segments,
    }
}
