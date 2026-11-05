//! Test-only helpers for the M7b PA-CMD CLI tests (MILESTONE_7 "M7b brief"; SPEC Y-1, Y-2, Y-6,
//! Y-7, D-2, D-3).
//!
//! `reference_pacmd` is the PA-CMD cycle of Y-7 written as a plain loop from the SPEC and the brief,
//! on library code: `RingPolymerSim`, `LennardJones`, `Rng`, the M7a controls (`set_thermostat`,
//! `resample_momenta`) and the M7b integrator option `with_adiabatic_frequency` (tested on its own
//! in `pimd-integrators`). H̃_P (Y-6) is computed here by hand (RP-4 matrix and mass factors
//! written from the SPEC), not with `pimd_estimators::hamiltonian_dynamical`. The correlation
//! arithmetic comes from the locked M7a helpers in `dyn_common` (hand-written Y-3/Y-4).
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use std::f64::consts::PI;

use pimd_cli::{Config, DynamicsMethod};
use pimd_core::{Beads, Rng};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat};
use pimd_potentials::LennardJones;

use crate::common::{initial_state, lj_params};
use crate::dyn_common::{RefDyn, c_seg};

/// SPEC U-2 (CODATA 2018), written here: 1 Eh = 219474.6313632 cm⁻¹.
pub const CM_PER_HARTREE: f64 = 219474.6313632;

/// The PA-CMD `[dynamics]` section of the cluster tests (cycle 6 + 12 = 18 steps, as the M7a
/// sections of `dyn_common`): Ω = 412.5 cm⁻¹ (Ω·dt ≈ 0.155 at dt = 2 fs; the cluster's ω_k at
/// P = 4, 20 K are 79–111 cm⁻¹), λ = 1 (the M7b test value).
pub const DYN_PACMD: &str = r#"
[dynamics]
method = "pa-cmd"
cmd_omega_cm = 412.5
cmd_lambda = 1.0
decorrelation_steps = 6
segment_steps = 12
corr_stride = 3
n_lags = 4
"#;

/// `config hash` lines of `dyn_common::CLUSTER` + `DYN_TRPMD` and + `DYN_RPMD` as printed in
/// `run.log` by `pimd-rs` built at e6008c9 (`main`, M7a merged), recorded 2026-11-04.
pub const CLUSTER_TRPMD_HASH_E6008C9: &str = "0x5faeb95810b7ab13";
pub const CLUSTER_RPMD_HASH_E6008C9: &str = "0x71b9af229dd4ed43";

/// 64-bit FNV-1a of (checkpoint.bin, segments.csv, correlations.csv) written by the same e6008c9
/// runs of `CLUSTER` + `DYN_TRPMD` and `CLUSTER` + `DYN_RPMD` (2 672, 177/178, 626 bytes).
pub const CLUSTER_TRPMD_FNV1A_E6008C9: [u64; 3] = [
    0xead3_3b23_271f_96fc,
    0x22d5_3353_9c95_8166,
    0x472f_9ad4_969d_8b35,
];
pub const CLUSTER_RPMD_FNV1A_E6008C9: [u64; 3] = [
    0x8193_9145_e004_2fb1,
    0xf8f1_6762_52b4_caf2,
    0x32c9_a3a7_e3ec_fe4c,
];

/// C from RP-4, row-major P × P (C[j][k] at j·P + k).
pub fn ref_matrix(p: usize) -> Vec<f64> {
    let pf = p as f64;
    let mut c = vec![0.0; p * p];
    for j in 0..p {
        let jf = j as f64;
        c[j * p] = 1.0 / pf.sqrt();
        for m in 1..=(p - 1) / 2 {
            let arg = 2.0 * PI * (m as f64) * jf / pf;
            c[j * p + 2 * m - 1] = (2.0 / pf).sqrt() * arg.cos();
            c[j * p + 2 * m] = (2.0 / pf).sqrt() * arg.sin();
        }
        if p.is_multiple_of(2) {
            c[j * p + p - 1] = if j.is_multiple_of(2) { 1.0 } else { -1.0 } / pf.sqrt();
        }
    }
    c
}

/// Y-6 mass factors: s_0 = 1, s_k = (ω_k/Ω)² with ω_k = 2 ω_P sin(mπ/P) (k = 2m − 1, 2m) and
/// 2 ω_P for the last column of even P, ω_P = P/β (RP-2, RP-4).
pub fn ref_mass_factors(p: usize, beta: f64, omega: f64) -> Vec<f64> {
    let pf = p as f64;
    let wp = pf / beta;
    let mut s = vec![1.0; p];
    for m in 1..=(p - 1) / 2 {
        let w = 2.0 * wp * ((m as f64) * PI / pf).sin();
        s[2 * m - 1] = (w / omega).powi(2);
        s[2 * m] = (w / omega).powi(2);
    }
    if p.is_multiple_of(2) && p > 1 {
        s[p - 1] = (2.0 * wp / omega).powi(2);
    }
    s
}

/// H̃_P of Y-6 by hand: Σ_k Σ_i |p̃_{k,i}|²/(2 m_i s_k) with p̃_k = Σ_j C_{jk} p_j, plus the E-5
/// springs Σ_j Σ_i ½ m_i ω_P² |q_{j,i} − q_{j+1,i}|² and Σ_j V_j.
pub fn ref_h_dyn(state: &RingState, beta: f64, energies: &[f64], s: &[f64]) -> f64 {
    let (q, p) = (&state.q, &state.p);
    let (nb, n) = (q.n_beads(), q.n_atoms());
    let c = ref_matrix(nb);
    let wp = nb as f64 / beta;
    let mut h: f64 = energies.iter().sum();
    for i in 0..n {
        let m = state.masses[i];
        for k in 0..nb {
            for d in 0..3 {
                let pt: f64 = (0..nb).map(|j| c[j * nb + k] * p.get(j, i)[d]).sum();
                h += pt * pt / (2.0 * m * s[k]);
            }
        }
        for j in 0..nb {
            let (a, b) = (q.get(j, i), q.get((j + 1) % nb, i));
            for d in 0..3 {
                h += 0.5 * m * wp * wp * (a[d] - b[d]).powi(2);
            }
        }
    }
    h
}

/// The amended Y-2 transformation by hand: bead momenta `p` → normal modes (RP-4 matrix of the
/// test) → mode k × √s_k → beads.
pub fn amended(p: &Beads, s: &[f64]) -> Beads {
    let (nb, n) = (p.n_beads(), p.n_atoms());
    let c = ref_matrix(nb);
    let mut out = Beads::zeros(nb, n);
    for i in 0..n {
        for d in 0..3 {
            let pt: Vec<f64> = (0..nb)
                .map(|k| (0..nb).map(|j| c[j * nb + k] * p.get(j, i)[d]).sum::<f64>() * s[k].sqrt())
                .collect();
            for j in 0..nb {
                let mut v = out.get(j, i);
                v[d] = (0..nb).map(|k| c[j * nb + k] * pt[k]).sum();
                out.set(j, i, v);
            }
        }
    }
    out
}

/// Y-3 by hand: centroid velocity v̄_i = (1/P) Σ_j p_{j,i}/m_i and centroid q̄_i = (1/P) Σ_j q_{j,i}.
fn centroids(sim: &RingPolymerSim) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let st = sim.state();
    let (p, n) = (st.q.n_beads(), st.q.n_atoms());
    let mut v = vec![[0.0; 3]; n];
    let mut q = vec![[0.0; 3]; n];
    for i in 0..n {
        for j in 0..p {
            let (a, b) = (st.p.get(j, i), st.q.get(j, i));
            for d in 0..3 {
                v[i][d] += a[d] / st.masses[i];
                q[i][d] += b[d];
            }
        }
        for d in 0..3 {
            v[i][d] /= p as f64;
            q[i][d] /= p as f64;
        }
    }
    (v, q)
}

/// (Ω in Eh/ħ, λ) of a PA-CMD config.
pub fn omega_lambda(cfg: &Config) -> (f64, f64) {
    match cfg.dynamics.as_ref().expect("[dynamics]").method {
        DynamicsMethod::PaCmd { omega, lambda } => (omega, lambda),
        ref other => panic!("not a pa-cmd config: {other:?}"),
    }
}

/// The run of a PA-CMD config as a plain loop (Y-7 = Y-1 with Y-6 masses for the whole run):
/// one `Rng::from_seed(seed)` (D-2) for the initial momenta, an integrator with PILE-L(τ_0) and Ω
/// for the `n_equil` steps and every later phase; then n_steps/c cycles of (a)
/// `decorrelation_steps` PILE-L(τ_0) steps, (b) the amended Y-2 redraw, (c) `segment_steps` steps
/// with `Trpmd { λ }`, centroid samples at segment steps 0, corr_stride, … (Y-3); H̃_P/N (Y-6, by
/// hand) right after the redraw and at the segment end.
///
/// Initial momenta (owner decision on QUESTIONS Q32, 2026-11-05): the amended Y-2 draw, i.e. the
/// MILESTONE_4/5 initial draw (P·N·3 normals, atom i outer, bead j, dimension d inner, variance
/// m_i P/β) transformed to normal modes, mode k scaled by √s_k, transformed back (as i-PI's
/// thermal initialisation with `dynm3`). Here: the M4/M5 initial state (`common::initial_state`),
/// then its momenta replaced by `resample_momenta` with Ω from a fresh `Rng::from_seed(seed)` (the
/// same draws; the redraw is tested on its own in pimd-integrators), and checked against the
/// amended draw written out by hand (`amended`, test RP-4 matrix) to 1e-12.
pub fn reference_pacmd(cfg: &Config) -> RefDyn {
    let s = &cfg.sampling;
    let dy = cfg.dynamics.as_ref().expect("[dynamics] config");
    let (omega, lambda) = omega_lambda(cfg);
    let n = cfg.system.n_atoms;
    let mut rng = Rng::from_seed(s.seed);
    let state = initial_state(cfg, &mut rng);
    let sampling = Thermostat::PileL { tau0: Some(s.tau0) };
    let dynamics = Thermostat::Trpmd { lambda };
    let ig = RingPolymerIntegrator::new(s.dt, s.beta, s.n_beads, sampling)
        .unwrap()
        .with_adiabatic_frequency(omega)
        .unwrap();
    let pot = LennardJones::new(lj_params(cfg)).unwrap();
    let sk = ref_mass_factors(s.n_beads, s.beta, omega);
    let want_p = amended(&state.p, &sk);
    let mut sim = RingPolymerSim::new(state, cfg.system.cell, ig, Box::new(pot)).unwrap();
    let mut init = Rng::from_seed(s.seed);
    sim.resample_momenta(&mut init);
    assert_eq!(
        init.snapshot(),
        rng.snapshot(),
        "initial draw: not P·N·3 normals"
    );
    let scale = (cfg.system.masses.iter().fold(0.0f64, |a, &m| a.max(m)) * s.n_beads as f64
        / s.beta)
        .sqrt();
    for (a, b) in sim.state().p.as_slice().iter().zip(want_p.as_slice()) {
        assert!(
            (a - b).abs() <= 1e-12 * scale,
            "initial amended draw: {a:e} vs {b:e}"
        );
    }
    for _ in 0..s.n_equil {
        sim.step(&mut rng).unwrap();
    }
    let c = dy.decorrelation_steps + dy.segment_steps;
    assert_eq!(s.n_steps % c, 0);
    let nf = n as f64;
    let mut out = RefDyn {
        cvv_seg: Vec::new(),
        cxx_seg: Vec::new(),
        h: Vec::new(),
        q: Beads::zeros(1, 1),
        p: Beads::zeros(1, 1),
    };
    for _ in 0..s.n_steps / c {
        sim.set_thermostat(sampling).unwrap();
        for _ in 0..dy.decorrelation_steps {
            sim.step(&mut rng).unwrap();
        }
        sim.resample_momenta(&mut rng);
        let h0 = ref_h_dyn(sim.state(), s.beta, sim.energies(), &sk) / nf;
        sim.set_thermostat(dynamics).unwrap();
        let (mut vs, mut qs) = (Vec::new(), Vec::new());
        let (v, q) = centroids(&sim);
        vs.push(v);
        qs.push(q);
        for k in 1..=dy.segment_steps {
            sim.step(&mut rng).unwrap();
            if k % dy.corr_stride == 0 {
                let (v, q) = centroids(&sim);
                vs.push(v);
                qs.push(q);
            }
        }
        assert_eq!(vs.len(), dy.segment_steps / dy.corr_stride + 1);
        let h1 = ref_h_dyn(sim.state(), s.beta, sim.energies(), &sk) / nf;
        out.cvv_seg.push(c_seg(&vs, dy.n_lags));
        out.cxx_seg.push(c_seg(&qs, dy.n_lags));
        out.h.push((h0, h1));
    }
    out.q = sim.state().q.clone();
    out.p = sim.state().p.clone();
    out
}
