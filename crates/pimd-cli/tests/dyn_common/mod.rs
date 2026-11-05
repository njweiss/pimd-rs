//! Test-only helpers for the M7 `[dynamics]` CLI tests (MILESTONE_7; SPEC Y-1 … Y-5, D-2, D-3).
//!
//! `reference_dynamics` is the segment cycle of Y-1 written as a plain loop from the SPEC and the
//! MILESTONE_7 clarifications, on merged library code (`RingPolymerSim`, `LennardJones`,
//! `hamiltonian`, `Rng`) plus the two M7 controls (`set_thermostat`, `resample_momenta`, tested on
//! their own in `pimd-integrators`). The centroid samples (Y-3), the per-segment correlation
//! (Y-4), the segment mean and σ (Y-4) and the Green–Kubo integral (Y-5) are computed here by
//! hand, not with `pimd_estimators::CentroidCorrelation` or `green_kubo`.
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use pimd_cli::{Config, DynamicsMethod};
use pimd_core::Rng;
use pimd_estimators::hamiltonian;
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, Thermostat};
use pimd_potentials::LennardJones;

use crate::common::{initial_state, lj_params};

/// The 8-atom gas-phase neon cluster of the M7 CLI tests, without `[dynamics]` (the exact text
/// whose `config hash` and `checkpoint.bin` were recorded with `pimd-rs` built at cf79a4d).
/// Cycle of the `[dynamics]` variants: 6 + 12 = 18 steps; n_steps = 54 = 3 cycles.
pub const CLUSTER: &str = r#"
[system]
n_atoms = 8
mass_amu = 20.1797
positions_angstrom = [
  [0.00, 0.02, -0.03], [3.12, -0.04, 0.01], [0.03, 3.08, 0.05], [3.10, 3.13, -0.02],
  [-0.02, 0.05, 3.11], [3.07, 0.01, 3.09], [0.04, 3.11, 3.06], [3.13, 3.06, 3.12],
]
symbols = ["Ne", "Ne", "Ne", "Ne", "Ne", "Ne", "Ne", "Ne"]

[potential]
kind = "lj"
epsilon_kelvin = 35.6
sigma_angstrom = 2.749
cutoff_sigma = 2.5
skin_sigma = 0.3
shift = true

[sampling]
temperature_kelvin = 20.0
n_beads = 4
dt_fs = 2.0
tau0_fs = 50.0
n_equil = 10
n_steps = 54
stride = 3
seed = 777

[output]
checkpoint_every = 18
"#;

/// `config hash` of `CLUSTER` as printed in `run.log` by `pimd-rs` built at cf79a4d (M6 merged),
/// `pimd-rs cluster.toml --out <dir>`, recorded 2026-11-04.
pub const CLUSTER_HASH_CF79A4D: &str = "0x5590f06b7e080cce";

/// 64-bit FNV-1a of the `checkpoint.bin` (2 480 bytes) written by the same cf79a4d run of
/// `CLUSTER`.
pub const CLUSTER_CKPT_FNV1A_CF79A4D: u64 = 0x6112_f525_279e_f0e2;

/// `config hash` of `common::BASE` printed by `pimd-rs` at cf79a4d.
pub const BASE_HASH_CF79A4D: &str = "0xbc768f3a57024bdf";

/// FNV-1a of the `checkpoint.bin` (9 576 bytes) of the cf79a4d run of `common::BASE`.
pub const BASE_CKPT_FNV1A_CF79A4D: u64 = 0x5671_dd5d_9b1b_dd46;

/// The `[dynamics]` section used by most tests: TRPMD λ = 0.5, 6 decorrelation steps, segments of
/// 12 steps sampled every 3 (L = 5), 4 lags.
pub const DYN_TRPMD: &str = r#"
[dynamics]
method = "trpmd"
trpmd_lambda = 0.5
decorrelation_steps = 6
segment_steps = 12
corr_stride = 3
n_lags = 4
"#;

/// The same cycle with RPMD.
pub const DYN_RPMD: &str = r#"
[dynamics]
method = "rpmd"
decorrelation_steps = 6
segment_steps = 12
corr_stride = 3
n_lags = 4
"#;

/// 8 neon atoms in a periodic 12.5 Å box (cutoff 2σ + skin 0.25σ ≤ L/2, C-4), started from the
/// cluster positions with atoms 1, 3, 5 shifted by −L in x and atom 6 by +L in z, so their stored
/// (unwrapped, C-5) positions lie outside [0, L) and differ from wrapped ones (Y-3: q̄ "as stored,
/// not wrapped"). With an RDF, so the version-2 checkpoint has its RDF part.
pub const BOX8: &str = r#"
[system]
n_atoms = 8
mass_amu = 20.1797
box_angstrom = 12.5
positions_angstrom = [
  [0.00, 0.02, -0.03], [-9.38, -0.04, 0.01], [0.03, 3.08, 0.05], [-9.40, 3.13, -0.02],
  [-0.02, 0.05, 3.11], [-9.43, 0.01, 3.09], [0.04, 3.11, 15.56], [3.13, 3.06, 3.12],
]

[potential]
kind = "lj"
epsilon_kelvin = 35.6
sigma_angstrom = 2.749
cutoff_sigma = 2.0
skin_sigma = 0.25
shift = true

[sampling]
temperature_kelvin = 20.0
n_beads = 4
dt_fs = 2.0
tau0_fs = 50.0
n_equil = 10
n_steps = 54
stride = 3
seed = 778

[output]
rdf_bins = 10
rdf_rmax_angstrom = 6.0
checkpoint_every = 18
"#;

/// The correlations.csv header (MILESTONE_7 "Outputs").
pub const CORR_HEADER: &str =
    "lag,t_fs,c_vv_au,c_vv_sem_au,d_au,d_angstrom2_per_ps,c_xx_bohr2,c_xx_sem_bohr2";

/// The segments.csv header (MILESTONE_7 "Outputs").
pub const SEG_HEADER: &str = "segment,step,h_start_eh,h_end_eh";

/// The result of the reference cycle.
pub struct RefDyn {
    /// Per-segment C_seg of v̄ and q̄ (Y-4), one row per segment, n_lags values each.
    pub cvv_seg: Vec<Vec<f64>>,
    pub cxx_seg: Vec<Vec<f64>>,
    /// (H_P/N right after the redraw, H_P/N at the segment end), Eh, per segment.
    pub h: Vec<(f64, f64)>,
    /// Final bead positions and momenta.
    pub q: pimd_core::Beads,
    pub p: pimd_core::Beads,
}

/// Mean and σ (sample SD with n − 1 over √n) of the rows, per lag (Y-4); σ NaN for one row.
pub fn mean_sem(rows: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let n = rows.len();
    let k = rows[0].len();
    let nf = n as f64;
    let mean: Vec<f64> = (0..k)
        .map(|t| rows.iter().map(|r| r[t]).sum::<f64>() / nf)
        .collect();
    let sem = (0..k)
        .map(|t| {
            if n < 2 {
                return f64::NAN;
            }
            let ss: f64 = rows.iter().map(|r| (r[t] - mean[t]).powi(2)).sum();
            (ss / (nf - 1.0)).sqrt() / nf.sqrt()
        })
        .collect();
    (mean, sem)
}

/// Y-5 by hand: D(τ_n) = (1/3) Σ_{k<n} (C_k + C_{k+1})/2 · Δ, D(0) = 0.
pub fn trapezoid_d(c: &[f64], dt_lag: f64) -> Vec<f64> {
    let mut d = vec![0.0; c.len()];
    for k in 1..c.len() {
        d[k] = d[k - 1] + 0.5 * (c[k - 1] + c[k]) * dt_lag / 3.0;
    }
    d
}

/// Y-4 for one segment, by hand: C(τ) = (1/(N (L − τ))) Σ_s Σ_i x_i(s)·x_i(s + τ).
pub fn c_seg(x: &[Vec<[f64; 3]>], n_lags: usize) -> Vec<f64> {
    let l = x.len();
    let n = x[0].len();
    (0..n_lags)
        .map(|tau| {
            let mut s = 0.0;
            for t in 0..l - tau {
                for i in 0..n {
                    for d in 0..3 {
                        s += x[t][i][d] * x[t + tau][i][d];
                    }
                }
            }
            s / (n * (l - tau)) as f64
        })
        .collect()
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

/// The run of a `[dynamics]` config as a plain loop (Y-1, Y-2, Y-3, MILESTONE_7 "Production with
/// `[dynamics]`"): one `Rng::from_seed(seed)` (D-2) for the initial momenta (MILESTONE_4/5
/// initial state), the `n_equil` PILE-L(τ_0) steps and every later draw in the order they occur;
/// then n_steps/c cycles of (a) `decorrelation_steps` PILE-L(τ_0) steps, (b) the Y-2 redraw,
/// (c) `segment_steps` steps with `Off` (RPMD) or `Trpmd { λ }` (TRPMD), with centroid samples at
/// segment steps 0, corr_stride, 2·corr_stride, … (s = 0 right after the redraw).
pub fn reference_dynamics(cfg: &Config) -> RefDyn {
    let s = &cfg.sampling;
    let dy = cfg.dynamics.as_ref().expect("[dynamics] config");
    let n = cfg.system.n_atoms;
    let mut rng = Rng::from_seed(s.seed);
    let state = initial_state(cfg, &mut rng);
    let sampling = Thermostat::PileL { tau0: Some(s.tau0) };
    let dynamics = match dy.method {
        DynamicsMethod::Rpmd => Thermostat::Off,
        DynamicsMethod::Trpmd { lambda } => Thermostat::Trpmd { lambda },
        DynamicsMethod::PaCmd { .. } => unreachable!("PA-CMD is tested in pacmd_run.rs"),
    };
    let ig = RingPolymerIntegrator::new(s.dt, s.beta, s.n_beads, sampling).unwrap();
    let pot = LennardJones::new(lj_params(cfg)).unwrap();
    let mut sim = RingPolymerSim::new(state, cfg.system.cell, ig, Box::new(pot)).unwrap();
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
        q: sim.state().q.clone(),
        p: sim.state().p.clone(),
    };
    for _ in 0..s.n_steps / c {
        sim.set_thermostat(sampling).unwrap();
        for _ in 0..dy.decorrelation_steps {
            sim.step(&mut rng).unwrap();
        }
        sim.resample_momenta(&mut rng);
        let h0 = hamiltonian(sim.state(), s.beta, sim.energies()) / nf;
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
        let h1 = hamiltonian(sim.state(), s.beta, sim.energies()) / nf;
        out.cvv_seg.push(c_seg(&vs, dy.n_lags));
        out.cxx_seg.push(c_seg(&qs, dy.n_lags));
        out.h.push((h0, h1));
    }
    out.q = sim.state().q.clone();
    out.p = sim.state().p.clone();
    out
}

/// 64-bit FNV-1a (as recorded for the cf79a4d checkpoints).
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Byte length of the version-2 checkpoint layout (MILESTONE_4 "Checkpoint", unchanged through
/// M6): magic (8), 9 u64 header words (version, hash, P, N, step, recorded, Σ E_prim, Σ E_vir,
/// RNG length), the RNG state (56 bytes: 32-byte key, 8-byte stream, 16-byte word position),
/// q, p, forces (P·N·3 f64 each), energies (P f64), the RDF flag and, with an RDF,
/// n_bins, r_max, n_samples and n_bins counts.
pub fn v2_len(p: usize, n: usize, rdf_bins: Option<usize>) -> usize {
    8 + 9 * 8 + 56 + 3 * p * n * 3 * 8 + p * 8 + 8 + rdf_bins.map_or(0, |b| 3 * 8 + b * 8)
}

/// The trailing M7 block of a `[dynamics]` checkpoint after the version-2 layout of `v2` bytes:
/// two u64-length-prefixed buffers (C_vv, then C_xx). Panics unless they fill the file exactly.
pub fn trailing_buffers(bytes: &[u8], v2: usize) -> (Vec<u8>, Vec<u8>) {
    let u = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) as usize;
    let l1 = u(v2);
    let b1 = bytes[v2 + 8..v2 + 8 + l1].to_vec();
    let at = v2 + 8 + l1;
    let l2 = u(at);
    let b2 = bytes[at + 8..at + 8 + l2].to_vec();
    assert_eq!(
        at + 8 + l2,
        bytes.len(),
        "bytes after the two correlation buffers"
    );
    (b1, b2)
}
