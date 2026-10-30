//! Test-only helpers for the `pimd-cli` integration tests (MILESTONE_4).
//!
//! The reference run (`reference_run`) is written here from the SPEC and the MILESTONE_4
//! clarifications, using only merged M1–M3 library code: `RingPolymerSim` (BAOAB + PILE-L, I-1,
//! T-1), `LennardJones` (P-3), `Rdf` (E-7), the energy estimators (E-1 … E-4) and `Rng` (D-2).
//! The initial lattice, the momentum draw order, the recording rule and the observables'
//! definitions are re-derived from the brief, independently of the run driver under test.
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

use std::path::Path;

use pimd_cli::Config;
use pimd_core::units::K_B_HARTREE_PER_KELVIN;
use pimd_core::{Beads, Rng};
use pimd_estimators::{Rdf, potential_mean, total_primitive, total_virial};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat};
use pimd_potentials::LennardJones;

/// 1 Eh in eV, CODATA 2018 (27.211386245988 eV; NIST). Hand value for checking the `*_mev`
/// columns; see QUESTIONS.md Q13 (the constant is not in `pimd_core::units` yet).
pub const EV_PER_HARTREE: f64 = 27.211386245988;

/// The observables.csv header (MILESTONE_4 "Outputs").
pub const OBS_HEADER: &str =
    "step,e_prim_eh,e_vir_eh,v_eh,t_est_k,e_prim_mev,e_vir_mev,e_prim_k,e_vir_k";

/// Small neon system for the fast tests: N = 32 on the pinned lattice (n_side = 4, spacing
/// 3.125 Å > σ), box 12.5 Å, cutoff 2σ + skin 0.25σ = 6.185 Å ≤ L/2 = 6.25 Å (C-4), P = 4.
pub const BASE: &str = r#"
[system]
n_atoms = 32
mass_amu = 20.1797
box_angstrom = 12.5

[potential]
kind = "lj"
epsilon_kelvin = 35.6
sigma_angstrom = 2.749
cutoff_sigma = 2.0
skin_sigma = 0.25
shift = true

[sampling]
temperature_kelvin = 30.0
n_beads = 4
dt_fs = 1.0
tau0_fs = 100.0
n_equil = 20
n_steps = 40
stride = 5
seed = 12345

[output]
rdf_bins = 20
rdf_rmax_angstrom = 6.0
checkpoint_every = 10
"#;

/// Replaces the value of `key` (the line `key = …`) in `toml`; adds `key = value` at the end of
/// `[section]` when the key is absent. Panics if neither applies.
pub fn set(toml: &str, section: &str, key: &str, value: &str) -> String {
    let mut out = Vec::new();
    let mut found = false;
    for line in toml.lines() {
        let t = line.trim_start();
        if t.starts_with(&format!("{key} ")) || t.starts_with(&format!("{key}=")) {
            out.push(format!("{key} = {value}"));
            found = true;
        } else {
            out.push(line.to_string());
        }
    }
    if !found {
        let pos = out
            .iter()
            .position(|l| l.trim() == format!("[{section}]"))
            .unwrap_or_else(|| panic!("section [{section}] not in config"));
        out.insert(pos + 1, format!("{key} = {value}"));
    }
    out.join("\n") + "\n"
}

/// Removes the line `key = …` from `toml`; panics if absent.
pub fn remove(toml: &str, key: &str) -> String {
    let before = toml.lines().count();
    let out: Vec<&str> = toml
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            !(t.starts_with(&format!("{key} ")) || t.starts_with(&format!("{key}=")))
        })
        .collect();
    assert_eq!(out.len() + 1, before, "key {key} not found exactly once");
    out.join("\n") + "\n"
}

/// Parses `toml`, panicking with the error on failure.
pub fn parse(toml: &str) -> Config {
    Config::from_toml_str(toml).unwrap_or_else(|e| panic!("config rejected: {e:?}"))
}

/// A rayon pool with exactly `n` threads (overrides RAYON_NUM_THREADS).
pub fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

/// A fresh temporary directory.
pub fn tmp() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

/// Reads a whole file as bytes, panicking with the path.
pub fn bytes(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Reads a whole file as text.
pub fn text(path: &Path) -> String {
    String::from_utf8(bytes(path)).unwrap()
}

/// Header and rows of a comma-separated file with no quoting.
pub fn read_csv(path: &Path) -> (String, Vec<Vec<String>>) {
    let t = text(path);
    let mut lines = t.lines();
    let header = lines.next().unwrap_or("").to_string();
    let rows = lines
        .filter(|l| !l.is_empty())
        .map(|l| l.split(',').map(str::to_string).collect())
        .collect();
    (header, rows)
}

/// Rows of `observables.csv` as numbers (column 0, the step, as f64 too).
pub fn observables(dir: &Path) -> Vec<Vec<f64>> {
    let (header, rows) = read_csv(&dir.join("observables.csv"));
    assert_eq!(header, OBS_HEADER);
    rows.iter()
        .map(|r| r.iter().map(|v| v.parse::<f64>().unwrap()).collect())
        .collect()
}

/// One recorded production step of the reference run, every value per atom where the brief says
/// so (MILESTONE_4 "Outputs").
#[derive(Debug, Clone)]
pub struct RefRow {
    pub step: usize,
    /// E_prim / N, Eh (E-4).
    pub e_prim: f64,
    /// E_vir / N, Eh (E-4).
    pub e_vir: f64,
    /// V̄ / N, Eh (E-3).
    pub v: f64,
    /// T_est in K: k_B T_est = Σ_{j,i} |p_{j,i}|²/m_i / (3 N P²).
    pub t_est: f64,
}

pub struct RefRun {
    pub rows: Vec<RefRow>,
    /// (bin centres, g) from `Rdf`, accumulated at the recorded steps.
    pub rdf: Option<(Vec<f64>, Vec<f64>)>,
}

/// Initial lattice (MILESTONE_4 "Initial state"): n_side = ⌈N^(1/3)⌉ computed exactly in
/// integers, a = L/n_side per axis, site (ix, iy, iz) at ((ix + ½)a_x, (iy + ½)a_y, (iz + ½)a_z),
/// ix fastest, then iy, then iz; the first N sites.
pub fn lattice(n: usize, l: [f64; 3]) -> Vec<[f64; 3]> {
    let mut side = 1;
    while side * side * side < n {
        side += 1;
    }
    let a = l.map(|x| x / side as f64);
    let mut out = Vec::with_capacity(n);
    'fill: for iz in 0..side {
        for iy in 0..side {
            for ix in 0..side {
                if out.len() == n {
                    break 'fill;
                }
                out.push([
                    (ix as f64 + 0.5) * a[0],
                    (iy as f64 + 0.5) * a[1],
                    (iz as f64 + 0.5) * a[2],
                ]);
            }
        }
    }
    out
}

/// The initial ring-polymer state of the brief: every bead of atom i at lattice site i;
/// momenta N(0, m P/β) per component (RP-3) from `rng`, atom i outer, bead j, dimension d inner.
pub fn initial_state(cfg: &Config, rng: &mut Rng) -> RingState {
    let n = cfg.system.n_atoms;
    let p = cfg.sampling.n_beads;
    let m = cfg.system.mass;
    let sites = lattice(n, cfg.system.cell.lengths());
    let mut q = Beads::zeros(p, n);
    let mut mom = Beads::zeros(p, n);
    let s = (m * p as f64 / cfg.sampling.beta).sqrt();
    for i in 0..n {
        for j in 0..p {
            q.set(j, i, sites[i]);
            let mut v = [0.0; 3];
            for x in &mut v {
                *x = s * rng.normal();
            }
            mom.set(j, i, v);
        }
    }
    RingState {
        q,
        p: mom,
        masses: vec![m; n],
    }
}

/// The run of the brief written as a plain loop: one `Rng::from_seed(seed)` for the initial
/// momenta and then the thermostat (D-2), `RingPolymerSim` with PILE-L(τ_0) and the matrix
/// transform, `LennardJones` in the config cell; `n_equil` steps recording nothing, then
/// production steps s = 1 … n_steps recorded when s % stride == 0, after the step.
pub fn reference_run(cfg: &Config) -> RefRun {
    let s = &cfg.sampling;
    let n = cfg.system.n_atoms;
    let p = s.n_beads;
    let beta = s.beta;
    let mut rng = Rng::from_seed(s.seed);
    let state = initial_state(cfg, &mut rng);
    let ig = RingPolymerIntegrator::new(s.dt, beta, p, Thermostat::PileL { tau0: Some(s.tau0) })
        .unwrap();
    let pot = LennardJones::new(cfg.potential).unwrap();
    let cell = cfg.system.cell;
    let mut sim = RingPolymerSim::new(state, Some(cell), ig, Box::new(pot)).unwrap();
    let mut rdf = cfg
        .output
        .rdf
        .as_ref()
        .map(|r| Rdf::new(r.n_bins, r.r_max, cell, n).unwrap());
    for _ in 0..s.n_equil {
        sim.step(&mut rng).unwrap();
    }
    let nf = n as f64;
    let mut rows = Vec::new();
    for step in 1..=s.n_steps {
        sim.step(&mut rng).unwrap();
        if step % s.stride != 0 {
            continue;
        }
        let st = sim.state();
        let e = sim.energies();
        let mut p2m = 0.0;
        for j in 0..p {
            for i in 0..n {
                let v = st.p.get(j, i);
                p2m += (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) / st.masses[i];
            }
        }
        let kt = p2m / (3.0 * nf * (p * p) as f64);
        rows.push(RefRow {
            step,
            e_prim: total_primitive(&st.q, &st.masses, beta, e) / nf,
            e_vir: total_virial(&st.q, sim.forces(), beta, e) / nf,
            v: potential_mean(e) / nf,
            t_est: kt / K_B_HARTREE_PER_KELVIN,
        });
        if let Some(r) = rdf.as_mut() {
            r.accumulate(&st.q).unwrap();
        }
    }
    RefRun {
        rows,
        rdf: rdf.map(|r| (r.bin_centers(), r.g())),
    }
}

/// |a − b| / max(|b|, floor).
pub fn rel(a: f64, b: f64, floor: f64) -> f64 {
    (a - b).abs() / b.abs().max(floor)
}

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
/// samples; asserts σ_4len ≤ 1.5 σ_len (blocks of `len` already uncorrelated, up to the noise of
/// σ itself) and returns the larger of the two. Series that are identically zero give σ = 0.
pub fn mean_and_sigma(x: &[f64], len: usize, what: &str) -> (f64, f64) {
    let (m, s1) = block_mean_sigma(x, len);
    let (_, s4) = block_mean_sigma(x, 4 * len);
    assert!(
        s4 <= 1.5 * s1 || s1 == 0.0 && s4 == 0.0,
        "{what}: block length {len} too short (sigma {s1:e} vs {s4:e} at 4x)"
    );
    (m, s1.max(s4))
}
