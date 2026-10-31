//! The run driver: initial state, equilibration, production with recording, checkpoint/restart
//! (MILESTONE_4; I-1, T-1, E-1 … E-4, E-7, D-2, D-3).

use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use pimd_core::units::{BOHR_IN_ANGSTROM, HARTREE_IN_EV, K_B_HARTREE_PER_KELVIN};
use pimd_core::{Beads, Rng, RngState};
use pimd_estimators::{Rdf, potential_mean, total_primitive, total_virial};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat};
use pimd_potentials::LennardJones;

use crate::config::{Config, PotentialConfig, put_u64};
use crate::error::CliError;

/// Results of a run (or of the run continued by `run_from_checkpoint`).
#[derive(Debug, Clone, PartialEq)]
pub struct RunSummary {
    /// Production steps completed (the config's `n_steps`).
    pub n_steps: usize,
    /// ⟨E_prim⟩ (E-4) per atom, Eh/atom, over every recorded production step.
    pub mean_e_prim: f64,
    /// ⟨E_vir⟩ (E-4) per atom, Eh/atom, over every recorded production step.
    pub mean_e_vir: f64,
    /// (bin centres in bohr, g) when the config has an RDF (E-7), else `None`.
    pub rdf: Option<(Vec<f64>, Vec<f64>)>,
}

/// Runs `config` from the initial lattice: `n_equil` equilibration steps, then `n_steps`
/// production steps. Writes `observables.csv`, `rdf.csv` (with an RDF), `checkpoint.bin` and
/// `run.log` into `out_dir` (created if missing).
///
/// The summary means are NaN when no production step was recorded (`n_steps < stride`).
///
/// Errors: `CliError::Io` for file-system errors; errors of the core crates.
pub fn run(config: &Config, out_dir: &Path) -> Result<RunSummary, CliError> {
    let s = &config.sampling;
    let mut rng = Rng::from_seed(s.seed);
    let state = initial_state(config, &mut rng);
    let sim = new_sim(config, state)?;
    let rdf = config
        .output
        .rdf
        .as_ref()
        .map(|r| {
            Rdf::new(
                r.n_bins,
                r.r_max,
                periodic_cell(config),
                config.system.n_atoms,
            )
        })
        .transpose()?;
    let mut d = Driver {
        sim,
        rng,
        rdf,
        step: 0,
        n_recorded: 0,
        sum_e_prim: 0.0,
        sum_e_vir: 0.0,
    };
    fs::create_dir_all(out_dir)?;
    write_log(config, out_dir, "initial lattice")?;
    // Equilibration records nothing.
    for _ in 0..s.n_equil {
        d.sim.step(&mut d.rng)?;
    }
    d.production(config, out_dir)
}

/// Continues the run saved in `checkpoint` until production step `config.sampling.n_steps`,
/// writing the outputs of the continued part into `out_dir`. The result is bitwise identical to
/// an uninterrupted `run` (D-3).
///
/// Restarting into the directory that holds the earlier run's outputs overwrites its
/// `observables.csv` and `run.log` (they then cover only the continued part); this is not
/// prevented (N1).
///
/// Errors: `CliError::Checkpoint` for a corrupt file, a config whose hash or RDF settings
/// (n_bins, r_max) differ from the checkpoint's, or a checkpoint beyond `n_steps`; otherwise as
/// `run`.
pub fn run_from_checkpoint(
    config: &Config,
    checkpoint: &Path,
    out_dir: &Path,
) -> Result<RunSummary, CliError> {
    let bytes = fs::read(checkpoint)?;
    let d = Driver::decode(config, &bytes)?;
    if d.step > config.sampling.n_steps {
        return Err(CliError::Checkpoint(format!(
            "checkpoint is at production step {}, beyond n_steps = {}",
            d.step, config.sampling.n_steps
        )));
    }
    fs::create_dir_all(out_dir)?;
    write_log(
        config,
        out_dir,
        &format!("checkpoint at production step {}", d.step),
    )?;
    d.production(config, out_dir)
}

/// Everything a run carries from step to step; exactly what `checkpoint.bin` stores.
struct Driver {
    sim: RingPolymerSim,
    rng: Rng,
    rdf: Option<Rdf>,
    /// Production steps completed.
    step: usize,
    /// Recorded production steps.
    n_recorded: usize,
    /// Σ E_prim/N over recorded steps, Eh.
    sum_e_prim: f64,
    /// Σ E_vir/N over recorded steps, Eh.
    sum_e_vir: f64,
}

/// The observables.csv header (MILESTONE_4 "Outputs").
const OBS_HEADER: &str =
    "step,e_prim_eh,e_vir_eh,v_eh,t_est_k,e_prim_mev,e_vir_mev,e_prim_k,e_vir_k";

/// Checkpoint magic and format version.
const MAGIC: &[u8; 8] = b"PIMDCKPT";
const VERSION: u64 = 2;

/// The initial ring-polymer state (MILESTONE_4 "Initial state"): every bead of atom i at lattice
/// site i; momenta N(0, m·P/β) per component (RP-3), drawn atom i outer, bead j, dimension d inner.
fn initial_state(config: &Config, rng: &mut Rng) -> RingState {
    let n = config.system.n_atoms;
    let p = config.sampling.n_beads;
    if config.system.positions.is_some() {
        unimplemented!("M5: initial positions from the config");
    }
    let masses = config.system.masses.clone();
    let sites = lattice(n, periodic_cell(config).lengths());
    let mut q = Beads::zeros(p, n);
    let mut mom = Beads::zeros(p, n);
    for (i, &site) in sites.iter().enumerate() {
        let sd = (masses[i] * p as f64 / config.sampling.beta).sqrt();
        for j in 0..p {
            q.set(j, i, site);
            let mut v = [0.0; 3];
            for x in &mut v {
                *x = sd * rng.normal();
            }
            mom.set(j, i, v);
        }
    }
    RingState { q, p: mom, masses }
}

/// The config's periodic cell (M5 stub: the gas phase is not implemented yet).
fn periodic_cell(config: &Config) -> pimd_core::Cell {
    match config.system.cell {
        Some(c) => c,
        None => unimplemented!("M5: gas phase"),
    }
}

/// The first `n` sites of a simple cubic lattice with n_side = ⌈n^(1/3)⌉ (exact, in integers),
/// spacing L/n_side per axis, sites at (i + ½)·a, i_x fastest, then i_y, then i_z. Bohr.
fn lattice(n: usize, l: [f64; 3]) -> Vec<[f64; 3]> {
    let mut side = 1;
    while side * side * side < n {
        side += 1;
    }
    let a = l.map(|x| x / side as f64);
    (0..n)
        .map(|k| {
            let idx = [k % side, (k / side) % side, k / (side * side)];
            [0, 1, 2].map(|d| (idx[d] as f64 + 0.5) * a[d])
        })
        .collect()
}

/// `RingPolymerSim` with BAOAB + PILE-L(τ_0), the configured transform, and LJ in the config cell.
fn new_sim(config: &Config, state: RingState) -> Result<RingPolymerSim, CliError> {
    let s = &config.sampling;
    let ig = RingPolymerIntegrator::with_transform(
        s.dt,
        s.beta,
        s.n_beads,
        Thermostat::PileL { tau0: Some(s.tau0) },
        s.normal_modes,
    )?;
    let pot = match &config.potential {
        PotentialConfig::Lj(lj) => LennardJones::new(*lj)?,
        PotentialConfig::Socket { .. } => unimplemented!("M5: socket potential in the driver"),
    };
    Ok(RingPolymerSim::new(
        state,
        Some(periodic_cell(config)),
        ig,
        Box::new(pot),
    )?)
}

/// `run.log`: version, seed, start point, config hash and the parsed config. Nothing
/// machine- or time-dependent.
fn write_log(config: &Config, out_dir: &Path, start: &str) -> Result<(), CliError> {
    let log = format!(
        concat!(
            "pimd-rs {}\nseed = {}\nstart = {start}\n",
            "config hash = {:#018x}\nconfig (atomic units) = {config:#?}\n",
        ),
        env!("CARGO_PKG_VERSION"),
        config.sampling.seed,
        config.hash(),
        start = start,
        config = config,
    );
    fs::write(out_dir.join("run.log"), log)?;
    Ok(())
}

impl Driver {
    /// Production from `self.step` to `n_steps`: step s is recorded when s % stride == 0, after
    /// the step; checkpoints every `checkpoint_every` steps and at the end.
    fn production(mut self, config: &Config, out_dir: &Path) -> Result<RunSummary, CliError> {
        let s = &config.sampling;
        if config.output.trajectory_every.is_some() {
            unimplemented!("M5: trajectory output");
        }
        let mut obs = BufWriter::new(fs::File::create(out_dir.join("observables.csv"))?);
        writeln!(obs, "{OBS_HEADER}")?;
        while self.step < s.n_steps {
            self.sim.step(&mut self.rng)?;
            self.step += 1;
            if self.step.is_multiple_of(s.stride) {
                self.record(config, &mut obs)?;
            }
            if self.step.is_multiple_of(config.output.checkpoint_every) && self.step < s.n_steps {
                obs.flush()?;
                self.write_checkpoint(config, out_dir)?;
            }
        }
        obs.flush()?;
        self.write_checkpoint(config, out_dir)?;
        let rdf = self.rdf.as_ref().map(|r| (r.bin_centers(), r.g()));
        if let Some((r, g)) = &rdf {
            let mut f = BufWriter::new(fs::File::create(out_dir.join("rdf.csv"))?);
            writeln!(f, "r_bohr,r_angstrom,g")?;
            for (r, g) in r.iter().zip(g) {
                writeln!(f, "{},{},{}", r, r * BOHR_IN_ANGSTROM, g)?;
            }
            f.flush()?;
        }
        let n = self.n_recorded as f64;
        Ok(RunSummary {
            n_steps: s.n_steps,
            mean_e_prim: self.sum_e_prim / n,
            mean_e_vir: self.sum_e_vir / n,
            rdf,
        })
    }

    /// One observables row (per-atom E_prim, E_vir, V̄ in Eh, T_est in K, then meV and K
    /// columns, via `units` only, U-3), the running sums, and an RDF sample.
    fn record(&mut self, config: &Config, obs: &mut impl Write) -> Result<(), CliError> {
        let beta = config.sampling.beta;
        let st = self.sim.state();
        let e = self.sim.energies();
        let (p, n) = (st.q.n_beads(), st.q.n_atoms());
        let nf = n as f64;
        let e_prim = total_primitive(&st.q, &st.masses, beta, e) / nf;
        let e_vir = total_virial(&st.q, self.sim.forces(), beta, e) / nf;
        let v = potential_mean(e) / nf;
        // k_B T_est = Σ_{j,i} |p_{j,i}|²/m_i / (3 N P²).
        let mut p2m = 0.0;
        for j in 0..p {
            for (i, &m) in st.masses.iter().enumerate() {
                let x = st.p.get(j, i);
                p2m += (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) / m;
            }
        }
        let t_est = p2m / (3.0 * nf * (p * p) as f64) / K_B_HARTREE_PER_KELVIN;
        let mev = HARTREE_IN_EV * 1000.0;
        writeln!(
            obs,
            "{},{},{},{},{},{},{},{},{}",
            self.step,
            e_prim,
            e_vir,
            v,
            t_est,
            e_prim * mev,
            e_vir * mev,
            e_prim / K_B_HARTREE_PER_KELVIN,
            e_vir / K_B_HARTREE_PER_KELVIN
        )?;
        self.sum_e_prim += e_prim;
        self.sum_e_vir += e_vir;
        self.n_recorded += 1;
        if let Some(r) = self.rdf.as_mut() {
            r.accumulate(&st.q)?;
        }
        Ok(())
    }

    /// Writes `checkpoint.bin` (via a temporary file and a rename, so an interrupted write never
    /// leaves a truncated checkpoint).
    fn write_checkpoint(&self, config: &Config, out_dir: &Path) -> Result<(), CliError> {
        let tmp = out_dir.join("checkpoint.bin.tmp");
        fs::write(&tmp, self.encode(config))?;
        fs::rename(&tmp, out_dir.join("checkpoint.bin"))?;
        Ok(())
    }

    /// Checkpoint format, all little-endian: magic `PIMDCKPT`, version (u64), config hash (u64),
    /// P, N, production step, recorded steps (u64), Σ E_prim/N, Σ E_vir/N (f64), RNG state length
    /// (u64) and bytes, q, p, forces (P·N·3 f64 each), energies (P f64), RDF flag (u64 0/1) and,
    /// with an RDF, n_bins (u64), r_max (f64, bohr), n_samples (u64) and the counts (n_bins f64).
    /// Depends only on the state, the accumulators, the config hash and the RDF settings.
    fn encode(&self, config: &Config) -> Vec<u8> {
        let st = self.sim.state();
        let (p, n) = (st.q.n_beads(), st.q.n_atoms());
        let rng = self.rng.snapshot().to_bytes();
        let mut w = Vec::with_capacity(128 + rng.len() + 8 * (3 * p * n * 3 + p));
        let u = put_u64;
        w.extend_from_slice(MAGIC);
        u(&mut w, VERSION);
        u(&mut w, config.hash());
        u(&mut w, p as u64);
        u(&mut w, n as u64);
        u(&mut w, self.step as u64);
        u(&mut w, self.n_recorded as u64);
        u(&mut w, self.sum_e_prim.to_bits());
        u(&mut w, self.sum_e_vir.to_bits());
        u(&mut w, rng.len() as u64);
        w.extend_from_slice(&rng);
        for b in [&st.q, &st.p, self.sim.forces()] {
            for &x in b.as_slice() {
                u(&mut w, x.to_bits());
            }
        }
        for &x in self.sim.energies() {
            u(&mut w, x.to_bits());
        }
        // `self.rdf` is built from `config.output.rdf` (n_bins, r_max); it holds the counts.
        match (&self.rdf, &config.output.rdf) {
            (Some(r), Some(rc)) => {
                u(&mut w, 1);
                u(&mut w, rc.n_bins as u64);
                u(&mut w, rc.r_max.to_bits());
                u(&mut w, r.n_samples() as u64);
                for &c in r.counts() {
                    u(&mut w, c.to_bits());
                }
            }
            _ => u(&mut w, 0),
        }
        w
    }

    /// Inverse of `encode` for `config`. The simulation is rebuilt from the saved positions and
    /// momenta; its initialising evaluation must reproduce the saved forces and energies bitwise
    /// (LJ results do not depend on neighbor-list history).
    ///
    /// Errors: `CliError::Checkpoint` for a malformed file (bad magic or version, wrong length),
    /// a config-hash mismatch, saved shapes or RDF settings that disagree with `config`, or saved
    /// forces and energies that a re-evaluation does not reproduce.
    fn decode(config: &Config, bytes: &[u8]) -> Result<Driver, CliError> {
        let bad = |msg: &str| CliError::Checkpoint(msg.to_string());
        let mut r = Reader { bytes, pos: 0 };
        if r.take(8)? != MAGIC {
            return Err(bad("not a pimd-rs checkpoint (bad magic)"));
        }
        let version = r.u64()?;
        if version != VERSION {
            return Err(CliError::Checkpoint(format!(
                "unsupported checkpoint version {version}"
            )));
        }
        if r.u64()? != config.hash() {
            return Err(bad("config hash differs from the checkpoint's"));
        }
        let (p, n) = (r.usize()?, r.usize()?);
        if p != config.sampling.n_beads || n != config.system.n_atoms {
            return Err(bad("bead or atom count differs from the config"));
        }
        let step = r.usize()?;
        let n_recorded = r.usize()?;
        let sum_e_prim = r.f64()?;
        let sum_e_vir = r.f64()?;
        let rng_len = r.usize()?;
        let rng = RngState::from_bytes(r.take(rng_len)?)
            .map_err(|e| CliError::Checkpoint(format!("RNG state: {e}")))?;
        let len = p
            .checked_mul(n)
            .and_then(|x| x.checked_mul(3))
            .ok_or_else(|| bad("shape overflow"))?;
        let mut beads = || -> Result<Beads, CliError> {
            let v = r.f64s(len)?;
            Beads::from_vec(p, n, v).map_err(|e| CliError::Checkpoint(e.to_string()))
        };
        let (q, mom, forces) = (beads()?, beads()?, beads()?);
        let energies = r.f64s(p)?;
        let rdf = match (r.u64()?, &config.output.rdf) {
            (0, None) => None,
            (1, Some(rc)) => {
                let n_bins = r.usize()?;
                let r_max = r.f64()?;
                // Q16: `[output]` is outside the config hash, so the RDF settings are checked here.
                if n_bins != rc.n_bins || r_max.to_bits() != rc.r_max.to_bits() {
                    return Err(bad(
                        "RDF settings (n_bins, r_max) differ from the checkpoint's",
                    ));
                }
                let n_samples = r.usize()?;
                let counts = r.f64s(n_bins)?;
                Some(
                    Rdf::from_parts(
                        rc.n_bins,
                        rc.r_max,
                        periodic_cell(config),
                        n,
                        counts,
                        n_samples,
                    )
                    .map_err(|e| CliError::Checkpoint(format!("RDF: {e}")))?,
                )
            }
            (0 | 1, _) => return Err(bad("RDF presence differs from the config's [output]")),
            _ => return Err(bad("corrupt RDF flag")),
        };
        if r.pos != bytes.len() {
            return Err(bad("trailing bytes after the checkpoint data"));
        }
        let state = RingState {
            q,
            p: mom,
            masses: config.system.masses.clone(),
        };
        let sim = new_sim(config, state)?;
        let same = |a: &[f64], b: &[f64]| {
            a.iter()
                .map(|x| x.to_bits())
                .eq(b.iter().map(|x| x.to_bits()))
        };
        if !same(sim.forces().as_slice(), forces.as_slice()) || !same(sim.energies(), &energies) {
            return Err(bad(
                "saved forces or energies do not match a re-evaluation at the saved positions; \
                 the file is corrupt or was written by a different build or machine",
            ));
        }
        Ok(Driver {
            sim,
            rng: Rng::restore(&rng),
            rdf,
            step,
            n_recorded,
            sum_e_prim,
            sum_e_vir,
        })
    }
}

/// Bounds-checked little-endian reader over a checkpoint; running past the end is
/// `CliError::Checkpoint` (truncated file).
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, k: usize) -> Result<&'a [u8], CliError> {
        let end = self
            .pos
            .checked_add(k)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| CliError::Checkpoint("truncated checkpoint".to_string()))?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u64(&mut self) -> Result<u64, CliError> {
        Ok(le_u64(self.take(8)?))
    }

    fn usize(&mut self) -> Result<usize, CliError> {
        usize::try_from(self.u64()?)
            .map_err(|_| CliError::Checkpoint("count out of range".to_string()))
    }

    fn f64(&mut self) -> Result<f64, CliError> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn f64s(&mut self, k: usize) -> Result<Vec<f64>, CliError> {
        let raw = self.take(
            k.checked_mul(8)
                .ok_or_else(|| CliError::Checkpoint("count out of range".to_string()))?,
        )?;
        Ok(raw
            .chunks_exact(8)
            .map(|c| f64::from_bits(le_u64(c)))
            .collect())
    }
}

/// The u64 encoded little-endian in the 8 bytes `b`.
fn le_u64(b: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(b);
    u64::from_le_bytes(a)
}
