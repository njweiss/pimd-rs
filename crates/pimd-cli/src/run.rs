//! The run driver: initial state, equilibration, production with recording, checkpoint/restart
//! (MILESTONE_4; I-1, T-1, E-1 … E-4, E-7, D-2, D-3), and the RPMD/TRPMD/PA-CMD segment cycle
//! of a `[dynamics]` run (MILESTONE_7; Y-1 … Y-7, T-4).

use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Duration;

use pimd_core::units::{
    BOHR_IN_ANGSTROM, FS_IN_ATOMIC_TIME, HARTREE_IN_EV, HARTREE_IN_WAVENUMBERS,
    K_B_HARTREE_PER_KELVIN,
};
use pimd_core::{Beads, Rng, RngState};
use pimd_estimators::{
    CentroidCorrelation, Rdf, centroid_velocity, green_kubo, hamiltonian, hamiltonian_dynamical,
    potential_mean, total_primitive, total_virial,
};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat};
use pimd_potentials::{LennardJones, Potential};
use pimd_socket::SocketPotential;

use crate::config::{
    Config, DynamicsConfig, DynamicsMethod, GAS_PHASE_NEEDS_POSITIONS, PotentialConfig, invalid,
    put_u64,
};
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
    /// Centroid correlation functions of a `[dynamics]` run (Y-4, Y-5), else `None`.
    pub correlations: Option<Correlations>,
}

/// Centroid time-correlation functions of a `[dynamics]` run (MILESTONE_7).
#[derive(Debug, Clone, PartialEq)]
pub struct Correlations {
    /// Lag spacing corr_stride·dt, ħ/Eh.
    pub dt_lag: f64,
    /// Segments averaged.
    pub n_segments: usize,
    /// C_vv(τ) (Y-4), (bohr/(ħ/Eh))², length n_lags.
    pub c_vv: Vec<f64>,
    /// σ of `c_vv` (Y-4).
    pub c_vv_sem: Vec<f64>,
    /// (C_xx, σ) in bohr² (Y-4); `None` with a cell.
    pub c_xx: Option<(Vec<f64>, Vec<f64>)>,
    /// Running Green–Kubo D(τ) (Y-5), bohr²/(ħ/Eh).
    pub diffusion: Vec<f64>,
}

/// Runs `config` from its initial positions (or the initial lattice): `n_equil` equilibration
/// steps, then `n_steps` production steps. Writes `observables.csv`, `rdf.csv` (with an RDF),
/// `trajectory.xyz` (with `trajectory_every`), `checkpoint.bin` and `run.log` into `out_dir`
/// (created if missing). A socket potential binds its address and waits for its clients first.
///
/// With `[dynamics]`, production is the segment cycle of Y-1 (decorrelation steps with PILE-L,
/// the Y-2 redraw, a segment with `Off` or `Trpmd`), and the run also writes `segments.csv` (one
/// row per segment) and `correlations.csv` (at the end), MILESTONE_7 "Outputs". With
/// `method = "pa-cmd"` the Y-6 masses apply to the whole run (Y-7) and the initial momenta are
/// the amended Y-2 draw (M7b "Clarifications", Q32).
///
/// The summary means are NaN when no production step was recorded (`n_steps < stride`).
///
/// Errors: `CliError::Io` for file-system errors; errors of the core crates.
pub fn run(config: &Config, out_dir: &Path) -> Result<RunSummary, CliError> {
    let s = &config.sampling;
    let mut rng = Rng::from_seed(s.seed);
    let state = initial_state(config, &mut rng)?;
    let rdf = match (&config.output.rdf, config.system.cell) {
        (Some(r), Some(cell)) => Some(Rdf::new(r.n_bins, r.r_max, cell, config.system.n_atoms)?),
        _ => None,
    };
    let mut sim = RingPolymerSim::new(
        state,
        config.system.cell,
        integrator(config)?,
        potential(config)?,
    )?;
    if sim.integrator().adiabatic_frequency().is_some() {
        // Q32: the amended Y-2 draw of the same P·N·3 normals that `initial_state` drew (a fresh
        // RNG from the seed repeats them; `rng` has already advanced past them).
        sim.resample_momenta(&mut Rng::from_seed(s.seed));
    }
    let dynamics = config
        .dynamics
        .as_ref()
        .map(|dy| DynState::new(dy, config.system.n_atoms))
        .transpose()?;
    let mut d = Driver {
        sim,
        rng,
        rdf,
        dynamics,
        step: 0,
        n_recorded: 0,
        sum_e_prim: 0.0,
        sum_e_vir: 0.0,
    };
    fs::create_dir_all(out_dir)?;
    let start = if config.system.positions.is_some() {
        "initial positions"
    } else {
        "initial lattice"
    };
    write_log(config, out_dir, start)?;
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
    /// Correlation accumulators of a `[dynamics]` run.
    dynamics: Option<DynState>,
    /// Production steps completed.
    step: usize,
    /// Recorded production steps.
    n_recorded: usize,
    /// Σ E_prim/N over recorded steps, Eh.
    sum_e_prim: f64,
    /// Σ E_vir/N over recorded steps, Eh.
    sum_e_vir: f64,
}

/// The correlation accumulators of a `[dynamics]` run (Y-4) and the open segment's H_P.
struct DynState {
    /// C_vv of the centroid velocities v̄ (Y-3).
    c_vv: CentroidCorrelation,
    /// C_xx of the stored (unwrapped) centroids q̄ (Y-3); accumulated with a cell too.
    c_xx: CentroidCorrelation,
    /// H_P/N right after the current segment's redraw, Eh.
    h_start: f64,
}

impl DynState {
    fn new(dy: &DynamicsConfig, n_atoms: usize) -> Result<DynState, CliError> {
        Ok(DynState {
            c_vv: CentroidCorrelation::new(dy.n_lags, n_atoms)?,
            c_xx: CentroidCorrelation::new(dy.n_lags, n_atoms)?,
            h_start: f64::NAN,
        })
    }
}

/// The correlations.csv header (MILESTONE_7 "Outputs").
const CORR_HEADER: &str =
    "lag,t_fs,c_vv_au,c_vv_sem_au,d_au,d_angstrom2_per_ps,c_xx_bohr2,c_xx_sem_bohr2";

/// The segments.csv header (MILESTONE_7 "Outputs").
const SEG_HEADER: &str = "segment,step,h_start_eh,h_end_eh";

/// The observables.csv header (MILESTONE_4 "Outputs").
const OBS_HEADER: &str =
    "step,e_prim_eh,e_vir_eh,v_eh,t_est_k,e_prim_mev,e_vir_mev,e_prim_k,e_vir_k";

/// Checkpoint magic and format version.
const MAGIC: &[u8; 8] = b"PIMDCKPT";
const VERSION: u64 = 2;

/// The initial ring-polymer state (MILESTONE_4 "Initial state", MILESTONE_5 "Initial state"):
/// every bead of atom i at `system.positions[i]` when given, else at lattice site i; momenta
/// N(0, m_i·P/β) per component (RP-3), drawn atom i outer, bead j, dimension d inner.
fn initial_state(config: &Config, rng: &mut Rng) -> Result<RingState, CliError> {
    let n = config.system.n_atoms;
    let p = config.sampling.n_beads;
    let masses = config.system.masses.clone();
    let lattice_sites;
    let sites: &[[f64; 3]] = match (&config.system.positions, config.system.cell) {
        (Some(x), _) => x,
        (None, Some(cell)) => {
            lattice_sites = lattice(n, cell.lengths());
            &lattice_sites
        }
        // Rejected by `Config::from_toml_str`; reachable only with a hand-built `Config`.
        (None, None) => return Err(CliError::Parse(GAS_PHASE_NEEDS_POSITIONS.to_string())),
    };
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
    Ok(RingState { q, p: mom, masses })
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

/// The sampling thermostat PILE-L(τ_0) (T-1, T-2).
fn sampling_thermostat(config: &Config) -> Thermostat {
    Thermostat::PileL {
        tau0: Some(config.sampling.tau0),
    }
}

/// The segment thermostat of `[dynamics]` (Y-1): `Off` for RPMD, `Trpmd` for TRPMD (T-4) and
/// PA-CMD (Y-7).
fn segment_thermostat(dy: &DynamicsConfig) -> Thermostat {
    match dy.method {
        DynamicsMethod::Rpmd => Thermostat::Off,
        DynamicsMethod::Trpmd { lambda } | DynamicsMethod::PaCmd { lambda, .. } => {
            Thermostat::Trpmd { lambda }
        }
    }
}

/// Ω (Eh/ħ) of a `method = "pa-cmd"` run, else `None`.
fn adiabatic_frequency(config: &Config) -> Option<f64> {
    match config.dynamics.as_ref()?.method {
        DynamicsMethod::PaCmd { omega, .. } => Some(omega),
        _ => None,
    }
}

/// BAOAB + PILE-L(τ_0) with the configured transform, and the Y-6 masses for PA-CMD (Y-7).
fn integrator(config: &Config) -> Result<RingPolymerIntegrator, CliError> {
    let s = &config.sampling;
    let ig = RingPolymerIntegrator::with_transform(
        s.dt,
        s.beta,
        s.n_beads,
        sampling_thermostat(config),
        s.normal_modes,
    )?;
    Ok(match adiabatic_frequency(config) {
        Some(omega) => ig.with_adiabatic_frequency(omega)?,
        None => ig,
    })
}

/// The configured potential: LJ in-process, the socket server once its clients have connected
/// (K-1), or a MACE export in-process (P-6, feature `torch`).
fn potential(config: &Config) -> Result<Box<dyn Potential>, CliError> {
    Ok(match &config.potential {
        PotentialConfig::Lj(lj) => Box::new(LennardJones::new(*lj)?),
        PotentialConfig::Socket {
            address,
            n_clients,
            timeout_s,
        } => {
            let timeout = Duration::try_from_secs_f64(*timeout_s)
                .map_err(|_| invalid("potential.timeout_s", *timeout_s))?;
            Box::new(SocketPotential::bind_and_accept(
                address,
                config.system.n_atoms,
                *n_clients,
                timeout,
            )?)
        }
        #[cfg(feature = "torch")]
        PotentialConfig::Mace {
            model,
            device,
            dtype,
            threads,
            deterministic,
            beads_per_graph,
            ..
        } => {
            // Validated against `pimd_torch::atomic_number` by `Config::from_toml_str`; a
            // hand-built config with bad symbols is refused here.
            let symbols = crate::config::need(config.system.symbols.as_deref(), "symbols")?;
            let z = symbols
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    pimd_torch::atomic_number(s).ok_or_else(|| invalid("system.symbols", i as f64))
                })
                .collect::<Result<Vec<u8>, CliError>>()?;
            let mace = pimd_torch::MacePotential::load(
                model,
                &z,
                *device,
                *dtype,
                *threads,
                *deterministic,
            )?;
            // P-6 (MILESTONE_8): chunks of c beads when set; one graph of all beads otherwise.
            Box::new(match beads_per_graph {
                Some(c) => mace.with_beads_per_graph(*c)?,
                None => mace,
            })
        }
    })
}

/// `run.log`: version, seed, start point, config hash, Ω and λ for PA-CMD (M7b
/// "Clarifications") and the parsed config. Nothing machine- or time-dependent.
fn write_log(config: &Config, out_dir: &Path, start: &str) -> Result<(), CliError> {
    let mut log = format!(
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
    if let Some(DynamicsMethod::PaCmd { omega, lambda }) =
        config.dynamics.as_ref().map(|d| &d.method)
    {
        log.push_str(&format!(
            "pa-cmd: Omega = {} cm^-1 = {omega} Eh/hbar, cmd_lambda = {lambda}\n",
            omega * HARTREE_IN_WAVENUMBERS
        ));
    }
    fs::write(out_dir.join("run.log"), log)?;
    Ok(())
}

impl Driver {
    /// Production from `self.step` to `n_steps`: step s is recorded when s % stride == 0, after
    /// the step; a `trajectory.xyz` frame at every step s with s % trajectory_every == 0;
    /// checkpoints every `checkpoint_every` steps and at the end. With `[dynamics]`, each step
    /// is placed in the Y-1 cycle by `self.step` (`dynamics_before`, `dynamics_after`).
    fn production(mut self, config: &Config, out_dir: &Path) -> Result<RunSummary, CliError> {
        let s = &config.sampling;
        let mut obs = BufWriter::new(fs::File::create(out_dir.join("observables.csv"))?);
        writeln!(obs, "{OBS_HEADER}")?;
        let mut segs = match &config.dynamics {
            Some(_) => {
                let mut f = BufWriter::new(fs::File::create(out_dir.join("segments.csv"))?);
                writeln!(f, "{SEG_HEADER}")?;
                Some(f)
            }
            None => None,
        };
        // (trajectory_every, writer)
        let mut traj = match config.output.trajectory_every {
            Some(t) => Some((
                t,
                BufWriter::new(fs::File::create(out_dir.join("trajectory.xyz"))?),
            )),
            None => None,
        };
        while self.step < s.n_steps {
            if let Some(dy) = &config.dynamics {
                self.dynamics_before(config, dy)?;
            }
            self.sim.step(&mut self.rng)?;
            self.step += 1;
            if let Some((t, f)) = traj.as_mut()
                && self.step.is_multiple_of(*t)
            {
                self.write_frame(config, f)?;
            }
            if self.step.is_multiple_of(s.stride) {
                self.record(config, &mut obs)?;
            }
            if let (Some(dy), Some(f)) = (&config.dynamics, segs.as_mut()) {
                self.dynamics_after(config, dy, f)?;
            }
            if self.step.is_multiple_of(config.output.checkpoint_every) && self.step < s.n_steps {
                obs.flush()?;
                if let Some(f) = segs.as_mut() {
                    f.flush()?;
                }
                if let Some((_, f)) = traj.as_mut() {
                    f.flush()?;
                }
                self.write_checkpoint(config, out_dir)?;
            }
        }
        obs.flush()?;
        if let Some((_, f)) = traj.as_mut() {
            f.flush()?;
        }
        if let Some(f) = segs.as_mut() {
            f.flush()?;
        }
        self.write_checkpoint(config, out_dir)?;
        let correlations = match (&config.dynamics, &self.dynamics) {
            (Some(dy), Some(ds)) => {
                let c = correlations(config, dy, ds);
                write_correlations(&c, out_dir)?;
                Some(c)
            }
            _ => None,
        };
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
            correlations,
        })
    }

    /// Before production step `self.step + 1` (Y-1): at a cycle start, the sampling thermostat;
    /// after `decorrelation_steps` steps of the cycle, the Y-2 redraw, H_P/N at the segment
    /// start, the segment thermostat and centroid sample s = 0 (Y-3).
    fn dynamics_before(&mut self, config: &Config, dy: &DynamicsConfig) -> Result<(), CliError> {
        let pos = self.step % (dy.decorrelation_steps + dy.segment_steps);
        if pos == 0 {
            self.sim.set_thermostat(sampling_thermostat(config))?;
        }
        if pos == dy.decorrelation_steps {
            self.sim.resample_momenta(&mut self.rng);
            let h = self.h_per_atom(config);
            self.sim.set_thermostat(segment_thermostat(dy))?;
            self.sample()?;
            if let Some(ds) = self.dynamics.as_mut() {
                ds.h_start = h;
            }
        }
        Ok(())
    }

    /// After production step `self.step` (Y-1, Y-3): a centroid sample at every `corr_stride`-th
    /// segment step; at the cycle end, H_P/N, the segment's C_seg (Y-4) and its segments.csv row
    /// (segment counted from 0, the production step, H_P/N at start and end; Q30).
    fn dynamics_after(
        &mut self,
        config: &Config,
        dy: &DynamicsConfig,
        segs: &mut impl Write,
    ) -> Result<(), CliError> {
        let cycle = dy.decorrelation_steps + dy.segment_steps;
        // Steps of the current cycle completed, 1 … cycle.
        let pos = (self.step - 1) % cycle + 1;
        if pos <= dy.decorrelation_steps {
            return Ok(());
        }
        if (pos - dy.decorrelation_steps).is_multiple_of(dy.corr_stride) {
            self.sample()?;
        }
        if pos == cycle {
            let h_end = self.h_per_atom(config);
            if let Some(ds) = self.dynamics.as_mut() {
                ds.c_vv.end_segment()?;
                ds.c_xx.end_segment()?;
                writeln!(
                    segs,
                    "{},{},{},{}",
                    self.step / cycle - 1,
                    self.step,
                    ds.h_start,
                    h_end
                )?;
            }
        }
        Ok(())
    }

    /// H_P/N for `segments.csv`: `hamiltonian_dynamical` (Y-6) under PA-CMD, else `hamiltonian`.
    fn h_per_atom(&self, config: &Config) -> f64 {
        let n = config.system.n_atoms as f64;
        let (ig, beta) = (self.sim.integrator(), config.sampling.beta);
        let (st, e) = (self.sim.state(), self.sim.energies());
        if ig.adiabatic_frequency().is_some() {
            hamiltonian_dynamical(st, beta, e, ig.modes(), ig.mass_factors()) / n
        } else {
            hamiltonian(st, beta, e) / n
        }
    }

    /// One centroid sample (Y-3): v̄ into C_vv, the stored q̄ into C_xx.
    fn sample(&mut self) -> Result<(), CliError> {
        if let Some(ds) = self.dynamics.as_mut() {
            let st = self.sim.state();
            ds.c_vv.push(&centroid_velocity(&st.p, &st.masses))?;
            ds.c_xx.push(&st.q.centroid())?;
        }
        Ok(())
    }

    /// One `trajectory.xyz` frame (MILESTONE_5 "Trajectory output"): `P·N`, then
    /// `step=<s> beads=<P> atoms=<N> units=angstrom`, then P·N lines `symbol x y z`, bead-major,
    /// unwrapped positions in Å (C-5, U-3), symbol `X` without `symbols`.
    fn write_frame(&self, config: &Config, f: &mut impl Write) -> Result<(), CliError> {
        let q = &self.sim.state().q;
        let (p, n) = (q.n_beads(), q.n_atoms());
        writeln!(f, "{}", p * n)?;
        writeln!(f, "step={} beads={p} atoms={n} units=angstrom", self.step)?;
        let symbols = config.system.symbols.as_deref();
        for j in 0..p {
            for i in 0..n {
                let sym = symbols.map_or("X", |v| v[i].as_str());
                let x = q.get(j, i).map(|c| c * BOHR_IN_ANGSTROM);
                writeln!(f, "{sym} {} {} {}", x[0], x[1], x[2])?;
            }
        }
        Ok(())
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
        fs::write(&tmp, self.encode(config)?)?;
        fs::rename(&tmp, out_dir.join("checkpoint.bin"))?;
        Ok(())
    }

    /// Checkpoint format, all little-endian: magic `PIMDCKPT`, version (u64), config hash (u64),
    /// P, N, production step, recorded steps (u64), Σ E_prim/N, Σ E_vir/N (f64), RNG state length
    /// (u64) and bytes, q, p, forces (P·N·3 f64 each), energies (P f64), RDF flag (u64 0/1) and,
    /// with an RDF, n_bins (u64), r_max (f64, bohr), n_samples (u64) and the counts (n_bins f64).
    /// With `[dynamics]` (MILESTONE_7 "Checkpoint block") this version-2 layout is followed by the
    /// C_vv and then the C_xx `CentroidCorrelation::to_bytes` buffers, each prefixed by its length
    /// (u64). Depends only on the state, the accumulators, the config hash and the RDF settings.
    ///
    /// Errors: `CliError::Estimator` if a segment is open (checkpoints fall on cycle ends).
    fn encode(&self, config: &Config) -> Result<Vec<u8>, CliError> {
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
        if let Some(ds) = &self.dynamics {
            for c in [&ds.c_vv, &ds.c_xx] {
                let b = c.to_bytes()?;
                u(&mut w, b.len() as u64);
                w.extend_from_slice(&b);
            }
        }
        Ok(w)
    }

    /// Inverse of `encode` for `config`. The simulation is rebuilt from the saved positions,
    /// momenta, forces and energies with `RingPolymerSim::from_parts`, without re-evaluating
    /// (Q18). The potential (for a socket: binding and accepting clients) is built only after the
    /// whole file has been checked.
    ///
    /// Errors: `CliError::Checkpoint` for a malformed file (bad magic or version, wrong length),
    /// a config-hash mismatch, or saved shapes or RDF settings that disagree with `config`;
    /// `CliError::Integrator` from `RingPolymerSim::from_parts` (e.g. non-finite saved forces or
    /// energies, Q23); otherwise errors of building the potential.
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
                let cell = config
                    .system
                    .cell
                    .ok_or_else(|| bad("RDF without a periodic cell"))?;
                Some(
                    Rdf::from_parts(rc.n_bins, rc.r_max, cell, n, counts, n_samples)
                        .map_err(|e| CliError::Checkpoint(format!("RDF: {e}")))?,
                )
            }
            (0 | 1, _) => return Err(bad("RDF presence differs from the config's [output]")),
            _ => return Err(bad("corrupt RDF flag")),
        };
        let dynamics = match &config.dynamics {
            None => None,
            Some(dy) => {
                // Samples are not saved: the checkpoint must fall on a cycle end.
                if !step.is_multiple_of(dy.decorrelation_steps + dy.segment_steps) {
                    return Err(bad("checkpoint is not at the end of a dynamics cycle"));
                }
                let mut corr = || -> Result<CentroidCorrelation, CliError> {
                    let len = r.usize()?;
                    let bytes = r.take(len)?;
                    let c = CentroidCorrelation::from_bytes(bytes)
                        .map_err(|e| CliError::Checkpoint(format!("correlation block: {e}")))?;
                    if c.mean().len() != dy.n_lags {
                        return Err(bad("correlation block: n_lags differs from the config"));
                    }
                    // n_atoms is the second u64 of the `to_bytes` layout; from_bytes checked
                    // the length.
                    let n_atoms = bytes
                        .get(8..16)
                        .and_then(|b| b.try_into().ok())
                        .map(u64::from_le_bytes);
                    if n_atoms != Some(n as u64) {
                        return Err(bad("correlation block: n_atoms differs from the config"));
                    }
                    Ok(c)
                };
                let (c_vv, c_xx) = (corr()?, corr()?);
                Some(DynState {
                    c_vv,
                    c_xx,
                    h_start: f64::NAN,
                })
            }
        };
        if r.pos != bytes.len() {
            return Err(bad("trailing bytes after the checkpoint data"));
        }
        let state = RingState {
            q,
            p: mom,
            masses: config.system.masses.clone(),
        };
        let sim = RingPolymerSim::from_parts(
            state,
            config.system.cell,
            integrator(config)?,
            potential(config)?,
            forces,
            energies,
        )?;
        Ok(Driver {
            sim,
            rng: Rng::restore(&rng),
            rdf,
            dynamics,
            step,
            n_recorded,
            sum_e_prim,
            sum_e_vir,
        })
    }
}

/// The run's `Correlations` from the accumulators (Y-4, Y-5); C_xx is not reported with a cell.
fn correlations(config: &Config, dy: &DynamicsConfig, ds: &DynState) -> Correlations {
    let dt_lag = dy.corr_stride as f64 * config.sampling.dt;
    let c_vv = ds.c_vv.mean();
    let diffusion = green_kubo(&c_vv, dt_lag);
    Correlations {
        dt_lag,
        n_segments: ds.c_vv.n_segments(),
        c_vv_sem: ds.c_vv.sem(),
        c_vv,
        c_xx: match config.system.cell {
            None => Some((ds.c_xx.mean(), ds.c_xx.sem())),
            Some(_) => None,
        },
        diffusion,
    }
}

/// `correlations.csv` (MILESTONE_7 "Outputs"): one row per lag; t in fs, D also in Å²/ps, C_xx
/// columns `NaN` with a cell; conversions through `pimd_core::units` only (U-3).
fn write_correlations(c: &Correlations, out_dir: &Path) -> Result<(), CliError> {
    let mut f = BufWriter::new(fs::File::create(out_dir.join("correlations.csv"))?);
    writeln!(f, "{CORR_HEADER}")?;
    // bohr²/(ħ/Eh) → Å²/ps: (Å per bohr)² × (ħ/Eh per fs) × (fs per ps).
    let d_conv = BOHR_IN_ANGSTROM * BOHR_IN_ANGSTROM * FS_IN_ATOMIC_TIME * 1000.0;
    for k in 0..c.c_vv.len() {
        let (xx, xx_sem) = match &c.c_xx {
            Some((x, s)) => (x[k], s[k]),
            None => (f64::NAN, f64::NAN),
        };
        writeln!(
            f,
            "{},{},{},{},{},{},{},{}",
            k,
            k as f64 * c.dt_lag / FS_IN_ATOMIC_TIME,
            c.c_vv[k],
            c.c_vv_sem[k],
            c.diffusion[k],
            c.diffusion[k] * d_conv,
            xx,
            xx_sem
        )?;
    }
    f.flush()?;
    Ok(())
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
