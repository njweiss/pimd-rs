# Milestone 4: Quantum liquid and release candidate

Crates touched: `pimd-cli` (new), `pimd-core` (RNG snapshot, `Cell::min_length`, FFT transform option), `pimd-integrators` (transform option), `pimd-potentials` (LJ parallel granularity), plus validation scripts and README. Spec sections: 2 (D-2, D-3), 6, 7, 8, 9 (P-3).

## Goal

A runnable engine: config file in, trajectory and observables out, with bitwise-reproducible checkpoint/restart. Validate liquid-neon PIMD with Lennard-Jones against i-PI, and show the quantum vs. classical shift. This ends the milestones that gate release R1.

## Already done

M1–M3 merged. i-PI available via `pixi run -e ref` (version recorded in `pixi.lock`).

## In scope

| ID | Item |
| --- | --- |
| D-2, D-3 | RNG snapshot/restore; restart is bitwise identical to an uninterrupted run |
| I-1, T-1 | Full PIMD run driver using LJ |
| E-1 … E-4, E-7 | Observables recorded during a run |
| U-3 | Unit conversion only at config/output boundary |
| (new) | Config parsing, run driver, observables output, checkpointing |
| RP-4 | Optional FFT normal-mode transform behind `sampling.normal_modes` |
| D-3 | `LennardJones` parallelises over all (bead, atom) pairs, results bitwise unchanged |

## Public API (stubs)

```rust
// pimd_core::rng (additions)
#[derive(Clone, PartialEq, Debug)]
pub struct RngState { /* opaque, serialisable via Vec<u8> */ }
impl Rng {
    pub fn snapshot(&self) -> RngState;
    pub fn restore(state: &RngState) -> Rng;
}
impl RngState {
    pub fn to_bytes(&self) -> Vec<u8>;
    pub fn from_bytes(bytes: &[u8]) -> Result<RngState, CoreError>;
}

// pimd_core::cell (addition)
impl Cell { pub fn min_length(&self) -> f64; }                  // shortest edge, bohr

// pimd_core::normal_modes (additions)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Transform { #[default] Matrix, Fft }
impl NormalModes {
    pub fn with_transform(n_beads: usize, beta: f64, transform: Transform)
        -> Result<NormalModes, CoreError>;                         // `new` == with_transform(.., Matrix)
    pub fn transform(&self) -> Transform;
}

// pimd_integrators (addition)
impl RingPolymerIntegrator {
    pub fn with_transform(dt: f64, beta: f64, n_beads: usize, thermostat: Thermostat,
        transform: Transform) -> Result<Self, IntegratorError>;   // `new` == with_transform(.., Matrix)
}

// pimd_core::units (addition, Q13)
pub const HARTREE_IN_EV: f64 = 27.211386245988;                 // CODATA 2018

// pimd_estimators::Rdf (additions, Q14)
impl Rdf {
    pub fn counts(&self) -> &[f64];                               // accumulated Σ C_b/P per bin
    pub fn from_parts(n_bins: usize, r_max: f64, cell: Cell, n_atoms: usize,
        counts: Vec<f64>, n_samples: usize) -> Result<Rdf, EstimatorError>;
        // validates as `new` (InvalidValue); counts.len() != n_bins → ShapeMismatch
}

// pimd_cli::error
pub enum CliError {
    Parse(String),                                   // TOML syntax, missing field, unknown key
    InvalidValue { what: String, value: f64 },       // `what` names the field (e.g. "sampling.n_beads")
    Checkpoint(String),                              // corrupt file, bad magic/version, config-hash mismatch
    Io(std::io::Error),
    Core(CoreError), Potential(PotentialError), Integrator(IntegratorError), Estimator(EstimatorError), // #[from]
}

// pimd_cli::config
pub struct Config { /* see "Config format" */ }
impl Config {
    pub fn from_toml_str(s: &str) -> Result<Config, CliError>;     // validates; converts units (U-3)
}

// pimd_cli::run
pub struct RunSummary {
    pub n_steps: usize,
    pub mean_e_prim: f64,        // Eh per atom (see Clarifications)
    pub mean_e_vir: f64,
    pub rdf: Option<(Vec<f64>, Vec<f64>)>,   // (bin centers in bohr, g)
}
pub fn run(config: &Config, out_dir: &std::path::Path) -> Result<RunSummary, CliError>;
pub fn run_from_checkpoint(config: &Config, checkpoint: &std::path::Path, out_dir: &std::path::Path)
    -> Result<RunSummary, CliError>;
```

Allowed new dependencies: `serde`, `toml`, `clap` (binary only); `rustfft` in `pimd-core` for the FFT transform, scalar planner only (owner-approved); `matplotlib` in the pixi `ref` environment for `validation/make_plots.py` (owner-approved). A fixed, hand-written checkpoint format (little-endian f64/u64, header with magic + version) is preferred over a serialization framework; `serde` is for config only. Test-only: `tempfile`.

## Config format (TOML, user units)

```toml
[system]
n_atoms = 108
mass_amu = 20.1797              # neon
box_angstrom = 17.0             # cubic; or three values
[potential]
kind = "lj"
epsilon_kelvin = 35.6
sigma_angstrom = 2.749
cutoff_sigma = 2.5
skin_sigma = 0.3
shift = true
[sampling]
temperature_kelvin = 30.0
n_beads = 16
dt_fs = 1.0
tau0_fs = 100.0
n_equil = 5000
n_steps = 20000
stride = 10
seed = 12345
normal_modes = "matrix"         # optional; "matrix" (default) or "fft"
[output]
rdf_bins = 100                  # rdf_bins and rdf_rmax_angstrom optional together; absent → RunSummary.rdf = None
rdf_rmax_angstrom = 8.0
checkpoint_every = 1000
```

Unknown keys are rejected (`CliError::Parse`).

Binary (`clap`): `pimd-rs <config.toml> --out <dir> [--restart <checkpoint>]`.

## Clarifications

- **Initial state:** atoms on a simple cubic lattice: n_side = ⌈N^(1/3)⌉, spacing a = L/n_side per axis, site (i_x, i_y, i_z) at ((i_x + ½)a_x, (i_y + ½)a_y, (i_z + ½)a_z), sites ordered with i_x fastest, then i_y, then i_z; the first N sites are filled. Gaussian initial momenta of variance m·P/β (RP-3) drawn from `Rng`, all beads of an atom initialised at the same position, in atom-major draw order (atom i outer, bead j, dimension d inner). Initial layout and draw order are part of the contract.
- **Observables:** `mean_e_prim` and `mean_e_vir` are total-energy estimators (E-4) averaged per atom, in Eh/atom, over production steps every `stride` steps; the output files report kelvin and meV as well, via `units` only. Equilibration steps (`n_equil`) record nothing. Production steps are numbered s = 1 … n_steps; a step is recorded (observables row, averages, RDF sample) when s % stride == 0, after the step.
- **Restart determinism:** a run of 2K steps equals a run of K steps → checkpoint → K more, bitwise, including RNG state, accumulated observables and the RDF accumulator. Checkpoints contain everything needed (positions, momenta, forces, energies, RNG, step counter, accumulators); a mismatching config (hash) is an error (`CliError::Checkpoint`). The config hash is 64-bit FNV-1a over a canonical little-endian encoding of the parsed config (in SI-free internal units), covering `[system]`, `[potential]` and `[sampling]` except `n_steps`; `[output]` (including `checkpoint_every`) is excluded. A run always writes a checkpoint at its end. `run_from_checkpoint` continues until production step `n_steps` of the given config. Neighbor lists are not checkpointed: LJ results do not depend on list history (pairs with r ≥ r_c are skipped, neighbors summed in ascending order).
- **Outputs:** `observables.csv` with header `step,e_prim_eh,e_vir_eh,v_eh,t_est_k,e_prim_mev,e_vir_mev,e_prim_k,e_vir_k`, all energies per atom; V is V̄ (E-3) per atom; k_B·T_est = Σ_{j,i} |p_{j,i}|²/m_i / (3 N P²). Numbers are written with Rust's `{}` for f64 (shortest round-trip), so files carry exact values. `rdf.csv` (`r_bohr,r_angstrom,g`), `checkpoint.bin`, `run.log` (seed, versions, config echo). No output contains timestamps or anything machine- or time-dependent, so repeated runs are byte-identical. After a restart, `observables.csv` in the new `out_dir` holds only rows after the checkpoint; the restart test compares the concatenation (header once) with the uninterrupted file.

- **Details pinned by the tests (Q15, owner-confirmed):**
  1. `Config` has public fields `system: SystemConfig { n_atoms, mass (m_e), cell: Cell }`, `potential: LjParams` (atomic units), `sampling: SamplingConfig { temperature_kelvin, beta, n_beads, dt, tau0 (ħ/Eh), n_equil, n_steps, stride, seed, normal_modes: Transform }`, `output: OutputConfig { rdf: Option<RdfConfig { n_bins, r_max (bohr) }>, checkpoint_every }`; all derive `Debug, Clone, PartialEq`, as does `RunSummary`. `[output]` is required.
  2. `InvalidValue.what` is the TOML `"section.key"` (e.g. `"sampling.temperature_kelvin"`); a C-4 violation names `"potential.cutoff_sigma"` or `"potential.skin_sigma"`. Negative integer fields are serde type errors (`Parse`).
  3. One `Rng::from_seed(seed)` draws the initial momenta and then drives the thermostat (D-2).
  4. `checkpoint.bin` depends only on state, accumulators and the config hash (no `n_steps`, no paths), so uninterrupted and restarted runs write identical bytes at the same final step.
  5. Corrupt checkpoints (empty, truncated, halved, magic at offset 0 zeroed, unrelated text) give `CliError::Checkpoint`, not `Io`.
  6. n_side = ⌈N^(1/3)⌉ is computed exactly with integers (N = 27 gives 3).
  7. `rdf_bins` and `rdf_rmax_angstrom` given alone is an error; a two-value `box_angstrom` is an error.
  8. `run` creates `out_dir` if it does not exist.
- **After review (Q16–Q18, N1, N2, N5):** the checkpoint stores the RDF's n_bins and r_max and a restart with different RDF settings is `CliError::Checkpoint` (Q16). No overflow checks on n_atoms / n_beads (Q17, declined by the owner). On restart the driver re-evaluates the saved positions and requires bitwise-equal forces and energies (kept for M4; to be revisited before M5, Q18); the error message says the file is corrupt or was written by a different build or machine (N5). With an RDF configured, n_atoms = 1 is `InvalidValue { what: "system.n_atoms" }` at parse time (N2). Restarting into the directory that holds the earlier run's outputs overwrites its `observables.csv` and `run.log`; this is documented on `run_from_checkpoint` and in the binary's help, not prevented (N1).
- **meV columns (Q13):** converted with `units::HARTREE_IN_EV` × 1000.
- **RDF in checkpoints (Q14):** the driver saves `Rdf::counts()` and `n_samples()` and restores with `Rdf::from_parts`; there is no second E-7 histogram in `pimd-cli`.
- **`Cell::min_length()` (owner-approved after M3):** new public method returning the shortest cell edge (bohr); replaces the three copies of the L_min fold in `NeighborList::build`, `Rdf::new` and `LennardJones::evaluate`. Bit-identical.
- **Last evaluation (Q8, decided after M3):** no change in M4. No M4 observable needs the virial (pressure is not recorded, and P-5 gives no virial at P > 1). `VelocityVerlet::step` keeps its signature. When pressure becomes an observable, `RingPolymerSim` keeps the last virial alongside its forces and energies (one field and an accessor) rather than `step` returning an `Evaluation`.
- **Normal-mode transform option (decided after M3):** config key `sampling.normal_modes = "matrix" | "fft"`, default `"matrix"` (the current RP-4 transform). D-3 is unchanged; bitwise guarantees hold per transform setting and per machine (see the D-3 note in SPEC). The setting is part of the config hash, so a checkpoint restarts only under the same setting. The FFT path produces exactly the RP-4 columns (centroid, then cos/sin pairs, then the alternating mode for even P) with the RP-4 normalisation, for any P ≥ 1 including odd P; `frequencies()` and `matrix()` are unchanged. Profiled after M3 (neon LJ, N = 108, release, 1 thread): step 0.54 / 2.1 / 4.0 ms at P = 4 / 16 / 32, of which the transforms are about 1 % / 3.5 % / 7.5 %; the option is kept for larger P and cheap potentials. Use `rustfft`'s scalar planner (`FftPlannerScalar`) so results do not depend on CPU features. If the transform option needs more than additive constructors (`with_transform`) and an internal dispatch in `NormalModes`, stop and report: the owner keeps it only if it does not require a major refactor.
- **LJ parallel granularity (decided after M3):** in M3 `LennardJones::evaluate` parallelises over atoms inside a sequential bead loop; at N = 108 four threads were slower than one (P = 32: 5.0 vs 4.0 ms per step). M4 parallelises over all (bead, atom) pairs (and list rebuilds over beads), keeping each atom's terms and the per-bead index-order reduction exactly as in M3, so results are bitwise unchanged (D-3).

## Required tests

Cargo tests (fast; small systems and short runs):

- **Config:** parsing of a full valid config; missing fields, negative values, `n_beads = 0`, cutoff + skin > L/2 all rejected with distinct errors; unit conversion checked against hand values (e.g. `35.6 K` → 35.6·k_B Eh).
- **RNG snapshot:** `restore(snapshot())` continues the same sequence; `to_bytes`/`from_bytes` round trip; corrupt bytes rejected.
- **Restart (D-3):** 2K steps vs K + checkpoint + K steps for a small system (N = 32, P = 4): bitwise-identical final state, observables file contents and RDF; also with a 1-thread and a 4-thread pool.
- **Reproducibility:** same config, two runs, identical outputs (bytes); different seed differs.
- **Normal-mode transform:** `fft` and `matrix` agree to 1e-13 relative (forward and inverse, P ∈ {1, 2, 3, 4, 8, 16, 32}, including odd P and the RP-4 column order) and give the same thermostat draws; each setting alone is bitwise identical across 1 and 4 threads; changing the setting changes the config hash.
- **P = 1 limit (RP-6):** with `n_beads = 1`, the recorded observables equal those of a loop written in the test from `RingPolymerSim` (P = 1, PILE-L, `LennardJones`, same seed, same initial state) and the M2/M3 estimators, to 1e-12 relative per recorded value. (`VelocityVerlet` cannot host the O step inside the drift, so it is not the reference.)
- **LJ granularity (D-3):** a P-bead `evaluate` equals P separate single-bead evaluations bitwise (energies, forces), at 1 and 4 threads.
- **Smoke:** a 50-step run executes and writes all output files.

Long validation (`validation/`, run with `CHECK_LONG=1 ./scripts/check.sh` via the cargo feature `long-validation`; no `#[ignore]`):

- **vs i-PI:** the same LJ neon system (same ε, σ, cutoff and shift policy, same β, P, N, box), i-PI with LAMMPS `fix ipi` (`pair_style lj/cut`) as the LJ client (WORKFLOW Phase 0), PILE-L with matching thermostat settings (`pile_lambda = 1` so that γ_k = 2ω_k, T-2; i-PI splits the step differently from B A O A B, which is allowed with dt small enough that the splitting bias stays inside the error bars, stated in a comment). Compare ⟨E_vir⟩, ⟨E_prim⟩ and g(r) within 5σ_combined (block averaging on both sides). The i-PI inputs, a generator script (`gen_ipi_neon.py`, accepting `--out`) and the reference observables (JSON with error bars, i-PI and LAMMPS versions in the `versions` block) live in `crates/pimd-cli/tests/data/`, so they are locked with the tests and regenerated by `validation/verify_reference_data.py`; regenerating it takes about 34 min, so `verify_reference_data.py` runs `gen_ipi_neon.py` only with `CHECK_IPI=1` (run before merging M4 and whenever i-PI, LAMMPS or the generator changes); `CHECK_LONG=1` alone still runs `long_neon.rs` against the committed JSON. Any systematic disagreement goes to `QUESTIONS.md`.
- **Quantum vs. classical:** P = 1 vs P = 16 at a low temperature where the effect is visible: g(r) first-peak height and ⟨E⟩ shifts have the expected sign (quantum delocalisation lowers the peak and raises the energy at fixed T); quote the numbers with error bars.
- **Convergence in P:** P ∈ {4, 8, 16, 32}: ⟨E_vir⟩ approaches a limit with O(1/P²) differences; the virial estimator's block-averaged error is smaller than the primitive's at P ≥ 8.

README/benchmark deliverables (checked by script, not by tests):

- Validation plots regenerated by `validation/make_plots.py` (no hand edits).
- Performance table: atom-steps/s for the LJ core vs LAMMPS (same system, same thread count) and thread scaling for 1, 2, 4, 8 threads, generated by `validation/benchmark.sh` with machine details printed. LAMMPS uses OpenMP threads if the build has the OMP package, otherwise MPI ranks; the table says which. Target (reported, not gated): at N = 108, P = 16, 4 threads at least 2× faster than 1.

## Mutant ideas (physics-bug set from WORKFLOW.md applies)

Checkpoint omitting the RNG position; accumulators not stored; restart off by one step; unit conversion factor error (k_B or fs → a.u.); initial momentum variance m/β; observable averaged per bead instead of per atom; RDF accumulated before equilibration ends; config hash ignoring a field.

## Out of scope

Para-H₂ and Silvera-Goldman (stretch, only after parameters are cross-checked from two independent sources), socket backend, ML potentials, dynamics (RPMD/CMD/TRPMD), PBC for non-orthorhombic cells, Python bindings.

## Acceptance

- [x] Cargo tests pass; `check.sh` exits 0, including thread-count runs
- [x] Long validation passes; agreement levels recorded in `validation/RESULTS.md` (`CHECK_LONG=1 CHECK_IPI=1` passed on e2bdfd7, 54.5 min, all four reference datasets regenerated and matching)
- [x] Plots and performance table reproduced by scripts (`validation/run_neon.py`, `make_plots.py`, `benchmark.sh`; benchmark taken under background load, see RESULTS.md)
- [x] Review done; no blocking findings; `QUESTIONS.md` has no open entries (should-fix Q16 fixed, Q17 declined, Q18 deferred to M5)
- [x] **R1 decision by owner:** hold (2026-10-30); work continues with M5
