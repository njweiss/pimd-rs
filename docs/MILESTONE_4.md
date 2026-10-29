# Milestone 4: Quantum liquid and release candidate

Crates touched: `pimd-cli` (new), `pimd-core` (RNG snapshot), plus validation scripts and README. Spec sections: 2 (D-2, D-3), 6, 7, 8, 9 (P-3).

## Goal

A runnable engine: config file in, trajectory and observables out, with bitwise-reproducible checkpoint/restart. Validate liquid-neon PIMD with Lennard-Jones against i-PI, and show the quantum vs. classical shift. This ends the milestones that gate release R1.

## Already done

M1–M3 merged. i-PI available via `pixi run -e ref` (version recorded in `pixi.lock`).

## In scope

| ID | Item |
|---|---|
| D-2, D-3 | RNG snapshot/restore; restart is bitwise identical to an uninterrupted run |
| I-1, T-1 | Full PIMD run driver using LJ |
| E-1 … E-4, E-7 | Observables recorded during a run |
| U-3 | Unit conversion only at config/output boundary |
| (new) | Config parsing, run driver, observables output, checkpointing |

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

Allowed new dependencies: `serde`, `toml`, `clap` (binary only). A fixed, hand-written checkpoint format (little-endian f64/u64, header with magic + version) is preferred over a serialization framework; `serde` is for config only. Test-only: `tempfile`.

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
[output]
rdf_bins = 100
rdf_rmax_angstrom = 8.0
checkpoint_every = 1000
```

## Clarifications

- **Initial state:** atoms on a simple cubic lattice with Gaussian initial momenta of variance m·P/β (RP-3) drawn from `Rng`, all beads of an atom initialised at the same position, in atom-major draw order (atom i outer, bead j, dimension d inner). Initial layout and draw order are part of the contract.
- **Observables:** `mean_e_prim` and `mean_e_vir` are total-energy estimators (E-4) averaged per atom, in Eh/atom, over production steps every `stride` steps; the output files report kelvin and meV as well, via `units` only.
- **Restart determinism:** a run of 2K steps equals a run of K steps → checkpoint → K more, bitwise, including RNG state, accumulated observables and the RDF accumulator. Checkpoints contain everything needed (positions, momenta, forces, energies, RNG, step counter, accumulators); a mismatching config (hash) is an error.
- **Outputs:** `observables.csv` (step, E_prim, E_vir, V, T-estimate columns), `rdf.csv`, `checkpoint.bin`, `run.log` (seed, versions, config echo).

## Required tests

Cargo tests (fast; small systems and short runs):
- **Config:** parsing of a full valid config; missing fields, negative values, `n_beads = 0`, cutoff + skin > L/2 all rejected with distinct errors; unit conversion checked against hand values (e.g. `35.6 K` → 35.6·k_B Eh).
- **RNG snapshot:** `restore(snapshot())` continues the same sequence; `to_bytes`/`from_bytes` round trip; corrupt bytes rejected.
- **Restart (D-3):** 2K steps vs K + checkpoint + K steps for a small system (N = 32, P = 4): bitwise-identical final state, observables file contents and RDF; also with a 1-thread and a 4-thread pool.
- **Reproducibility:** same config, two runs, identical outputs (bytes); different seed differs.
- **P = 1 limit:** with `n_beads = 1`, observables agree with a classical NVT run assembled from M3 components (VelocityVerlet + PILE-L on one bead, same seed) to 1e-12 relative per recorded value.
- **Smoke:** a 50-step run executes and writes all output files.

Long validation (`validation/`, run with `CHECK_LONG=1 ./scripts/check.sh` via the cargo feature `long-validation`; no `#[ignore]`):
- **vs i-PI:** the same LJ neon system (same ε, σ, cutoff and shift policy, same β, P, N, box), i-PI with LAMMPS `fix ipi` (`pair_style lj/cut`) as the LJ client (WORKFLOW Phase 0), PILE-L with matching thermostat settings. Compare ⟨E_vir⟩, ⟨E_prim⟩ and g(r) within 5σ_combined (block averaging on both sides). The i-PI input and output files and a generator script are stored under `validation/data/` with the i-PI version; reference observables are stored as JSON with error bars. Any systematic disagreement goes to `QUESTIONS.md`.
- **Quantum vs. classical:** P = 1 vs P = 16 at a low temperature where the effect is visible: g(r) first-peak height and ⟨E⟩ shifts have the expected sign (quantum delocalisation lowers the peak and raises the energy at fixed T); quote the numbers with error bars.
- **Convergence in P:** P ∈ {4, 8, 16, 32}: ⟨E_vir⟩ approaches a limit with O(1/P²) differences; the virial estimator's block-averaged error is smaller than the primitive's at P ≥ 8.

README/benchmark deliverables (checked by script, not by tests):
- Validation plots regenerated by `validation/make_plots.py` (no hand edits).
- Performance table: atom-steps/s for the LJ core vs LAMMPS (same system, same thread count) and thread scaling for 1, 2, 4, 8 threads, generated by `validation/benchmark.sh` with machine details printed.

## Mutant ideas (physics-bug set from WORKFLOW.md applies)

Checkpoint omitting the RNG position; accumulators not stored; restart off by one step; unit conversion factor error (k_B or fs → a.u.); initial momentum variance m/β; observable averaged per bead instead of per atom; RDF accumulated before equilibration ends; config hash ignoring a field.

## Out of scope

Para-H₂ and Silvera-Goldman (stretch, only after parameters are cross-checked from two independent sources), socket backend, ML potentials, dynamics (RPMD/CMD/TRPMD), PBC for non-orthorhombic cells, Python bindings.

## Acceptance

- [ ] Cargo tests pass; `check.sh` exits 0, including thread-count runs
- [ ] Long validation passes; agreement levels recorded in `validation/RESULTS.md`
- [ ] Plots and performance table reproduced by scripts
- [ ] Review done; no blocking findings; `QUESTIONS.md` has no open entries
- [ ] **R1 decision by owner:** publish or hold
