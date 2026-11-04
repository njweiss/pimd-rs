# Milestone 7: Ring-polymer dynamics (RPMD, TRPMD; PA-CMD in M7b)

Status: **approved (2026-11-04)**: M7a as below (SPEC T-4 and §8a Y-1 … Y-5 added); M7b
outline only, briefed after M7a merges.

| Phase | Flow | Output |
| --- | --- | --- |
| M7a | A/B/C | RPMD and TRPMD segments, centroid time-correlation functions, Green–Kubo diffusion, `[dynamics]` in the config |
| M7b | A/B/C, briefed after M7a merges | PA-CMD (outline at the end of this file) |

Crates touched in M7a: `pimd-integrators` (`Thermostat::Trpmd`, `RingPolymerSim::set_thermostat`,
`RingPolymerSim::resample_momenta`), `pimd-estimators` (centroid correlation, Green–Kubo),
`pimd-cli` (`[dynamics]`, outputs, checkpoint block), plus validation generators. `pimd-core`,
`pimd-potentials`, `pimd-socket` and `pimd-torch` are unchanged. Spec sections: 6 (I-1, I-3), 7
(T-1, T-2, new T-4), new 8a (Y-1 … Y-5), 2 (D-2, D-3, D-4), 11.

## Goal

pimd-rs computes approximate quantum time-correlation functions by ring-polymer dynamics: the
Kubo-transformed autocorrelation of the centroid velocity (and of the centroid position for
external-field potentials), and from the velocity autocorrelation the self-diffusion coefficient.
Validated against exact results (harmonic), the classical limit (P = 1), and i-PI on the same
potentials (a 1D double well served through the socket, and liquid neon with LAMMPS as in M4).

## Already done

M1–M6 merged; R1 on hold. Relevant pieces:

- BAOAB with the exact free propagator (I-1, I-2); `Thermostat::Off` conserves H_P (T-3);
  PILE-L with optional centroid friction (`tau0: None` ⇒ γ_0 = 0, T-2).
- `pimd_estimators::hamiltonian` (H_P), `Beads::centroid`, `VelocityVerlet` (I-4), `Harmonic`
  and `DoubleWell` (P-2), the socket backend (K-1 … K-6) and its Python clients.
- Checkpoint format version 2 (M4, unchanged through M6); the config hash (M4, M5).
- i-PI 3.3.1 in `pixi run -e ref`. Checked in its source (2026-11-04): PILE-L sets the
  internal-mode time constant to τ_k = 1/(2λω_k) (`ipi/engine/thermostats.py`, `pile_lambda`),
  the same convention as T-4 below, so i-PI runs use the same λ; `normal_modes` mode `pa-cmd`
  takes one target frequency Ω and scales internal-mode masses by (ω_k/Ω)² (M7b reference).

## Decisions taken (owner, 2026-11-04)

1. M7 is split: M7a RPMD + TRPMD + correlation machinery (validation rungs 1–4); M7b PA-CMD.
2. Correlation functions come from **segments**: thermostatted PIMD between segments, momenta
   redrawn from the canonical distribution at each segment start, every recorded step of a
   segment used as a time origin. Redrawing makes the average canonical (a single NVE ring-polymer
   trajectory samples one H_P shell and is not ergodic for the harmonic test); how far apart
   segments are only affects the error bars, and segments are the blocks for σ.
3. Pluggable thermostat as a new enum variant `Thermostat::Trpmd { lambda }`, default λ = 1/2.
   Source: Rossi, Ceriotti, Manolopoulos, J. Chem. Phys. 140, 234116 (2014) (arXiv:1406.1074),
   checked in the full text on 2026-11-04: PILE critical damping γ̃_c = 2ω̃ (Eq. 28) scaled as
   γ = λγ_c (Eq. 30), λ = 1 critical damping, λ = 0 standard RPMD; "λ = 1/2 is close to optimum
   for all of the problems we have studied", used throughout their Sec. III. The paper also
   requires the centroid to stay unthermostatted (a centroid thermostat "would invalidate" the
   TRPMD correlation-function results), as T-4 does. No trait.
4. Correlations computed in Rust (`pimd-estimators`) and written by the CLI.
5. Only centroid velocity and centroid position autocorrelations. No dipoles or spectra.
6. Double-well rung uses the existing separable `DoubleWell` (P-2, b > 0 unchanged); its x
   component is an exact 1D problem.
7. Literature rung: liquid neon vs i-PI is the gate; para-H₂ is an optional experiment after M7,
   once Silvera–Goldman parameters are cross-checked in two independent sources (SPEC §11).
8. PA-CMD (M7b): the adiabatic frequency Ω is a required config input with no built-in formula
   (as i-PI); the usual Ω = P^{P/(P−1)}/β could not be confirmed against a primary source.

## SPEC additions (added 2026-11-04)

Section 7, after T-3:

- **T-4** `Thermostat::Trpmd { lambda }`, λ > 0 and finite: the O step of T-1 with γ_0 = 0 and
  γ_k = 2λω_k for every mode with ω_k > 0. Draw order and draw count as T-1 (P·N·3 normals,
  including the k = 0 draws, whose coefficient is zero). `Trpmd { lambda: 1.0 }` is
  `PileL { tau0: None }` bitwise.

New section 8a "Dynamics" (prefix Y):

- **Y-1** Segment cycle. After `n_equil` thermostatted steps, production is a sequence of
  cycles, each: (a) `decorrelation_steps` steps with the sampling thermostat (PILE-L, T-1/T-2);
  (b) momentum redraw (Y-2); (c) `segment_steps` steps with the dynamics thermostat (`Off` for
  RPMD, `Trpmd` for TRPMD). The forces at the end of one phase are reused by the next (I-3:
  switching the thermostat never evaluates). One `pimd_core::Rng` drives every draw in the
  order they occur (D-2).
- **Y-2** Momentum redraw: every bead momentum p_{j,i,d} ← sqrt(m_i·P/β)·ξ, ξ ~ N(0, 1), drawn
  atom i outermost, then bead j, then dimension d (the order of the initial draw); P·N·3 draws.
  Positions and forces are unchanged.
- **Y-3** Centroid samples: v̄_i = (1/P) Σ_j p_{j,i}/m_i (bohr per ħ/Eh) and q̄_i = (1/P) Σ_j q_{j,i}
  (bohr, as stored, not wrapped). Within a segment, sample s = 0, 1, …, L − 1 is taken at segment
  step s·`corr_stride` (s = 0 right after the redraw, before the first dynamics step), so
  L = `segment_steps`/`corr_stride` + 1.
- **Y-4** Correlation estimator. For one segment with samples x(0) … x(L − 1) of N atoms and lag
  τ = 0 … `n_lags` − 1 (n_lags ≤ L):
  C_seg(τ) = (1/(N·(L − τ))) Σ_{s=0}^{L−1−τ} Σ_i x_i(s)·x_i(s + τ)
  (dot product over x, y, z; summed s outermost, then i, then d). The estimate is the mean of
  C_seg over segments (equal weights, in segment order); its σ is the sample standard deviation of
  the C_seg (n − 1 denominator) over √n_segments. C_vv uses v̄, C_xx uses q̄. Lag τ is the time
  τ·`corr_stride`·dt. Expectation at τ = 0: C_vv(0) = 3/(β m) for equal masses; C_xx(0) = ⟨|q̄|²⟩
  of the finite-P ensemble.
- **Y-5** Green–Kubo: D(τ_n) = (1/3) ∫_0^{τ_n} C_vv dt by the trapezoid rule on the lag grid
  (D(0) = 0), bohr²/(ħ/Eh). Only meaningful for a single species; the brief's systems have one.

Section 11 roadmap note updated: M7 split into M7a/M7b as above.

## M7a: in scope

| ID | Item |
| --- | --- |
| T-4 | `Thermostat::Trpmd { lambda }` |
| Y-1, Y-2 | `RingPolymerSim::set_thermostat`, `RingPolymerSim::resample_momenta`; the cycle in the CLI |
| Y-3, Y-4, Y-5 | `CentroidCorrelation`, `centroid_velocity`, `green_kubo` |
| D-2, D-3, D-4 | Draw order across phases; bitwise results across thread counts and restarts |
| (new) | `[dynamics]` config section, `correlations.csv`, `segments.csv`, checkpoint block |

## Public API (stubs)

```rust
// pimd_integrators
pub enum Thermostat {
    Off,
    PileL { tau0: Option<f64> },
    /// TRPMD (T-4): γ_0 = 0, γ_k = 2·lambda·ω_k; lambda > 0, finite.
    Trpmd { lambda: f64 },
}

impl RingPolymerSim {
    /// Replaces the O step's thermostat (Y-1). Keeps state, forces, energies, dt, β, P and the
    /// transform; no `evaluate`, no draws.
    /// Errors: `IntegratorError::InvalidValue` for a bad τ_0 or λ (as `RingPolymerIntegrator::new`);
    /// the simulation is then unchanged.
    pub fn set_thermostat(&mut self, thermostat: Thermostat) -> Result<(), IntegratorError>;

    /// Redraws every bead momentum from N(0, m_i·P/β) (Y-2): atom i outer, bead j, dimension d
    /// inner; P·N·3 normal draws. Positions, forces and energies unchanged.
    pub fn resample_momenta(&mut self, rng: &mut Rng);
}

// pimd_estimators
/// Centroid velocities v̄_i = (1/P) Σ_j p_{j,i}/m_i, length N, bohr per ħ/Eh (Y-3).
/// Panics if `masses.len()` differs from the atom count of `p`.
pub fn centroid_velocity(p: &Beads, masses: &[f64]) -> Vec<[f64; 3]>;

/// Running Green–Kubo integral D(τ_n) = (1/3) ∫ C_vv by the trapezoid rule (Y-5); same length as
/// `c_vv`, D(0) = 0. `dt_lag` is the lag spacing in ħ/Eh. Units bohr²/(ħ/Eh) for C_vv in
/// (bohr/(ħ/Eh))².
pub fn green_kubo(c_vv: &[f64], dt_lag: f64) -> Vec<f64>;

/// Segment-averaged autocorrelation of a per-atom 3-vector (Y-4).
#[derive(Debug, Clone, PartialEq)]
pub struct CentroidCorrelation { /* n_lags, n_atoms, current segment's samples, Σ C_seg, Σ C_seg², n_segments */ }

impl CentroidCorrelation {
    /// Errors: `EstimatorError::InvalidValue` for n_lags = 0 or n_atoms = 0.
    pub fn new(n_lags: usize, n_atoms: usize) -> Result<CentroidCorrelation, EstimatorError>;
    /// Appends one sample of the current segment.
    /// Errors: `EstimatorError::ShapeMismatch` if `x.len()` ≠ n_atoms.
    pub fn push(&mut self, x: &[[f64; 3]]) -> Result<(), EstimatorError>;
    /// Closes the current segment: adds its C_seg (Y-4) and clears the samples.
    /// Errors: `EstimatorError::InvalidValue` if it has fewer than n_lags samples (nothing changes).
    pub fn end_segment(&mut self) -> Result<(), EstimatorError>;
    pub fn n_segments(&self) -> usize;
    /// Mean over segments, length n_lags; NaN before the first segment.
    pub fn mean(&self) -> Vec<f64>;
    /// σ of the mean (Y-4), length n_lags; NaN with fewer than two segments.
    pub fn sem(&self) -> Vec<f64>;
    /// Accumulators only (n_lags, n_atoms, Σ, Σ², n_segments), little-endian; valid only between
    /// segments (no samples pending). Errors: `InvalidValue` if samples are pending.
    pub fn to_bytes(&self) -> Result<Vec<u8>, EstimatorError>;
    /// Errors: `EstimatorError::InvalidValue` for a malformed or truncated buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<CentroidCorrelation, EstimatorError>;
}
```

`pimd_cli`: `Config` gains `pub dynamics: Option<DynamicsConfig>`;

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicsMethod { Rpmd, Trpmd { lambda: f64 } }

#[derive(Debug, Clone, PartialEq)]
pub struct DynamicsConfig {
    pub method: DynamicsMethod,
    pub decorrelation_steps: usize, // ≥ 0
    pub segment_steps: usize,       // ≥ 1, multiple of corr_stride
    pub corr_stride: usize,         // ≥ 1
    pub n_lags: usize,              // 1 ≤ n_lags ≤ segment_steps/corr_stride + 1
}

#[derive(Debug, Clone, PartialEq)]
pub struct Correlations {
    pub dt_lag: f64,          // ħ/Eh
    pub n_segments: usize,
    pub c_vv: Vec<f64>, pub c_vv_sem: Vec<f64>,   // (bohr/(ħ/Eh))²
    pub c_xx: Option<(Vec<f64>, Vec<f64>)>,       // bohr²; None with a cell
    pub diffusion: Vec<f64>,                      // bohr²/(ħ/Eh), running (Y-5)
}
```

`RunSummary` gains `pub correlations: Option<Correlations>` (`None` without `[dynamics]`).
`CliError` gains no variant.

Allowed new dependencies: none.

## Config format (addition)

```toml
[dynamics]                    # optional; absent ⇒ M6 behaviour, unchanged
method = "trpmd"              # "rpmd" or "trpmd"
trpmd_lambda = 0.5            # only with "trpmd"; default 0.5
decorrelation_steps = 200     # PILE-L steps before each segment (tau0_fs from [sampling])
segment_steps = 2000
corr_stride = 4
n_lags = 400                  # lags 0 … n_lags − 1, spacing corr_stride·dt
```

## Clarifications

- **Production with `[dynamics]`.** `n_steps` counts every production step (decorrelation and
  segment steps alike) and must be a multiple of the cycle length c = `decorrelation_steps` +
  `segment_steps`; n_segments = n_steps/c. `checkpoint_every` must be a multiple of c, so
  checkpoints fall on cycle ends. Violations are `CliError::InvalidValue` naming
  `"sampling.n_steps"` or `"output.checkpoint_every"`. `observables.csv` and `trajectory.xyz` keep
  recording every production step at their strides, format unchanged. `trpmd_lambda` with
  `method = "rpmd"`, `n_lags` beyond L, `segment_steps` not a multiple of `corr_stride`, and
  unknown keys are errors naming the `"dynamics.<key>"` (Q29): a key that does not belong
  (`trpmd_lambda` with RPMD, an unknown key) may be `CliError::Parse` with the key in the message
  or `InvalidValue` naming it; an unknown `method` may be either; the non-multiple is
  `InvalidValue("dynamics.segment_steps")`; range rules are `InvalidValue` naming their own key.
- **No change without `[dynamics]`.** Config hash, checkpoint bytes (version 2 layout),
  `observables.csv`, `run.log` content apart from the config echo, and every M4–M6 test are
  unchanged. The `[dynamics]` fields enter the hash only when the section is present (appended
  after the existing fields).
- **Checkpoint block.** With `[dynamics]`, the version-2 layout is followed by a trailing block:
  the two `CentroidCorrelation::to_bytes` buffers (C_vv, then C_xx; C_xx is still accumulated with
  a cell but not reported), each length-prefixed (u64). The run's position in the cycle follows
  from the step counter. A checkpoint of a `[dynamics]` run cannot be continued without it and
  vice versa (config hash).
- **Outputs.** `correlations.csv`, written at the end of a run (also after a restart, from the
  restored accumulators): header
  `lag,t_fs,c_vv_au,c_vv_sem_au,d_au,d_angstrom2_per_ps,c_xx_bohr2,c_xx_sem_bohr2`; C_xx columns
  `NaN` with a cell; f64 written with `{}`. Conversions only from `pimd_core::units` constants
  (U-3). `segments.csv`: `segment,step,h_start_eh,h_end_eh` with H_P per atom
  (`pimd_estimators::hamiltonian`/N) right after the redraw and at the segment end; with RPMD the
  difference is the integrator's energy error (T-3). One row per segment; `segment` counts from 0
  and `step` is the production step at the segment end (Q30). After a restart both files in the new
  `out_dir` hold only rows after the checkpoint (as `observables.csv`).
- **P = 1.** RPMD at P = 1 is classical NVE MD; TRPMD at P = 1 has no internal modes and equals
  RPMD bitwise (the k = 0 coefficient is 1 and its noise term zero, but the draws still happen,
  T-4: the RNG streams differ, the trajectories within a segment do not).
- **Socket and MACE potentials** work unchanged (the dynamics only changes the O step and adds
  redraws).

## Required tests

Deterministic (default `check.sh`):

1. **T-4.** `Trpmd { lambda: 1.0 }` and `PileL { tau0: None }` give bitwise identical
   trajectories and RNG states over 50 steps (LJ cluster, P = 4); for λ = 0.5 the per-mode
   c_k = exp(−λ·2ω_k·dt) are checked through one O step on a state with known normal-mode momenta
   and a known draw sequence; λ ≤ 0, NaN, ∞ rejected.
2. **`set_thermostat`.** A sim switched to thermostat X and stepped equals a sim built with X
   from the same state, bitwise; switching performs no `evaluate` (counting potential) and no
   draws (RNG state unchanged); a rejected thermostat leaves the sim unchanged.
3. **`resample_momenta`.** Bitwise equal to N(0, m_i P/β) draws in the Y-2 order from a cloned
   RNG; exactly P·N·3 draws; positions, forces, energies untouched.
4. **P = 1 is classical.** `RingPolymerSim` (P = 1, `Off`) and `VelocityVerlet` from the same
   state agree to 1e-10 relative in positions and momenta over 200 steps (LJ cluster).
5. **Harmonic centroid decouples.** For `Harmonic`, the centroid of an `Off` trajectory (P = 8)
   follows the exact discrete map of velocity Verlet for frequency ω applied to (q̄(0), p̄(0)) to
   1e-12 over 1 000 steps; the same holds under `Trpmd` (internal-mode noise does not reach the
   centroid).
6. **`CentroidCorrelation`.** Hand-computed C_seg for small sequences (N = 2, L = 4, n_lags = 3),
   including the (L − τ) normalisation; mean and sem over three segments; NaN rules; errors
   (shape, short segment leaves state unchanged, n_lags = 0); `to_bytes`/`from_bytes` round
   trip, pending samples refused, truncated buffers rejected.
7. **`centroid_velocity`, `green_kubo`.** Hand values; trapezoid on cos(ωt) against
   sin(ωt)/(3ω) within the trapezoid error bound; D(0) = 0.
8. **Config.** Valid `[dynamics]` parses to `DynamicsConfig`; each rule in "Clarifications"
   gives the named `InvalidValue`; every `[dynamics]` field is covered by the hash (restart under
   a changed field is `CliError::Checkpoint`, the M5 test pattern; the hash itself is
   crate-private); for a config without the section the `config hash` line of `run.log` equals the
   value printed by `main` at cf79a4d (recorded in the test).
9. **Run and restart.** A short `[dynamics]` run (LJ cluster, P = 4, 3 cycles) is byte-identical
   across repeated runs and thread counts 1 and 4; 2K cycles equals K cycles → checkpoint → K
   cycles, bitwise, including `correlations.csv` and the concatenated `segments.csv`; output
   headers as specified; a run without `[dynamics]` writes a `checkpoint.bin` whose FNV-1a equals
   the value from `main` at cf79a4d for the same config (recorded in the test).

Statistical (`long-validation`; tolerance policy of WORKFLOW, σ from segments):

10. **Harmonic, exact** (one atom, `Harmonic`, βω = 8, P = 8, RPMD and TRPMD λ = 0.5): C_xx(τ) and
    C_vv(τ) at every lag within 5σ of the exact finite-dt result (C_xx(0)·[Mⁿ]_qq and
    C_vv(0)·[Mⁿ]_pp, with M the velocity-Verlet map for ω, and C_xx(0) = 3/(βmω²),
    C_vv(0) = 3/(βm)); the mean |h_end − h_start| of the RPMD `segments.csv` falls by a factor in
    [3.5, 4.5] when dt halves. Tests 10 and 11 drive the Y-1 cycle through the library, with
    |h_end − h_start| computed by the `segments.csv` definition; the CLI has no `Harmonic` or
    `DoubleWell` potential, and its cycle is pinned by test 9's reference-cycle comparison and by
    test 12 (Q30).
11. **Double well vs i-PI** (one atom, `DoubleWell` with barrier ≈ 3 k_BT, P = 16, RPMD and TRPMD
    λ = 0.5): C_xx(τ) and C_vv(τ) agree with i-PI driving the same potential through
    `ipi_client.py` (new backend `doublewell`) under the same segment protocol, within 5σ of the
    combined error at every lag. The generator (`gen_ipi_doublewell.py`, `CHECK_IPI=1`) also
    writes the exact quantum Kubo-transformed C_xx(t) from a DVR of the 1D Hamiltonian, reported
    in RESULTS.md with the RPMD and TRPMD curves; not a gate (RPMD is approximate for t > 0, and
    at t = 0 it is exact only as P → ∞).
12. **Liquid neon vs i-PI** (M4 system, N = 108, P = 16, 30 K, TRPMD λ = 0.5 and RPMD): C_vv(τ)
    within 5σ at every lag and D(τ_max) within 5σ, against i-PI + LAMMPS under the same protocol
    (`gen_ipi_neon_dynamics.py`, `CHECK_IPI=1`); C_vv(0) within 5σ of 3/(βm). Wall time of the
    generator stated in its header; if over 2 h, the protocol is shortened in the brief, not the
    tolerance.

For 11 and 12 the i-PI side runs the cycle as a sequence of i-PI runs (PILE-L for the
decorrelation steps, then NVE or PILE-L with `pile_lambda` and a centroid τ of zero friction for
the segment, momenta re-initialised thermally between them), with its own seeds. If i-PI cannot
reproduce the cycle as specified, the Phase A author stops and writes a `QUESTIONS.md` entry.

## Mutant ideas (physics-bug set from WORKFLOW.md applies)

Mandatory items that apply: PILE friction γ_k = ω_k instead of 2ω_k (here: T-4 with γ_k = λω_k);
c_k = exp(−γ dt/2); noise variance m/β instead of mP/β (in Y-2); forces not reused (an `evaluate`
in `set_thermostat`). New: centroid thermostatted under TRPMD; C_seg normalised by L instead of
L − τ; only s = 0 used as time origin; v̄ from bead 0 instead of the centroid, or without 1/m;
redraw missing (caught by test 3, not by the statistics); lag time off by `corr_stride`;
Green–Kubo without the 1/3 or with the rectangle rule; C_xx with wrapped positions.

## Out of scope

PA-CMD (M7b); dipoles, IR spectra, other operators; GLE or other thermostats; reaction rates and
flux-side correlation functions; quasi-centroid MD; finite-size corrections to D (reported only);
para-H₂ (optional experiment after M7); periodic MACE (M8).

## Acceptance

- [x] Owner approves this brief, the SPEC additions (T-4, Y-1 … Y-5) and the public API
- [x] TRPMD λ = 1/2 checked against Rossi et al. (2014) full text (Eqs. 28, 30); citation fixed
- [ ] M7a: all tests pass; `check.sh` exits 0, including thread-count runs
- [ ] M7a: long validation passes (`CHECK_LONG=1 CHECK_IPI=1`); agreement levels and the DVR
      comparison added to `validation/RESULTS.md`
- [ ] M7a: review done; no blocking findings; `QUESTIONS.md` has no open entries
- [ ] M7b briefed and approved (after M7a merges)

## M7b outline (PA-CMD; to be briefed after M7a)

- Dynamical masses: internal modes k ≥ 1 get mass m_i·(ω_k/Ω)² in the A step, so every internal
  mode oscillates at Ω; the centroid keeps m_i. Statics (sampling, estimators) unchanged: the
  thermostatted cycle still uses physical masses. Needs a SPEC rule extending I-2 to per-mode
  masses, and a public way to pass them (`RingPolymerIntegrator` option or a new constructor).
- Thermostat during segments: PILE-L on internal modes at γ = 2λΩ, centroid NVE (as i-PI's
  `pa-cmd` with `pile_lambda`); momentum redraw from the scaled-mass distribution.
- Config: `method = "pa-cmd"`, `cmd_omega_cm` (required), `cmd_lambda`.
- Time step: dt must resolve Ω; the brief states the bound and a test that the segment energy
  error grows as Ω·dt rises.
- Validation: harmonic exact (centroid decouples, as in M7a); double well and neon vs i-PI
  `pa-cmd`; the curvature problem is not tested (no molecules with bends in scope).
