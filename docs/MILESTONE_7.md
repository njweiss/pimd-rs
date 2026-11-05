# Milestone 7: Ring-polymer dynamics (RPMD, TRPMD; PA-CMD in M7b)

Status: M7a **approved and merged (2026-11-04)** (SPEC T-4 and §8a Y-1 … Y-5 added); M7b
**approved (2026-11-04) and merged (2026-11-05)**, brief at the end of this file.

| Phase | Flow | Output |
| --- | --- | --- |
| M7a | A/B/C | RPMD and TRPMD segments, centroid time-correlation functions, Green–Kubo diffusion, `[dynamics]` in the config |
| M7b | A/B/C | PA-CMD (brief at the end of this file) |

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
- [x] M7a: all tests pass; `check.sh` exits 0, including thread-count runs
- [x] M7a: long validation passes (`CHECK_LONG=1 CHECK_IPI=1`); agreement levels and the DVR
      comparison added to `validation/RESULTS.md`
- [x] M7a: review done; no blocking findings; `QUESTIONS.md` has no open entries
- [x] M7b briefed and approved (after M7a merges)

## After review (M7a, 2026-11-04)

Phase C at 7a64ac1: no blocking or should-fix findings. Accepted behaviour beyond the brief:

- **Allocation (Q17 policy).** A segment sample buffer (L·N·3 f64) that cannot be allocated is
  `InvalidValue("dynamics.segment_steps")` at parse time; a cycle length
  `decorrelation_steps + segment_steps` that overflows is `InvalidValue("dynamics.decorrelation_steps")`.
- **Restart refusals.** `CliError::Checkpoint` for a `[dynamics]` checkpoint not at a cycle end,
  and for a correlation block that fails `from_bytes` or whose n_lags or n_atoms differs from the
  config (only reachable with a corrupt or hand-made file; the hash covers these fields).
- **SI prefixes (U-3).** A pure SI-prefix factor (×1000 for fs → ps, as for Eh → meV) may be
  written inline next to the `units` constants it scales; physical conversion factors stay in
  `units`.

Nits fixed after review: TRPMD added to the O-step and draw-count docs (D-4), the n_atoms check
above, and the zero clamp of a rounding-negative variance documented on `sem`.

## M7b brief (PA-CMD)

Status: **approved (2026-11-04), merged (2026-11-05)**; Y-6, Y-7 and the Y-2 amendment added to SPEC. M7a is merged (main at e6008c9); M7b runs its own A/B/C
with tests committed as `tests(m7b):`.

Crates touched: `pimd-integrators` (dynamical masses in the integrator), `pimd-estimators`
(H_P with dynamical masses), `pimd-cli` (`method = "pa-cmd"`), plus two i-PI generators.
`pimd-core`, `pimd-potentials`, `pimd-socket` and `pimd-torch` are unchanged. Spec sections: 6
(I-2), 7 (T-1, T-2, T-4), 8a (Y-2, new Y-6, Y-7).

### Goal

Partially adiabatic centroid MD (PA-CMD): the internal ring-polymer modes get dynamical masses
that put them all at one adiabatic frequency Ω, and are thermostatted during the segments; the
centroid keeps its physical mass and runs NVE. Same segment cycle, correlation functions and
outputs as M7a. Validated against the exact harmonic result and i-PI's `pa-cmd` mode on the
double well (two Ω) and liquid neon.

### M7b decisions (owner, 2026-11-04)

1. **Whole-run masses.** With PA-CMD the dynamical masses apply to the whole run (equilibration,
   decorrelation and segments), as in i-PI. Positions sample the same distribution as with
   physical masses (masses do not enter the configurational distribution); there is no mass
   switch at segment boundaries. This replaces the outline's "the thermostatted cycle still uses
   physical masses".
2. **API.** Ω is an optional property of `RingPolymerIntegrator`, set once at construction; the
   free propagator, the O step and `resample_momenta` read it. `set_thermostat` keeps it.
3. **Thermostat rules generalised.** PILE-L and TRPMD use each mode's dynamical frequency
   (ω_k without Ω, Ω with it) and dynamical mass. Without Ω every M7a rule and result is
   unchanged, bitwise. PA-CMD segments use `Trpmd { lambda: cmd_lambda }` (centroid NVE, internal
   modes γ = 2λΩ); decorrelation uses PILE-L (internal γ = 2Ω).
4. **`cmd_lambda` is required** (no default). Literature checked 2026-11-04, no consensus value:
   Rossi, Ceriotti, Manolopoulos (2014)'s λ = 1/2 is for TRPMD; the original PA-CMD (Hone,
   Rossky, Voth, J. Chem. Phys. 124, 154103 (2006), second-hand) and DL_POLY Quantum 2.1
   (arXiv:2412.17216) use Nosé–Hoover chains on the internal modes, not Langevin; Castro,
   Trenins, Kapil, Rossi (arXiv:2508.16246, elevated-temperature variant) use γ_k = 2λΩ with
   λ = 0.001–0.01 ("under-damped regime to minimize artificial coupling"). These values go in the
   config documentation.
5. **Ω is required** (`cmd_omega_cm`, cm⁻¹, via `units::HARTREE_IN_WAVENUMBERS`; no default).
   Ω = P^{P/(P−1)}/β (ħ = 1) is documented as a suggested starting value once the path integral
   is converged in P; for large P it tends to ω_P = P/β, so its time-step cost is about that of
   plain PIMD. Published runs pick Ω by hand (16 000 cm⁻¹ in Rossi et al. 2014 for q-TIP4P/F
   water; 24 000 cm⁻¹ in Castro et al.). Constraints stated in the docs: Ω well above the
   physical frequencies of interest (adiabatic separation), Ω·dt well below π (all internal
   modes sit at Ω, so a resonance would hit every mode at once); check results at two Ω.
6. **Redraw.** Normals drawn in the Y-2 order, transformed to normal modes, mode k scaled by √s_k:
   the same draw order and count as Y-2, the exact scaled-mass distribution (orthogonal
   transform of iid normals). RPMD and TRPMD keep the M7a code path.
7. **H_P for `segments.csv`** under PA-CMD uses normal-mode kinetic energy with dynamical masses
   (new estimator function); RPMD/TRPMD keep `hamiltonian`. Nothing else in the outputs depends on
   momenta (E_prim, E_vir, t_est use positions and forces).
8. **Validation.** Harmonic exact; double well vs i-PI `pa-cmd` at Ω_h and 2Ω_h (Ω_h the
   heuristic value); liquid neon vs i-PI `pa-cmd` at Ω_h. Not in scope: the curvature problem,
   `wmax-cmd`.

### SPEC additions (added 2026-11-04)

Section 8a, after Y-5:

- **Y-6** Dynamical masses (PA-CMD). With an adiabatic frequency Ω > 0 (Eh/ħ), mode k of atom i
  has dynamical mass m_i·s_k with s_0 = 1 and s_k = (ω_k/Ω)² for ω_k > 0, and dynamical
  frequency Ω_k = ω_k/√s_k (= Ω for k ≥ 1, 0 for k = 0). These replace m_i and ω_k in I-2 (free
  propagation), in the noise variance of T-1 ((1 − c_k²)·m_i·s_k·P/β) and in the friction of
  T-2/T-4 (γ_k = 2Ω_k, γ_k = 2λΩ_k). The spring constants m_i·s_k·Ω_k² = m_i·ω_k² and hence the
  configurational distribution are unchanged; normal-mode momenta have variance m_i·s_k·P/β. The
  dynamical Hamiltonian is H̃_P = Σ_k Σ_i |p̃_{k,i}|²/(2 m_i s_k) + springs + Σ_j V(q_j). Without Ω
  (s ≡ 1) nothing changes. P = 1: s = [1], Ω has no effect.
- **Y-2** (amended) With Ω: draw z_{j,i,d} = sqrt(m_i·P/β)·ξ in the Y-2 order, transform to normal
  modes, multiply mode k by √s_k, transform back. Draw order and count as Y-2.
- **Y-7** PA-CMD cycle: Y-1 with Y-6 masses for the whole run; sampling thermostat PILE-L
  (T-1/T-2 with Y-6), segment thermostat `Trpmd { lambda }` (T-4 with Y-6: centroid NVE).
  Correlations Y-3 … Y-5 unchanged (the centroid mass is physical, so v̄ = (1/P) Σ_j p_j/m_i
  still holds).

### M7b: in scope

| ID | Item |
| --- | --- |
| Y-6 | `RingPolymerIntegrator::with_adiabatic_frequency`, mass factors in I-2, T-1, T-2, T-4 |
| Y-2 (amended) | `resample_momenta` with Ω |
| Y-6 | `hamiltonian_dynamical` |
| Y-7 | `method = "pa-cmd"` in `[dynamics]`; the cycle in the CLI |
| D-2, D-3, D-4 | Draw order unchanged; bitwise results across thread counts and restarts |

### Public API (stubs)

```rust
// pimd_integrators
impl RingPolymerIntegrator {
    /// PA-CMD dynamical masses (Y-6) with adiabatic frequency `omega` (Eh/ħ); recomputes c_k with
    /// the dynamical frequencies. Errors: `IntegratorError::InvalidValue` unless omega is
    /// positive and finite.
    pub fn with_adiabatic_frequency(self, omega: f64) -> Result<Self, IntegratorError>;
    /// Ω, or `None` (physical masses).
    pub fn adiabatic_frequency(&self) -> Option<f64>;
    /// s_k (Y-6), length P, in `NormalModes` column order; all 1.0 without Ω.
    pub fn mass_factors(&self) -> &[f64];
}

impl RingPolymerSim {
    /// The integrator (for Ω, the mass factors and the modes).
    pub fn integrator(&self) -> &RingPolymerIntegrator;
}
// set_thermostat keeps Ω; resample_momenta follows the amended Y-2.

// pimd_estimators
/// H̃_P with dynamical masses (Y-6): Σ_k Σ_i |p̃_{k,i}|²/(2 m_i s_k) with p̃ = `modes.to_normal(p)`,
/// plus springs and Σ energies as `hamiltonian`. Eh.
/// Panics on the shape mismatches of `hamiltonian`, or if `mass_factors.len()` ≠ P.
pub fn hamiltonian_dynamical(
    state: &RingState, beta: f64, energies: &[f64], modes: &NormalModes, mass_factors: &[f64],
) -> f64;
```

`pimd_cli`: `DynamicsMethod` gains `PaCmd { omega: f64 /* Eh/ħ */, lambda: f64 }`. No other
type changes. Allowed new dependencies: none.

### Config format (addition)

```toml
[dynamics]
method = "pa-cmd"
cmd_omega_cm = 400.0          # Ω in cm⁻¹, required; suggested start P^{P/(P−1)}/β (see docs)
cmd_lambda = 1.0              # internal-mode friction γ = 2λΩ during segments, required
decorrelation_steps = 250
segment_steps = 500
corr_stride = 2
n_lags = 251
```

### Clarifications

- `cmd_omega_cm` and `cmd_lambda` are required with `"pa-cmd"` and rejected with the other
  methods (as `trpmd_lambda`, Q29 rules); `trpmd_lambda` is rejected with `"pa-cmd"`. Ω and λ
  positive and finite, else `InvalidValue("dynamics.cmd_omega_cm")` / `("dynamics.cmd_lambda")`.
  No check of Ω·dt (documented, not enforced).
- Ω enters the integrator for equilibration and production. The config hash appends the method
  tag, Ω and λ for `"pa-cmd"`; M7a configs hash as on main at e6008c9 (recorded in the test), and
  configs without `[dynamics]` as at cf79a4d.
- Checkpoint layout unchanged (bead momenta are stored as before; Ω comes from the config).
- `segments.csv` H_P values use `hamiltonian_dynamical`/N under PA-CMD, `hamiltonian`/N
  otherwise.
- `run.log` echoes Ω in cm⁻¹ and Eh/ħ, and λ.
- Initial momenta of a `"pa-cmd"` run (Q32): the amended Y-2 draw, i.e. the M4/M5 initial draw
  (P·N·3 normals, same order) transformed to normal modes, mode k scaled by √s_k, transformed
  back (as i-PI with `dynm3`). Without `[dynamics]` and for rpmd/trpmd the initial draw is
  unchanged.
- Adding `PaCmd` allows one `unreachable!` arm in the M7a test helper `dyn_common/mod.rs`
  (`reference_dynamics`), nothing else in a locked file (Q31, as Q28).
- Points pinned by the tests (Q33): a missing or misplaced `cmd_*` key is `Parse` naming the key or
  `InvalidValue("dynamics.<key>")`; the run.log echo is checked by value, no line format; P = 1
  with Ω is bitwise equal to P = 1 without Ω; "unchanged without Ω" is pinned by FNV-1a
  fingerprints recorded with the e6008c9 code; `with_adiabatic_frequency` works with any
  thermostat and survives `set_thermostat`; i-PI's 8-digit cm⁻¹ factor (1e-8 relative) needs no
  allowance; the panic messages of `hamiltonian_dynamical` are not specified.

### Required tests

Deterministic:

1. **Mass factors and free propagation (Y-6, I-2).** `mass_factors` = [1, (ω_k/Ω)², …] by hand
   (P = 4, 8); Ω ≤ 0, NaN, ∞ rejected. `free_propagate` with Ω over t = 2π/Ω returns every
   internal mode to its start (1e-12) while the centroid drifts by t·p̄/m; one step against I-2
   written out with m_i·s_k and Ω. Without Ω, integrators are bitwise as on main.
2. **O step with Ω (T-1, T-2, T-4 with Y-6).** Per-mode c_k = exp(−2Ω dt) (PILE-L internal),
   exp(−2λΩ dt) (TRPMD), centroid as before; noise amplitude sqrt((1 − c_k²) m_i s_k P/β) through a
   known draw sequence; draw order and count unchanged; `set_thermostat` keeps Ω (a switched sim
   equals one built with Ω and the target thermostat, bitwise).
3. **Redraw with Ω (amended Y-2).** Equal, to 1e-12, to the Y-2 draws from a cloned RNG passed
   through the RP-4 matrix built in the test and scaled by √s_k; P·N·3 draws; positions, forces,
   energies untouched; without Ω bitwise as M7a.
4. **`hamiltonian_dynamical`.** Hand value (P = 4, N = 2); equals `hamiltonian` to 1e-12 relative
   for s ≡ 1; under `Off` with Ω (LJ cluster, P = 4) the mean |ΔH̃| over a fixed trajectory falls
   by a factor in [3.5, 4.5] when dt halves.
5. **Harmonic centroid decouples with Ω.** As M7a test 5, with Ω set and under `Trpmd`: the
   centroid follows the velocity-Verlet map to 1e-12 over 1 000 steps.
6. **Config.** `"pa-cmd"` parses to `PaCmd` with Ω = cmd_omega_cm/HARTREE_IN_WAVENUMBERS; the
   rules above; Ω and λ covered by the hash; M7a and no-`[dynamics]` hashes unchanged.
7. **Run and restart.** A short PA-CMD run (LJ cluster, P = 4, 3 cycles) is byte-identical
   across repeats and 1/4 threads; restart at a cycle end is bitwise; the CLI cycle equals a
   plain-loop reference cycle (as M7a `run_matches_reference_cycle`) including `segments.csv`
   with H̃_P.

Statistical (`long-validation`):

8. **Harmonic, exact** (M7a test 10 system: ω = 1, β = 8, P = 8 so ω_P = 1, dt = 0.1; Ω = 4,
   Ω·dt = 0.4; λ = 1): C_xx and C_vv within 5σ of the M7a exact finite-dt expressions at every
   lag (the centroid is unchanged by Y-6). Under `Off` with Ω = 2, 4, 8 (Ω·dt = 0.2, 0.4, 0.8)
   at fixed dt, the mean segment |ΔH̃| is strictly increasing, each step resolved by > 5σ.
9. **Double well vs i-PI `pa-cmd`** (M7a system, P = 16, β = 1.5, dt = 0.05, λ = 1) at
   Ω_h = 16^{16/15}/β ≈ 12.83 Eh/ħ (Ω·dt ≈ 0.64) and 2Ω_h: C_xx and C_vv within 5σ_c of i-PI at
   every lag, bias allowance 0. The DVR Kubo C_xx (already in `ipi_doublewell.json`) and the
   Ω_h-vs-2Ω_h difference are reported in RESULTS.md, not gated.
10. **Liquid neon vs i-PI + LAMMPS `pa-cmd`** (M4/M7a system, Ω_h ≈ 401 cm⁻¹, Ω·dt ≈ 0.15,
    λ = 1): C_vv within 5σ_c at every lag, D(τ_max) within 5σ_c; C_vv(0) within 5σ of 3/(βm);
    ⟨E_vir⟩ of the run within 5σ_c of the M4 i-PI value in `ipi_neon.json` (statics unchanged by
    Y-6).

New generators `gen_ipi_doublewell_cmd.py` and `gen_ipi_neon_cmd.py` (the locked M7a
generators and data stay as they are); i-PI `normal_modes` with `<frequencies style="pa-cmd"
units="inversecm">` and `pile_l` with `pile_lambda` = λ during segments, same scripted cycle as
M7a. Phase A confirms in i-PI's source that its cm⁻¹ conversion is the same angular-frequency
convention (ħΩ = Ω_cm/HARTREE_IN_WAVENUMBERS Eh). Expected wall times about 36 and 16 min; over
2 h ⇒ shorten the protocol in the brief, not the tolerance.

λ = 1 in the tests is a test value (distinct from TRPMD's 0.5 so that mixing the two keys is
caught), not a recommendation.

### Mutant ideas

s_k = ω_k/Ω (not squared); Ω applied to the centroid; T-1 noise with m_i instead of m_i·s_k;
friction from ω_k instead of Ω; redraw without the √s_k scaling; `segments.csv` with the bead
kinetic energy under PA-CMD; cm⁻¹ conversion with an extra 2π; equilibration without Ω;
`set_thermostat` dropping Ω; `cmd_lambda` read from `trpmd_lambda`.

### Out of scope

`wmax-cmd` and other mass schemes; the curvature problem (no bent molecules in scope);
Nosé–Hoover chains on the internal modes; Ω from a formula (decision 5); a run-time Ω·dt check;
elevated-temperature CMD variants.

### Acceptance (M7b)

- [x] Owner approves this brief, Y-6, Y-7, the Y-2 amendment and the public API
- [x] All tests pass; `check.sh` exits 0
- [x] Long validation passes (`CHECK_LONG=1 CHECK_IPI=1`); results in `validation/RESULTS.md`
- [x] Review done; no blocking findings; `QUESTIONS.md` has no open entries
