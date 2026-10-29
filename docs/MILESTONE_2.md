# Milestone 2: Ring-polymer sampling

Crates touched: `pimd-integrators` (new), `pimd-estimators` (new), `pimd-potentials` (add `DoubleWell`). Spec sections: 6, 7, 8 (E-1 … E-5, E-9), 9 (P-2 `DoubleWell`).

## Goal

A ring-polymer integrator (BAOAB with exact free propagation and PILE-L thermostat) and the energy estimators, validated on the harmonic oscillator (exact finite-P results, E-9) and an anharmonic reference.

## Already done

M1 merged. Error enum `IntegratorError` and empty modules exist (Phase 0 addendum). There is no `EstimatorError` (Q4).

## In scope

| ID | Item |
|---|---|
| I-1 | Step is B A O A B |
| I-2 | Exact free propagation per mode, including ω_k = 0 |
| I-3 | One `evaluate` per step, forces reused; one extra at construction |
| I-5 | dt validation |
| T-1 | PILE-L O step including exact draw order |
| T-2 | γ_0 = 1/τ_0 or 0; γ_k = 2ω_k |
| T-3 | `Thermostat::Off` conserves H_P with O(dt²) error |
| RP-3 | Momentum variance mP/β; H_P definition |
| RP-6 | P = 1 equals classical |
| E-1 … E-5 | Estimators |
| E-9 | Harmonic finite-P exact reference |
| P-2 | `DoubleWell` |

## Public API (stubs)

```rust
// pimd_integrators
pub struct RingState { pub q: Beads, pub p: Beads, pub masses: Vec<f64> }   // bead representation, shape [P,N,3]
pub enum Thermostat { Off, PileL { tau0: Option<f64> } }                     // tau0 > 0 when Some

pub struct RingPolymerIntegrator { /* dt, beta, NormalModes, thermostat, gamma_k, c_k */ }
impl RingPolymerIntegrator {
    pub fn new(dt: f64, beta: f64, n_beads: usize, thermostat: Thermostat)
        -> Result<Self, IntegratorError>;                                    // I-5, beta > 0, tau0 > 0
    pub fn free_propagate(&self, state: &mut RingState, t: f64);             // I-2, time t, any sign
    pub fn thermostat_step(&self, state: &mut RingState, rng: &mut Rng);     // T-1, uses dt
    pub fn modes(&self) -> &NormalModes;
}

pub struct RingPolymerSim { /* state, forces, energies, integrator, potential, cell */ }
impl RingPolymerSim {
    pub fn new(
        state: RingState,
        cell: Option<Cell>,
        integrator: RingPolymerIntegrator,
        potential: Box<dyn Potential>,
    ) -> Result<Self, IntegratorError>;          // performs the initialising evaluate; masses positive finite (S-2) else InvalidValue
    pub fn step(&mut self, rng: &mut Rng) -> Result<(), IntegratorError>;    // I-1, I-3
    // new and step: an Evaluation with forces shape != [P,N,3] or energies.len() != P is ShapeMismatch (P-1)
    pub fn state(&self) -> &RingState;
    pub fn forces(&self) -> &Beads;              // -grad V at current positions
    pub fn energies(&self) -> &[f64];            // V(q_j) per bead at current positions
}

// pimd_estimators  (all totals over atoms and components, in Eh)
// Each panics if input shapes disagree (masses.len() != N, q/forces shapes differ, energies.len() != P); documented (Q4)
pub fn spring_energy(q: &Beads, masses: &[f64], beta: f64) -> f64;                    // E-5
pub fn kinetic_primitive(q: &Beads, masses: &[f64], beta: f64) -> f64;                // E-1
pub fn kinetic_virial(q: &Beads, forces: &Beads, beta: f64) -> f64;                   // E-2
pub fn potential_mean(energies: &[f64]) -> f64;                                       // E-3
pub fn total_primitive(q: &Beads, masses: &[f64], beta: f64, energies: &[f64]) -> f64; // E-4
pub fn total_virial(q: &Beads, forces: &Beads, beta: f64, energies: &[f64]) -> f64;    // E-4
pub fn hamiltonian(state: &RingState, beta: f64, energies: &[f64]) -> f64;             // H_P of RP-3

// pimd_potentials
impl DoubleWell { pub fn new(a: f64, b: f64) -> Result<DoubleWell, PotentialError>; } // a, b > 0; V per component a x^4 - b x^2
```

Allowed new dependencies: none beyond M1's. Test-only: `approx`; a small eigen-solver dev-dependency (`nalgebra`) is allowed *only* if the author prefers it over precomputed JSON reference data.

## Required tests

All hand/analytic expectations are derived in comments from SPEC rule IDs. The test builds its own C and mode frequencies from RP-4 (never from `NormalModes`).

- **Free propagation (I-2):** against an independent per-mode formula for P ∈ {1, 2, 3, 4, 8}; composition (t₁ then t₂ equals t₁+t₂) and reversal (t then −t) to 1e-12; conservation of the free ring Hamiltonian Σp²/2m + springs exactly (1e-12 relative); centroid moves uniformly; ω_0 = 0 branch is drift (also tested at P = 1).
- **Thermostat (T-1/T-2):**
  - `Off` is the identity.
  - With τ₀ = `None`, the centroid momentum is unchanged exactly.
  - Draw-order test: a second `Rng` with the same seed reproduces the momenta predicted by the test's own T-1 formula, drawing in order k (outer), atom, dimension. This pins the contract.
  - Variance test: start from p = 0 with N = 4000 atoms, P = 4 (several τ₀ values), one O step. The per-mode variance of the new p̃ equals (1 − c_k²)·m·P/β within 5σ (σ of a sample variance = var·√(2/(n−1))), using c_k from T-2.
- **NVE energy conservation (T-3):** harmonic, P = 8, Off. H_P is conserved with max deviation scaling as dt²: halving dt reduces the max |H_P(t) − H_P(0)| over a fixed duration by a factor in [3, 5].
- **Sim boundary (S-2, P-1, C-5; added after review):** bad masses rejected by `new`; wrongly shaped `Evaluation` rejected by `new` and `step`; a `Some(cell)` reaches every `evaluate` and the ring is never wrapped.
- **Evaluation count (I-3):** a counting `Potential` wrapper (defined in the test) sees exactly 1 call at construction and exactly 1 per step.
- **P = 1 equals classical (RP-6):** with Off, P = 1, the trajectory matches an independent velocity-Verlet written in the test to 1e-12 relative over 200 steps (harmonic and `DoubleWell`).
- **Estimators:** spring energy, primitive and virial kinetic, and totals against hand-computed small examples (P = 2, N = 1; hand calculations in comments); for P = 1 both kinetic estimators equal 3N/(2β) at any positions (the spring term and q − q̄ vanish).
- **Harmonic sampling (E-9):** PILE-L (τ₀ chosen by the author, e.g. ~100 dt), P ∈ {1, 4, 16}, N small, β·ω in the quantum regime (e.g. βω ≈ 4). Time-averaged `total_primitive` and `total_virial` match E-9 within 5σ (block averaged) plus a stated dt-bias allowance. Also: the sample variance of `total_virial` is smaller than that of `total_primitive` at P = 16 by a margin the reference validates (quoted in a comment).
- **Equipartition-type check (T-1/RP-3):** at equilibrium, ⟨p̃_k²⟩ = m·P/β per component within 5σ for a long free ring-polymer run with PILE-L.
- **DoubleWell:** energy and force at hand-computed points; finite-difference force check. Exact-diagonalization comparison: 1D, single particle. The author chooses and documents mass, a, b and β so that the well is clearly quantum (βω_well ≈ 2–6) and the barrier is a few kT, and so that P = 32 and P = 64 are in the O(1/P²) regime (check with the reference). Compare the time-averaged `total_virial` at P = 32 and P = 64 to the exact quantum thermal energy from the generator script (`gen_double_well.py`: numpy diagonalization in a grid or harmonic-oscillator basis, with basis convergence shown). The tolerance accounts for the discretization error visible in the difference between the two P values.

## Mutant ideas (physics-bug set from WORKFLOW.md applies)

ω_P = P/β → √P/β; E-1 without 1/P; virial using E_spring; noise variance m/β; γ_k = ω_k; c_k = exp(−γ dt/2); O step after the last B; A steps using dt instead of dt/2; ω_k = 0 branch dividing by zero or not drifting; forces recomputed twice; thermostat draw order swapped (atom outer, mode inner); wrong sign in rotation (sin term).

## Out of scope

LJ, neighbor lists, cells in integrators beyond passing them through, RDF, pressure, CLI, Langevin on single replica (use P = 1 PILE-L), config, restarts.

## Acceptance

- [x] All tests pass; `check.sh` exits 0
- [x] E-9 reproduced for P = 1, 4, 16
- [x] Review done (mandatory for this milestone); no blocking findings
