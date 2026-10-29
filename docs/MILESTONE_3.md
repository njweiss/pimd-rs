# Milestone 3: Classical core

Crates touched: `pimd-core` (neighbor list), `pimd-potentials` (`LennardJones`), `pimd-integrators` (`VelocityVerlet`), `pimd-estimators` (RDF, pressure). Spec sections: 2 (D-3), 4 (C-4, C-5), 6 (I-4), 8 (E-7, E-8), 9 (P-3, P-5), 10.

## Goal

A classical Lennard-Jones MD core that matches LAMMPS on identical input, is bitwise deterministic across thread counts, and provides RDF and pressure. This is also the potential used by the quantum-liquid milestone (M4).

## Already done

M1 and M2 merged. Reference tooling (`pixi run -e ref lmp`) works. `EstimatorError` (`InvalidValue`, `ShapeMismatch`) exists again (Phase 0 addendum; Q4 covered only the `f64` energy estimators).

## In scope

| ID | Item |
|---|---|
| N-1 | Verlet neighbor list, full pairs, sorted |
| N-2 | Rebuild criterion with unwrapped displacements and skin/2 |
| N-3 | One list per bead in `LennardJones` |
| P-3 | LJ energy, force, optional shift, no tail correction |
| P-5 | Virial for P = 1 |
| C-4 | Cutoff + skin ≤ L_min/2 check |
| I-4 | `VelocityVerlet` |
| D-3 | Bitwise determinism across thread counts |
| E-7 | RDF |
| E-8 | Pressure |

## Public API (stubs)

```rust
// pimd_core::neighbor
pub struct NeighborList { /* cutoff, skin, pairs, reference positions */ }
impl NeighborList {
    pub fn new(cutoff: f64, skin: f64) -> Result<NeighborList, CoreError>; // cutoff > 0, skin >= 0
    pub fn build(&mut self, positions: &[[f64; 3]], cell: Option<&Cell>) -> Result<(), CoreError>; // C-4 check when cell is Some
    pub fn needs_rebuild(&self, positions: &[[f64; 3]]) -> bool;           // N-2
    pub fn neighbors(&self, i: usize) -> &[usize];                          // N-1: sorted ascending, j != i
    pub fn n_atoms(&self) -> usize;
}

// pimd_potentials
pub struct LjParams { pub epsilon: f64, pub sigma: f64, pub cutoff: f64, pub skin: f64, pub shift: bool }
pub struct LennardJones { /* params, per-bead neighbor lists */ }
impl LennardJones { pub fn new(params: LjParams) -> Result<LennardJones, PotentialError>; }
impl Potential for LennardJones { /* P-3, P-5, N-3; per-atom accumulation in fixed order (D-3) */ }

// pimd_integrators
pub struct VelocityVerlet { /* dt */ }
impl VelocityVerlet {
    pub fn new(dt: f64) -> Result<VelocityVerlet, IntegratorError>;          // I-5
    /// `forces` holds F at the current positions on entry and F at the new positions on exit.
    /// Returns the potential energy at the new positions. Requires state with exactly one bead.
    pub fn step(
        &self,
        state: &mut RingState,
        forces: &mut Beads,
        potential: &mut dyn Potential,
        cell: Option<&Cell>,
    ) -> Result<f64, IntegratorError>;
}

// pimd_estimators
pub struct Rdf { /* bins, counts, n_atoms, cell, n_samples */ }
impl Rdf {
    pub fn new(n_bins: usize, r_max: f64, cell: Cell, n_atoms: usize) -> Result<Rdf, EstimatorError>; // r_max <= L_min/2
    pub fn accumulate(&mut self, q: &Beads) -> Result<(), EstimatorError>;   // all beads of one configuration
    pub fn n_samples(&self) -> usize;
    pub fn bin_centers(&self) -> Vec<f64>;                                   // (r_b + r_{b+1})/2
    pub fn g(&self) -> Vec<f64>;                                             // E-7, averaged over samples
}
pub fn kinetic_classical(masses: &[f64], p: &Beads) -> f64;                  // sum p^2/(2m), P = 1
pub fn pressure_classical(
    masses: &[f64], p: &Beads, virial: [[f64; 3]; 3], cell: &Cell,
) -> Result<f64, EstimatorError>;                                            // E-8
```

Allowed new dependency: `rayon` in `pimd-potentials` (and `pimd-core` if the implementer needs it for the list build). Test-only: `approx`; thread-pool control via `rayon::ThreadPoolBuilder` in tests.

## Clarifications

- **Parallelism and D-3:** `LennardJones` computes each atom's force and energy contribution from its own full neighbor list (every pair visited twice; per-atom energy is half the pair sum), then reduces per-atom values in index order. No Newton's-third-law scatter across threads.
- **Cutoff handling:** pairs with r ≥ r_c contribute nothing. With `shift`, V(r) − V(r_c) inside r_c; forces never shifted.
- **P > 1:** energies per bead, forces per bead; `virial` is `None`.
- **Neighbor list rebuild inside `evaluate`:** `LennardJones::evaluate` calls `needs_rebuild` per bead and rebuilds when true or when the atom count changed. Positions passed are unwrapped (C-5).

## Required tests

- **Neighbor list vs brute force:** seeded random configurations (open and periodic, N ≤ 300, including atoms near faces and pairs separated by almost L/2 and by more than L/2 along one axis). The test computes `neighbors(i)` via its own brute-force min-image distance and compares exactly (membership and order). Rebuild criterion tested by displacing one atom by just under and just over skin/2, including a displacement that crosses the periodic boundary in unwrapped coordinates.
- **LJ analytic:** two-atom energy and force at several r (hand-computed values in comments), cutoff edge (just inside and just outside r_c), shifted vs unshifted, no tail correction, Newton's third law (net force zero) on random configurations.
- **Finite-difference force check** on random periodic configurations to 1e-6 relative.
- **LAMMPS single point** (`gen_lj_snapshot.py`, run via `pixi run -e ref`): fixed 108- or 256-atom configuration with velocities; `lj/cut` with `pair_modify shift yes` and `no`; per-atom forces, total energy, virial/pressure from `compute pressure`; match to 1e-10 relative (document the LAMMPS version in the data file). Also P = 1 virial → `pressure_classical` against LAMMPS pressure at the same snapshot.
- **LAMMPS trajectory:** 20 steps of NVE from the same snapshot; positions match to 1e-8 absolute (LJ units). Document that agreement degrades with chaos beyond this horizon.
- **NVE drift scaling:** liquid-like state (e.g. ρ* = 0.8, T* ≈ 1.0, N = 256, rc = 2.5σ shifted); fluctuation or drift of total energy over a fixed duration scales as dt² (halving dt reduces by a factor in [3, 5]).
- **Thermodynamic comparison with LAMMPS:** equilibrated NVE (or NVT) averages of potential energy and pressure at the same state point from the engine and from LAMMPS, block-averaged error bars, agreement within 5σ_combined. Published EOS values (e.g. Johnson–Zollweg–Gubbins 1993) may be added only if the author can cite exact table entries; otherwise LAMMPS is the reference. Long runs are integration tests gated by the Cargo feature `long-validation` (see WORKFLOW.md); a short smoke version stays in the default `cargo test`.
- **RDF:** hand-counted two- and three-atom example with exactly computable bins; ideal-gas statistical test (uniform random points, g = 1 within 5σ, enough samples); LAMMPS `compute rdf` on a static snapshot matched to 1e-10 (convention differences are logged in `QUESTIONS.md`, not worked around); periodic distances via the test's own min-image.
- **Pressure:** hand-computed two-body example; ideal-gas limit (no interactions: p = 2K/(3V)).
- **Determinism (D-3):** LJ forces and energies from a 1-thread pool and a 4-thread pool are bitwise equal (`to_bits()`); a 200-step VelocityVerlet run is bitwise equal across thread counts.
- **Error paths:** cutoff + skin > L_min/2 rejected; non-positive ε, σ, cutoff, negative skin rejected; P > 1 gives `virial = None`; `subset = Some` → `Unsupported`.

## Mutant ideas (physics-bug set from WORKFLOW.md applies)

Shift applied beyond r_c; force truncated differently from energy; skin check with skin not skin/2; wrapped instead of unwrapped displacement in rebuild; min image with round-half-up; reduction in thread-scheduling order; missing factor 2 in RDF normalization; pressure virial sign; kinetic term using Σmv² instead of ½.

## Out of scope

Thermostats on classical runs, cell lists beyond what the list build needs, neighbor lists for non-orthorhombic cells, long-range corrections, config, restarts.

## Acceptance

- [ ] All tests pass; `check.sh` exits 0 (including the thread-count runs)
- [ ] LAMMPS comparisons documented with version and agreement levels
- [ ] Review done; no blocking findings
