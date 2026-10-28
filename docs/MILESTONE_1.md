# Milestone 1: Foundations

Crates touched: `pimd-core`, `pimd-potentials`. Spec sections: 1, 2, 3, 4, 5, 9 (P-1, P-2 for `Harmonic`, P-4).

## Goal

The building blocks every later milestone depends on: units, cell, bead storage, RNG, normal modes, the `Potential` trait, and the harmonic potential. Nothing here integrates dynamics.

## Already done (Phase 0)

- Workspace skeleton, empty modules, error enums (`CoreError`, `PotentialError`) with the variants used below.
- `pimd_core::units` constants (U-2) as `pub const f64`.
- `Beads` (storage and trivial accessors only, S-1): `zeros`, `from_vec`, `n_beads`, `n_atoms`, `get`, `set`, `as_slice`, `as_mut_slice`.
- `Cell` struct definition with `lengths()` and `volume()`.

## In scope

| ID | Item |
|---|---|
| U-2 | Constants exist and match the spec values exactly (already done; tested anyway) |
| U-4 | `beta_from_kelvin` |
| D-2 | `Rng` seeded determinism and normal/uniform draws |
| C-1 | `Cell::new` validation |
| C-2 | `Cell::min_image` |
| C-3 | `Cell::wrap` |
| RP-1 | `NormalModes::new` accepts P ≥ 1 |
| RP-2 | `omega_p()` = P/β |
| RP-4 | Matrix and frequencies in the exact column order |
| RP-5 | CᵀSC = diag(ω_k²) |
| RP-6 | P = 1 gives C = [1], ω = [0] |
| S-1 | Bead indexing (via `Beads::centroid` and the transforms) |
| P-1, P-2 | `Potential` trait and `Harmonic` |
| P-4 | `Harmonic` returns `Unsupported` for `subset = Some` |

## Public API (stubs: bodies `unimplemented!()`)

```rust
// pimd_core::units
pub fn beta_from_kelvin(t_kelvin: f64) -> Result<f64, CoreError>; // 1/(k_B T); error if T <= 0 or non-finite

// pimd_core::cell
impl Cell {
    pub fn new(lengths: [f64; 3]) -> Result<Cell, CoreError>;   // each length > 0 and finite
    pub fn min_image(&self, d: [f64; 3]) -> [f64; 3];           // C-2
    pub fn wrap(&self, r: [f64; 3]) -> [f64; 3];                // C-3
}

// pimd_core::beads
impl Beads {
    pub fn centroid(&self) -> Vec<[f64; 3]>;                    // per atom, mean over beads
}

// pimd_core::rng
pub struct Rng { /* ChaCha8Rng */ }
impl Rng {
    pub fn from_seed(seed: u64) -> Rng;     // ChaCha8Rng::seed_from_u64
    pub fn normal(&mut self) -> f64;        // StandardNormal
    pub fn uniform(&mut self) -> f64;       // [0, 1)
}

// pimd_core::normal_modes
pub struct NormalModes { /* n_beads, omega_p, matrix, freqs */ }
impl NormalModes {
    pub fn new(n_beads: usize, beta: f64) -> Result<NormalModes, CoreError>;  // P >= 1, beta > 0 finite
    pub fn n_beads(&self) -> usize;
    pub fn omega_p(&self) -> f64;
    pub fn frequencies(&self) -> &[f64];           // length P, aligned with columns (RP-4)
    pub fn matrix(&self) -> &[f64];                // row-major P x P: C[j][k] at j*P + k
    pub fn to_normal(&self, beads: &Beads) -> Beads;   // q~_k = sum_j C_jk q_j, same shape, mode-major
    pub fn to_beads(&self, modes: &Beads) -> Beads;    // inverse (transpose)
    // to_normal/to_beads panic if the input's n_beads() != P (programming error, documented; Q2)
}

// pimd_potentials
pub enum PotentialError { Unsupported, InvalidParameter { what, value }, ShapeMismatch { expected: usize, actual: usize } } // atom counts (P-1, Q2)
pub struct Evaluation { pub energies: Vec<f64>, pub forces: Beads, pub virial: Option<[[f64; 3]; 3]> }
pub trait Potential {
    fn evaluate(
        &mut self,
        positions: &Beads,
        cell: Option<&Cell>,
        subset: Option<&[usize]>,
    ) -> Result<Evaluation, PotentialError>;
}
pub struct Harmonic { /* masses, omega */ }
impl Harmonic {
    pub fn new(masses: Vec<f64>, omega: f64) -> Result<Harmonic, PotentialError>; // masses > 0 finite, omega >= 0 finite
}
impl Potential for Harmonic { /* per bead: V = sum_i 1/2 m_i omega^2 |q_i|^2; force -m_i omega^2 q_i */ }
```

Allowed dependencies: `rand`, `rand_chacha`, `rand_distr`, `thiserror`. Test-only: `approx`.

## Required tests (minimum; the author may add more)

- **Units:** each constant equals its spec value; `beta_from_kelvin(300.0)` equals 1/(3.1668115634556e-6·300) to 1 ulp-level relative 1e-14; errors for 0, negative, NaN, infinity.
- **Cell:** `new` rejects non-positive/non-finite lengths. `min_image` and `wrap` tested on: hand-picked values, exactly ±L/2 and 0 (min_image(L/2) = −L/2), multiples of L, negative inputs; property tests over seeded random vectors: result in range, and (input − result)/L is an integer within 1e-12 per component.
- **Rng:** same seed → identical sequences; different seeds differ; 200 000 normals have mean within 5/√n of 0 and variance within 5√(2/n) of 1 (fixed seed); uniforms lie in [0,1) with mean within 5σ of 0.5. No golden values.
- **NormalModes:**
  - The test builds C independently from the RP-4 formula and compares `matrix()` elementwise to 1e-14 for P ∈ {1, 2, 3, 4, 5, 8, 9, 16}.
  - CᵀC = I to 1e-13.
  - Builds S independently (RP-5) and checks CᵀSC = diag(frequencies()²) to 1e-11 relative to ω_P².
  - Frequencies equal the RP-4 formula; `omega_p() = P/β`.
  - Round trip: `to_beads(to_normal(x)) = x` to 1e-13 on seeded random beads, P ∈ {1, 2, 3, 8}, N = 3.
  - q̃_0 = √P · centroid (S-1 layout), and `Beads::centroid` against a hand-computed example.
  - Spring energy invariance: Σ_j ½ m ω_P²|q_j − q_{j+1}|² = Σ_k ½ m ω_k² q̃_k² to 1e-12 relative.
  - P = 1: matrix [1.0], frequencies [0.0].
  - Errors for P = 0, β ≤ 0, NaN.
- **Harmonic:** energies and forces against hand-computed values at a few points (hand calculation in a comment); linearity of forces; per-bead independence (beads differing in position give independent energies); `Unsupported` for `subset = Some`; `ShapeMismatch` when `positions.n_atoms()` differs from the number of masses (P-1, Q2); `Some(cell)` gives results bitwise equal to `None` (P-2, Q3); error on non-positive mass, negative ω.
- **Finite-difference check** (independent): forces = −(numerical gradient of energies) to 1e-6 on random beads.

## Mutant ideas (at least four, from different rules)

ω_P = P/β → √P/β; sin dropped from frequencies; cos/sin columns swapped; centroid column scaled by 1/P; min_image round-half-up; wrap into (−L/2, L/2]; normal draws taken from uniform; Harmonic force with wrong sign or missing mass factor; `to_beads` using C instead of Cᵀ.

## Out of scope

Any integration, thermostat, estimator, neighbor list, LJ, PBC in potentials, config parsing, rayon.

## Acceptance

- [ ] All tests pass; `check.sh` exits 0
- [ ] Reviewer traceability table has no MISSING or UNTESTED rows
