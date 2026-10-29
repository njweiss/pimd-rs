# pimd-rs specification

Authoritative. Rules carry IDs (e.g. `RP-3`); briefs, tests and reviews cite them. If code or tests disagree with this file, that is a finding: record it in `docs/QUESTIONS.md`, do not work around it.

## 0. Purpose and non-goals

A validated, open-source path-integral MD (PIMD/RPMD) engine in Rust with a model-agnostic potential interface, used later for an ML-RPMD study. Correctness is demonstrated against analytic results and reference codes (LAMMPS, OpenMM, i-PI).

Non-goals: training MLIPs, a general-purpose MD package, electrostatics (Ewald/PME), holonomic constraints, GUI/workflow tooling, Python bindings (deferred until an analysis need appears).

## 1. Units and constants

- **U-1** All internal quantities are Hartree atomic units: ħ = 1, length in bohr, mass in electron masses, energy in hartree (Eh), time in ħ/Eh. Temperature enters only through β.
- **U-2** Constants (CODATA 2018), defined once in `pimd_core::units`: k_B = 3.1668115634556e-6 Eh/K; 1 amu = 1822.888486209 m_e; 1 bohr = 0.529177210903 Å; 1 fs = 41.341373336561 a.u. of time; 1 Eh = 219474.6313632 cm⁻¹; 1 Eh = 627.5094740631 kcal/mol.
- **U-3** No unit constants or conversion factors outside `units`. Conversion happens only at I/O boundaries (config parsing, output, model adapters).
- **U-4** β = 1/(k_B T), with T in kelvin and β in Eh⁻¹. Non-positive or non-finite T is an error.

## 2. Precision, randomness, determinism

- **D-1** f64 everywhere in the core. Casting to f32 happens only inside model adapters.
- **D-2** All randomness comes from `pimd_core::Rng`: ChaCha8 (`rand_chacha`) seeded from a `u64`; standard normals from `rand_distr::StandardNormal`. Draw order is part of the contract (see T-1); it is reproducible for fixed dependency versions (pinned by `Cargo.lock`).
- **D-3** For a fixed seed and config, results are bitwise identical across runs and across thread counts. Parallel code computes per-atom (or fixed-size-chunk) partial results and combines them in a fixed order. Forbidden: float reductions whose order depends on scheduling, `HashMap`/`HashSet` iteration that can affect numbers, wall-clock or address-dependent behaviour.
- **D-4** Every stochastic step documents its draw order.

## 3. Data layout and sign conventions

- **S-1** `Beads` stores shape `[P, N, 3]`, bead-major, row-major: element (j, i, d) is at index `(j*N + i)*3 + d`.
- **S-2** Masses: one per atom, in m_e, positive and finite.
- **S-3** Forces are −∇V in Eh/bohr. They never include ring-polymer spring forces.
- **S-4** Per-bead energy is V(q_j) only; springs are not included.

## 4. Cell and boundaries

- **C-1** `Cell` is orthorhombic: three positive finite lengths in bohr. A cell is passed as `Option<&Cell>` everywhere; `None` means open boundaries.
- **C-2** `min_image`: component-wise, each result component lies in [−L/2, L/2).
- **C-3** `wrap`: each result component lies in [0, L).
- **C-4** Pair potentials with a cutoff require `cutoff + skin ≤ L_min/2` when a cell is present, otherwise error (`CoreError::InvalidValue` from `NeighborList::build`, `PotentialError::InvalidParameter` from `evaluate`; Q7).
- **C-5** Ring-polymer positions are never wrapped by integrators (each ring stays continuous). Potentials apply minimum image internally. Ring-polymer estimators (E-1 … E-5) use plain differences; the RDF (E-7) uses the minimum image (Q9).

## 5. Ring polymer

- **RP-1** A ring polymer has P ≥ 1 beads with cyclic indexing (bead P ≡ bead 0).
- **RP-2** ω_P = P/β (ħ = 1).
- **RP-3** Sampled distribution is exp(−(β/P) H_P), with
  H_P = Σ_j Σ_i p²_{j,i}/(2 m_i) + Σ_j Σ_i ½ m_i ω_P² |q_{j,i} − q_{j+1,i}|² + Σ_j V(q_j).
  Consequently bead and normal-mode momenta have variance m_i·P/β per Cartesian component (not m_i/β). Physical kinetic energy comes from estimators E-1/E-2, never from Σ p²/2m.
- **RP-4** Normal-mode transform: a real orthogonal P×P matrix C (row j = bead, column k = mode) with q̃_k = Σ_j C_{jk} q_j (same for momenta). Column order is fixed, because thermostat draw order (T-1) depends on it:
  - k = 0: centroid, C_{j0} = 1/√P, ω_0 = 0.
  - for m = 1..⌊(P−1)/2⌋: k = 2m−1 has C_{jk} = √(2/P)·cos(2π m j/P) and k = 2m has C_{jk} = √(2/P)·sin(2π m j/P); both have ω_k = 2 ω_P sin(mπ/P).
  - if P is even: the last column k = P−1 has C_{jk} = (−1)^j/√P and ω_k = 2 ω_P.
  `frequencies()` is aligned with the columns. Note q̃_0 = √P·q̄ (q̄ the centroid). `to_normal`/`to_beads` panic if the input's bead count differs from P (a programming error, not a recoverable condition).
- **RP-5** Cᵀ·S·C = diag(ω_k²), where S = ω_P²·Σ_j (e_j − e_{j+1})(e_j − e_{j+1})ᵀ is the spring matrix, so that Σ_j ½ m ω_P² (q_j − q_{j+1})² = Σ_k ½ m ω_k² q̃_k².
- **RP-6** For P = 1 the ring polymer is a classical particle: ω_0 = 0, no springs, and every ring-polymer rule reduces to its classical counterpart.

## 6. Integrators

- **I-1** One step of length dt is B(dt/2) A(dt/2) O(dt) A(dt/2) B(dt/2).
  - B: p ← p + (dt/2)·F, with F from S-3.
  - A: exact free ring-polymer propagation (I-2) for time dt/2, applied in normal-mode space.
  - O: thermostat step (T-1), or identity if the thermostat is off.
- **I-2** Free propagation for time t, per normal mode k and Cartesian component, with mass m:
  - ω_k > 0: q̃ ← cos(ω_k t) q̃ + sin(ω_k t)/(m ω_k) p̃; p̃ ← −m ω_k sin(ω_k t) q̃ + cos(ω_k t) p̃ (using the old q̃ in both).
  - ω_k = 0: q̃ ← q̃ + t p̃/m; p̃ unchanged.
- **I-3** Exactly one `Potential::evaluate` call per step, after the second A, at the new positions. Its forces are reused by the next step's first B. One extra evaluation initialises the state.
- **I-4** `VelocityVerlet` (single replica, P = 1): standard velocity Verlet with dt in atomic time units.
- **I-5** dt must be positive and finite.

## 7. Thermostats

- **T-1** PILE-L O step, per normal mode k and Cartesian component (i, d):
  p̃_k ← c_k p̃_k + sqrt((1 − c_k²)·m_i·P/β)·ξ, with c_k = exp(−γ_k·dt) and ξ ~ N(0,1).
  Draw order: k = 0..P−1 (in `NormalModes` column order) outermost, then atom i, then dimension d; one normal per (k, i, d) per O step.
- **T-2** γ_0 = 1/τ_0 for configured τ_0 (`None` ⇒ γ_0 = 0). For every mode with ω_k > 0, γ_k = 2 ω_k.
- **T-3** `Thermostat::Off` applies the identity and draws no random numbers; with it, the dynamics conserve H_P with error O(dt²).

## 8. Estimators

All return totals over atoms and Cartesian components, in Eh. The energy estimators (E-1 … E-5, H_P) return plain `f64` and panic if their input shapes disagree (a programming error). Estimators with real input validation (E-7 RDF, E-8 pressure) return `Result<_, EstimatorError>`.

- **E-1** Primitive kinetic: K_prim = 3NP/(2β) − E_spring/P, with E_spring from E-5. (The 1/P is required because ω_P = P/β and the ring is sampled at β/P; it is the β-derivative of ln Z_P.)
- **E-2** Virial kinetic: K_vir = 3N/(2β) − (1/(2P)) Σ_j Σ_i (q_{j,i} − q̄_i)·F_{j,i}, with q̄ the centroid and F from S-3.
- **E-3** Potential: V̄ = (1/P) Σ_j V(q_j).
- **E-4** Total energy estimators: E_prim = K_prim + V̄; E_vir = K_vir + V̄. Both have expectation −∂ ln Z_P/∂β of the finite-P system.
- **E-5** Spring energy: Σ_j Σ_i ½ m_i ω_P² |q_{j,i} − q_{j+1,i}|².
- **E-6** Virial estimator uses −∇V forces only (S-3).
- **E-7** RDF: `n_bins` equal bins on [0, r_max) (pairs with r ≥ r_max are not counted; a distance below r_max that rounds past the last bin goes into the last bin), r_max ≤ L_min/2, cell required; a configuration with P = 0 is `ShapeMismatch` (Q12). With C_b the number of pair distances (i < j) over all beads falling in bin b and V_shell(b) = (4π/3)(r_{b+1}³ − r_b³): g_b = 2·C_b·V / (P·N·(N−1)·V_shell(b)). An ideal gas gives g = 1 in expectation. Requires N ≥ 2 (else `InvalidValue`); before any sample g is all zeros (Q10).
- **E-8** Classical pressure (P = 1): p = (2K + tr W)/(3V), with K = Σ_i |p_i|²/(2 m_i), virial W = Σ_{i<j} r_ij ⊗ f_ij, r_ij = r_i − r_j, f_ij the force on i from j.
- **E-9** Exact finite-P reference (derived from Z_P = (P/β)^P / Π_k Ω_k per Cartesian component, Ω_k² = ω_k² + ω², and verified numerically against −∂ ln Z_P/∂β and Monte Carlo): for N atoms in `Harmonic` with frequency ω, ⟨E_prim⟩ = ⟨E_vir⟩ = (3N/β) Σ_{k=0}^{P−1} ω²/(ω_k² + ω²), independent of mass. P = 1 gives 3N/β; P → ∞ gives 3N·(ω/2)·coth(βω/2). Statistical tests must use this finite-P value, not the P → ∞ limit.

## 9. Potentials

- **P-1** `Potential::evaluate(&mut self, positions: &Beads, cell: Option<&Cell>, subset: Option<&[usize]>) -> Result<Evaluation, PotentialError>`; `Evaluation { energies: Vec<f64> /*[P]*/, forces: Beads, virial: Option<[[f64;3];3]> }`. If `positions.n_atoms()` differs from the number of atoms the potential was built for, it returns `Err(PotentialError::ShapeMismatch { expected, actual })` (atom counts).
- **P-2** `Harmonic`: V = Σ_i ½ m_i ω² |q_i|². `DoubleWell`: separable over every Cartesian component, V(x) = a x⁴ − b x², a, b > 0. Both are external fields, not periodic: they ignore `cell`.
- **P-3** `LennardJones`: V(r) = 4ε[(σ/r)¹² − (σ/r)⁶] for r < r_c, 0 otherwise. If `shift`, V(r_c) is subtracted inside r < r_c. No tail corrections. Forces are the exact derivative (discontinuous at r_c when unshifted, matching LAMMPS `lj/cut`).
- **P-4** Native potentials return `Err(PotentialError::Unsupported)` when `subset` is `Some`.
- **P-5** `virial` is `Some(W)` for `LennardJones` when P = 1 (W as in E-8, in Eh) and `None` when P > 1; other potentials may return `None`.

## 10. Neighbor lists

- **N-1** `NeighborList` is a Verlet list of full pairs: `neighbors(i)` lists every j ≠ i with r_ij < `cutoff + skin` (strict; minimum image if a cell is given), sorted ascending.
- **N-2** A rebuild is required when any atom has moved more than `skin/2` since the last build (displacements in unwrapped coordinates), or when the atom count differs from the last build (Q10).
- **N-3** `LennardJones` keeps one neighbor list per bead, and rebuilds them when the cell passed to `evaluate` differs from the one they were built with, including `None` ↔ `Some` (Q11).

## 11. Roadmap

| Milestone | Content | Flow |
|---|---|---|
| M1 | Foundations: units, cell, beads, RNG, normal modes, `Potential`, `Harmonic` | A/B/C |
| M2 | Ring-polymer sampling: free propagator, BAOAB + PILE-L, estimators, `DoubleWell` | A/B/C |
| M3 | Classical core: LJ, neighbor lists, velocity Verlet, rayon, RDF, pressure, vs LAMMPS | A/B/C |
| M4 | Quantum liquid: config + run driver, checkpoint/restart, LJ neon PIMD vs i-PI, README plots, perf table | A/B/C |
| **R1** | **Release gate: owner decides whether to publish** | |
| M5 | i-PI-protocol socket backend; bridge validation (LJ served from Python); GFN2-xTB; engine vs i-PI on xTB | A/B/C |
| M6 | In-process torch backend (spike first), MACE-OFF23 adapter, gas-phase ML-RPMD study | engine: A/B/C; study: experiment log |
| M7 | Dynamics: RPMD, CMD, TRPMD with a pluggable thermostat; time-correlation validation | A/B/C |
| M8 | Solvated ligand: PBC with larger batches, ML/MM decision | decide at start |

Notes for later milestones:
- Para-H₂ with Silvera-Goldman is a stretch inside R1 or M4 follow-up, only after parameters are cross-verified from at least two independent sources.
- MACE-OFF23 weights are not shipped; the repo only interfaces with a user-downloaded file. Confirm the licence terms on the model card before citing it in the README.
- M7 validation ladder: harmonic Kubo-transformed autocorrelation (exact); P = 1 equals classical; 1D double well vs exact eigenstate correlation functions; literature para-H₂/neon diffusion and VACF.

## 12. Open questions

- In-process backend choice (`tch-rs` expected), decided by the M6 spike.
- Reference-code version pins in the pixi lockfile.
