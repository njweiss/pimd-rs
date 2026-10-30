# Open questions

Numbered entries. Test authors and implementers add entries; only the owner resolves them. Format:

```
## Q<n> (open | resolved) — <title>
Milestone: <N>   Raised by: test author | implementer | reviewer
Spec text: <rule ID and quote>
Test or code: <file:line>
What seems wrong / ambiguous:
Proposed resolution:
Owner decision:
```

Number new entries from Q13. Once an entry is resolved and its decision is written into SPEC, WORKFLOW or a
brief, it moves to the index below; the full text stays in git history.

## Q13 (open) — No Eh → meV constant in `units` for the `*_mev` output columns
Milestone: 4   Raised by: test author
Spec text: U-2 lists k_B, amu, bohr, fs, cm⁻¹ and kcal/mol only; U-3 "No unit constants or conversion factors outside `units`". MILESTONE_4 "Outputs": `e_prim_mev`, `e_vir_mev` "via `units` only".
Test or code: crates/pimd-cli/tests/run_outputs.rs `unit_columns_consistent`; crates/pimd-cli/tests/common/mod.rs `EV_PER_HARTREE`
What seems wrong / ambiguous: the driver cannot write meV without an Eh → eV constant, and adding one to `pimd_core::units` is a public addition not listed in the brief (CLAUDE.md rule 4). The test checks `*_mev = *_eh × 27211.386245988` to 1e-12 relative, using the CODATA 2018 value written in the test, and does not name a constant.
Proposed resolution: add `pub const HARTREE_IN_EV: f64 = 27.211386245988;` (CODATA 2018) to `pimd_core::units` and to the U-2 list.
Owner decision:

## Q14 (open) — `Rdf` has no way to save and restore its accumulator
Milestone: 4   Raised by: test author
Spec text: MILESTONE_4 "Restart determinism": checkpoints contain "accumulators", including "the RDF accumulator"; restart must be bitwise.
Test or code: crates/pimd-cli/tests/restart.rs (`check_restart` compares `RunSummary::rdf` and rdf.csv bitwise); crates/pimd-estimators/src/rdf.rs
What seems wrong / ambiguous: `pimd_estimators::Rdf` keeps `counts` and `n_samples` private and has no constructor from saved state, so the driver can only checkpoint the RDF by duplicating the histogram in `pimd-cli` (the throwaway reference did this) or by a new public API on `Rdf`. Either works for the tests; the second avoids a second copy of E-7.
Proposed resolution: add `Rdf::counts(&self) -> &[f64]` and `Rdf::from_parts(n_bins, r_max, cell, n_atoms, counts: Vec<f64>, n_samples: usize) -> Result<Rdf, EstimatorError>` (validating like `new`), or state that pimd-cli keeps its own E-7 histogram.
Owner decision:

## Q15 (open) — Details the M4 tests pin that the brief leaves open
Milestone: 4   Raised by: test author
Spec text: MILESTONE_4 "Public API" (`pub struct Config { /* see "Config format" */ }`), "Clarifications".
Test or code: crates/pimd-cli/src/config.rs (stub); crates/pimd-cli/tests/{config,restart,run_outputs,reference_loop}.rs
What seems wrong / ambiguous: the tests need these choices; please confirm or change them (the tests follow the answer):
1. `Config` layout: public fields `system: SystemConfig { n_atoms, mass (m_e), cell: Cell }`, `potential: LjParams` (atomic units), `sampling: SamplingConfig { temperature_kelvin, beta, n_beads, dt, tau0 (ħ/Eh), n_equil, n_steps, stride, seed, normal_modes: Transform }`, `output: OutputConfig { rdf: Option<RdfConfig { n_bins, r_max (bohr) }>, checkpoint_every }`; all derive `Debug, Clone, PartialEq`. `RunSummary` also derives `Debug, Clone, PartialEq`.
2. `InvalidValue.what` is `"section.key"` of the TOML key (e.g. `"sampling.temperature_kelvin"`); the C-4 violation names `"potential.cutoff_sigma"` or `"potential.skin_sigma"`; `value` is only checked to be negative for negative inputs (user or converted units both pass). Negative values of integer fields are serde type errors, hence `Parse`; only float fields are tested for `InvalidValue`.
3. One `Rng::from_seed(seed)` draws the initial momenta and then drives the thermostat (D-2: one source).
4. `checkpoint.bin` depends only on the simulation state, accumulators and config hash (no n_steps, no paths), so the uninterrupted and the restarted run write identical bytes at the same final step; the restart tests compare the final checkpoints byte for byte.
5. Corrupt checkpoints tested: empty, truncated by one byte, halved, first 8 bytes zeroed (magic assumed at offset 0, as in "header with magic + version"), unrelated text; all must give `CliError::Checkpoint` (not `Io`).
6. n_side = ⌈N^(1/3)⌉ is computed exactly (integer); a test with N = 27 requires n_side = 3 (glibc `cbrt(27.0)` = 3.0000000000000004).
7. `rdf_bins`/`rdf_rmax_angstrom` given alone is an error (variant not checked); `box_angstrom` with two values is an error.
8. `run` creates `out_dir` if it does not exist.
Proposed resolution: confirm 1–8 as written (add them to MILESTONE_4 "Clarifications").
Owner decision:

## Resolved (index)

Code and tests cite these numbers. Each decision now lives where the last column says.

| Q | Decision | Codified in |
|---|---|---|
| Q1 | i-PI's LJ client is LAMMPS `fix ipi` with `pair_style lj/cut` | WORKFLOW Phase 0; MILESTONE_4 "vs i-PI" |
| Q2 | Atom-count mismatch → `PotentialError::ShapeMismatch`; normal-mode transforms panic on bead-count mismatch | SPEC P-1, RP-4; MILESTONE_1 |
| Q3 | `Harmonic` and `DoubleWell` ignore the cell (tested in M1/M2) | SPEC P-2 |
| Q4 | Energy estimators (E-1 … E-5, H_P) return `f64` and panic on shape mismatch; `EstimatorError` is used only by M3's fallible estimators | SPEC §8; MILESTONE_2, MILESTONE_3 |
| Q5 | `validation/verify_reference_data.py` regenerates and compares all reference JSON | WORKFLOW "Long validation" |
| Q6 | `Thermostat::Off` draws no random numbers | SPEC T-3 |
| Q7 | C-4 violated in `LennardJones::evaluate` → `PotentialError::InvalidParameter` (doc widened) | SPEC C-4; MILESTONE_3 |
| Q8 | `VelocityVerlet::step` keeps its M3 signature; M4's run driver keeps the last `Evaluation` | MILESTONE_3, MILESTONE_4 |
| Q9 | Plain differences for ring-polymer estimators only; the RDF uses the minimum image | SPEC C-5 |
| Q10 | Strict `<` at cutoff + skin; `needs_rebuild` true on atom-count change; `kinetic_classical` panics for P ≠ 1; `Rdf::new` needs N ≥ 2; `g` before samples is zeros | SPEC N-1, N-2, E-7; MILESTONE_3 |
| Q11 | `LennardJones` rebuilds its lists when the cell changes (incl. `None` ↔ `Some`); found in M3 review | SPEC N-3; MILESTONE_3 |
| Q12 | RDF bins are [0, r_max), rounding past the last bin is clamped into it; P = 0 → `ShapeMismatch` | SPEC E-7; MILESTONE_3 |
