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

## Q1 (resolved) — i-PI "LJ driver" is not in the conda-forge i-PI package
Milestone: 4   Raised by: Phase 0 bootstrap
Spec text: WORKFLOW Phase 0: "tasks `lmp` (LAMMPS), `ipi` (i-PI, with its LJ driver)"
Test or code: pixi.toml `[feature.ref.tasks]`
What seems wrong / ambiguous: conda-forge i-pi 3.3.1 ships only `i-pi-py_driver`; its modes have no periodic
Lennard-Jones (only `Spherical_LJ`, a confined model). The Fortran `i-pi-driver -m lj` is not packaged.
Proposed resolution: use LAMMPS as the i-PI client (`fix ipi` with `pair_style lj/cut`; present in the pinned
lammps 2025.07.22 build). This also reuses the same LJ implementation validated against in M3. Alternatively
build the Fortran driver from the i-PI source tag matching pixi.lock. The `ipi` task currently runs `i-pi` only.
Owner decision (2026-10-28): use LAMMPS `fix ipi` with `pair_style lj/cut` as the i-PI LJ client.
The `ipi` task stays `i-pi`; the client is launched with the `lmp` task.

## Q2 (resolved) — Shape mismatches have no error path in `Harmonic::evaluate` or `NormalModes` transforms
Milestone: 1   Raised by: test author
Spec text: P-1 "`Potential::evaluate(...) -> Result<Evaluation, PotentialError>`"; S-2 "Masses: one per atom";
MILESTONE_1 `to_normal(&self, beads: &Beads) -> Beads`, `to_beads(&self, modes: &Beads) -> Beads`.
Test or code: crates/pimd-potentials/src/error.rs (`PotentialError` has only `Unsupported`, `InvalidParameter`);
crates/pimd-core/src/normal_modes.rs.
What seems wrong / ambiguous: (a) `Harmonic::evaluate` with `positions.n_atoms() != masses.len()` has no
fitting `PotentialError` variant, so the implementer must either panic (library code, against the
"typed errors, no unwrap" convention) or misuse `InvalidParameter`. (b) `to_normal`/`to_beads` with
`n_beads() != P` cannot return an error with the briefed signature, so they can only panic.
No M1 test exercises either case.
Proposed resolution: add `PotentialError::ShapeMismatch { expected, actual }` (mirroring `CoreError`) and test it;
document that `to_normal`/`to_beads` panic on a bead-count mismatch (a programming error), or make them
return `Result<Beads, CoreError>`.
Owner decision (2026-10-28): accept as proposed. Add `PotentialError::ShapeMismatch { expected, actual }` (atom counts),
stated in SPEC P-1 and tested in M1; `to_normal`/`to_beads` keep their signatures and document a panic on a bead-count mismatch.

## Q3 (resolved) — What `Harmonic` does with `cell = Some(..)`
Milestone: 1   Raised by: test author
Spec text: C-1 "A cell is passed as `Option<&Cell>` everywhere"; P-2 "`Harmonic`: V = Σ_i ½ m_i ω² |q_i|²";
MILESTONE_1 out of scope: "PBC in potentials".
Test or code: crates/pimd-potentials/tests/harmonic.rs (all tests pass `cell = None`).
What seems wrong / ambiguous: the spec does not say whether `Harmonic` ignores a cell (an external trap is not
periodic), applies `wrap`/`min_image` to q, or returns `Unsupported`. M1 tests only use `None`.
Proposed resolution: state in P-2 that `Harmonic` (and `DoubleWell`) ignore `cell`, and add a test in a later
milestone that `Some(cell)` gives the same result as `None`.
Owner decision (2026-10-28): accept as proposed. SPEC P-2 now states that `Harmonic` and `DoubleWell` ignore `cell`;
the `Some(cell)` == `None` test is added in a later milestone.
Amended (2026-10-28, after M1 review): the test is added in M1 instead (`harmonic::cell_is_ignored`).

## Q4 (resolved) — `EstimatorError` is never returned by the briefed estimator API
Milestone: 2   Raised by: Phase 0 addendum
Spec text: MILESTONE_2 "Error enums `IntegratorError`, `EstimatorError` ... exist"; Public API: every
`pimd_estimators` function returns plain `f64` (e.g. `spring_energy(q, masses, beta) -> f64`).
Test or code: crates/pimd-estimators/src/error.rs (`EstimatorError::ShapeMismatch`).
What seems wrong / ambiguous: with `f64` returns there is no error path, so a wrong `masses.len()`,
mismatched `q`/`forces` shapes or `energies.len() != P` can only panic, and `EstimatorError` is dead.
Options: (a) return `Result<f64, EstimatorError>` from each estimator; (b) keep `f64`, document a panic on
shape mismatch (a programming error, as decided for the transforms in Q2) and drop `EstimatorError`.
Proposed resolution: (b). Estimators are called every sample step with inputs the simulation built itself,
so a mismatch is a bug, not a recoverable condition; it matches Q2 and keeps the briefed signatures.
Owner decision (2026-10-28): (b). Estimators keep `f64` returns and document a panic on shape mismatch;
`EstimatorError` is removed and the M2 brief updated.

## Q5 (resolved) — `validation/verify_reference_data.py` does not exist, but M2 adds the first reference data
Milestone: 2   Raised by: test author
Spec text: WORKFLOW "Long validation": `CHECK_LONG=1 ./scripts/check.sh` runs the long tests "together with
`validation/verify_reference_data.py`, which regenerates reference data into a temporary directory and compares it
with the committed JSON."
Test or code: crates/pimd-estimators/tests/data/gen_double_well.py, double_well.json; scripts/check.sh (skips the
step when the script is absent).
What seems wrong / ambiguous: M2 commits the first generated reference data, so the regeneration check now has
something to check, but no brief says who writes the script or how it compares (tolerances, which keys).
Without it, nothing detects a hand-edited JSON or a generator that no longer reproduces it.
Proposed resolution: the owner (or a small chore commit outside the locked tests) adds a generic script that runs
every `crates/*/tests/data/gen_*.py --out <tmpdir>` (the M2 generator already accepts `--out`) and compares each
JSON leaf numerically with the committed file (rtol 1e-9, atol 1e-12; strings compared exactly except the
`versions` block, which is reported, not compared). Runtime of the M2 generator: about 50 s.
Owner decision (2026-10-29): accept as proposed. `validation/verify_reference_data.py` added in a chore commit.

## Q6 (resolved) — Does `Thermostat::Off` consume random numbers?
Milestone: 2   Raised by: test author
Spec text: T-1 "one normal per (k, i, d) per O step"; T-3 "`Thermostat::Off` applies the identity"; I-1 "O: thermostat
step (T-1), or identity if the thermostat is off"; D-4 "Every stochastic step documents its draw order."
Test or code: crates/pimd-integrators/tests/thermostat.rs `off_is_identity_and_draws_nothing`.
What seems wrong / ambiguous: the spec does not say outright whether the identity O step still draws P·N·3
normals (keeping the stream aligned with a PILE-L run) or draws none. The test pins "draws none" (Off is not a
stochastic step, so it has no draw order), and also requires the state to be bitwise unchanged.
Proposed resolution: state in T-3 "`Thermostat::Off` draws no random numbers".
Owner decision (2026-10-29): accept as proposed. SPEC T-3 now reads "`Thermostat::Off` applies the identity and
draws no random numbers".
