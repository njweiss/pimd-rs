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

Number new entries from Q7. Once an entry is resolved and its decision is written into SPEC, WORKFLOW or a
brief, it moves to the index below; the full text stays in git history.

## Q7 (open) — Error variant for C-4 inside `LennardJones::evaluate`
Milestone: 3   Raised by: test author
Spec text: C-4 "Pair potentials with a cutoff require `cutoff + skin ≤ L_min/2` when a cell is
present, otherwise error."; MILESTONE_3 `NeighborList::build` returns `CoreError` (C-4 check).
Test or code: crates/pimd-potentials/tests/lennard_jones.rs `cell_too_small_is_rejected`
What seems wrong / ambiguous: `LennardJones` learns the cell only in `evaluate`, which returns
`PotentialError`. None of its variants fits: `InvalidParameter` is documented as "a constructor
parameter was out of range", `Unsupported` and `ShapeMismatch` mean other things (P-4, P-1), and
there is no variant wrapping `CoreError`. `NeighborList::build` itself is tested to return
`CoreError::InvalidValue` (the existing variant for an out-of-range value).
The test currently asserts only `Err(_)` that is neither `Unsupported` nor `ShapeMismatch`, so any
choice passes.
Proposed resolution: either (a) reuse `PotentialError::InvalidParameter { what: "cutoff + skin
(bohr) > L_min/2", value: cutoff + skin }` and widen its doc to "a parameter, or a parameter
combined with the cell, is out of range"; or (b) add `PotentialError::CellTooSmall { required:
f64, half_min_length: f64 }` (public API change). The test author prefers (a): no new variant,
and the message carries the numbers. Tighten the test to the chosen variant afterwards.
Owner decision:

## Q8 (open) — `VelocityVerlet::step` returns no virial
Milestone: 3   Raised by: test author
Spec text: MILESTONE_3 `VelocityVerlet::step(...) -> Result<f64, IntegratorError>` "Returns the
potential energy at the new positions"; P-5 / E-8 need the virial for the pressure.
Test or code: crates/pimd-estimators/tests/lj_thermo.rs `run`
What seems wrong / ambiguous: the step evaluates the potential (and LJ computes W at P = 1) but
drops the virial, so sampling the pressure needs one extra `evaluate` per sample at the same
positions. In the thermodynamic test (a sample every 10 steps) that costs 10 %; sampling every
step would double the cost. Not blocking M3: the tests do the extra evaluation. The M4 run driver
will meet the same problem.
Proposed resolution: for M4, have `step` return the `Evaluation` (energies, virial; forces still
written into `forces`), or add a `RingPolymerSim`-style classical driver that keeps the last
evaluation. Leave the M3 signature as briefed.
Owner decision:

## Q9 (open) — C-5 "Estimators use plain differences" vs E-7 minimum image
Milestone: 3   Raised by: test author
Spec text: C-5 "Estimators use plain differences."; E-7 RDF "cell required", MILESTONE_3 "periodic
distances via the test's own min-image".
Test or code: crates/pimd-estimators/tests/rdf.rs (hand-counted examples use the minimum image)
What seems wrong / ambiguous: read literally, C-5 forbids the minimum image in every estimator,
but the RDF needs it. The intent is clearly that ring-polymer estimators (centroid, springs,
E-1/E-2) use plain differences of unwrapped bead positions.
Proposed resolution: reword C-5 to "Ring-polymer estimators (E-1 … E-5) use plain differences;
the RDF (E-7) uses the minimum image." The tests already follow this reading.
Owner decision:

## Q10 (open) — Edge cases the M3 brief does not pin (not tested)
Milestone: 3   Raised by: test author
Spec text: N-1 "within `cutoff + skin`"; MILESTONE_3 `kinetic_classical` "P = 1"; E-7.
Test or code: none (deliberately untested)
What seems wrong / ambiguous: (1) N-1: whether a pair at exactly r = cutoff + skin is listed; the
tests keep every pair at least 1e-9 (relative) away from that distance. (2)
`NeighborList::needs_rebuild` with a different atom count than the last build (the LJ brief says
rebuild "when the atom count changed", so `true` or a panic are both plausible). (3)
`kinetic_classical` given P > 1 (panic, or sum over all beads?). (4) `Rdf::new` with
`n_atoms < 2` (E-7 divides by N(N − 1)). (5) `Rdf::g` before any sample (0/0).
Proposed resolution: (1) strict `<`, matching `r ≥ r_c` contributes nothing in P-3; (2) `true`;
(3) panic, like the other energy estimators on bad shapes (Q4); (4) `InvalidValue`; (5) all zeros.
If accepted, tests can be added in a later Phase A.
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
