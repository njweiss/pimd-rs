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

(No open entries.)

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
