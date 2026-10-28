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

## Q1 (open) — i-PI "LJ driver" is not in the conda-forge i-PI package
Milestone: 4   Raised by: Phase 0 bootstrap
Spec text: WORKFLOW Phase 0: "tasks `lmp` (LAMMPS), `ipi` (i-PI, with its LJ driver)"
Test or code: pixi.toml `[feature.ref.tasks]`
What seems wrong / ambiguous: conda-forge i-pi 3.3.1 ships only `i-pi-py_driver`; its modes have no periodic
Lennard-Jones (only `Spherical_LJ`, a confined model). The Fortran `i-pi-driver -m lj` is not packaged.
Proposed resolution: use LAMMPS as the i-PI client (`fix ipi` with `pair_style lj/cut`; present in the pinned
lammps 2025.07.22 build). This also reuses the same LJ implementation validated against in M3. Alternatively
build the Fortran driver from the i-PI source tag matching pixi.lock. The `ipi` task currently runs `i-pi` only.
Owner decision:
