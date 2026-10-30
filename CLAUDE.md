# CLAUDE.md: pimd-rs

Rust path-integral MD engine with a model-agnostic MLIP hook. This file is the short version; the authority is in `docs/`.

## Where things are

- `docs/SPEC.md`: authoritative rules with IDs (U-, D-, S-, C-, RP-, I-, T-, E-, P-, K-, N-). Code and tests cite these IDs.
- `docs/WORKFLOW.md`: the three-phase process (tests, implement, review), tolerance policy, mutant list, signature-revision path.
- `docs/MILESTONE_N.md`: the brief for each milestone: in-scope IDs, exact signatures, required tests.
- `docs/QUESTIONS.md`: anything ambiguous or wrong. Add an entry; do not work around it.
- `.claude/commands/`: `/write-tests N`, `/implement N`, `/review N`.

## Hard rules

1. Do only what the active phase and the brief allow. Do not start another milestone.
2. Never edit, add, delete or ignore tests, reference data or generator scripts under `crates/*/tests/` during implementation. `scripts/check.sh` enforces it.
3. Never loosen a tolerance to get green. If a tolerance is wrong, write a `QUESTIONS.md` entry with numbers.
4. Ask before adding a dependency, changing a public signature, or changing a unit or normal-mode convention.
5. If code and spec disagree, the spec wins; record the disagreement.

## Commands

```bash
./scripts/check.sh              # the pass/fail gate (fmt, clippy, tests at 1 and 4 threads, test lock)
CHECK_LONG=1 ./scripts/check.sh # adds long validation (feature `long-validation`) and reference-data checks
CHECK_LONG=1 CHECK_IPI=1 ./scripts/check.sh # also regenerates the i-PI reference (~34 min); before merging
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
pixi run -e ref <cmd>           # LAMMPS, i-PI, OpenMM, tblite, ASE, numpy
pixi run -e ml <cmd>            # PyTorch CUDA, MACE, ASE
pixi run test-validation        # same as CHECK_LONG=1 ./scripts/check.sh
```

## Conventions (summary; see the spec for the rules)

- Atomic units internally (U-1); constants and conversions only in `pimd_core::units` (U-2, U-3).
- f64 in the core; f32 only inside model adapters (D-1).
- All randomness from `pimd_core::Rng` with the documented draw order (D-2, D-4).
- Bitwise determinism across thread counts: per-atom partial results combined in fixed order, no scatter reductions (D-3).
- `Option<&Cell>` everywhere; one code path for gas phase and PBC (C-1).
- ω_P = P/β, sampling at β/P, momentum variance m·P/β (RP-2, RP-3). Estimators E-1 … E-5, with the 1/P in E-1.
- Library crates return typed errors; no `unwrap()` outside tests.
- Public items carry doc comments stating units and shapes (positions `[P, N, 3]`, bohr).
- Never integrate stiff ring springs with plain Verlet; use the exact free propagator (I-2).
- The engine never imports a model; models sit behind `Potential` (P-1).

## Status

| Milestone | State |
|---|---|
| Phase 0 bootstrap | done |
| M1 Foundations | merged |
| M2 Ring-polymer sampling | merged |
| M3 Classical core | merged |
| M4 Quantum liquid, then release gate R1 | merged; R1: hold (2026-10-30) |
| M5 Socket backend, GFN2-xTB | briefed (draft) |
| M6–M8 | not briefed yet (see SPEC §11) |

Update this table when a milestone merges.

## Do not

- Implement electrostatics, constraints or MLIP training.
- Add Python bindings before an analysis need appears.
- Hardcode paths, units or model-specific logic in the core crates.
- Commit large trajectories or model weights (`.gitignore` and document download steps).
