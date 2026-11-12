---
description: Phase A. Write stubs and tests for a milestone (no implementation)
argument-hint: <milestone number>
---
You are the TEST AUTHOR for milestone $ARGUMENTS. You do not implement the milestone.

1. Read CLAUDE.md, docs/WORKFLOW.md, docs/SPEC.md and docs/MILESTONE_$ARGUMENTS.md. Work on a branch
   named `milestone-$ARGUMENTS` (create it if needed).
2. Create the stubs: the exact public signatures from the brief, bodies `unimplemented!()`.
   Types, constants, trivial accessors and data files the brief calls "already done" may be real.
3. Write the tests under `crates/*/tests/`. Rules:
   - Every test's doc comment starts with the SPEC rule ID(s) it covers (e.g. `/// RP-5`), so the
     reviewer can build the traceability table. Every rule in the brief's "In scope" table is
     covered by at least one test.
   - Expected values come from the sources the brief or spec names: SPEC rules, a hand derivation
     written out in a comment, an exact enumeration in the test, or a reference code (LAMMPS, i-PI,
     numpy) run through a generator script. Never from running the code under test, and never
     copied from your own reference implementation's output.
   - Checkers are independent of the implementation: build the normal-mode matrix from the RP-4
     formula in the test, write your own minimum-image, brute-force neighbor search, velocity
     Verlet and block averaging. Do not call the function under test to compute its expectation.
   - Reference data goes in `crates/<crate>/tests/data/` as JSON, each file produced by a committed
     generator script next to it (`gen_*.py`, run with `pixi run -e ref`), recording tool versions
     and seeds. Data and generators are part of the locked tests.
   - Statistical tests follow the tolerance policy in docs/WORKFLOW.md: fixed seeds, block-averaged
     error bars, 5 sigma plus a stated discretization-bias allowance validated at dt and dt/2, and a
     comment saying where each expected number comes from. Use the finite-P value (E-9), never the
     P -> infinity limit, for harmonic sampling checks.
   - Long runs are gated by the Cargo feature `long-validation`; never `#[ignore]`.
4. Validate the tests. Outside the repo (a scratch directory or `git worktree` you delete after),
   write a throwaway reference implementation and confirm every test passes. Then make mutants of
   the reference: at least four, each breaking a different rule from the brief, and every item of
   the physics-bug mutant list in docs/WORKFLOW.md that applies to this milestone. Confirm each is
   caught by at least one test. If a mutant survives, add or tighten a test. Do not commit the
   reference or the mutants.
5. In the repo, confirm: the stubs compile; every new test that exercises stubbed code FAILS (by
   panic, not by compile error); tests of already-done code pass; `cargo fmt --check` and
   `cargo clippy --workspace --all-targets -- -D warnings` are clean.
6. Commit with a message starting `tests(m$ARGUMENTS):`. The implementer session locates the
   test commit by that prefix, so it must be the commit that contains the final tests and data.
7. Report: test counts per file; the mutation table (mutant, test that caught it); every
   assumption you made; reference tool versions. Anything ambiguous, or any spec or brief problem
   you hit (including an awkward signature), goes in docs/QUESTIONS.md. Then stop. Do not start the
   implementation.
