# Workflow

How milestones are built. Three separate Claude Code sessions per engine milestone (A: tests, B: implement, C: review), with locks so the implementer cannot weaken the tests.

## Roles and files

| File | Purpose |
|---|---|
| `docs/SPEC.md` | Authoritative rules with IDs. Changed only by the owner. |
| `docs/MILESTONE_N.md` | Brief: in-scope rule IDs, exact public signatures, required tests, mutant ideas, out-of-scope. |
| `docs/QUESTIONS.md` | Anything ambiguous, contradictory or wrong. Numbered. Only the owner resolves entries. |
| `.claude/commands/write-tests.md`, `implement.md`, `review.md` | The three phase prompts, run as `/write-tests N`, `/implement N`, `/review N`. |
| `scripts/check.sh` | The single pass/fail gate. |

## Phase 0: bootstrap (once, interactive)

Done by the owner with Claude, before M1. Creates:
- Cargo workspace (`crates/pimd-core`, `pimd-potentials`, `pimd-integrators`, `pimd-estimators`; `pimd-cli` arrives in M4), `rust-toolchain.toml` (pinned stable), `Cargo.lock` committed, MIT `LICENSE`, `.gitignore`.
- `pixi.toml` with environments `ref` (LAMMPS, i-PI, OpenMM, tblite, ASE, numpy, scipy) and `ml` (PyTorch CUDA, MACE, ASE), and tasks `lmp` (LAMMPS), `ipi` (i-PI; its Lennard-Jones client is LAMMPS `fix ipi` with `pair_style lj/cut`, started through `lmp`, because the conda-forge i-PI ships no LJ driver) and `test-validation` (= `CHECK_LONG=1 scripts/check.sh`).
- `scripts/check.sh` (provided), `docs/*` (provided), CI stub running `scripts/check.sh`.
- Whatever each brief lists under "Already done" (types, error enums, constants).
- First commit on `main`. Milestone branches are `milestone-N`.

## Phase A: tests (`/write-tests N`)

1. Stubs with the exact public signatures from the brief; bodies `unimplemented!()`.
2. Tests under `crates/*/tests/`. Reference data under `crates/<crate>/tests/data/` (JSON), each file produced by a generator script committed next to it (`gen_*.py`, run through `pixi run -e ref`). The generator and data are part of the tests and are locked with them.
3. Validate the tests outside the repo: throwaway reference implementation passes everything; at least four mutants, each caught. Mutant suggestions are in each brief; physics-bug mutants are mandatory (see below).
4. Commit with prefix `tests(mN):`. Report. Stop.

### Tolerance policy

- MD samples are autocorrelated. Error bars come from block averaging (blocks long enough that the block means are uncorrelated, checked by comparing two block lengths), not naive standard errors.
- Tolerance is 5σ of the block-averaged estimate, plus a stated discretization-bias allowance (validate the reference at dt and dt/2 and keep bias below a third of the tolerance).
- Use fixed seeds. If a test fails on one seed by chance, the tolerance is wrong or the code is: never reseed to pass.
- Comment, for every expected number, where it comes from (SPEC rule, hand derivation, reference-code run with version, exact enumeration).

### Long validation

Long statistical runs are ordinary integration tests gated by the Cargo feature `long-validation` (declared in every crate that has them), never `#[ignore]`. `scripts/check.sh` skips them by default; `CHECK_LONG=1 ./scripts/check.sh` (or `pixi run test-validation`) runs them in release mode together with `validation/verify_reference_data.py`, which regenerates reference data into a temporary directory and compares it with the committed JSON. It runs every `crates/*/tests/data/gen_*.py --out <tmpdir>` (generators must accept `--out`); keys and list lengths must match, numbers within rtol 1e-9 / atol 1e-12, other values exactly; the top-level `versions` block is reported, not compared. Generators listed as slow in the script (currently `gen_ipi_neon.py`, about 34 min) run only when `CHECK_IPI=1` is also set; they are still locked like all reference data.

### Mandatory physics-bug mutants

The mutant set must include relevant cases from this list (the brief marks which apply):
- ω_P = P/β replaced by √P/β, or by 1/β
- `sin` missing, or mπ/P replaced by 2mπ/P, in the mode frequencies
- cos/sin columns swapped, or the centroid column normalised by 1/P
- primitive and virial estimators swapped, or the 1/P missing in E-1
- PILE friction γ_k = ω_k instead of 2ω_k; c_k = exp(−γ dt/2) instead of exp(−γ dt)
- noise variance m/β instead of mP/β
- O step before B instead of in the middle; dt vs dt/2 in the A steps
- spring or potential force applied twice, or forces not reused (two evaluations per step)
- minimum image with round-half-up instead of the spec interval; wrap into (−L/2, L/2]
- LJ shift applied beyond r_c, or force not truncated with the potential
- neighbor list skin check using skin instead of skin/2, or wrapped instead of unwrapped displacement

## Phase B: implement (`/implement N`)

The implementer locks the test commit and works until `./scripts/check.sh` exits 0. It cannot edit tests or reference data. If a test looks wrong, or a signature turns out to be unworkable, it stops and writes a `QUESTIONS.md` entry.

### Signature revisions

Exploratory numerical code sometimes has awkward APIs that only show up when implementing. Revision path:
1. Implementer stops and logs the problem in `QUESTIONS.md` (what is awkward, proposed signature).
2. Owner decides, edits the brief, and re-runs Phase A for the affected tests (a new `tests(mN):` commit; the older test commit is superseded).
3. Implementation resumes under the new lock.

The implementer never changes a signature unilaterally.

## Phase C: review (`/review N`)

Read-only. The traceability table keyed on SPEC rule IDs, plus determinism, test-integrity and robustness checks. Mandatory for any milestone touching integrators or thermostats (M2, M4). Optional but encouraged otherwise.

After review: run simplify on the diff, then `./scripts/check.sh` again. The locked tests must still pass unmodified.

## Merging

Owner merges `milestone-N` into `main` only when: `check.sh` is green, review has no blocking findings, `QUESTIONS.md` has no open entries for this milestone, and the milestone's acceptance items in the brief are ticked.

## Experiments (M6 study and other exploratory work)

The A/B/C flow is for engine milestones. Studies use a lighter flow:
- A short plan in `docs/experiments/<name>.md`: question, hypothesis, system, protocol, success/failure criteria written before running.
- Scripts and configs committed; results (tables, figures) regenerated by script.
- A log appended as work proceeds (what was run, with seed, versions, outcome), including failures and negative results.
- The engine is not modified during a study except through a normal milestone.
- A short writeup at the end; claims must cite the log entries that support them.

## Release gate R1

After M4 the owner decides whether to publish: README with validation plots and performance table, CI green, licence in place, MACE weights not included.
