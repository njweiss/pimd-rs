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

Number new entries from Q41. Once an entry is resolved and its decision is written into SPEC, WORKFLOW or a
brief, it moves to the index below; the full text stays in git history.

## Q40 (open) — Details pinned by the M9b `pimd-traj` tests
Milestone: 9 (M9b)   Raised by: test author
Spec text: MILESTONE_9 M9b "Command", "Wrapping", "Errors", required tests 1–6; SPEC C-2, C-3.
Test or code: `crates/pimd-cli/tests/traj.rs`, `traj_common/mod.rs`, `long_traj_ase.rs`; stub `crates/pimd-cli/src/bin/pimd-traj.rs`
What seems wrong / ambiguous: the brief leaves these open; the tests pin the choices below.
1. **Output layout.** Line 2 is `step=<s>`, then ` Lattice="L 0 0 0 L 0 0 0 L" pbc="T T T"` (box given,
   also with `--no-wrap`), then ` Properties=…`, single spaces, no trailing space; every line ends in
   `\n`. L and coordinates use `{}` (L = 10 → `10`). Rows are `symbol x y z` or `symbol x y z bead`.
   A centroid row takes the symbol of the atom's bead-0 row.
2. **Wrap shift.** The shift is −floor(c/L)·L per component, an exact lattice vector, with c the
   centroid of the molecule's lowest-index atom. It is applied to that molecule's beads and centroids.
   The test 1–3 inputs are multiples of 0.25 Å with L = 10 Å, so centroids and shifts are exact in f64
   whatever the summation order or shift formula. Test 6 compares to 1e-12 Å.
3. **Molecules.** Molecules come from the first kept frame of the joined sequence and stay fixed.
   An edge needs a C-2 distance strictly below `--bond-angstrom` (1.75 Å apart with
   `--bond-angstrom 1.75` gives two molecules). Molecules are connected components, so chains count.
   Taken literally, a molecule that is split across the box in the input (bonded only through the
   minimum image) stays split: all of its atoms get the first atom's single shift. Test
   `bonds_use_minimum_image` pins this; it is the only case where the minimum image changes the
   result for an unwrapped input. Making such molecules whole would be a different rule (out of
   scope as written).
4. **`--every n`.** Frames are counted from 0 over the joined sequence, after restart drops. Index
   0, n, 2n, … are kept. Counting per file is wrong (test `restart_joining_every`).
5. **P and N.** The first frame fixes P and N. Any later frame with a different P or N is an error,
   in a later file or later in the same file.
6. **Error messages.** The message names the path as given (`Path::display`) and a 1-based line
   number, as `line <k>` or `<path>:<k>`. The status is non-zero and not a panic (≠ 101). For
   frame-level errors, the line may be the frame's count line or its header line: malformed header,
   count ≠ P·N, non-increasing step, P/N mismatch. A malformed atom line names that line exactly.
   A partial last frame in the last file may name its count line, its header line, the file's last
   line or the line after it. With two bad lines, only the first is named. Line numbers count
   within each file. `--box-angstrom` ≤ 0, `-0`, `inf`, `NaN` and `--every 0` must fail cleanly; the
   message is not checked.
7. **Malformed lines.** A malformed atom line is one without exactly four fields, with a
   non-numeric coordinate, or empty. A malformed header is one missing `units=`, missing `step=`, or
   with a non-numeric step or beads value. Whether other `units=` values, extra fields or non-finite
   coordinates are errors is not tested.
8. **Test 6, "`--centroids` output".** This is the default output (no flag). The command line in
   "Command" has no `--centroids` flag, and none was added. The engine run goes through
   `pimd_cli::run`, the library the `pimd-rs` binary calls (M4 "Binary" test). The test also checks
   both outputs against the test's own checker of molecules and shifts, at the default bond and at
   `--bond-angstrom 3.5`.
9. **Test 4.** The test runs ASE from `.pixi/envs/ref/bin/python`, or `pixi run -e ref python` when
   that file is absent, under the feature `long-validation`. ASE 3.29.0 and numpy 2.5.4 were used
   when it was written. With a box, ASE must read cell diag(L, L, L) and pbc all true. Without one,
   it must read a zero cell and pbc all false. Without `--beads` there is no `bead` array.
10. **Stub.** The stub is binary-only: `[[bin]] pimd-traj` in `crates/pimd-cli/Cargo.toml` and
    `src/bin/pimd-traj.rs`, clap arguments as in "Command", then `unimplemented!()`. No library API
    was added. Argument types: `--every` is `usize`, `--box-angstrom` is `Option<f64>`,
    `--bond-angstrom` is `f64` with default 1.6.
Proposed resolution: accept 1–10 as pinned. On 3, if the owner wants split input molecules made
whole, that needs a brief change and a new test commit.
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
| Q13 | `units::HARTREE_IN_EV` = 27.211386245988 (CODATA 2018) for the meV output columns | SPEC U-2; MILESTONE_4 |
| Q14 | `Rdf::counts()` and `Rdf::from_parts(..)` let the driver checkpoint the RDF without a second E-7 histogram | MILESTONE_4 |
| Q15 | Eight details pinned by the M4 tests (Config layout, error naming, one RNG, byte-identical checkpoints, corrupt-checkpoint cases, exact cube root, RDF keys together, `run` creates `out_dir`) | MILESTONE_4 "Clarifications" |
| Q16 | The checkpoint stores the RDF's n_bins and r_max; a restart whose RDF settings differ is `CliError::Checkpoint` (`[output]` stays outside the hash); found in M4 review | MILESTONE_4 "Restart determinism" |
| Q17 | Amended in M5 (was declined in M4): sizes that cannot be allocated are `InvalidValue` at parse time (P·N·3 or P² f64 values over `isize::MAX` bytes); sizes that merely exceed memory still fail by OOM | MILESTONE_5 "After review" |
| Q18 | Restart keeps the bitwise re-evaluation check in M4; revisit before M5 (non-bitwise potentials would make every restart fail; alternative: restore saved forces into `RingPolymerSim`) | MILESTONE_4 "Clarifications" |
| Q19 | `trajectory_every` stands alone: a frame at every production step with s % trajectory_every == 0, independent of `stride` | MILESTONE_5 "Clarifications" |
| Q20 | `potential.timeout_s` is at most 1e9 s; larger values are `InvalidValue` at parse time (an unrepresentable deadline panicked); found in M5 review | MILESTONE_5 "After review" |
| Q21 | The client's FORCEREADY extra string is read and discarded in bounded chunks, never allocated at the client's stated length; found in M5 review | MILESTONE_5 "After review" |
| Q22 | Each `system.symbols` entry is non-empty and has no whitespace, else `InvalidValue` (it would corrupt `trajectory.xyz`); found in M5 review | MILESTONE_5 "After review" |
| Q23 | `RingPolymerSim::from_parts` rejects non-finite given forces or energies as `InvalidValue`; found in M5 review | MILESTONE_5 "After review" |
| Q24 | `PotentialConfig::Mace { model, device, dtype, threads, deterministic }` (feature `torch` only; "cuda" = device 0); bad strings are `InvalidValue` or `Parse` naming the field; when the model bytes are hashed is the implementer's choice | MILESTONE_6 "Clarifications" |
| Q25 | `pimd-cli` has its own `build.rs` with libtorch's rpath and `--no-as-needed` (link args do not propagate to dependent crates) | MILESTONE_6 "Clarifications" |
| Q26 | `PIMD_MACE_MODEL` names the MACE `.model` file; the long test exports it to a temporary directory | MILESTONE_6 "Clarifications" |
| Q27 | A float32 export is refused for `Dtype::F64` (`Backend`); found in M6b review (N6) | SPEC P-6; MILESTONE_6 "After review" |
| Q28 | Adding `Thermostat::Trpmd` allows one `unreachable!` arm in the M2 test `baoab.rs`, nothing else | MILESTONE_7 decision 3; commit 28cf388 |
| Q29 | `[dynamics]` key errors: misplaced or unknown key is `Parse` (key in message) or `InvalidValue` naming it; non-multiple is `InvalidValue("dynamics.segment_steps")`; range rules name their own key | MILESTONE_7 "Clarifications" |
| Q30 | Required tests 10 and 11 drive the Y-1 cycle through the library; `segments.csv` counts segments from 0, `step` = production step at the segment end | MILESTONE_7 "Clarifications", tests 10–11 |
| Q31 | Adding `DynamicsMethod::PaCmd` allows one `unreachable!` arm in the M7a helper `dyn_common/mod.rs`, nothing else | MILESTONE_7 M7b "Clarifications"; commit d6c369b |
| Q32 | A `pa-cmd` run's initial momenta use the amended Y-2 draw (√s_k per mode); other runs unchanged | MILESTONE_7 M7b "Clarifications" |
| Q33 | Eight details pinned by the M7b tests (key errors, run.log by value, P = 1 bitwise, e6008c9 fingerprints, Ω with any thermostat, cm⁻¹ rounding, panic messages) | MILESTONE_7 M7b "Clarifications" |
| Q34 | Test 7 float32 energy tolerance is 0.25 eV per bead (M6 5e-3 eV is below float32 resolution at 660 atoms); force tolerance unchanged | MILESTONE_8 M8a "Clarifications" |
| Q35 | `gen_m8_boxes.py` is bitwise reproducible; `m8_boxes.json` supersedes the spike's xyz boxes | MILESTONE_8 M8a "Clarifications" |
| Q36 | Test 8 design (periodic client, protocol, P = 8 production 15 ps, σ from three block lengths, slow/model-gated generators) | MILESTONE_8 M8a "Clarifications" |
| Q37 | Details pinned by the M8a config tests (xyz parsing, error naming, hash as on 3e6a5bd, run.log echo, chunked run differs) | MILESTONE_8 M8a "Clarifications" |
| Q38 | M9a test 2 uses a mixed-sign ≈1e6 eV E0 table; v2 float32 tolerance 2^-24·2^8·max(|E_int|, N eV) per bead; float64 row 1e-8 eV absolute | MILESTONE_9 M9a "Clarifications" |
| Q39 | Details pinned by the M9a tests (determinism per thread count, machine-specific v1 bits, ±∞/missing E0 → `Backend`, stub bit equality, script refusal without traceback, test 6 forces) | MILESTONE_9 M9a "Clarifications" |
