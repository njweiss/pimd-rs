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

Number new entries from Q34. Once an entry is resolved and its decision is written into SPEC, WORKFLOW or a
brief, it moves to the index below; the full text stays in git history.

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
