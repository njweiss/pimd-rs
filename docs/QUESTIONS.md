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

Number new entries from Q24. Once an entry is resolved and its decision is written into SPEC, WORKFLOW or a
brief, it moves to the index below; the full text stays in git history.

## Q24 (open) — M6b: Rust shape of `PotentialConfig` for `kind = "mace"`
Milestone: 6   Raised by: test author
Spec text: MILESTONE_6 "Config" gives the TOML keys and the hash coverage but no Rust signature.
Test or code: crates/pimd-cli/src/config.rs (`PotentialConfig::Mace` stub); crates/pimd-cli/tests/config_mace.rs, run_mace.rs
What seems wrong / ambiguous: the tests need to read the parsed values, and `Config` derives `Debug, Clone, PartialEq`. Unspecified: the variant's fields, whether it exists without the feature, when the model bytes are read for the hash (parse time or hash time), and whether bad `device`/`dtype`/`symbols` strings are `InvalidValue` or `Parse`.
Proposed resolution (stubbed and tested this way): `#[cfg(feature = "torch")] PotentialConfig::Mace { model: PathBuf, device: pimd_torch::Device, dtype: pimd_torch::Dtype, threads: usize, deterministic: bool }`, with `Device`/`Dtype` deriving `Debug, Clone, Copy, PartialEq, Eq`; "cuda" = `Device::Cuda(0)`. The tests destructure with `..`, so the implementer may add a field (e.g. the FNV-1a of the model bytes read at parse time) without a test change; they never parse a config whose model file is missing. String-valued errors are accepted as `InvalidValue` naming the field or `Parse` mentioning its key (as in M5); `threads = 0` must be `InvalidValue { what: "potential.threads" }`; without the feature any error whose message contains "torch".
Owner decision:

## Q25 (open) — M6b: `pimd-cli` needs its own build script for libtorch
Milestone: 6   Raised by: test author
Spec text: MILESTONE_6 "Build isolation": "`pimd-torch/build.rs` acts only when `CARGO_FEATURE_TORCH` is set: rpath `$CONDA_PREFIX/lib`, `-Wl,--no-as-needed,-ltorch_cuda` …".
Test or code: crates/pimd-cli/build.rs (added with the stubs); crates/pimd-torch/build.rs
What seems wrong / ambiguous: `cargo:rustc-link-arg` from `pimd-torch/build.rs` applies only to pimd-torch's own targets; it does not reach the `pimd-rs` binary or the `pimd-cli` test binaries (torch-sys's build.rs says the same about its own flags). Without the rpath those binaries cannot load `libtorch_cpu.so`, and without `--no-as-needed` they see no CUDA.
Proposed resolution: `crates/pimd-cli/build.rs` (std only, no-op without the feature `torch`) emits the same two link arguments; already added so the torch tests of pimd-cli can run. The build.rs files and `src/deterministic.cpp` are real (not stubs) because the tests cannot link otherwise; the implementer may change them.
Owner decision:

## Q26 (open) — M6b: what `PIMD_MACE_MODEL` points to in the long test
Milestone: 6   Raised by: test author
Spec text: MILESTONE_6 "Required tests: Long": "needs `PIMD_MACE_MODEL`, skips with a message otherwise … in-process vs the socket path with Python MACE clients".
Test or code: crates/pimd-torch/tests/long_mace_off23.rs
What seems wrong / ambiguous: the in-process side needs a TorchScript export, the Python clients the MACE model file; one variable is named.
Proposed resolution (tested this way): `PIMD_MACE_MODEL` is the MACE model file (e.g. `ff_models/mace_off23/MACE-OFF23_small.model`, as in the spike); the test exports it into a temporary directory with `crates/pimd-torch/scripts/export_mace.py` (about 10 s) through the `ml` interpreter, so both sides use the same weights and nothing derived is written into the repository.
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
