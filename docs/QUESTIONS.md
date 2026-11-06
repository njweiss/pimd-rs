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

## Q34 (open) — Test 7: the M6 float32 energy tolerance is below float32 resolution for 660 atoms
Milestone: 8a   Raised by: test author
Spec text: MILESTONE_8 test 7: "float32 CUDA (chunks of 2) vs that reference to the M6 float32
tolerances" (M6: |ΔE| ≤ 5e-3 eV, |ΔF| ≤ 1e-4·max |F|).
Test or code: crates/pimd-torch/tests/long_mace_off23_pbc.rs, `F32_ENERGY_TOL_EV`
What seems wrong / ambiguous: the NMA + 216 box has |E| ≈ 4.5635e5 eV per bead (the total includes
the atomic reference energies). One float32 ulp at that magnitude is 2^-5 = 0.031 eV (spike Q1), so
5e-3 eV cannot be met by any float32 evaluation. Measured with the reference implementation
(MACE-OFF23 small, CUDA, chunks of 2 and of 3): max |ΔE| = 0.106 eV over the 4 beads (3.4 ulp,
2.3e-7 relative); max |ΔF| = 1.75e-5·max |F| (the M6 force tolerance 1e-4 holds). The test uses
|ΔE| ≤ 0.25 eV (8 ulp, 5.5e-7 relative) and keeps the M6 force tolerance.
Proposed resolution: accept 0.25 eV per bead (or state the tolerance relative, e.g. 1e-6·|E|) for
float32 energies of large systems; decision 6 already treats the float32 energy as ≈ 3e-3 Eh
(0.08 eV) of rounding.
Owner decision:

## Q35 (open) — Box generator: OpenMM reproducibility, and the spike's boxes are not reproducible
Milestone: 8a   Raised by: test author
Spec text: MILESTONE_8 "Reference data": "If OpenMM's box is not bitwise reproducible across runs,
the generator check compares coordinates to a stated tolerance; the test author decides and
records it."
Test or code: crates/pimd-cli/tests/data/gen_m8_boxes.py, m8_boxes.json
What seems wrong / ambiguous: decision taken: the generator is made bitwise reproducible, so
validation/verify_reference_data.py compares it with its usual rtol 1e-9 / atol 1e-12 (no special
tolerance). Ported unchanged from spikes/m8/build_box.py, two concurrent runs gave NMA boxes that
differed by up to 19 Å after equilibration (the water box was identical): `Modeller.addHydrogens`
places the new hydrogens at positions drawn from Python's unseeded `random` before minimising
them, and `addSolvent` then depends on them. The generator seeds `random` (8000) before each box,
and runs OpenMM's CPU platform with Threads = 1 and DeterministicForces = true. Two concurrent runs
then gave byte-identical JSON (192 s each). Consequence: the committed boxes differ from
spikes/m8/data/nma_w216.xyz and water_w64.xyz (thread count and hydrogen placement), and those
spike files cannot be regenerated. M8b should start from `m8_boxes.json` (nma_w216).
Proposed resolution: confirm; M8b's plan cites m8_boxes.json rather than the spike's xyz files.
Owner decision:

## Q36 (open) — Test 8 (vs i-PI): design choices for the owner to confirm
Milestone: 8a   Raised by: test author
Spec text: MILESTONE_8 test 8: "the 64-water box, MACE-OFF23 small, 300 K, P = 1 and P = 8:
pimd-rs in-process (float32 CUDA, chunked) against i-PI driving one Python MACE client (float32
CUDA) through the socket, ⟨E_vir⟩ and g_OO within 5σ_combined, bias allowance 0 (same splitting and
dt). Run lengths set so that σ is useful".
Test or code: crates/pimd-cli/tests/long_water_mace_ipi.rs, data/gen_ipi_water_mace.py,
clients/mace_pbc_client.py; scripts/check.sh; validation/verify_reference_data.py
What seems wrong / ambiguous: choices the brief leaves open:
1. Client: the locked M6 client (pimd-torch/tests/clients/mace_client.py) is gas phase only, so a
   periodic sibling `pimd-cli/tests/clients/mace_pbc_client.py` is added (same M5 protocol code). It
   switches on torch's deterministic algorithms and disables the TorchScript profiling executor:
   without the latter, mace's scripted modules give different float32 results in the first two
   calls (up to 2e-6 eV/Å); with both, results are bitwise across calls and processes, so the i-PI
   reference regenerates bitwise (a shortened run gave byte-identical JSON twice). The client must
   be started with `pixi run -e ml python` (triton needs `ptxas` on PATH for the periodic virial
   path; the bare `.pixi/envs/ml/bin/python` fails).
2. i-PI's cell is given as a diagonal matrix (mode 'manual'): mode 'abc' sends off-diagonal
   elements of 1.4e-15 bohr.
3. Protocol: PILE-L τ_0 = 100 fs; dt 0.5 fs; P = 1: 2.5 ps + 20 ps; P = 8: 2.5 ps + 15 ps (not the
   spike's 20 ps: with one f32 client i-PI takes ≈ 0.3–0.37 s per P = 8 step alone, so 20 ps would
   be ≈ 4.6 h); stride 10, trajectory every 40 steps; g_OO 60 bins of 0.1 Å on [0, 6) Å. The
   committed run took 4.3 h wall (P = 8: 15 581 s, P = 1: 4 312 s, concurrent), of which about
   1.5 h overlapped other GPU jobs of the test author that slowed it 2–3×; alone it is ≈ 3.3 h.
   Resulting errors per molecule: σ(E_vir) = 3.3e-5 Eh (P = 1), 7.3e-5 Eh (P = 8), against a
   quantum shift ⟨E_vir⟩_8 − ⟨E_vir⟩_1 = 0.0138 Eh (≈ 190 σ_combined).
4. Statistics: σ = the largest block standard error over blocks of 0.5, 1 and 2 ps, with no
   assertion on their ratio (with 7–10 blocks at 2 ps the ratio is itself noisy, and a failed
   assertion after 4 h of i-PI would waste the run). Bias allowance 0 as the brief says; no dt/2
   run was made.
5. Wiring: the test lives in pimd-cli (it runs `pimd_cli::run` with `positions_xyz` and
   `beads_per_graph`); check.sh's CHECK_LONG + CHECK_TORCH step now also runs
   `cargo test -p pimd-cli --release --features torch,long-validation --test long_water_mace_ipi`
   (≈ 2 h on the GPU). verify_reference_data.py registers gen_ipi_water_mace.py as slow (CHECK_IPI)
   and skips the two generators that need the model (gen_ipi_water_mace.py,
   gen_mace_off23_nma.py) unless PIMD_MACE_MODEL is set. Both generators record the model's SHA-256
   and FNV-1a; the tests check the FNV-1a of their own PIMD_MACE_MODEL against it.
6. MACE-OFF23 small has r_max = 4.5 Å (export_mace.py), not 5 Å; the spike's "32 waters would
   violate r_max < L/2" does not hold (L_32 = 9.87 Å, L/2 = 4.93 Å). The brief's 64-water box is
   kept.
Proposed resolution: confirm, or say which choice to change (the i-PI reference would have to be
regenerated, ≈ 4 h).
Owner decision:

## Q37 (open) — Details pinned or left open by the M8a config tests
Milestone: 8a   Raised by: test author
Spec text: MILESTONE_8 "M8a: Config" (`positions_xyz`, `beads_per_graph`, run.log).
Test or code: crates/pimd-cli/tests/config_m8.rs, config_m8_mace.rs, run_m8.rs
What seems wrong / ambiguous: points the brief leaves open, and how the tests treat them:
1. xyz columns are separated by any run of spaces; a count line must be a non-negative integer
   alone on the line ("32 atoms", "32.0", "-32", empty are malformed); an empty atom line, a
   comma-separated line and a line without a symbol are malformed; every test file ends with one
   newline after the last atom line, and nothing after the last atom line is tested.
2. A directory given as `positions_xyz` is `CliError::Io` (like a missing file).
3. A symbol unknown to `atomic_number` in the xyz file with kind = "mace" may be named
   `system.symbols` (the existing rule) or `system.positions_xyz`.
4. `beads_per_graph` with a negative, fractional, string or boolean value is `Parse` naming the key
   or `InvalidValue("potential.beads_per_graph")`.
5. "Hash as on main" is pinned by the `config hash` printed by pimd-rs built at main 3e6a5bd for the
   tiny-model water config (P = 4: 0xb14892014363f1fd; P = 1: 0xfc9b4ad40c4fcee0).
6. run.log's chunk-size echo is checked loosely: the c = 3 and c = 2 logs differ outside the hash
   line, and the log mentions "beads_per_graph", "beads per graph" or "chunk". A config without the
   key may still print `beads_per_graph: None` in the config echo (run.log content apart from the
   hash line is not pinned by any older test).
7. Test 6 asserts that the c = 3 run's trajectory and observables differ from the unchunked run's
   (the 4-bead and 3 + 1-bead graphs round differently with this libtorch build; checked with the
   reference implementation; the checkpoints differ anyway through the config hash). The spec only says they agree to tolerances; the assertion is what catches a
   CLI that ignores `beads_per_graph`.
Proposed resolution: confirm, or say which point should be pinned differently.
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
