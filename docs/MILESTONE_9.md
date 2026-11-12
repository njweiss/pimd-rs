# Milestone 9: float64 reference energies for float32 MACE; trajectory export for viewers

Status: **brief approved (owner, 2026-11-12)**; decisions 1–3 confirmed. M9a tests next. Two independent engine phases, each A/B/C:

| Phase | Flow | Output |
| --- | --- | --- |
| M9a | A/B/C | MACE export v2: interaction energy from the model, reference energies summed in float64 by the adapter |
| M9b | A/B/C | `pimd-traj`: `trajectory.xyz` → extended XYZ (centroids or all beads, molecules wrapped) for OVITO, VMD, ASE |

Crates touched: `pimd-torch` (M9a; export script and adapter), `pimd-cli` (M9b; a second binary).
Spec: P-6 (M9a sentence). `pimd-core`, `pimd-potentials`, `pimd-integrators` and
`pimd-estimators` are unchanged. No new dependencies.

## Goal

Remove the constant float32 energy offset found in M8b (Q4: +0.093 eV per 660-atom box, the same
at every P), so absolute float32 energies are usable. Make trajectories viewable in standard
tools without a custom reader.

## Already done

M1–M8 merged. Relevant to M9:

- **Source of the offset (checked 2026-11-12, CPU, scratch script).** MACE-OFF's
  `ScaleShiftMACE.forward` computes `total_energy = e0 + inter_e`, with
  `e0 = scatter_sum(node_e0)` in the model dtype. Forces are the gradient of `inter_e` alone.
  For NMA + 216 waters, Σ E0 = −454 069.78 eV. A float32 ulp there is 0.031 eV.
  - Rounding the E0 table to float32 alone: −0.005 eV.
  - torch's float32 sum of the 660 E0-sized node values: +0.062 eV, against +0.093 eV observed. The offset is a few ulp of summation at E0 magnitude.
  - The interaction energy is orders of magnitude smaller, so its float32 sum is precise to well below 1e-3 eV.
- `trajectory.xyz` (MILESTONE_5 "Trajectory output"): per frame, `P·N`, then
  `step=<s> beads=<P> atoms=<N> units=angstrom`, then P·N `symbol x y z` lines, bead-major, Å,
  unwrapped (a ring polymer is never split across the box). After a restart, a new file holds the
  frames after the checkpoint; `studies/m8-nma-water/run.sh` keeps earlier ones in `part<k>/`.
- The existing export `crates/pimd-torch/tests/data/tiny_mace_f64.pt` is locked test data (v1
  interface) and must keep loading with bitwise-unchanged results.

## Decisions (owner, 2026-11-12)

1. **The integrator is not in M9.** The M8b writeup's proposal 2 claimed that a Cayley free
   propagator (BCOCB) would remove the P growth of the BAOAB step-size bias. The harmonic desk
   check (`studies/m8-nma-water/cayley_bias.py`; same mode model as `ebias_harmonic.py`, 300 K,
   Δt = 0.5 fs) does not support that claim:

   | P | E_prim error (BAOAB / BCOCB) | E_vir error (BAOAB / BCOCB) |
   | ---: | --- | --- |
   | 8 | +0.7 / +0.6 | +9.5 / +7.1 |
   | 16 | −11.8 / −8.8 | +20.4 / +15.2 |
   | 32 | −62.4 / −45.8 | +27.2 / +20.1 |
   | 64 | −188 / −131 | +29.9 / +21.8 |

   Units are 10⁻⁶ Eh per atom.
   - Cayley cuts the bias by about 25 % and keeps the P growth of E_prim.
   - The E_vir bias, the estimator the studies use, already levels off with P under BAOAB.
   - Both schemes sample the free ring polymer exactly. Cayley's benefit is stability at large Δt, which these runs don't need.
   - The writeup's proposal 2 is corrected to match.
2. **M9a keeps v1 exports working.** An export without the v2 marker loads and evaluates exactly
   as on main, so the locked tests and existing exports stay valid; v2 is the export script's new
   default.
3. **M9b is a separate binary** (`pimd-traj`, in `pimd-cli`), so the `pimd-rs` command line is
   unchanged, rather than a Python script. A P = 32 trajectory of this size is several GB, and a
   Rust stream handles it in one pass with tests under the A/B/C flow.

## M9a: float64 reference energies

### In scope

| ID | What |
| --- | --- |
| P-6 (sentence) | v2 export: energy = interaction energy (model dtype, cast to f64) + Σ_i E0(Z_i) in f64 |
| export | `export_mace.py` writes v2 by default |

### Export interface v2 (`crates/pimd-torch/scripts/export_mace.py`)

- `forward` returns `out["interaction_energy"]` in place of `out["energy"]`. Forces and virials
  are unchanged; MACE takes them from the interaction energy already.
- `atomic_energies() -> Tensor [S] float64`: E0 per species in eV, in the order of
  `atomic_numbers()`. It is stored as float64 in both the float64 and float32 exports.
- `interface_version() -> int` returns 2. Without this method the export is v1.
- The script refuses a model without `atomic_energies_fn`, or whose outputs lack
  `interaction_energy`, with a clear message.
- `--interface 1` keeps the old output.

### Adapter (`MacePotential`)

- At `load`, before any cast to float32, read `interface_version()`.
  - If it is 2, read `atomic_energies()` into a `Vec<f64>`.
  - Its length must equal `atomic_numbers()`'s and every value must be finite. Otherwise `Backend`, and nothing is changed (the M6 order: validate first, settings last).
- Bead energy: E_j = f64(E_int,j)·(eV→Eh) + Σ_i E0(Z_i)·(eV→Eh). The sum runs in f64 in atom order, so the result is the same for every bead and chunk (D-3). The conversion goes through `units` only (U-3).
- Forces, errors and chunking are as P-6. No public signature changes; the config hash already covers the model file's bytes.

### SPEC change (applied with this brief's approval)

**P-6**, after "Energies are per bead (S-4), forces −∇V from the model's autograd (S-3).":

> With a v2 export (`interface_version()` = 2) the model returns the interaction energy and the
> adapter adds the reference energies Σ_i E0(Z_i) from `atomic_energies()` in f64; a v1 export's
> energy is used as returned.

### Required tests

Deterministic (tiny model, CPU, `CHECK_TORCH=1`; v2 exports of the tiny model from a committed
generator next to the existing one):

1. **v2 = v1 in float64.** For each of: gas phase, the 4-water box, P = 3, and c = 2:
   - energies agree to 1e-12 relative;
   - forces are bitwise equal.
2. **float32 offset removed.** The test uses a tiny model generated with E0 values large enough
   that the v1 float32 energy misses the float64 energy by ≥ 10 float32 ulp of |E|.
   - The v2 float32 energy is within a tolerance set by the interaction energy's magnitude. The test author derives that tolerance and states it in the test. The v1 float32 energy is outside it.
   - Forces are bitwise equal to v1's float32 forces.
3. **v1 unchanged.** `tiny_mace_f64.pt` gives the same bits as on main (values recorded with
   main's code).
4. **Errors.**
   - An `atomic_energies()` length mismatch, or a NaN in it: `Backend` at load, with libtorch settings unchanged.
   - `interface_version()` = 3: `Backend`.
5. **Determinism.** Threads 1 and 4 bitwise; every bead's E0 sum identical.

Long (`CHECK_LONG=1 CHECK_TORCH=1`, `PIMD_MACE_MODEL`, CUDA rows skip without CUDA):

6. **MACE-OFF23 small, NMA + 216 waters, v2** against the M8a test 7 Python float64 reference:
   - float64 CPU: |ΔE| ≤ 1e-12·|E|.
   - float32 CUDA (chunks of 2): |ΔE| ≤ 5e-3 eV per bead (the M6 tolerance; v1 is at 0.106 eV, Q34) and forces within the M6 1e-4·max |F|.

   The test author records the measured |ΔE|.

### Mutant ideas

- E0 added twice (the model's total returned with v2);
- E0 summed in f32;
- E0 indexed by the atomic number instead of the species index;
- E0 left in eV;
- v2 detection inverted;
- `atomic_energies()` read after the cast (f32-rounded);
- validation after the settings change.

### Clarifications (owner, 2026-11-12; Q38–Q39, test commit 8a6710d)

- **Test 2 (Q38):**
  - Test 2 uses a mixed-sign E0 table: H +1000000.03, C −2000000.05, N −1500000.3 and O −2002069.23 eV. Every term is about 1e6 eV and the totals are about −2,000 to −8,000 eV. The v1 float32 offset is then ≥ 327 ulp. With same-sign E0 the tiny systems stay below 1 ulp.
  - v2 float32 tolerance: 2^-24·2^8·max(|E_int,j|, N·1 eV) per bead.
  - The float64 row uses 1e-8 eV absolute, because the partial sums of 6e6 eV round at ≈ 5e-10 eV.
- **Test details (Q39):**
  1. "Threads 1 and 4 bitwise" means bitwise within each libtorch thread count (D-3, as M6/M8a). The constant-energy stub must also agree across thread counts, and for every bead and chunk size.
  2. Test 3's bits are main's (aa3fe6d) on this workstation; re-record them with main's code on another CPU or libtorch build.
  3. A v2 export without `atomic_energies()`, or with ±∞ in it, is `Backend` at load.
  4. The stub's float32 energies are bitwise equal to its float64 energies. Its energy equals (E_int + Σ E0)/27.211386245988 to 1e-14 relative.
  5. A model without `atomic_energies_fn` is refused by `export_mace.py`:
     - the script exits non-zero and writes no file;
     - stderr names `atomic_energies_fn`, with no traceback.

     `--interface 1` output is bitwise equal through the adapter to `tiny_mace_f64.pt`. The "lacks interaction_energy" refusal is untested.
  6. Test 6 also checks forces: 1e-10·max|F| in float64. It asserts that the default export has `interface_version()` = 2.
- **Review (owner, 2026-11-12):**
  - Any `interface_version()` other than 2, including 1, is `Backend`. v1 means the method is absent.
  - An export with `atomic_energies()` but no `interface_version()` is `Backend`, not v1.
  - A non-finite E0 sum is `Backend`.
  - `export_mace.py` refuses a multi-head model (an E0 table with more than one row), naming the head count.

### Out of scope (M9a)

- Re-running the M8b study.
- Changing float32 force precision.
- Models other than `ScaleShiftMACE`/`MACE` exports.

After merge, the float64 and float32 exports under `ff_models/exports/` are regenerated as v2. They are not committed.

## M9b: `pimd-traj`, trajectory export for viewers

### Command

```
pimd-traj <trajectory.xyz>... --out <file.extxyz> [--beads] [--box-angstrom L] [--every n]
          [--no-wrap] [--bond-angstrom 1.6]
```

- **Inputs:** one or more `trajectory.xyz` files in run order, for example `part0/… part1/… ./…`.
  - A frame whose step is ≥ the next file's first step is dropped. This is the restart-joining rule of `studies/m8-nma-water/health.py`.
  - P and N must agree across files. Otherwise the error names the file and the frame.
- **Output: extended XYZ**, one frame per kept input frame (every n-th; default n = 1). Line 2
  holds `step=<s>`, then `Lattice="L 0 0 0 L 0 0 0 L" pbc="T T T"` when the box is given, and
  `Properties=species:S:1:pos:R:3`, or `species:S:1:pos:R:3:bead:I:1` with `--beads`. Å, written
  with `{}`.
- **Default: centroids.** N atoms, each the mean of its P beads. The input is unwrapped, so no
  minimum image is needed.
- **`--beads`:** all P·N atoms, bead-major, with a `bead` column 0 … P−1. A ring polymer then shows
  as a cloud in OVITO, colour-coded by bead.
- **Wrapping** (box given, without `--no-wrap`):
  - Molecules are the connected components of the first kept frame's centroids, with edges between atoms closer than `--bond-angstrom` (default 1.6 Å) under minimum image (C-2).
  - Each frame, each molecule's beads and centroids are shifted together by one lattice vector, so the molecule's first atom's centroid lies in [0, L)³. Molecules and ring polymers are never split.
  - Without a box: no wrap and no `Lattice`/`pbc`.
- **Errors:** the run stops on the first error with a message naming the file and the line. Errors are:
  - a malformed header or line;
  - a count line ≠ P·N;
  - a step that does not increase within a file;
  - L ≤ 0 or non-finite;
  - n = 0.

  A partial last frame in a file other than the last one is dropped silently, since it lies past the next file's first step. In the last file it is an error.

### Required tests (deterministic, `pimd-cli/tests/`)

1. **Format.** A hand-written 3-frame, P = 2, N = 4 input (two 2-atom molecules) gives
   byte-for-byte the hand-written expected output, for each of:
   - centroids;
   - `--beads`;
   - `--every 2`;
   - `--no-wrap`;
   - no box.
2. **Wrapping.** A molecule straddling the boundary stays whole. A molecule drifted by +2L comes
   back with one shift applied to all its beads. A ring polymer whose centroid is in the box but
   some beads are outside is not split.
3. **Restart joining.** Two files with overlapping steps produce each step once, from the
   earlier file below the later file's first step. A mismatched P or N is an error.
4. **Readability.** The output of test 1 read back by ASE `ase.io.read(..., index=":")` (long
   test, `pixi run -e ref`) gives the same positions, cell and `bead` array.
5. **Errors.** Each error case above, with the message naming the file and the line.
6. **End-to-end.** A short engine run (LJ, P = 4, periodic) exported with `--beads`: the centroids
   of the output equal the `--centroids` output to 1e-12 Å (after the same wrap).

### Mutant ideas

- Centroid over atoms instead of beads (axis swapped);
- per-atom wrapping (molecules split);
- the wrap shift per bead instead of per molecule;
- the restart rule `>` in place of `≥`;
- `--every` counting from 1;
- bead column off by one;
- `Lattice` written as row sums;
- the bond cutoff without minimum image.

### Out of scope (M9b)

- PDB, DCD or other formats;
- non-orthorhombic cells;
- topology files;
- wrapping by molecule centre of mass;
- reading the box from a config;
- a Python wrapper.

## Out of scope

Integrator changes (decision 1); the MACE-OFF23 medium-model check (on hold); more M8b solute
sampling; electrostatics; NPT.

## Acceptance

- [x] Owner approves this brief, decisions 1–3, the P-6 sentence and the `pimd-traj` command line
- [ ] M9a: all tests pass; `check.sh` and `CHECK_TORCH=1` exit 0; long validation passes; review
      done with no blocking findings; `QUESTIONS.md` has no open entries
- [ ] M9b: the same
- [ ] `ff_models/exports/` regenerated as v2 (not committed); README updated for v2 exports and
      `pimd-traj`
