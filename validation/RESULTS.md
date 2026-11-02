# Validation results

Agreement of pimd-rs with every reference used in milestones M1–M6b, and the performance table.
M1–M4 numbers from branch `milestone-4`, M5 numbers from branch `milestone-5`, M6b numbers from branch `milestone-6` (pimd-rs 0.1.0, rustc 1.96.0). Every number below has one of these sources:

- **[run]**: printed by the long validation suite, run for this file with
  `RAYON_NUM_THREADS=4 cargo test --workspace --release --all-features -- --nocapture`
  (the release test step of `CHECK_LONG=1 ./scripts/check.sh`; 224 tests passed, 0 failed,
  12 min 48 s wall);
- **[plots]**: `validation/run_neon.py` + `validation/make_plots.py`, written to
  [`plots/neon_summary.json`](plots/neon_summary.json);
- **[json]**: a committed reference-data file under `crates/*/tests/data/`;
- **[comment]**: the doc comment of the named test (tolerances and their derivation).

Tolerance policy (`docs/WORKFLOW.md`): 5σ of the block-averaged estimate (blocks checked against
4× longer blocks, σ the larger) plus a stated discretisation-bias allowance; σ_c is the combined
σ of engine and reference.

## Reference versions

| Code | Version | Used for |
|---|---|---|
| LAMMPS | 22 Jul 2025 Update 6, conda `2025.07.22 cpu_py312_hdb376b3_mpi_openmpi_6` | M3 single point, trajectory, thermodynamics; LJ client of i-PI (M4); benchmark |
| i-PI | 3.3.1 (conda `pyh5ded981_0`) | M4 liquid-neon PIMD; M5 water/GFN2-xTB PIMD |
| tblite | 0.7.0 (conda `py312h0536e37_2`) | M5 GFN2-xTB force client (`ipi_client.py xtb`), for both codes |
| Python / numpy / scipy | 3.12.16 / 2.5.4 / 1.18.1 | reference generators (double well: numpy + scipy) |
| matplotlib | 3.11.2 | plots only |
| PyTorch / mace-torch / e3nn | 2.13.0 (CUDA 13.0) / 0.3.17 / 0.4.4 | M6b Python MACE reference (`mace_ref.json`, `mace_client.py`); libtorch for the `tch` 0.26.0 crate |
| MACE-OFF23 | small (`MACE-OFF23_small.model`, not committed) | M6b long validation |

Versions from the `versions` blocks of the committed JSON files [json] and `pixi.lock`.

## M1 Foundations

No external reference code. Units equal their CODATA 2018 values exactly; the normal-mode matrix,
frequencies and spring matrix are compared with an independent construction from the SPEC
formulas; neighbor lists equal a brute-force search exactly (`crates/pimd-core/tests/`, default
suite).

## M2 Ring-polymer sampling

| Comparison | Reference | pimd-rs [run] | Agreement | Tolerance [comment] |
|---|---|---|---|---|
| Harmonic, P = 1, ⟨E⟩ (E_prim = E_vir) | E-9 exact: 3 | 3.00373 ± 0.00388 | +0.96σ | 5σ + 0.005 = 0.0244 |
| Harmonic, P = 4, ⟨E_prim⟩ | E-9 exact: 5.6 | 5.59191 ± 0.00433 | −1.87σ | 0.0266 |
| Harmonic, P = 4, ⟨E_vir⟩ | 5.6 | 5.59966 ± 0.00501 | −0.07σ | 0.0300 |
| Harmonic, P = 16, ⟨E_prim⟩ | E-9 exact: 6.178180 | 6.16693 ± 0.00725 | −1.55σ | 0.0413 |
| Harmonic, P = 16, ⟨E_vir⟩ | 6.178180 | 6.18345 ± 0.00461 | +1.14σ | 0.0280 |
| Harmonic, P = 16, var(E_vir)/var(E_prim) | — | 0.6731 / 4.7281 = 0.142 | — | < 0.3 |
| Double well, P = 32, ⟨E_vir⟩ | finite-P exact −3.012905 [json] | −3.016219 ± 0.00413 | −3.31e-3 (0.80σ) | 5σ + 3e-3 |
| Double well, P = 32, vs exact quantum −3.007976 [json] | | | −8.24e-3 | 5σ + 3e-3 + 4.93e-3 |
| Double well, P = 64, ⟨E_vir⟩ | finite-P exact −3.009212 [json] | −3.012258 ± 0.00405 | −3.05e-3 (0.75σ) | 5σ + 3e-3 |
| Double well, P = 64, vs exact quantum | −3.007976 | | −4.28e-3 | 5σ + 3e-3 + 1.24e-3 |
| Free ring polymer, PILE-L, ⟨p̃_k²/m⟩, 8 modes | P/β = 2 | 1.99821 … 2.00633 | within 1.89σ (all modes) | 5σ |

Atomic units (Eh). Harmonic: N = 4, ω = 1, β = 4, dt = 0.05, 50 000 a.u. sampled
(`crates/pimd-estimators/tests/harmonic_sampling.rs`). Double well: one atom in 3D, a = 1/8, b = 1,
β = 2, exact values from sinc-DVR diagonalisation and the finite-P transfer matrix
(`double_well.json`, numpy 2.5.4, scipy 1.18.1; `double_well_sampling.rs`). The finite-P exact
values converge as 1/P² (ratio of discretisation errors P = 32 / P = 64: 3.986 [json]).

## M3 Classical core (LAMMPS)

| Comparison | pimd-rs vs LAMMPS [run] | Tolerance [comment] |
|---|---|---|
| Single point, 256-atom LJ liquid, shifted: total energy | −1181.0262416971264 vs −1181.0262416971216 (4.0e-15 rel) | 1e-10 rel |
| same, forces (768 components) | max 3.1e-14 rel | 1e-10 rel |
| Single point, unshifted: energy / forces | 1.8e-15 rel / 3.1e-14 rel | 1e-10 rel |
| NVE trajectory, 20 steps, dt = 0.005 τ (`fix nve`) | max \|Δx\| 8.9e-16, max \|Δv\| 2.8e-14 | 1e-8 abs |
| NVE energy conservation, fluctuation ratio dt = 0.01 / 0.005 | 0.2541 / 0.06256 = 4.06 (dt² gives 4) | [3, 5] |
| Thermo smoke (first 400 steps): ⟨V⟩ / ⟨p⟩ | 7.6e-15 / 7.3e-14 rel | 1e-9 rel |
| Thermo long (200 000 steps NVE): ⟨V⟩ | −1193.3956 ± 0.108 vs −1193.2938 ± 0.109 [json] (−0.66σ_c) | 5σ_c ≈ 0.77 |
| Thermo long: ⟨p⟩ | 1.832914 ± 0.00247 vs 1.832718 ± 0.00228 [json] (+0.06σ_c) | 5σ_c ≈ 0.017 |

LJ reduced units (ε = σ = m = 1), ρ* = 0.8, T* ≈ 1, r_c = 2.5σ. Tests:
`crates/pimd-potentials/tests/lj_lammps.rs`, `crates/pimd-integrators/tests/vv_lennard_jones.rs`,
`crates/pimd-estimators/tests/lj_thermo.rs`; LAMMPS data in `lj_snapshot.json` and
`lj_thermo.json`. The dt/2 check of the thermo bias (Δ⟨V⟩ = −0.12, Δ⟨p⟩ = −0.0002, unresolved) is
from the reference implementation [comment].

## M4 Liquid neon, PIMD

State point (from `crates/pimd-cli/tests/data/ipi_neon.json`, configs in
[`configs/`](configs/)): N = 108 Ne in a 14.1 Å cube (ρ* = 0.80), LJ ε = 35.6 K, σ = 2.749 Å,
r_c = 2.2σ shifted, T = 30 K, dt = 2 fs, BAOAB + PILE-L (τ_0 = 100 fs, γ_k = 2ω_k), 5 000
equilibration and 200 000 production steps (400 ps), recorded every 10 steps. Engine data: the
release `pimd-rs` binary, 80 restarted legs of 2 500 steps per P (`run_neon.py`); blocks of 250
samples (5 ps) checked against 1 000. i-PI 3.3.1 with LAMMPS `fix ipi` as LJ client, same
splitting and thermostat, seed 31415 [json]. Energies per atom in K. All numbers [plots]; the
same numbers were printed by `crates/pimd-cli/tests/long_neon.rs` in the [run], which passed.

### pimd-rs vs i-PI, P = 16

| Quantity | pimd-rs | i-PI + LAMMPS [json] | Difference |
|---|---|---|---|
| ⟨E_vir⟩ | −96.721 ± 0.208 | −97.189 ± 0.183 | +0.47 K (1.69σ_c) |
| ⟨E_prim⟩ | −96.699 ± 0.251 | −97.291 ± 0.256 | +0.59 K (1.65σ_c) |
| g(r), 70 bins to 7 Å | | | worst bin 2.03σ_c (r = 2.55 Å); no bin above 3σ_c |

Tolerance 5σ_c, bias allowance 0 (same integrator and dt on both sides; the dt vs dt/2 change
measured with the reference implementation is 0.08 ± 0.53 K in ⟨E_vir⟩ [comment]).
Plots: [energies](plots/neon_p16_energy_vs_ipi.png), [g(r) and residuals](plots/neon_p16_rdf_vs_ipi.png).

### Quantum (P = 16) vs classical (P = 1)

| Quantity | P = 1 | P = 16 | Shift |
|---|---|---|---|
| ⟨E_vir⟩ | −112.683 ± 0.178 | −96.721 ± 0.208 | +15.96 K (58.2σ_c) |
| ⟨E_prim⟩ | −112.683 ± 0.178 | −96.699 ± 0.251 | +15.98 K (51.9σ_c) |
| g(r) at the classical first peak, r = 2.95 Å | 2.7322 ± 0.0070 | 2.4631 ± 0.0041 | −0.269 (33.1σ_c) |

Both have the expected sign: quantum delocalisation raises the energy and lowers the first peak.
Plot: [g(r)](plots/neon_quantum_vs_classical_rdf.png).

### Convergence in P

| P | ⟨E_vir⟩ | σ(K_prim) | σ(K_vir) |
|---|---|---|---|
| 4 | −99.001 ± 0.215 | 0.096 | 0.026 |
| 8 | −97.693 ± 0.185 | 0.113 | 0.036 |
| 16 | −96.721 ± 0.208 | 0.129 | 0.032 |
| 32 | −96.744 ± 0.208 | 0.169 | 0.036 |

Weighted fit ⟨E_vir⟩ = E∞ + a/P²: E∞ = −96.80 ± 0.13 K, a = −36.6 ± 4.3 K; residuals +0.42,
−1.72, +1.08, +0.45 σ. E(32) − E(4) = +2.26 K (7.5σ_c). The virial kinetic estimator's block
error is 3–5× smaller than the primitive's at every P (K = E − V̄, which removes the shared slow
potential-energy fluctuations). Plot: [⟨E_vir⟩ vs 1/P²](plots/neon_convergence_in_p.png).

These M4 numbers equal, to the printed digits, those recorded in the header of `long_neon.rs`
for the reference implementation.

## M5 Socket backend, GFN2-xTB

Numbers printed by `RAYON_NUM_THREADS=4 cargo test --release -p pimd-socket --features
long-validation --test bridge -- --nocapture` and `cargo test --release -p pimd-cli --features
long-validation,socket-validation --test long_water_xtb -- --nocapture` on branch `milestone-5`
(commit 823ef01), after `CHECK_LONG=1 CHECK_IPI=1 ./scripts/check.sh` had passed on the same
commit (95 min wall, both i-PI references regenerated bit-identically) [run]. Tolerances and their
derivation: the doc comments of `crates/pimd-socket/tests/bridge.rs` and
`crates/pimd-cli/tests/long_water_xtb.rs` [comment].

### Bridge: Python `lj` client through the socket vs in-process `LennardJones`

Per evaluation, worst relative error over a 200-step PIMD trajectory driven by the in-process
forces (tolerance 1e-12):

| System | Clients | Energy | Forces | Virial |
|---|---|---|---|---|
| Liquid neon (M4 system, N = 108), P = 1 | 1 | 9.9e-16 | 9.4e-16 | 3.1e-15 |
| Liquid neon, P = 4 | 2 | 1.1e-15 | 1.2e-15 | — (P > 1, K-6) |
| 13-atom neon cluster, gas phase, P = 4 | 3 | 5.9e-16 | 4.7e-16 | — |

50-step PIMD run through the socket vs in-process, same state and seed (tolerance max |Δq| ≤
1e-10 bohr, |ΔV|/|V| ≤ 1e-12): neon P = 4, max |Δq| = 1.4e-14 bohr, |ΔV|/|V| = 0; cluster
P = 4, 4.4e-16 bohr, 4.1e-16. The two codes differ by summation order only; over 3 000 steps
the deviation grew roughly linearly to 2.9e-13 bohr (neon), far inside the chaos horizon
[comment].

### Water on GFN2-xTB: pimd-rs vs i-PI, P = 16

One gas-phase water molecule, 300 K, P = 16, dt = 0.5 fs, BAOAB + PILE-L (τ_0 = 20 fs,
γ_k = 2ω_k), 4 000 equilibration and 400 000 production steps (200 ps) per code, four
single-threaded stateless `xtb` clients on both sides, blocks of 4 000 steps (2 ps) checked
against 4× longer [json]. Per molecule; geometry bead-averaged.

| Quantity | pimd-rs | i-PI [json] | Difference |
|---|---|---|---|
| ⟨E_vir⟩ (Eh) | −5.049908 ± 0.000023 | −5.049943 ± 0.000022 | +1.10σ_c |
| ⟨E_prim⟩ (Eh) | −5.049985 ± 0.000047 | −5.050054 ± 0.000048 | +1.03σ_c |
| ⟨V̄⟩ (Eh) | −5.061468 ± 0.000017 | −5.061498 ± 0.000015 | +1.32σ_c |
| ⟨r_OH⟩ (bohr) | 1.849247 ± 0.000422 | 1.849065 ± 0.000432 | +0.30σ_c |
| ⟨θ_HOH⟩ (°) | 107.0878 ± 0.0149 | 107.0927 ± 0.0154 | −0.23σ_c |
| r_OH histogram, 25 bins 1.4–2.4 bohr | | | worst bin 2.19σ_c |

Tolerance 5σ_c, bias allowance 0 (same splitting and dt on both sides). The dt vs dt/2 check
with the reference implementation found no resolved bias (E_vir +6.2e-5 ± 3.3e-5 Eh, 1.9σ)
[comment]. Run times: pimd-rs 914 s for this run; the i-PI reference 2 195 s in the gate. These
numbers equal, to the printed digits, those recorded in the header of `long_water_xtb.rs` for
the reference implementation.

## M6 Torch backend, MACE (M6b)

Numbers printed by `PIMD_MACE_MODEL=ff_models/mace_off23/MACE-OFF23_small.model
RAYON_NUM_THREADS=4 pixi run -e ml env LIBTORCH_USE_PYTORCH=1 cargo test -p pimd-torch --release
--features torch,long-validation -- --nocapture` on branch `milestone-6` (commit eb2f32d), after
`CHECK_LONG=1 CHECK_IPI=1 CHECK_TORCH=1 ./scripts/check.sh` had passed on the same commit (99 min
wall; all 7 reference generators, including both MACE ones, reproduced the committed files)
[run]. Tolerances and their derivation: the doc comments of `crates/pimd-torch/tests/*.rs`
[comment]. The MACE-OFF23 weights are not in the repository (`ff_models/` is ignored); the test
reads them from `PIMD_MACE_MODEL` and exports them to a temporary file.

### MACE-OFF23 small: in-process `MacePotential` vs Python MACE through the socket

Gas-phase water at the M5 start geometry, P = 8, 300 K, dt = 0.5 fs, BAOAB + PILE-L. In
process: `MacePotential`, CPU, float64, 1 libtorch thread. Through the socket: four
`mace_client.py` clients (`MACECalculator` on the same model file, float64, CPU).

| Check | Measured | Tolerance |
|---|---|---|
| Per evaluation, 100 steps: worst \|ΔE\|/\|E\| per bead | 1.86e-16 | 1e-12 |
| Per evaluation, 100 steps: worst max\|ΔF\|/max\|F\| per bead | 6.42e-15 | 1e-10 |
| 50-step trajectory, same state and seed: max \|Δq\| | 1.78e-15 bohr | 1e-10 bohr |
| 50-step trajectory: \|ΔV\|/\|V\| | 0 | 1e-12 |

### Tiny model vs Python MACE [json]

The committed tiny MACE model (2 896 parameters, r_max 5 Å) against `mace_ref.json`
(`MACECalculator`, mace's own neighbour list, float64 CPU). Water, ethanol and a periodic
4-water box (positions not wrapped into the cell, 224 of 416 edges cross the boundary); 8
displaced beads each, evaluated as one batch. Worst over the beads.

| System | f64 CPU \|ΔE\|/\|E\| | f64 CPU max\|ΔF\|/max\|F\| | f64 CUDA max\|ΔF\|/max\|F\| | f32 CPU \|ΔE\| (eV) | f32 CPU max\|ΔF\|/max\|F\| |
|---|---|---|---|---|---|
| Water | 0 | 1.19e-15 | 1.73e-15 | 2.96e-4 | 1.34e-6 |
| Ethanol | 0 | 3.93e-16 | 1.41e-15 | 8.09e-4 | 8.28e-7 |
| 4-water box | 0 | 8.64e-16 | 6.47e-15 | 5.74e-4 | 4.52e-6 |

Tolerances: float64 (CPU and CUDA) \|ΔE\| ≤ 1e-12·\|E\| and max\|ΔF\| ≤ 1e-10·max\|F\|; float32
against the float64 reference \|ΔE\| ≤ 5e-3 eV and max\|ΔF\| ≤ 1e-4·max\|F\|. f64 CUDA
\|ΔE\|/\|E\| = 0 for all three; f32 CUDA \|ΔE\| 2.96e-4, 8.09e-4, 5.74e-4 eV and max\|ΔF\|/max\|F\|
1.75e-6, 6.64e-7, 4.43e-6. The CUDA numbers need a GPU and are skipped without one (here: RTX
4060, CUDA 13.0).

Batching all beads in one call vs one call per bead (tolerance 1e-12): \|ΔE\|/\|E\| = 0, worst
max\|ΔF\|/max\|F\| 1.19e-15. Periodic box with atoms wrapped into [0, L) or shifted by lattice
vectors (−2 … 2 box lengths per axis) vs the unwrapped positions (tolerance 1e-12):
\|ΔE\|/\|E\| = 0, max\|ΔF\|/max\|F\| 2.98e-14 (wrapped) and 4.99e-14 (shifted).

## Performance

`validation/benchmark.sh` (3 min 23 s wall), N = 108 neon at the state point above, best of 3
timings. atom-steps/s = N · P · steps / s. pimd-rs: full BAOAB + PILE-L step, no RDF;
LAMMPS: `fix nve` + `fix langevin`, OPENMP package (`-sf omp`, one MPI rank, OpenMP threads).

- CPU: AMD Ryzen 5 5600X3D 6-Core Processor, 12 logical CPUs (6 cores, 2 threads/core), WSL2
  (Linux 5.15.167.4-microsoft-standard-WSL2)
- Load average at start 4.49, at end 8.54: four unrelated single-thread processes were running
  during the benchmark, so the 8-thread column (and LAMMPS's multi-thread columns) are
  depressed by contention; absolute numbers are indicative only.
- pimd-rs 0.1.0, rustc 1.96.0; LAMMPS 22 Jul 2025 Update 6 (conda 2025.07.22
  cpu_py312_hdb376b3_mpi_openmpi_6)

| threads | pimd-rs P = 16 | speedup | pimd-rs P = 1 | speedup | LAMMPS (openmp) | speedup | pimd-rs P = 1 / LAMMPS |
|---|---|---|---|---|---|---|---|
| 1 | 7.09e+05 | 1.00 | 5.28e+05 | 1.00 | 5.04e+06 | 1.00 | 0.10 |
| 2 | 1.26e+06 | 1.78 | 7.13e+05 | 1.35 | 4.44e+06 | 0.88 | 0.16 |
| 4 | 1.74e+06 | 2.45 | 6.76e+05 | 1.28 | 3.86e+06 | 0.77 | 0.18 |
| 8 | 1.29e+06 | 1.82 | 4.48e+05 | 0.85 | 1.51e+06 | 0.30 | 0.30 |

4 vs 1 thread at N = 108, P = 16: **2.45×** (MILESTONE_4 target ≥ 2×, reported, not gated: met).
At P = 1 (108 atoms) the single-thread LJ step of pimd-rs is about 10× slower than LAMMPS's,
and neither code gains from threads at this size; PIMD at P = 16 parallelises over (bead, atom)
pairs and scales to 4 threads.

## Reproducing

```bash
python3 validation/run_neon.py                        # ~11 min on the machine above (5 chains, 2 threads each)
pixi run -e ref python validation/make_plots.py       # seconds; writes validation/plots/
./validation/benchmark.sh                             # ~3.5 min; prints the table above
RAYON_NUM_THREADS=4 cargo test --workspace --release --all-features -- --nocapture
```
