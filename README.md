# pimd-rs

A path-integral molecular dynamics (PIMD) engine in Rust. It samples the quantum canonical
ensemble of distinguishable nuclei with ring polymers: BAOAB integration with the exact free
ring-polymer propagator, PILE-L thermostat, primitive and virial energy estimators, and the radial
distribution function. Potentials sit behind a model-agnostic `Potential` trait so that machine-learned
interatomic potentials can be added without changes to the engine; the current potentials are
Lennard-Jones (periodic, neighbor lists), harmonic, a quartic double well, external codes over the
i-PI socket protocol (e.g. GFN2-xTB), and MACE models in process through libtorch (feature
`torch`). Ring-polymer dynamics (RPMD, thermostatted RPMD and partially adiabatic centroid MD) gives Kubo-transformed centroid
velocity and position autocorrelations and the Green–Kubo diffusion coefficient. Atomic units internally, f64 throughout, and results that are bitwise identical across thread counts and
across checkpoint/restart.

## Status

| Milestone | Content | State |
|---|---|---|
| M1 Foundations | units, RNG, beads, cell, neighbor lists, normal modes | merged |
| M2 Ring-polymer sampling | BAOAB, PILE-L, estimators, harmonic and double-well validation | merged |
| M3 Classical core | Lennard-Jones, velocity Verlet, RDF, pressure, LAMMPS comparisons | merged |
| M4 Quantum liquid | `pimd-rs` binary, config, checkpoint/restart, liquid neon vs i-PI | merged |
| M5 Socket backend | i-PI socket client, water with GFN2-xTB | merged |
| M6 Torch backend | in-process MACE (TorchScript, CPU/CUDA), ML-PIMD study of water with MACE-OFF23 | merged |
| M7a Dynamics | RPMD and TRPMD segments, centroid correlation functions, Green–Kubo D; double well and liquid neon vs i-PI | merged |
| M7b PA-CMD | partially adiabatic centroid MD (dynamical masses, adiabatic frequency Ω); harmonic, double well and liquid neon vs i-PI | merged |
| M8a Periodic MACE chunks | bead-chunked MACE evaluation (`beads_per_graph`), xyz input; NMA + 216 waters vs Python MACE, 64 waters vs i-PI | merged |

Release gate R1 (after M4): on hold; not yet published. Not implemented: electrostatics,
constraints, non-orthorhombic cells, Python bindings.
The rules and conventions are in [`docs/SPEC.md`](docs/SPEC.md); the process is in
[`docs/WORKFLOW.md`](docs/WORKFLOW.md).

## Build and test

Requires the Rust toolchain pinned in `rust-toolchain.toml`. Reference codes (LAMMPS, i-PI,
numpy, matplotlib) come from the [pixi](https://pixi.sh) `ref` environment.

```bash
cargo build --release                 # binary: target/release/pimd-rs
./scripts/check.sh                    # pass/fail gate: fmt, clippy, tests at 1 and 4 threads
CHECK_LONG=1 ./scripts/check.sh       # adds the long statistical validation and reference-data checks
cargo test --workspace
pixi run -e ref <cmd>                 # LAMMPS, i-PI, OpenMM, tblite, ASE, numpy, matplotlib
```

## Running

```bash
pimd-rs <config.toml> --out <dir> [--restart <dir>/checkpoint.bin]
```

The config is TOML in user units (Å, K, fs, amu); unknown keys are rejected. Liquid neon at 30 K
with 16 beads (this is [`validation/configs/neon_p16.toml`](validation/configs/neon_p16.toml)):

```toml
[system]
n_atoms = 108
mass_amu = 20.1797
box_angstrom = 14.1            # cubic; or three values

[potential]
kind = "lj"
epsilon_kelvin = 35.6
sigma_angstrom = 2.749
cutoff_sigma = 2.2
skin_sigma = 0.3
shift = true

[sampling]
temperature_kelvin = 30.0
n_beads = 16
dt_fs = 2.0
tau0_fs = 100.0
n_equil = 5000
n_steps = 200000
stride = 10
seed = 2026
# normal_modes = "fft"         # optional; default "matrix"

[output]
rdf_bins = 70                  # rdf_bins and rdf_rmax_angstrom are optional, together
rdf_rmax_angstrom = 7.0
checkpoint_every = 2500
```

Outputs in `<dir>`: `observables.csv` (per-atom primitive and virial total energies, mean
potential and kinetic temperature at every `stride`-th production step, in Eh, meV and K),
`rdf.csv`, `checkpoint.bin` and `run.log`. Repeated runs give byte-identical files, and a
restarted run continues the same trajectory bitwise. Restarting into the directory of the earlier
run overwrites its `observables.csv` and `run.log`.

An optional `[dynamics]` section (`method = "rpmd"` or `"trpmd"`, `trpmd_lambda`,
`decorrelation_steps`, `segment_steps`, `corr_stride`, `n_lags`) turns production into cycles of
thermostatted sampling, a momentum redraw and a dynamics segment, and adds `correlations.csv`
(C_vv, C_xx, D(τ) with error bars) and `segments.csv` (H_P at each segment's start and end); see
[`docs/MILESTONE_7.md`](docs/MILESTONE_7.md).

`method = "pa-cmd"` (partially adiabatic centroid MD) requires `cmd_omega_cm` (the adiabatic
frequency Ω, cm⁻¹) and `cmd_lambda`. A suggested starting Ω, once the run is converged in P, is
P^{P/(P−1)}/β (ħ = 1); Ω should sit well above the physical frequencies of interest, Ω·dt well
below π, and results should be checked at two Ω. λ has no consensus value: Rossi et al. (2014)'s
1/2 is for TRPMD, while Castro, Trenins, Kapil and Rossi (arXiv:2508.16246) use 0.001–0.01. See
the M7b brief in [`docs/MILESTONE_7.md`](docs/MILESTONE_7.md).

### Viewing trajectories: `pimd-traj`

```bash
pimd-traj <trajectory.xyz>... --out <file.extxyz> [--beads] [--box-angstrom L] [--every n]
          [--no-wrap] [--bond-angstrom 1.6]
```

Converts `trajectory.xyz` (written with `[output] trajectory_every`) to extended XYZ that OVITO,
VMD and ASE read directly. It streams, so multi-GB trajectories are fine.

- **Inputs:** files from a restarted run go in run order (`part0/trajectory.xyz part1/… ./…`); a
  frame whose step is ≥ the next file's first step is dropped.
- **Output:** by default the N ring-polymer centroids per frame. `--beads` writes all P·N beads with
  a `bead` column (0 … P−1) to colour by. `--every n` keeps every n-th frame.
- **Box:** with `--box-angstrom L` (cubic) the output carries `Lattice`/`pbc`. Molecules (bonds
  shorter than `--bond-angstrom` in the first frame, minimum image) are made whole once, then
  shifted as a unit each frame so that their first atom's centroid lies in the box; neither
  molecules nor ring polymers are split.
  `--no-wrap` writes positions as they are.

See M9b in [`docs/MILESTONE_9.md`](docs/MILESTONE_9.md).

## Validation

[`validation/RESULTS.md`](validation/RESULTS.md) lists every reference comparison with its
agreement level, the versions of the reference codes, and the performance table. In short:

- Lennard-Jones energies, forces and virial match LAMMPS to round-off; NVE trajectories and
  thermodynamic averages match LAMMPS (M3).
- Ring-polymer sampling reproduces the exact finite-P harmonic energy and the exact double-well
  energy within the stated error bars (M2).
- Liquid-neon PIMD at 30 K (N = 108, P = 16) agrees with i-PI + LAMMPS within 1.7σ in ⟨E_vir⟩
  and ⟨E_prim⟩ and within 2.1σ in every g(r) bin; the quantum − classical energy shift
  (+16.0 K/atom) and the lowering of the first g(r) peak are resolved, and ⟨E_vir⟩ converges as
  1/P² over P = 4 … 32 (M4).
- At N = 108, P = 16, four threads are 2.45× faster than one on a 6-core desktop (loaded
  machine; see the performance section of RESULTS.md).

Plots (regenerated by `validation/run_neon.py` then `validation/make_plots.py`):
[energies vs i-PI](validation/plots/neon_p16_energy_vs_ipi.png),
[g(r) vs i-PI](validation/plots/neon_p16_rdf_vs_ipi.png),
[quantum vs classical g(r)](validation/plots/neon_quantum_vs_classical_rdf.png),
[convergence in P](validation/plots/neon_convergence_in_p.png).
The performance table comes from `validation/benchmark.sh`.

## Models

The M6 study ([`docs/experiments/m6-water-mace.md`](docs/experiments/m6-water-mace.md)) uses the
MACE-OFF23 small model through the torch backend. If you use it, cite:

D. P. Kovács, J. H. Moore, N. J. Browning, I. Batatia, J. T. Horton, Y. Pu, V. Kapil, W. C. Witt,
I.-B. Magdău, D. J. Cole, G. Csányi, "MACE-OFF: Short-Range Transferable Machine Learning Force
Fields for Organic Molecules", *J. Am. Chem. Soc.* (2025), doi:10.1021/jacs.4c07099
(arXiv:2312.15211, first released as MACE-OFF23).

The weights are not in this repository and are under their own licence. Download them from
[github.com/ACEsuit/mace-off](https://github.com/ACEsuit/mace-off) into `ff_models/mace_off23/`
(ignored by git) and export with
`pixi run -e ml python crates/pimd-torch/scripts/export_mace.py MODEL --out OUT.pt`.
The script writes export interface v2 by default: the model returns the interaction energy and
the engine adds the reference energies Σ E0 in f64, so float32 runs have no constant energy offset
(M8b found +0.093 eV per 660-atom box with the old export). `--interface 1` writes the old export,
which the engine still loads; exports made before M9 are v1 and are best regenerated.

By default the P beads are evaluated as one batched graph. On a GPU with limited memory, set
`potential.beads_per_graph = c` to evaluate them in consecutive chunks of c beads (SPEC P-6;
results depend on c to stated tolerances, and c < P enters the config hash). Safe sizes on an
8 GB card (M8 spike): about 190–200 k edges per graph for MACE-OFF23 small in float32, 95 k for
small in float64, 60 k for medium in float32 and 30 k for medium in float64; one bead of the
660-atom NMA + 216 water box has about 25 k edges with small (so c = 8 in float32). On WSL2 an
oversized graph does not fail but spills into shared system memory and runs several times slower;
there is no free-memory check.
Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the environment reduces
fragmentation (the engine does not set it). Positions and symbols can also be read from an xyz file
with `system.positions_xyz`.

### MACE through the socket client (e3nn or cuEquivariance)

`crates/pimd-torch/scripts/mace_client.py` (SPEC K-7) evaluates a MACE model file (the pickled
`.model`, not an export) as an i-PI socket client, one bead per request, with the same edges, Σ E0
handling and unit constants as the in-process adapter. Its point is NVIDIA's cuEquivariance
kernels, which cannot run inside the engine: about 4.5× faster per step at P = 32 with four clients
on the NMA + 216 water box ([`docs/experiments/m10-profiling-spike.md`](docs/experiments/m10-profiling-spike.md)).
Use the existing socket potential and start `n_clients` clients, each reading the atoms from the
same config:

```toml
[potential]
kind = "socket"
address = "unix:/tmp/pimd-mace.sock"
n_clients = 4
timeout_s = 600.0
```

```bash
for k in 1 2 3 4; do
  pixi run -e ml .venv-cueq/bin/python crates/pimd-torch/scripts/mace_client.py \
      ff_models/mace_off23/MACE-OFF23_small.model --config run.toml \
      --address unix:/tmp/pimd-mace.sock --backend cueq --dtype float32 --device cuda &
done
pimd-rs run.toml --out out/
```

Without cuEquivariance use `pixi run -e ml python` and `--backend e3nn` (the default; `--device`
defaults to `cpu`, `--dtype` to `float64`). Run clients from the directory you run `pimd-rs` from
when the config uses a relative `positions_xyz`. The clients exit when the run ends.

- **Installing cuEquivariance.** Its kernel packages are under NVIDIA's proprietary licence, so
  they are not part of the pixi environments and nothing of theirs is committed. Install them into
  a venv layered on the `ml` environment, `.venv-cueq/` (ignored by git), with the versions tested:

  ```bash
  pixi run -e ml python -m venv --system-site-packages .venv-cueq
  pixi run -e ml .venv-cueq/bin/python -m pip install \
      cuequivariance-torch==0.13.0 cuequivariance-ops-torch-cu13==0.13.0
  ```

  The pin is deliberate: mace-torch 0.3.17 declares older ranges for its own `cueq` extra, but
  0.13.0 is the version the M10 spike and the M10a tests ran with.
  Run the venv's python inside `pixi run -e ml` (Triton needs that environment's `ptxas`).
  `CHECK_CUEQ=1 ./scripts/check.sh` runs the cueq tests with it (`CUEQ_PYTHON` overrides the path).
- **Host memory limits K.** Each MACE client holds 1.2–2.3 GB of host RAM (MACE-OFF23 small on the
  660-atom box), so K ≤ host RAM / 2.3 GB: at most 4 clients on an 11 GB machine (K = 8 ran out of
  memory). They share one GPU.
- **`use_reduced_cg`.** `--backend cueq` converts the model with `mace.cli.convert_e3nn_cueq`, which
  trusts the model's `use_reduced_cg` flag. A model trained with mace ≥ 0.3.1x in an environment
  without cuEquivariance uses the full Clebsch–Gordan basis but stores `use_reduced_cg = True`; the
  conversion then fails with an einsum size error ("subscript a has size 7 for operand 1 which does
  not broadcast with previously seen size 9"). Set the attribute to `False` on the loaded model
  and save it again, or convert with mace's own tools in an environment that has cuEquivariance
  (QUESTIONS.md Q41). MACE-OFF23 converts as downloaded.
- **Reproducibility.** On CUDA the client is not bitwise reproducible with either backend (SPEC
  D-3): repeat runs agree to about 1e-8 Eh, and with the in-process adapter to the float32
  tolerances of the tests. The in-process backend stays bitwise reproducible.

## License

MIT; see [`LICENSE`](LICENSE).
