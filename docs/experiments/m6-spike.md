# M6a spike: in-process torch backend for MACE

Experiment record for MILESTONE_6 "M6a: spike" (WORKFLOW "Experiments"). Plan written before any
measurement; the log below is appended as work proceeds; the writeup and recommendation come last.
Code: `spikes/m6-torch/` (standalone Cargo project, not a workspace member, not maintained after
the spike). Nothing under `crates/` changes.

## Plan

### Question

Can pimd-rs evaluate a MACE model in-process through `tch` 0.26 and a TorchScript export, batched
over beads, on CPU and CUDA, with energies and forces matching Python MACE, and is that faster than
the M5 socket path with Python MACE clients? Which dtype, device, batching, determinism statement
and edge-building approach should M6b fix?

### Hypothesis

`tch` 0.26 links against the `ml` environment's libtorch 2.13.0 (`LIBTORCH_USE_PYTORCH=1`). A
MACE model compiled with `e3nn.util.jit.compile` (the route mace's own `run_train` and
`create_lammps_model` use) loads in `tch::CModule` and computes forces by autograd inside the
module. float64 on CPU agrees with Python MACE to rounding; batching all P beads into one graph is
much faster than P calls; CPU at a fixed libtorch thread count is bitwise reproducible; CUDA is not
bitwise reproducible by default because MACE's scatter sums use atomics. float64 on the RTX 4060 is
slow enough that CUDA only pays off in float32 or for large batches.

### Models and systems

- **Tiny model**: random MACE from `spikes/m6-torch/gen_tiny_mace.py` (fixed seed; elements H, C,
  N, O; r_max 5 Å; small channel counts). All correctness, edge and determinism results.
- **MACE-OFF23-small-like**: same script, randomly initialised with the hyperparameters of
  MACE-OFF23 small (96x0e, L = 0, correlation 3, two interactions, r_max 4.5 Å, 8 Bessel functions,
  radial MLP [64, 64, 64]). Timing only, labelled as such; its numbers stand in for MACE-OFF23 until
  real weights are supplied.
- **MACE-OFF23 small/medium**: only if `PIMD_MACE_MODEL` points to an existing file. Not on the
  machine at the start of the spike and not downloaded by the spike (licence not yet confirmed);
  the MACE-OFF23 parts of questions 3, 5 and 7 are otherwise **pending**.
- **Systems**: gas-phase water (the M5 start geometry, `crates/pimd-cli/tests/data/ipi_water_xtb.json`)
  and ethanol (C2H5OH, 9 atoms; coordinates written down in the log). For beads, each bead is the
  base geometry plus a fixed-seed Gaussian displacement (σ = 0.05 Å), so beads differ. For the PBC
  edge check: four water molecules in a 10.5 Å cubic box (> 2 r_max), positions deliberately
  unwrapped (some outside the box), as ring-polymer positions are in pimd-rs (C-5).

### Questions, measurements, criteria

1. **Build and run.** `cargo build --release` of `spikes/m6-torch` inside `pixi run -e ml` with
   `LIBTORCH_USE_PYTORCH=1`. Record: build success, clean build time, binary size, `ldd` output,
   how the library path is set (rpath vs `LD_LIBRARY_PATH`), whether the binary runs outside
   `pixi run` (plain shell), and whether `tch::Cuda::is_available()` is true.
   *Success*: CPU and CUDA both run. *Failure*: no build or link within the time box: stop and
   report the error output (brief: no switch to another approach).
2. **Export.** Script `export_mace.py`: load a MACE model, wrap it in a small `torch.nn.Module`
   with a positional-tensor interface (no dict marshalling in Rust), compile with
   `e3nn.util.jit.compile`, save. Record the input tensors (positions, node attributes, edge index,
   shifts, unit shifts, cell, batch, ptr), outputs (energy, forces by autograd inside the module;
   virial if `compute_virials` works in the export), metadata exposed to Rust (r_max, atomic
   numbers, dtype), and whether float32 is obtained by exporting a float32 model or by
   `CModule::to(kind)` in Rust. *Success*: a scripted procedure that runs unattended.
3. **Correctness.** For each system and bead set: energies and forces from Rust (tch, TorchScript,
   edges built in Rust) vs Python MACE (`MACECalculator` on the original model file, mace's own
   neighbour list), float64 and float32, CPU and CUDA. Report max |ΔE|/|E| and
   max|ΔF| / max|F|. *Success*: float64 ≤ 1e-12 (energy) and ≤ 1e-10 (forces) on CPU and CUDA;
   float32 vs float32-Python ≤ 1e-5 / 1e-4; float32 vs float64 reported (that is the float32
   error the M6b tolerance must cover). Anything larger is investigated, not explained away.
4. **Graph.** Rust edge builder: all ordered pairs i ≠ j with r_ij < r_max (gas phase); with an
   orthorhombic cell, minimum image per C-2 and shift = image correction (vector = q_j − q_i +
   shift), requiring r_max < L_min/2. Compare the edge set (as a set of (sender, receiver,
   shift)) with mace's `get_neighborhood` and the resulting E, F (same model, Python graph vs Rust
   graph). *Success*: identical edge sets; E, F equal to float64 rounding (≤ 1e-12 relative).
5. **Batching.** Time per evaluation (median of ≥ 20 after 5 warm-up calls; includes Rust edge
   building and host↔device copies) for P = 1, 8, 16, 32, water and ethanol, tiny and
   OFF23-small-like models; one batched graph vs P separate calls; CPU at 1 and 4 libtorch threads
   (`OMP_NUM_THREADS` and `tch::set_num_threads` set explicitly), CUDA; float32 and float64. Load
   average recorded with every table. No pass/fail; the numbers decide batching, default dtype and
   device.
6. **Determinism.** Twenty repeated identical calls: bitwise equal? CPU at 1 thread and at 4
   threads; CPU 1 vs 4 threads; CUDA default; CUDA with deterministic algorithms (via whatever
   `tch` exposes, else recorded as not reachable from tch and measured in Python), with its cost.
   *Outcome*: the exact D-3 note text for M6b.
7. **Baseline.** The merged engine binary (`target/release/pimd-rs`, built from this branch,
   unchanged) with `kind = "socket"`, four Python MACE clients (spike-local `mace_client.py`, which
   reuses the locked M5 protocol code of `crates/pimd-socket/tests/clients/ipi_client.py` by import,
   single-threaded, CPU), water, P = 16 (the M5 protocol), 300 K, dt 0.5 fs. Time per step from two
   run lengths (difference removes start-up). Agreement: the client records the first evaluations
   (positions in bohr, energy, forces in a.u.); the Rust spike evaluates the same positions in
   process and the differences are reported as in question 3. *Success*: agreement at the level of
   question 3 (float64), and a per-step time to compare with the in-process P = 16 number.

### What would change the plan

`tch` not building or linking: stop and report. `torch.jit` export of MACE failing on torch
2.13: record the error, try mace's own LAMMPS export once, then stop and report (no ONNX, no PyO3).

## Log

All runs on this machine (12 logical CPUs, RTX 4060 8 GB, WSL2 kernel 5.15.167.4), branch
`milestone-6`, from the repository root. Versions: Rust 1.96.0; tch 0.26.0 / torch-sys 0.26.0;
pixi `ml` env: Python 3.12.16, PyTorch 2.13.0 (CUDA 13.0 build, cuDNN 9.27), mace-torch 0.3.17,
e3nn 0.4.4, ASE 3.29.0, numpy 2.5.4, matscipy 1.3.1. Spike outputs go to `spikes/m6-torch/out/`
(ignored); MACE-OFF23 exports to `ff_models/exports/` (ignored).

### 2026-11-01 17:12 Start; state of the machine

Load average 0.01 0.02 0.03; GPU idle (1.4 GB used by other processes). `PIMD_MACE_MODEL` unset,
no MACE-OFF23 weights at the start. At 17:20 the orchestrator reported that the owner had
confirmed the licence and placed the weights in `ff_models/mace_off23/` (small, medium, large,
OFF23b medium) and `ff_models/mace_off24/` (commit 58d5ac8 ignores `ff_models/`); from then on
"MACE-OFF23 small" = `ff_models/mace_off23/MACE-OFF23_small.model` (treated as
`PIMD_MACE_MODEL`) and "medium" = `MACE-OFF23_medium.model`. Nothing was downloaded by the spike.

### Q1 build and run

- `spikes/m6-torch/Cargo.toml`: `tch = "0.26"`, `[workspace]` so the spike is its own root (the
  root `Cargo.toml` is untouched). Build:
  `LIBTORCH_USE_PYTORCH=1 pixi run -e ml bash -c 'cd spikes/m6-torch && cargo build --release'`.
- First build (no build.rs): **builds**; clean build 66 s wall (178 s CPU; mostly torch-sys
  compiling its C++ wrapper with the system g++ 13.3, against the env's headers, CXX11 ABI). 84
  packages in the spike's Cargo.lock (tch pulls in zip, safetensors, ndarray, half, image-less
  defaults, ...). Binary 0.95 MB (dynamically linked against libtorch).
- Problem 1, runtime path: the binary has **no rpath**; it fails both inside `pixi run -e ml` and
  outside with `error while loading shared libraries: libtorch_cpu.so: cannot open shared object
  file`. pixi (like conda) sets no `LD_LIBRARY_PATH`. `LD_LIBRARY_PATH=<env>/…/site-packages/torch/lib`
  is not enough (`libmkl_intel_lp64.so.3` not found: in this conda-forge build `torch/lib` holds
  symlinks to `$CONDA_PREFIX/lib`, and `$ORIGIN` resolves to the symlink directory).
  `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` works.
- Problem 2, CUDA: with the binary running, `tch::Cuda::is_available()` = **false**: `ldd` shows
  `libtorch_cuda.so` is not NEEDED (the linker's `--as-needed` drops it, nothing references a
  symbol in it directly). Known tch behaviour.
- Fix (`spikes/m6-torch/build.rs`, std only): `-Wl,--no-as-needed,-ltorch_cuda,--as-needed` and
  `-Wl,-rpath,$CONDA_PREFIX/lib` (RUNPATH). Result: `cuda available: true, 1 device`; the binary
  runs **outside pixi** with no environment variables (plain shell, `env -u LD_LIBRARY_PATH`);
  `ldd` resolves libtorch_cpu/cuda, c10, cudart 13, cublas 13, MKL and the env's libstdc++ from
  `$CONDA_PREFIX/lib`. Incremental rebuild 0.4 s. Runtime libraries: libtorch_cpu.so 329 MB,
  libtorch_cuda.so 594 MB (the whole `ml` env is 6.0 GB); nothing is copied, the binary depends on
  the env staying where it is (absolute RUNPATH).
- Default libtorch intra-op threads = 6 (half the logical CPUs), inter-op 12.
- Later addition (Q6): build.rs also compiles `shim.cpp` with the system g++ through
  `std::process::Command` (no `cc` crate, no new Rust dependency) to reach
  `at::globalContext().setDeterministicAlgorithms`, which tch 0.26 does not expose. It compiles
  against `$CONDA_PREFIX/include` in a few seconds.

**Answer Q1: yes**, CPU and CUDA, with two link flags. 

### Q2 export

- Models: `gen_tiny_mace.py` (seed 20261101): tiny = 2 896 parameters (23 kB of float64
  weights; the pickled module file is 274 kB because it pickles the e3nn/MACE objects, so it is
  not committed: the script is). Also `off23-small-like` (274 416 parameters) and
  `off23-medium-like` (651 920), random, for timing only.
- Problem: on torch 2.13 `import e3nn` fails (`torch.load` of e3nn's `constants.pt` with the new
  `weights_only=True` default: `Unsupported global: GLOBAL slice`). mace sets
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in `mace/__init__.py`, so every spike script imports
  `mace` first. Must be the same in the M6b export script.
- `export_mace.py`: loads the model file (`torch.load(weights_only=False)`), casts to the dtype,
  freezes parameters, wraps it in `PimdMace` and compiles with `e3nn.util.jit.compile` (=
  `torch.jit.script` for modules tagged `@compile_mode("script")`; the wrapper needs that tag, else
  `compile` returns it unscripted: first attempt failed with `'PimdMace' object has no attribute
  'save'`). Wrapper interface: `forward(positions [n,3], species [n] i64, edge_index [2,E] i64
  (sender i, receiver j), shifts [E,3], ptr [B+1] i64, compute_virials: bool, unit_shifts [E,3],
  cell [B,3,3]) -> (energy [B], forces [n,3], virial [B,3,3])`, eV and Å; exported methods
  `r_max()`, `atomic_numbers()`, `dtype_code()`. Inside: `batch` from `ptr`
  (`repeat_interleave`), one-hot `node_attrs`, MACE `forward(data, training=False,
  compute_force=True, compute_virials=…)`. Forces are autograd inside the module, so Rust calls it
  with grad mode on (tch default) and never under `no_grad`.
- The export dict MACE needs (from `mace/modules/utils.py::prepare_graph`, mace-torch 0.3.17):
  `positions, node_attrs, edge_index, shifts, unit_shifts, cell ([B*3,3]), batch, ptr`; `head`
  optional (absent = head 0). The positional wrapper avoids building a `GenericDict` IValue in
  Rust.
- Export takes about 10 s (tiny) to 30 s (OFF23 medium) on CPU. Sizes: tiny 0.37 MB (f64);
  MACE-OFF23 small 7.5 MB (f64) / 3.9 MB (f32); medium 18.5 MB / 9.4 MB. MACE-OFF23 exported
  without changes (10 species, Z = 1 6 7 8 9 15 16 17 35 53; r_max 4.5 Å small, 5.0 Å medium).
- float32: a float32 export and a float64 export cast in Rust with `CModule::to(device,
  Kind::Float)` give **bitwise identical** results (all 9 model/system combinations below). So one
  float64 export suffices; the adapter casts.
- Virial (`check_virial.py`, waterbox bead 0, P = 1): `compute_virials=True` works through the
  export; −W/V equals `MACECalculator.get_stress()` to 9e-20 (tiny) and 4e-18 eV/Å³ (OFF23 small).

### Q3, Q4 correctness and edges (`run_correctness.sh`; logs `out/q3-*.log`)

8 beads each of water (3 atoms), ethanol (9) and the 4-water box (12 atoms, L = 10.5 Å, 224 of
416 tiny-model edges cross the boundary), geometries from `make_geoms.py --seed 7` (base
geometries below). Reference: `ref_eval.py` = mace's `MACECalculator` on the original model file,
mace's neighbour list; Rust: `m6-torch-spike eval`, edges built in Rust, batched (one graph of 8
beads), libtorch 1 thread. Metrics: max over beads of |ΔE| and |ΔE|/|E|, max |ΔF| / max |F|.
Note |E| includes the atomic E0 (≈ 2 000–8 000 eV), so relative energy errors look small; absolute
|ΔE| is the meaningful number.

Base geometries (Å): water = M5 start (O 0 0 0.1173; H 0 ±0.7572 −0.4692); ethanol =
`ase.build.molecule("CH3CH2OH")`: C 1.168181 −0.400382 0; C 0 0.559462 0; O −1.190083 −0.227669 0;
H −1.946623 0.381525 0; H 0.042557 1.207508 ±0.886933; H 2.115891 0.144800 0;
H 1.128599 −1.037234 ±0.885881. Bead j: base + N(0, 0.05 Å) (`default_rng(7 + P)`).

| model | check | water | ethanol | waterbox |
|---|---|---|---|---|
| tiny | f64 CPU Rust vs Python: ΔE / ΔF rel | 0 / 5.4e-16 | 0 / 6.0e-16 | 0 / 1.1e-15 |
| tiny | f64 CUDA Rust vs Python CUDA | 0 / 7.2e-16 | 2.7e-12 eV / 3.8e-16 | 1.8e-12 eV / 1.5e-15 |
| tiny | f64 CUDA Rust vs Python CPU | 0 / 1.6e-15 | 2.7e-12 eV / 1.3e-15 | 1.8e-12 eV / 6.7e-15 |
| tiny | f32 CPU Rust vs Python f32 | 0 / 2.4e-7 | 0 / 3.1e-7 | 0 / 3.4e-7 |
| tiny | f32 CUDA Rust vs Python f32 CUDA | 0 / 4.4e-7 | 0 / 2.8e-7 | 0 / 1.4e-6 |
| tiny | f32 vs f64 (Rust CPU) | 3.0e-4 eV / 7.4e-7 | 8.1e-4 eV / 5.8e-7 | 5.7e-4 eV / 3.7e-6 |
| OFF23 small | f64 CPU Rust vs Python | 0 / 5.2e-16 | 0 / 1.6e-16 | 0 / 1.6e-16 |
| OFF23 small | f64 CUDA Rust vs Python CPU | 0 / 1.3e-15 | 0 / 5.9e-16 | 1.8e-12 eV / 1.2e-15 |
| OFF23 small | f32 CPU / CUDA vs Python f32 same device | 0 / 2.8e-7; 0 / 2.8e-7 | 0 / 1.8e-7; 9.8e-4 eV / 3.5e-7 | 0 / 1.3e-7; 9.8e-4 eV / 1.7e-7 |
| OFF23 small | f32 vs f64 (Rust CPU) | 7.7e-5 eV / 8.9e-7 | 1.5e-3 eV / 8.2e-7 | 5.2e-4 eV / 3.9e-6 |
| OFF23 medium | f64 CPU Rust vs Python | 0 / 8.6e-16 | 0 / 3.3e-16 | 0 / 1.7e-16 |
| OFF23 medium | f64 CUDA Rust vs Python CPU | 0 / 6.0e-16 | 2.7e-12 eV / 6.6e-16 | 1.8e-12 eV / 1.9e-15 |
| OFF23 medium | f32 CPU / CUDA vs Python f32 same device | 0 / 2.1e-7; 0 / 1.8e-7 | 0 / 8.8e-8; 1.5e-3 eV / 1.8e-7 | 0 / 8.3e-8; 0 / 2.5e-7 |
| OFF23 medium | f32 vs f64 (Rust CPU) | 1.3e-4 eV / 6.1e-7 | 1.5e-3 eV / 8.1e-7 | 5.4e-4 eV / 3.7e-6 |

Max |F|: tiny 0.14–0.29 eV/Å (random model, weak forces); OFF23 5.1 (water), 10.9 (ethanol),
11.1–11.5 eV/Å (box; displaced beads). Every float64 row meets the plan's criteria (≤ 1e-12
relative energy, ≤ 1e-10 relative forces) by orders of magnitude; float64 CPU energies are
bitwise equal to Python's. The 2.7e-12 eV CUDA energy differences are 1 ulp of |E| ≈ 4 000 eV
(atomics, see Q6). float32 vs float32-Python ≤ 1.4e-6 relative force (criterion 1e-4) and 0 or
1 ulp of energy (9.8e-4 / 1.5e-3 eV = 1 float32 ulp at 4 220 eV). **float32 error itself** (vs
float64): energy up to 1.5e-3 eV (dominated by representing a total of −4 220 eV in float32:
ulp 4.9e-4 eV) and forces up to 4e-6 relative (4.4e-5 eV/Å).

Q4 (edges): Rust edge sets equal mace's `get_neighborhood` sets exactly (sender, receiver, unit
shift; shift vectors bit-equal) for every system and model: 48 / 576 / 416 (tiny, OFF23 medium) and
352 (OFF23 small, r_max 4.5) edges over 8 beads. Evaluating the Rust model with mace's edge list
instead of the Rust one: ΔE = 0, ΔF ≤ 1.9e-15 relative (edge order differs, so the force sums
round differently). Batched (8 beads, one graph) vs a loop of 8 single-bead calls: ΔE = 0,
ΔF ≤ 4.9e-15 relative: **not bitwise equal** (see Q6).

**Answer Q3: Rust/TorchScript reproduces Python MACE to float64 rounding on CPU and CUDA, for the
tiny model and MACE-OFF23 small and medium.** **Answer Q4: the O(N²) Rust edge builder (C-2
minimum image, single image, r_max < L/2) produces mace's exact edge set.**

## Writeup and recommendation
