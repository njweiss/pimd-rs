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

## Writeup and recommendation
