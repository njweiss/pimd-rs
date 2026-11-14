# M10 spike: where does an ML-PIMD step go, and can the model run faster?

Experiment record (WORKFLOW "Experiments"). The plan is written and committed before any
measurement; the log is appended as work proceeds (including failures); results and the writeup,
with a recommendation on an optimization milestone and the budget of a dynamics study, come last,
and their claims cite log entries. Only merged engine code is used (`main` at 2221c70, M9 merged):
the `pimd-rs` binary built with `--features torch`, the M8 spike bench (`spikes/m8/bench`,
which calls `MacePotential::evaluate` unchanged) and Python probes in `spikes/m10/`. Nothing under
`crates/`, no workspace manifest, `scripts/check.sh`, `pixi.toml` or `pixi.lock` changes. A
package the project does not have (cuEquivariance, Q3) is installed only into a spike-local pixi
project `spikes/m10/cueq/` with its own manifest, git-ignored environment and lock committed. Outputs
in `spikes/m10/out/` (git-ignored); exports in `ff_models/exports/` (git-ignored).

## Plan

### What is already known (M8 spike, M8b; not re-measured)

- Cost is linear in edges while the graph fits: small float32 2.35–2.44 µs per edge, ≈ 61 ms per
  NMA + 216 bead; M8b production 72, 482, 963, 1 931 ms per step at P = 1, 8, 16, 32
  (m8-nma-water.md, Q5).
- Edge building ≤ 2.2 % of an evaluation at P ≥ 8 (8 % at P = 1); a cell list gains ≤ 1.5× on that
  share. `deterministic = false` saves 6 % (small f32). Chunk sizes c = 2–8 lie within 6 %; c = 1 is
  23 % slower (m8-spike.md, Q1, Q2).
- Graph memory ≈ 31–35 kB per edge (small f32); ≈ 6 GB usable on this 8 GB card under WSL2,
  beyond which CUDA spills into shared memory and slows 5–21×.

So the levers left are inside the model: its kernels and its numerics. The engine's own share of a
step is inferred (M8b step time ≈ M8 spike evaluation time) but never measured directly.

### Questions

1. **Engine share.** What fraction of a production step is outside `MacePotential::evaluate`
   (normal-mode transforms, propagation, thermostat, estimators, output)?
2. **Model profile.** Within one evaluation (small f32, NMA + 216, c = 8), how is the time split
   between forward and the autograd backward, and among the model's stages (embedding, radial
   basis and MLP, interaction tensor products and edge scatter, symmetric contraction, readout)?
   Is the GPU busy, or is there launch-bound idle time between kernels?
3. **Faster kernels.** Does MACE's cuEquivariance path (`mace.cli.convert_e3nn_cueq`, the
   `cuequivariance-torch` ops) install for this torch 2.13 / CUDA 13 build, how much faster and how
   much smaller per edge is it in Python on the same inputs, does it survive TorchScript export, and
   can a libtorch process load the result (the ops live in a separate shared library that the
   process must load before the module)?
4. **TF32.** What do TF32 matmuls (`allow_tf32`) gain, and what do they cost in energy and force
   error against float64?
5. **Dynamics-study budget.** Cost per step of (a) 216 pure waters, the M8 water box scaled from
   `water_w64.xyz` or built with `build_box.py`, at P = 1, 8, 32 on CUDA small f32; (b) one
   gas-phase H₂O at P = 1, 32, 64 on CPU (1 and 4 threads) and CUDA, for the cheap tier of a
   failure-mode comparison (see "After the spike").

### Hypotheses (written before measuring)

- **H1 engine share.** Steps take ≤ 3 % longer than the evaluation at P = 8 and P = 32 (M8b 482 ms
  per step against the spike's ≈ 490 ms evaluation suggest ≈ 0); no engine optimization is worth a
  milestone.
- **H2 profile.** Backward takes 55–65 % of an evaluation. The per-edge stages (radial MLP,
  interaction tensor product, scatter onto receivers) take ≥ 60 % of the GPU time and the per-node
  symmetric contraction ≤ 25 %. At 200 k edges per call the GPU is busy ≥ 90 % of the evaluation
  (not launch-bound); at P = 1 (25 k edges) it is not.
- **H3 cuEquivariance.** It installs (or fails for a reason we can name: no build for CUDA 13 or
  torch 2.13). In Python it is 1.5–3× faster per edge and ≥ 1.5× smaller in graph memory than the
  e3nn model, with forces matching the e3nn float32 model to its own float32 noise. Scripting
  works, but a libtorch process can only run it after loading the ops library, which the Rust
  adapter does not do: using it needs an engine milestone (a P-6 change) and a new runtime
  dependency.
- **H4 TF32.** ≤ 1.2× faster (e3nn's small irreps are not matmul-bound), and force RMS errors
  against float64 ≥ 5× those of plain float32: not worth having.
- **H5 budget.** 216 waters cost within 5 % of NMA + 216 per bead (same atom count and edges per
  atom). Gas-phase H₂O at P = 32 is ≤ 5 ms per step on 1 CPU thread and is faster on CPU than CUDA
  (CUDA floor 26–30 ms per call, m6-water-mace.md Q6), i.e. ≥ 1 ns of dynamics per CPU-day per
  thread.

### System, model, protocol

Model MACE-OFF23 small, float32 on CUDA (the production setting), `deterministic = true`, c = 8,
from the v2 export `ff_models/exports/m8b-off23-small-f64.pt` unless stated. Box: `spikes/m8/data/nma_w216.xyz`
(the M8b start). Load average and `nvidia-smi` memory use by others are recorded at the start of
each batch; the GPU runs one job at a time (owner rule).

**Q1.** The merged `pimd-rs` binary, PIMD NVT as M8b (300 K, 0.5 fs, PILE-L, τ0 = 100 fs, output
cadence as M8b), 200 steps at P = 8 and 40 at P = 32 after 20 warm-up steps; ms per step from the
wall time difference of a short and a long run (as in the M8b pilot). The bench's `evaluate` time at
the same P, c and geometry in the same session gives the denominator. Engine share = 1 − evaluate
/ step.

**Q2.** `spikes/m10/profile_mace.py` (`ml` env): the export loaded with `torch.jit.load`, edges
built with numpy as in `spikes/m8/probe_mem.py` (same P-6 edge set), P = 8 beads, 3 warm-up and 5
profiled calls under `torch.profiler` (CPU + CUDA activities, `record_shapes`), forces by autograd
as the adapter does. The same on the eager model (`--eager`, the original `.model` through the
export wrapper) to name the stages, since TorchScript blurs module boundaries. Reported: forward vs
backward GPU time; top kernels by GPU time grouped into the stages of Q2; GPU idle fraction
(1 − sum of kernel time / wall time); the same at P = 1.

**Q3.** In `spikes/m10/cueq/` (own pixi manifest; torch, mace-torch at the `ml` env's versions plus
`cuequivariance`, `cuequivariance-torch`, `cuequivariance-ops-torch` for the matching CUDA): (i)
install; (ii) convert the small model, run the eager e3nn and cueq models on the Q2 inputs in
float32 and float64, time (median of 5 after 3 warm-ups) and peak memory
(`torch.cuda.max_memory_allocated`) at P = 1, 8 and the largest P that fits; force difference
cueq − e3nn in float64 (must be ≤ 1e-8 eV/Å RMS, else it is not the same model) and in float32
against float64; (iii) `torch.jit.script` of the cueq model through the export wrapper (a spike
copy of `export_mace.py`, interface v2); (iv) load and run that file from Rust: a
spike-local Rust binary (standalone Cargo project, not a workspace member) that loads the ops shared library (`libloading`, spike-only dependency)
before `MacePotential::load`, then evaluates once and compares with Python. (iv) is attempted only
if (iii) succeeds. Failure at any step is a result, logged with the error.

**Q4.** As Q3 (ii) on the e3nn model with `torch.backends.cuda.matmul.allow_tf32 = True` and
`torch.backends.cudnn.allow_tf32 = True`: time, and energy and force errors against float64, at
P = 8.

**Q5.** (a) the bench at P = 1, 8, 32 on 216 waters; (b) the bench on one H₂O (M6 geometry) at
P = 1, 32, 64, CPU with `potential.threads` 1 and 4, and CUDA. ms per evaluation; for (b) also one
`pimd-rs` run of 2 000 steps at P = 32 on CPU (1 thread) for the per-step cost including the engine.

### Success and failure

| Q | Success | Failure / negative result |
|---|---|---|
| 1 | Engine share stated at P = 8 and 32 with the numbers it comes from | — |
| 2 | Forward/backward split, stage shares, idle fraction at P = 1 and 8 | Profiler cannot attribute TorchScript kernels to stages: eager profile only, stated |
| 3 | Each of (i)–(iv) reaches a logged outcome; speed, memory and error numbers for what ran | No build for this torch/CUDA: logged with the pip/conda error; Q3 ends |
| 4 | Speed and error at P = 8 | — |
| 5 | Per-step table for both systems | — |

### Decision rule (for the writeup)

An optimization milestone is recommended only for a lever that gives **≥ 1.5×** on small f32,
NMA + 216, P = 32 (c chosen for that lever), measured on the path the engine would use, with force
RMS error against float64 no more than 1.5× that of the current float32 path, and with its
dependency and determinism costs stated (D-3 holds only where the kernel is deterministic). If no
lever passes, the writeup says so and the dynamics study is planned at today's throughput.

### Compute cap

≤ 6 h of GPU wall time in total and ≤ 1 h of CPU-only runs. Estimated: Q1 30 min, Q2 30 min, Q3 1–3 h
(mostly install and conversion), Q4 15 min, Q5 1 h. If Q3's install alone exceeds 1 h, it stops
and is logged as not installable here.

### After the spike (not part of it)

A failure-mode comparison of the M7 methods (classical, RPMD, TRPMD, PA-CMD) on vibrational spectra
from the centroid velocity autocorrelation, in tiers of cost: a 1D Morse oscillator fitted to an O–H
stretch, with the exact quantum Kubo spectrum from eigenstates; gas-phase H₂O with MACE at 300 K
and a lower temperature, where the CMD curvature problem grows; and liquid water only if Q5 and any
speedup make it affordable. Its plan is written after this spike.

## Log

Machine: RTX 4060 8 GB (WSL2), 12 logical CPUs, 11 GB RAM. torch 2.13.0 (conda-forge, CUDA 13),
mace-torch 0.3.17, e3nn 0.4.4, ase 3.29.0, numpy 2.5.4. Engine: `main` at 2221c70 (spike branch
`milestone-10` at 2e2d5fe, plan only). Export `m8b-off23-small-f64.pt` regenerated as v2 on
2026-11-13 (M9). The GPU ran one job at a time throughout.

### 2026-11-13 11:46–12:00 Q1 engine share (`spikes/m10/q1.sh`; `out/q1/`)

Start: load 0.02, GPU 591 MiB used by others, 1 %. The M8b pilot configs (`configs/pilot/p{8,32}_n{40,200}.toml`, seeds 8708/8732,
c = 8, M8b start box) through the merged `pimd-rs`, in M8b's run environment (`RAYON_NUM_THREADS=4`,
`expandable_segments`). Wall: P = 8 22.82 / 98.12 s, P = 32 80.43 / 385.39 s → **470.7 and 1 906.0
ms per step**. During P = 32 n40 a 216-water box was being built on 2 CPU threads (Q5, below); the
P = 32 numbers match the M8b pilot (389.9 s for n200), so no rerun.

The bench (`spikes/m8/bench`, rebuilt against `main`) first failed: the M8b start file
`studies/m8-nma-water/out/start/nma_w216.xyz` has no `Lattice=` header (the box length is in the
config), and the bench needs one. A copy with the header (L = 18.760371262 Å from the pilot config)
is `out/boxes/nma_w216_m8b.xyz`; it, not the M8 spike's `spikes/m8/data/nma_w216.xyz` (a different
equilibration), is the box of every later measurement. Bench, c = 8, 10 timed calls: P = 8
198 454 edges, **475.5 ms** per evaluation (edges 12.7 ms); P = 32 793 844 edges, **1 901.5 ms**
(edges 57.8 ms, 3.0 %). Engine share = 1 − evaluate/step: **−1.0 % (P = 8), +0.2 % (P = 32)**, i.e.
zero within the noise. H1 holds.

### 2026-11-13 12:02 Q2 model profile (`profile_mace.py`; `out/q2/profile_P{8,1}.json`)

Python, same export cast to float32, P-6 edges from numpy (seed 8, σ = 0.05 Å beads), 3 warm-up +
5 profiled calls; no deterministic-algorithms switch (unlike the engine, which accounts for most of
the 475 vs 413 ms difference with the edge build; the M8 spike measured `deterministic = false` at
−6 %).
- **TorchScript, P = 8 (198 552 edges)**: 413.5 ms; kernels 411.9 of 416.8 ms wall per call, **GPU
  busy 98.8 %**. P = 1: 52.7 ms, busy 87.4 %. Not launch-bound at production size (H2 holds).
- **Top kernels, P = 8 (ms per call)**: elementwise binary (mul) 86.8, vectorised add 46.4,
  reductions 34.4, direct-copy elementwise 31.2, two sgemm 30.9 + 29.7, fill 23.9, device-to-device
  copy 23.5, cat 21.8, scatter_add 18.6. Memory-bound elementwise and reduction work dominates;
  matrix multiplies are ≈ 15 %.
- **Stages (eager, P = 8, 387.9 ms)**: forward GPU time per stage interactions[0] 45.9,
  interactions[1] 38.6, products[0] 33.2, products[1] 33.2, embedding + radial + spherical harmonics
  3.4, readouts and scale-shift < 0.1; forward ≈ 154 ms, so **the force backward ≈ 234 ms, 60 %**
  (H2 holds). The per-node symmetric contraction (`products`) is **43 % of the forward**, the
  per-edge interactions 55 %: H2's "symmetric contraction ≤ 25 %" **fails**. (The profiler's eager
  kernel total double-counts the stage annotations; the stage times and the TorchScript totals are
  the valid numbers.)

### 2026-11-13 12:05–12:20 Q3 cuEquivariance, Q4 TF32 (`q34.py`; `out/q34/results.jsonl`)

Deviation from the plan: cuEquivariance 0.13.0 (`cuequivariance`, `cuequivariance-torch`,
`cuequivariance-ops-torch-cu13`; pulls `cuequivariance-ops-cu13`, `nvidia-cublas` 13.10,
`nvidia-cuda-nvrtc` 13.5) went into a venv layered on the `ml` env (`spikes/m10/cueq/.venv`,
`--system-site-packages`; `requirements.lock` kept locally, removed from the repo by owner decision 2026-11-13) instead of a separate pixi project, to
avoid a second 5 GB torch install. The `ml` env is unchanged. Install took < 2 min (Q3 (i):
**installs**). Probes must run under `pixi run -e ml` (Triton needs the activated env's `ptxas`; a
bare interpreter fails in the backward with "Cannot find ptxas"). cuEquivariance logs "Failed to
get GPU information from pynvml: Not Supported. Using default values" (WSL2) and tunes for an
A6000 profile.

Errors are against a float64 e3nn reference computed one bead at a time. Energies are interaction
energies, eV; forces eV/Å; force RMS 0.42 (P = 1) and 2.70 (P = 8).

| variant | P | ms | µs/edge | graph MiB | max \|ΔE\| bead | ΔF RMS | ΔF max |
|---|---|---|---|---|---|---|---|
| export (engine path) f32 | 1 | 52.3 | 2.11 | 672 | 1.9e-4 | 1.47e-5 | 8.7e-5 |
| export f32 | 8 | 412.7 | 2.08 | 5 339 | 6.4e-4 | 1.60e-5 | 1.5e-4 |
| e3nn eager f32 | 1 | 55.6 | 2.24 | 625 | 1.9e-4 | 1.47e-5 | 8.6e-5 |
| e3nn eager f32 | 8 | 387.4 | 1.95 | 4 873 | 1.8e-3 | 1.60e-5 | 1.5e-4 |
| e3nn eager f64 | 1 | 342.1 | 13.79 | 1 231 | 2e-12 | 9e-16 | 3e-15 |
| e3nn f32 + TF32 | 8 | 367.3 | 1.85 | 4 873 | 1.5e-2 | **2.43e-3** | 3.0e-2 |
| export f32 + TF32 | 8 | 382.6 | 1.93 | 5 356 | 1.3e-2 | 2.46e-3 | 3.3e-2 |
| **cueq f32** | 1 | **22.0** | 0.89 | 253 | 5.8e-5 | 1.47e-5 | 8.7e-5 |
| **cueq f32** | 8 | **72.2** | 0.36 | 1 888 | 1.6e-3 | 1.60e-5 | 1.5e-4 |
| cueq f32 | 16 | 148.6 | 0.37 | 3 744 | 1.5e-3 | 1.57e-5 | 1.5e-4 |
| cueq f32 | 32 | 659.2 | 0.83 | 7 477 (spills) | 1.3e-3 | 1.60e-5 | 1.5e-4 |
| cueq f64 | 1 | 76.9 | 3.10 | 484 | 5e-13 | 1e-15 | 6e-15 |

- **Q4 TF32**: 1.05–1.08× faster, force errors 150× those of float32. H4 holds; rejected.
- **Q3 (ii) speed and accuracy**: cueq is the same model (float64 forces equal e3nn's to 1e-15
  eV/Å) and its float32 errors equal e3nn float32's. Per edge it is **5.4× faster than e3nn eager
  and 5.7× faster than the engine's export path at P = 8**, 4.4× in float64, 2.5× at P = 1
  (launch overhead). Graph memory is 9.5 kB per edge against 25–27 kB (2.6× smaller), so c = 16
  fits (3.7 GB) and P = 32 at c = 16 would take ≈ 2 × 148.6 ≈ 300 ms against 1 906 ms. A single
  P = 32 graph (7.5 GB) spills. H3's 1.5–3× is exceeded.
- **Q3 (iii) TorchScript**: `e3nn.util.jit.compile` of the wrapped cueq model succeeds and saves
  (`out/q34/cueq_small_f32.pt`), and modifies the eager module in place (a timing run after it
  failed; timings above are from unscripted processes). Loading the file fails in a fresh process
  ("Unknown builtin op: cuequivariance::uniform_1d"). After `import cuequivariance_ops_torch.uniform_1d`
  it loads, but **running it fails inside the TorchScript interpreter** in the symmetric
  contraction (`cuequivariance_torch/operations/symmetric_contraction.py`, via
  `torch/_library/autograd.py`): the op is a Python `torch.library.custom_op` with a Python
  autograd formula.
- **Q3 (iv) Rust**: not attempted, as the plan says when (iii) fails, and moot: the kernels live in
  `cuequivariance_ops_torch_ext.cpython-312-x86_64-linux-gnu.so`, a nanobind Python extension (one
  `PyInit` symbol, 134 undefined `Py*` symbols, links `libtorch_python`); `torch.ops.load_library` on
  it fails. A libtorch process without a Python interpreter cannot run these kernels. H3's
  prediction (needs an engine change) holds in a stronger form: loading a library is not enough.

### 2026-11-13 12:20–12:45 Q3 (v), added: cueq behind the merged socket backend (`cueq_client.py`, `q3v.sh`; `out/q3v/`)

Not in the plan; added because Q3 (iv) closed the in-process route and the socket backend (K-1 …
K-6, M5) needs no engine change. `cueq_client.py` is the locked M5 test client's `serve` loop
(imported read-only) around the eager cueq model in the v2 wrapper: one bead per request (K-4),
edges built on the GPU in float64, energy = interaction + Σ E0 in float64, ase.units constants. K
clients share the GPU; the M8b pilot configs with `[potential]` replaced by `kind = "socket"`,
`n_clients = K`; ms per step from the short/long wall difference as in Q1.

| P | K | ms per step | vs engine today |
|---|---|---|---|
| 8 | 1 | 235.4 | 2.0× |
| 8 | 2 | 151.3 | 3.1× |
| 8 | 4 | 102.3 | **4.6×** |
| 32 | 1 | 1 016.2 | 1.9× |
| 32 | 4 | 423.5 | **4.5×** |
| 32 | 8 | — | OOM: host RAM |
| 32 | 4, **e3nn** clients (control) | 2 014.2 | 0.95× |

- The control (the same client with the e3nn model, K = 4) is no faster than the engine: the gain is
  the cueq kernels, not client concurrency. (Load average 16 at the control's start, left over from
  the K = 8 failure; 4 at its end.)

### 2026-11-13 12:30–13:20 Q5 budget (`q5.sh`; `out/q5/`)

Deviation: the gas-phase cases use short/long `pimd-rs` runs (200 and 2 000 steps, M6 geometry,
300 K, 0.5 fs, τ0 = 20 fs, export `m6c-off23-small-f64.pt`) instead of the bench, which needs a
periodic box. (a) 216 waters, bench, small f32 CUDA, c = min(P, 8):

| P | edges | ms per evaluation | ms per bead | µs per edge |
|---|---|---|---|---|
| 1 | 24 440 | 70.3 | 70.3 | 2.88 |
| 8 | 195 406 | 467.5 | 58.4 | 2.39 |
| 32 | 781 988 | 1 858.1 | 58.1 | 2.38 |

Within 2 % of NMA + 216 per bead (H5 (a) holds). (b) One H₂O, ms per step:

| P | CPU f64, 1 thread | CPU f32, 1 thread | CPU f64, 4 threads | CUDA f64 |
|---|---|---|---|---|
| 1 | 10.3 | 7.6 | 7.5 | 39.0 |
| 32 | 219.9 | 105.6 | 86.7 | **49.5** |
| 64 | 472.3 | 231.6 | 227.3 | **85.8** |

H5 (b) **fails**: a 3-atom MACE evaluation costs ≈ 7 ms per bead on one CPU thread (fixed per-call
cost, linear in P), not ≤ 5 ms per P = 32 step, and CUDA is the fastest device from P = 32 on
(its per-call floor is shared by the batched beads). At P = 32 on CUDA, 1 ns takes 27 h.

### 2026-11-13 13:20–13:40 Q3 (vi), added: bitwise repeatability (`q3_determinism.py`; `out/q34/determinism.jsonl`, `out/q3det/`)

Added because the decision rule asks for the determinism cost. Ten evaluations of the same P = 8
inputs in one Python process, float32, with and without `torch.use_deterministic_algorithms`:
eager e3nn, eager cueq and the TorchScript export all differ bitwise between calls (forces by up to
1–2e-5 eV/Å) even with the switch, so Python is not a valid baseline. The engine is: three
`pimd-rs` runs of `pilot/p8_n40` (two here, one in Q1) give **bitwise identical** observables
(SPEC D-3 with `deterministic = true` holds on this machine; the adapter also disables the
profiling executor). Two runs of the socket + cueq route (P = 8, K = 1) **differ from step 1**:
ΔV = 3e-8 Eh, ΔT_est up to 1.5e-5 K by step 100. Its timing repeated at 261.9 ms per step against
235.4 (+11 %).

- Agreement with the engine (P = 8, K = 1, same seed): step-1 potential −25.410810437591643 vs
  −25.410810225537734 Eh (2.1e-7 Eh, the size of the ase vs `units` Hartree/bohr constant
  difference over this energy); step 40 T_est 302.24083 K in both. Same physics.
- K = 8 failed: the WSL2 host has 11 GB of RAM and each client holds 1.2–2.3 GB resident; the
  kernel OOM-killed clients (dmesg), the engine reported "Connection reset by peer" for bead 1
  (K-5). K = 4 is the most this machine runs.
- Per bead, one client costs 29–32 ms against cueq's 22 ms batched P = 1 graph: the remainder is
  the socket round trip, the GPU edge build and Python. More clients hide it.

## Results

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1 engine share ≤ 3 % | **holds** (−1.0 %, +0.2 %) | Q1 |
| H2 backward 55–65 %; GPU busy ≥ 90 % at P = 8 | **holds** (≈ 60 %; 98.8 %) | Q2 |
| H2 symmetric contraction ≤ 25 % of the forward | **fails** (43 %) | Q2 |
| H3 cueq installs; 1.5–3× faster | **holds, exceeded**: 5.4–5.7× per edge at P = 8, 2.6× less memory | Q3 (i), (ii) |
| H3 scripting works; Rust needs a library load | **fails in a stronger form**: the scripted model does not run even in Python; the ops need a Python interpreter | Q3 (iii), (iv) |
| H4 TF32 ≤ 1.2×, errors ≥ 5× | **holds** (1.05–1.08×; 150×) | Q4 |
| H5 (a) water ≈ NMA box per bead | **holds** (58.1 vs 59.4 ms) | Q5 |
| H5 (b) gas-phase H₂O ≤ 5 ms per P = 32 step on CPU | **fails** (220 ms; CUDA 49.5 ms is fastest) | Q5 |

Throughput of NMA + 216, small f32, P = 32 (ms per step): engine today 1 906; socket + cueq with 4
clients **424 (4.5×)**, no engine change, measured; cueq batched in chunks of 16 beads ≈ 300
(6.4×), measured in Python only and with no route into the engine.

## Writeup

**Where the time goes.** All of an ML-PIMD step is the MACE evaluation (Q1). Inside it, the GPU is
saturated with memory-bound elementwise work (Q2), 60 % of it in the force backward, and the
symmetric contraction is a larger share than expected. Engine work (cell lists, transforms, output)
cannot buy more than ≈ 3 % (edges), and the knobs the M8 spike measured (determinism 6 %, chunk
size 6 %) are small. TF32 is not worth its error (Q4).

**The one large lever is NVIDIA's cuEquivariance kernels**: the same model, the same float32
accuracy, 5.4–5.7× faster per edge and 2.6× smaller in memory at production size (Q3 (ii)). They
cannot run inside the engine: the kernels are a Python custom op in a Python extension, and the
TorchScript route fails even in Python (Q3 (iii), (iv)). Through the merged socket backend they give
**4.5× at P = 32** with four clients and no engine change (Q3 (v)); the e3nn control shows the gain
is the kernels. Costs:
- **Determinism.** The socket + cueq route is not bitwise reproducible (Q3 (vi)); runs agree to
  ≈ 1e-8 Eh, which is the D-3 status of `deterministic = false`. The engine's own path stays
  bitwise reproducible.
- **Dependencies and licence.** `cuequivariance` and `cuequivariance-torch` are Apache-2.0; the
  kernel packages `cuequivariance-ops-cu13` and `cuequivariance-ops-torch-cu13` are under the NVIDIA
  proprietary software licence. Using them in studies needs the owner's decision (as the model
  weights did); they stay out of the repository either way.
- **Memory and machine.** Each client holds 1.2–2.3 GB of host RAM; four is the limit on this
  11 GB WSL2 host (Q3 (v)). Timing varies by ≈ 10 % between repeats.
- **Units.** The spike client uses ase constants; a supported client must use the engine's
  `units` values (2e-7 Eh offset at this size, Q3 (v)).

**Recommendation.**
1. **Optimization milestone: yes, but small and outside the core.** The decision rule passes for
   socket + cueq (4.5× ≥ 1.5×, same force errors) with the determinism and licence costs above. A
   milestone M10a would add a supported MACE socket client (a script beside `export_mace.py`, using
   the engine's constants and the v2 E0 handling, e3nn or cueq), tests against the in-process
   adapter to float32 tolerance, and README/spec text on its reproducibility (D-3 note, as for
   other external clients). No engine change. Owner decisions first: the NVIDIA licence, and
   whether tolerance-level reproducibility is acceptable for production studies.
2. **A batched socket request** (several beads per message, a K-rule extension) would let one
   client reach the ≈ 6.4× of batched cueq with less RAM. It changes the wire protocol away from
   i-PI's; a candidate after M10a, only if 4.5× proves not enough.
3. **Embedding Python in the engine** (pyo3) to call cueq in-process: not recommended; it adds a
   large dependency for ≈ 1.4× over the socket route.

**Dynamics-study budget (Q5).** Liquid water, 216 molecules, P = 32: 10 ps of dynamics costs 10.3 h
today and ≈ 2.4 h with socket + cueq; a set of methods × several segments of 10–20 ps is days today
and about a day with the speedup. Gas-phase H₂O is not cheap: 49.5 ms per P = 32 step on CUDA
(f64), so 100 ps per method and temperature takes ≈ 2.8 h, and 1 ns 27 h. Several CPU runs can go
in parallel (≈ 87 ms per step at 4 threads, f64), limited by RAM. The tiered failure-mode plan
stands: 1D model first (no MLIP cost), gas-phase H₂O second (hours per method and temperature),
liquid water last, and the liquid tier is the one the speedup makes practical.

## Owner decisions (2026-11-13)

1. M10a (supported MACE socket client) approved; brief in docs/MILESTONE_10.md.
2. The NVIDIA proprietary licence of the cueq kernel packages is acceptable as long as it does not
   extend into the repository: no cueq package in `pixi.toml`/`pixi.lock`, nothing vendored; the
   kernels are installed out of the repo.
3. Tolerance-level reproducibility (≈ 1e-8 Eh between repeat runs) is acceptable for production
   studies on the socket + cueq route; the in-process path stays bitwise reproducible.

## M10a measurement (2026-11-13; `spikes/m10/m10a_q3v.sh`; `spikes/m10/out/m10a/`)

The spike's Q3 (v) at P = 32, K = 4, rerun with the supported client
(`crates/pimd-torch/scripts/mace_client.py`, commit 8e5477d) instead of `cueq_client.py`:
`--backend cueq --dtype float32 --device cuda` on `ff_models/mace_off23/MACE-OFF23_small.model`,
`--config` the generated run config (species from its `positions_xyz`,
studies/m8-nma-water/out/start/nma_w216.xyz), started from `.venv-cueq/` made by the README's own
install steps (cuequivariance-torch and cuequivariance-ops-torch-cu13 0.13.0, layered on the `ml`
env). The M8b pilot configs `p32_n40` / `p32_n200` with `[potential]` replaced by
`kind = "socket"`, `n_clients = 4`, as in Q3 (v). Load average 1.4 at the start, GPU idle.

| run | wall s |
|---|---|
| p32_n40 | 41.97 |
| p32_n200 | 108.11 |

**413.3 ms per step** (Δwall/Δsteps = 66.13 s / 160), against 423.5 ms for the spike client
(−2.4 %, within the ±10 % acceptance band) and 1 906 ms for the engine in-process (4.6×).

Energies: the same `p32_n40` pilot config in-process (export `m8b-off23-small-f64.pt`, float32,
CUDA, c = 8, deterministic; `out/m10a/inproc/p32_n40/`, bitwise equal to the spike's Q1 run
`out/q1/p32_n40/observables.csv`) against the client run's `observables.csv`. The files give the
bead-averaged V̄ per atom, so the comparison is N·|ΔV̄| (Eh per bead-averaged box energy), not per
bead:

| | step 1 | max over 40 steps |
|---|---|---|
| supported client (U-2 constants) | 5.0e-6 Eh (1.4e-4 eV) | 1.3e-5 Eh (3.6e-4 eV, step 29) |
| spike client (ase.units), for reference | 1.30e-4 Eh (3.5e-3 eV) | 1.44e-4 Eh (3.9e-3 eV) |

The test 7 tolerance (M9a v2 float32 bound, per bead 2^-16 · max(|E_int|, N eV), E_int ≈ −2281 eV)
is 0.035 eV = 1.3e-3 Eh; the supported client is ≈ 100× inside it over the whole run, while the
spike client's constant offset (2e-7 Eh per atom, ase's CODATA 2014 constants) is gone. The
per-bead check against the in-process adapter is test 7 itself (`long_mace_client.rs`, P = 4, same
box). E_prim and E_vir agree to the same size (5.0e-6 Eh at step 1). The trajectories are not
bitwise equal (D-3: float32 kernels and summation order differ), so the difference grows slowly
with the step.
