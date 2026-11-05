# M8 spike: periodic MACE-OFF for N-methylacetamide in water: cost, memory, stability

Experiment record for MILESTONE_8 "M8 spike" (WORKFLOW "Experiments"). The plan is written and
committed before any measurement; the log is appended as work proceeds (including failures); the
results tables and the writeup with recommendations for the M8a brief and M8b come last, and their
claims cite log entries. Only merged engine code is used (branch `milestone-8` at a03d6d4 = M7b
merged plus the M8 outline): the `pimd-rs` binary built with `--features torch`, and a small bench
binary in `spikes/m8/` that depends on the workspace crates `pimd-core`, `pimd-potentials` and
`pimd-torch` by path and calls `MacePotential::evaluate` unchanged. Nothing under `crates/`, no
workspace manifest, `scripts/check.sh`, `pixi.toml` or `pixi.lock` changes. Spike code:
`spikes/m8/` (standalone Cargo project, not a workspace member); outputs in `spikes/m8/out/`
(git-ignored); TorchScript exports in `ff_models/exports/` (git-ignored, never committed).

## Plan

### Question

Can the merged `MacePotential` (P-6: all-pairs minimum-image edges built on the CPU, all P beads in
one graph) run ML-PIMD of N-methylacetamide (NMA) in periodic water on MACE-OFF on this machine
(RTX 4060 8 GB, 12 logical CPUs, 11 GB RAM)? What does an evaluation cost and how much GPU memory
does it need as a function of N, P, model, dtype and device; what bead chunk size must M8a support;
are cell-list edges needed for speed, for memory, or neither; is a classical MACE-OFF23 run of the
solvated box stable; how is the start structure made offline; and what do the M8b production and
the M8a i-PI comparison cost?

### Hypotheses (written before measuring)

- **H1 cost.** In the liquid each atom has ≈ 50 neighbours within r_max = 5 Å (number density
  ≈ 0.1 Å⁻³), against 2–8 in the gas-phase M6 molecules, so cost per atom is 10–30× the M6 numbers
  and scales with the number of edges, i.e. linearly in P·N. CUDA float32 is the only configuration
  fast enough for production: ≥ 10× faster than float64 on CUDA and ≥ 30× faster than four CPU
  threads at P·N ≥ 5 000. Medium costs 3–5× small. CPU float64 at P = 32, N ≈ 660 takes minutes
  per evaluation (not usable for production; measured only where it fits the cap).
- **H2 edges.** All-pairs edge building is O(P·N²) but cheap per pair: ≤ 10 % of the evaluation
  time on CUDA float32 at N ≤ 660 for every P, and a few per cent or less on CPU. A cell-list build
  of the same edge set is ≥ 5× faster at N = 660. Edge buffers on the host are tens of MB at most,
  so cell lists are not needed for memory; whether they are needed for speed is decided by the
  measured share.
- **H3 memory.** Peak GPU memory is dominated by the model's per-edge activations kept for the
  autograd force pass, roughly linear in the number of edges; float32 halves it. A full P = 32,
  N ≈ 660 graph (≈ 1 M edges) does not fit in 8 GB for medium and probably not for small, so M8a
  needs bead chunking; the largest graph that fits is a few thousand atoms (medium) to ≈ 10 000
  atoms (small) in float32. Chunking at a chunk size that keeps the GPU busy (≥ 2 000 atoms per
  call) costs ≤ 10 % throughput against one graph.
- **H4 stability.** A classical (P = 1) NVT run of NMA + 216 waters with MACE-OFF23 at 300 K,
  dt = 0.5 fs, for 10 ps keeps NMA intact (all covalent bonds within 1.3× their start length, the
  amide bond trans) and no water dissociates; under `Thermostat::Off` (NVE segments) the drift of
  H/N is ≤ 1e-6 Eh per atom per ps in float64 at 0.5 fs; float32 drifts more (rounding noise in
  the forces). g_OO's first peak lies at 2.75–2.85 Å (experiment: ≈ 2.80 Å) with height 2.5–3.5
  (classical MACE water is known to be over-structured; sanity only).
- **H5 start structure.** OpenMM Modeller can build NMA as the capped dipeptide ACE–NME (Amber
  ff14SB has both caps, and ACE–NME is exactly NMA), solvate it with a chosen number of TIP3P
  waters (`numAdded`) in a cubic box, and a short classical NVT equilibration at the target density
  gives a start structure that MACE-OFF23 accepts without blowing up (no restrained MACE route
  needed).
- **H6 budget.** An M8b production at the recommended setting (P ≈ 16–32, ≈ 216 waters, CUDA
  float32) costs ≈ 0.3–1 s per step, so tens of ps per day per P value; a quantum-vs-classical study
  with a P-convergence series fits in a few days. The M8a i-PI comparison on a 64-water box is
  dominated by Python MACE evaluation time and fits in hours at P = 8.

### Models and systems

- **Models**: MACE-OFF23 small and medium (`ff_models/mace_off23/MACE-OFF23_{small,medium}.model`),
  MACE-OFF24 medium (`ff_models/mace_off24/MACE-OFF24_medium.model`) only if time allows. Exported
  once each with the merged `crates/pimd-torch/scripts/export_mace.py` (float64; the adapter casts
  to float32) to `ff_models/exports/m8-<model>-f64.pt`; size and sha256 logged. r_max read from the
  export.
- **Systems**: NMA (CH₃–C(=O)–NH–CH₃, 12 atoms, trans) + N_w waters, N_w = 64, 128, 216 (204, 396,
  660 atoms), cubic box at a fixed mass density of 0.997 g cm⁻³ (L from the total mass: ≈ 12.7,
  15.9, 18.8 Å; r_max < L/2 holds for all; note that pure water with 32 molecules, L ≈ 9.9 Å, would
  violate r_max < L/2 for r_max = 5 Å, so 64 waters is the smallest box for the M8a comparison).
  Plus pure water, 64 waters (L ≈ 12.4 Å), for the budget of the M8a i-PI comparison.
- **Beads**: P copies of the equilibrated structure, each displaced by a fixed-seed Gaussian
  (σ = 0.05 Å per component, as in the M6 spike; seed logged), so beads differ but the edge count
  is that of the liquid. P = 1 uses the structure itself.

### Protocol

All CUDA runs with `deterministic = true` (the P-6 default) unless stated; CPU runs at 4 libtorch
threads (`OMP_NUM_THREADS=4`). Each case runs in its own process (clean GPU memory accounting)
under `timeout`. Load average and `nvidia-smi` (memory used by others, utilisation) are recorded at
the start of each batch of runs; other users of the GPU (the Windows host showed 1.4 GB at the start)
are noted, not stopped.

**Q4 start structure** (first, because Q1–Q3 need the boxes). `spikes/m8/build_box.py` in the `ref`
env: NMA heavy atoms as an ACE–NME PDB written by the script, `Modeller.addHydrogens` (ff14SB),
`addSolvent(model="tip3p", numAdded=N_w, boxSize=L)` with L from the density above, then (ff14SB +
TIP3P, PME 9 Å, HBonds constraints, LangevinMiddle 300 K, 1 ps⁻¹, 2 fs) energy minimisation and
50 ps NVT at the fixed box; molecules made whole; written as extended xyz (Å, `Lattice=`) to
`spikes/m8/out/boxes/`. Seeds logged. *Success*: a script that runs unattended in minutes and gives
the requested N_w exactly; the 216-water file is copied to `spikes/m8/data/` if < 200 kB.
*Fallback* (if Modeller cannot build NMA offline): a script-built lattice of waters around NMA and
a short MACE P = 1 run with gradual heating; logged as the route used.

**Q1 cost and edge share** (`spikes/m8/bench`, `run_bench.py`). One bench call per case: load the
export through `MacePotential::load`, build the P-bead positions, 2 warm-up evaluations, then timed
evaluations (median of up to 10, at least 3, stopping after 60 s of timed calls), each a full
`MacePotential::evaluate` (edges, host→device, forward with autograd forces, device→host). Inside
the same process, a spike-local copy of the merged all-pairs edge build (`bead_edges` +
concatenation, same rayon parallelism) is timed alone (median of 10) to give the edge-building
share; a spike-local cell-list build is timed the same way and checked to give the same edge set (as
a sorted set of (sender, receiver, unit shift)) and the same count. Grid:
- CUDA, float32 and float64: N_w = 64, 128, 216 × P = 1, 8, 16, 32 × small, medium (all
  combinations until OOM; OOM is a result, recorded for Q2).
- CPU, 4 threads: P = 1 at all N_w, both models, both dtypes; P = 8 at N_w = 64 (small, both
  dtypes); more only if a case is under 10 s per evaluation.
- One CPU case at 1 and 12 threads (small, f32, N_w = 216, P = 1) for the thread scaling.
No pass/fail; the numbers decide device, dtype and model. Reported: ms per evaluation, ms per bead,
µs per edge, the edge count per bead, the edge-building share (all-pairs and cell list).

**Q2 memory.** GPU memory from `nvidia-smi --query-gpu=memory.used` read by the bench before
loading, after loading (CUDA context + model) and after the timed evaluations (libtorch's caching
allocator keeps the peak reserved, so this is the peak reserved memory plus the context). Reported:
peak graph memory = after-evaluation − after-load, against P·N and edges, per model and dtype.
Cross-check of the reading on four cases with a Python probe (`probe_mem.py`, `ml` env, same
export, same inputs, `torch.cuda.max_memory_allocated/reserved`). The largest graph that fits:
from the linear fit (bytes per edge) and confirmed by a direct run at the predicted largest P·N
and at 1.2× it (expected OOM). Then chunking through the merged code: P = 32, N_w = 216, the beads
split into consecutive chunks of c = 1, 2, 4, 8, 16 beads, each chunk one `evaluate` call (this is
what M8a's chunked evaluation will do inside the potential); ms per full evaluation and peak memory
against c. *Outcome*: the chunk size (beads per graph) M8a must support, and its cost.

**Q3 stability** (merged `pimd-rs` binary, `kind = "mace"`). Model, dtype and device: the ones Q1
suggests for production (expected MACE-OFF23 small or medium, CUDA float32), plus float64 for the
drift test if its cost allows. System: NMA + 216 waters from Q4, positions inline in the config
(`positions_angstrom`, generated by `make_configs.py`; there is no positions file in the merged
engine yet). (a) NVT, P = 1, 300 K, dt 0.5 fs, τ_0 = 100 fs (PILE-L, only the centroid at P = 1),
20 000 steps (10 ps), `trajectory_every = 20`, seed logged. (b) Energy drift: `[dynamics]
method = "rpmd"` at P = 1 (thermostatted decorrelation, then `Thermostat::Off` segments, H/N at
segment start and end in `segments.csv`), 4 segments of 1 ps at dt 0.5 fs, and the same at 1.0 fs
if (a) is clean, in float32 and (if affordable) float64. Analysis (`analyse_md.py`): NMA bond
lengths and the ω (C–N) torsion over time; every water O–H < 1.3 Å and every H assigned to one O;
no atom further than one box length from its start (unwrapped); T_est; drift of H/N per ps per
segment; g_OO from the trajectory (minimum image, 0–L/2). *Success*: H4's criteria. *Failure*
(NMA breaks, water dissociates, H/N jumps): logged as a negative result with the step and the
model/dtype, then one retry at dt = 0.25 fs to tell a time-step problem from a model problem.

**Q5 budget** (arithmetic from Q1 and Q3, plus two small measurements). (i) M8b: ms per step at the
recommended setting for P = 1, 8, 16, 32 (Q1 numbers; the engine adds its O(P·N) work, checked
against the Q3 run's wall time at P = 1), times the steps needed (equilibration + production at the
recommended dt, for 20 and 50 ps of production), per run and for the whole study (P = 1, P-series,
second seed). (ii) M8a i-PI comparison: Python MACE (`MACECalculator`, `ml` env, float64, CPU and
CUDA) on pure water, 64 molecules, single-structure evaluation time (median of 10), and i-PI's own
per-step overhead taken from the merged i-PI references (`validation/RESULTS.md`) if they give it,
else estimated and stated as such; projected wall time for ⟨E_vir⟩ and g_OO at P = 1 and P = 8 to
a stated σ.

### Success and failure, per question

| Q | Success | Failure / negative result |
|---|---|---|
| 1 cost | Every CUDA case that fits in memory and every planned CPU case measured; edge share stated per case | A model fails to export or load: recorded, the other models continue |
| 2 memory | Bytes per edge per model and dtype, the largest graph confirmed by a direct run, chunked throughput vs c | OOM even at P = 1 for a model/dtype: recorded; that model is out for M8 on this GPU |
| 3 stability | H4's criteria hold at 0.5 fs for the chosen model and dtype | NMA or water breaks, or drift above 1e-5 Eh per atom per ps: recorded, retry at 0.25 fs |
| 4 start | Unattended script, exact N_w, MACE accepts the result (Q3) | Modeller cannot build NMA: fallback route, logged |
| 5 budget | A table of wall times for M8b and the M8a comparison with the numbers they come from | — |

### Compute cap and what would change the plan

About 4 h of wall time for all measurements. Estimated: Q4 15 min; Q1+Q2 CUDA grid 48 cases ×
≲ 1.5 min ≈ 70 min; CPU cases ≲ 30 min; chunking and memory checks 20 min; Q3 (a) 10 ps at an
estimated 30–100 ms per step ≈ 10–35 min, (b) 4–8 ps ≈ 20–30 min, at most two dtypes; Q5 15 min.
If the CPU cases are much slower than estimated they are cut first (P = 1 only), then the float64
CUDA cases at P = 32, then MACE-OFF24. If the plan cannot fit in the cap, stop and report. No
engine change of any kind; an engine defect goes to `docs/QUESTIONS.md` via the orchestrator, not
into the spike.

## Log

All runs on the owner's machine (12 logical CPUs, 11 GB RAM, RTX 4060 8 GB, WSL2 kernel
5.15.167.4), from the repository root, branch `milestone-8`. Versions: Rust 1.96.0; tch 0.26.0 /
torch-sys 0.26.0 (the spike's `Cargo.lock` is a copy of the workspace lock, so every crate version
equals the merged engine's); pixi `ml` env: PyTorch 2.13.0, mace-torch 0.3.17 (cuequivariance not
installed, so no accelerated kernels); `ref` env: OpenMM 8.6.1. Engine binary:
`pixi run -e ml env LIBTORCH_USE_PYTORCH=1 cargo build --release --offline -p pimd-cli --features torch`
(no source change). Bench: `spikes/m8/bench` built the same way with
`--manifest-path spikes/m8/bench/Cargo.toml`.

### 2026-11-05 16:45 Start; state of the machine

Load average 0.01 0.03 0.03; GPU: 1 362 MiB used and 11 % utilisation with no compute process
visible from WSL (Windows host; it varied between 464 and 1 362 MiB during the spike and is
subtracted, see Q2);
no other heavy process (`ps`: editor and Claude processes only, < 2 % CPU each; nothing from polytopia_rl computing). Plan committed as 3fe3529 at 16:48.

### 2026-11-05 16:49–16:52 Q4 start structures (`build_box.py`; logs `out/logs/box_*.log`)

`pixi run -e ref python spikes/m8/build_box.py --waters W --seed 8000+W --out spikes/m8/out/boxes/nma_wW.xyz`
for W = 64, 128, 216, and `--waters 64 --no-solute --seed 8164` (pure water). A 5 ps test run first
(25 s wall) to check the route. Result: **the Modeller route works offline, unattended, with no
fallback**: ACE–NME heavy atoms + `addHydrogens` (ff14SB) give NMA with the expected 12 atoms in
the order C H H H C O N H C H H H; `addSolvent(numAdded=W)` gives exactly W waters; molecule
centres scaled into the cube fixed by 0.997 g cm⁻³; minimisation + 50 ps NVT (ff14SB + TIP3P,
PME, HBonds, Langevin 300 K). All four boxes in 2 min 36 s wall (run concurrently, CPU platform):

| box | atoms | L (Å) | E_pot TIP3P (kJ/mol) | file size |
|---|---|---|---|---|
| NMA + 64 | 204 | 12.6869 | −2 722.6 | 8.7 kB |
| NMA + 128 | 396 | 15.8240 | −5 255.9 | 16.9 kB |
| NMA + 216 | 660 | 18.7604 | −8 755.4 | 28.2 kB |
| water 64 | 192 | 12.4296 | −2 572.7 | 8.2 kB |

(−40.2 kJ/mol per water for pure TIP3P, the known TIP3P value.) `numAdded` and `boxSize` are
mutually exclusive in OpenMM 8.6.1, hence the affine rescaling of the molecule centres before the
minimisation. Whether MACE-OFF accepts the structure is Q3.

### 2026-11-05 16:52 Exports

`pixi run -e ml python crates/pimd-torch/scripts/export_mace.py <model> --out ff_models/exports/m8-<m>-f64.pt`
(14 s for all three). **r_max differs between the models**: MACE-OFF23 small 4.5 Å, MACE-OFF23
medium 5.0 Å, MACE-OFF24 medium 6.0 Å (all from the export's `r_max()`). With r_max < L/2
(P-6), OFF24 medium needs L > 12 Å, i.e. ≥ 60 waters; 64 waters (12.43 Å) still passes for all
three. Files (bytes, sha256 prefix): small 7 453 024 `465436ea…`, medium 18 468 923 `1098e60b…`,
OFF24 medium 18 472 315 `c87e5617…` (model files `165cce4c…`, `4842c52a…`, `e5ccf583…`).

### 2026-11-05 16:54–16:57 Bench build; first cases

First `cargo build` of the spike resolved a fresh lock and fetched one crate (`cc` 1.11.0) from
crates.io into the cargo cache: a deviation from "nothing downloaded outside pixi"; corrected
immediately by copying the workspace `Cargo.lock` into `spikes/m8/bench/` and rebuilding with
`--offline` (56 s; identical versions to the workspace, `cc` 1.10.0). Smoke test (small f32 CUDA):
NMA + 64, P = 1: 26 ms, 7 652 edges, cell-list edges identical to the all-pairs copy; NMA + 216,
P = 8: 649 ms, 201 114 edges (25 139 per bead, 38 per atom), **5.8 GB graph memory**; with
`deterministic = false` 608 ms (−6 %). Bead-0 energy differs between the deterministic and
non-deterministic float32 runs by 0.012 Eh (0.34 eV) on a total of −16 769 Eh (see Q1 float32
energy note).

### 2026-11-05 16:57–17:00 Q1/Q2 CUDA grid, first pass: GPU memory oversubscription

`python3 spikes/m8/run_bench.py cuda spikes/m8/out/bench_cuda.jsonl`. The case small f32 NMA + 128,
P = 32 (≈ 480 k edges) did **not** fail with out-of-memory: it ran at **24 396 ms per evaluation,
21× the linear extrapolation** from P = 16 (573 ms), with the GPU at 100 % and 7.8 GB reported.
WSL2 (WDDM) lets CUDA allocations spill into shared system memory instead of raising OOM, so on
this machine "does not fit" shows up as a 20× slowdown, not as an error. The grid was stopped
(killed at 17:01) and restarted with a 300 s per-case timeout and a skip rule (a case is skipped
when the same model/dtype/N at the next smaller P needed > 6 000 MiB of graph memory); completed
cases are kept.

### 2026-11-05 17:01–17:12 Q1/Q2 CUDA grid (`run_bench.py cuda`; `out/bench_cuda.jsonl`, table in Results)

Second pass with the skip rule; 48 cases, 26 measured, 4 failed, 18 skipped. Load average ≤ 1.2
(the bench itself); GPU background 1 170–1 330 MiB during the pass. Each case: 2 warm-up + up to
10 timed `evaluate` calls (≥ 3, 30 s budget), seed 8, σ = 0.05 Å, `deterministic = true`.
Observations, all from the JSON lines:
- **Cost is per edge and linear while the graph fits.** Edges per atom: 37–38 (small, 4.5 Å),
  51–52 (medium, 5.0 Å). µs per edge at P ≥ 8 (P = 1 carries the fixed per-call cost):
  small f32 2.35–2.44, small f64 14.8–16.7, medium f32 8.7–9.3 (P = 1 only), medium f64 60–69.
  Medium costs 3.6× small per edge and 4.9× per atom in float32 (more edges); float64 is 6× (small)
  to 7× (medium) slower than float32 on this GPU.
- **Memory is per edge too**: graph memory (after-evaluation − after-load, `nvidia-smi`) per edge:
  small f32 31–35 kB, small f64 60–69 kB, medium f32 105–108 kB, medium f64 207–216 kB. So a
  single NMA + 216 bead (25–34 k edges) takes 0.87 GB (small f32), 1.5 GB (small f64), 3.5 GB
  (medium f32) and 7.0 GB (medium f64).
- **What does not fit, and how it fails.** Above ≈ 7.1–7.2 GB of graph memory either the run
  spills into shared system memory and slows down 5–21× (small f32 NMA + 128 P = 32: 24.4 s,
  NMA + 216 P = 16: 14.9 s, 15× per edge; small f64 NMA + 216 P = 8: 14.6 s, 4.7×; medium f32
  NMA + 64 P = 8: 4.9 s, 6.3×) or libtorch raises an error inside the TorchScript forward (medium
  f32 NMA + 128 and NMA + 216 at P = 8, medium f64 NMA + 64 and NMA + 128 at P = 8; message
  truncated by the bench at 300 characters; the bench now keeps the tail of the message, see the
  chunk pass). Small f32 NMA + 64 at P = 32 (6.3 GB) was 35 % slower per edge than at P = 16
  (3.22 vs 2.39 µs): the edge of the usable range is soft.
- **Edges are cheap.** The copy of the merged all-pairs build takes 0.7–17.7 ms (12 rayon threads),
  ≤ 8 % of the evaluation (small f32, NMA + 216, P = 1: 6.1 of 76 ms) and ≤ 2.2 % in every other
  fitting case. The spike's cell list gives the identical edge list (same order, same shifts) in
  every case; it is slower than all-pairs below 400 atoms and, for small, 1.5× faster at 660 atoms
  (4.0 vs 6.1 ms at P = 1; 7.0 vs 10.5 ms at P = 8); for medium at 660 atoms it is slower (8.5 vs
  5.9 ms at P = 1).
- **float32 energies.** At P = 1 (same geometry), float32 − float64 bead energy: small −0.070,
  +0.060, +0.077 eV; medium −0.065, +0.072, +0.089 eV at 204, 396, 660 atoms. At 660 atoms both
  models' float32 energies are the same double (−16 771.37773777647 Eh): the float32 total of
  ≈ −456 000 eV has a spacing of 0.031 eV, so per-bead energies are quantised at that level and
  the error is 2–3 spacings (≈ 3e-3 Eh per bead, 5e-6 Eh per atom).

### 2026-11-05 17:12–17:26 Q2 chunking and the fit limit (`run_bench.py chunk`; `out/bench_chunk.jsonl`)

GPU background had dropped to 464–485 MiB at the start (Windows host), so up to ≈ 0.8 GB more
headroom than in the CUDA grid. NMA + 216, P = 32, beads in consecutive chunks of c, one merged
`evaluate` per chunk (2 warm-up + ≥ 3 timed full evaluations, 20 s budget):

| model, dtype | c = 1 | c = 2 | c = 4 | c = 8 |
|---|---|---|---|---|
| small f32 (ms per P = 32 evaluation; graph MiB) | 2 410 (868) | 2 075 (1 526) | 1 998 (3 848) | 1 967 (5 984) |
| medium f32 | 9 971 (3 568) | 12 023 (7 005) | — | — |
| small f64 | 12 417 (1 542) | 12 341 (3 862) | 13 226 (7 118) | — |
| medium f64 | 75 419 (7 050; 1 warm-up, 2 timed) | — | — | — |

- Chunking costs little once a chunk carries ≈ 50 k edges: small f32 c = 2, 4, 8 are within 6 %
  of each other; c = 1 is 23 % slower than c = 8 (per-call overhead of 25 k-edge graphs). For
  medium f32, c = 1 (34 k edges, 3.6 GB) is the only choice; c = 2 (7.0 GB) is already 21 % slower
  (start of the spill). float64 shows no gain from larger chunks.
- `deterministic = false`: small f32 c = 4 1 878 ms (−6 %), medium f32 c = 1 8 954 ms (−10 %).
- **Fit limit, single graph** (small f32, NMA + 216): P = 8 (201 k edges, 6.0 GB) 61 ms/bead;
  P = 10 (251 k, 7.1 GB) 142 ms/bead (2.3×, spilling); P = 12 (302 k) 300 ms/bead (4.9×). Medium
  f32: NMA + 128 P = 3 (62 k edges, 6.4 GB) 174 ms/bead = P = 1's 183, fits; NMA + 216 P = 2
  (69 k edges, 7.0 GB) 334 ms/bead vs 298 at P = 1 (+12 %, edge of the spill). The CUDA grid's
  medium failures at P = 8 needed 80–170 k edges, 8–17 GB.
- So on this 8 GB card **≈ 6 GB of graph memory is usable**, i.e. ≈ 190–200 k edges per call for
  small f32, ≈ 95 k small f64, ≈ 60 k medium f32, ≈ 30 k medium f64 (one NMA + 216 bead has 25 k
  edges with small, 34 k with medium).

### 2026-11-05 17:27–17:34 Q1 CPU cases (`run_bench.py cpu`; `out/bench_cpu.jsonl`)

**Ran concurrently with the Q3 NVT run** (which holds one CPU core plus short rayon bursts and the
GPU), so the CPU numbers may be a few per cent pessimistic; the 12-thread case is the most
affected. Load average 1.1 at the start. 4 libtorch threads unless stated; peak RSS from `VmHWM`.

| model, dtype | NMA + 64, P = 1 | NMA + 128, P = 1 | NMA + 216, P = 1 | NMA + 64, P = 8 |
|---|---|---|---|---|
| small f32 (ms; µs/edge; RSS GB) | 274; 36; 0.8 | 728; 49; 0.9 | 1 083; 43; 1.2 | 2 897; 47; 2.1 |
| small f64 | 575; 75; 1.0 | 1 223; 82; 1.4 | 2 133; 85; 2.0 | 5 873; 96; 3.8 |
| medium f32 | 1 524; 145; 1.5 | 3 202; 156; 2.3 | 5 866; 170; 3.3 | — |
| medium f64 | 3 037; 289; 2.5 | 6 398; 312; 4.0 | 11 101; 323; 6.0 | — |

Small f32 NMA + 216 P = 1 at 1 thread: 1 961 ms; at 12 threads: 1 246 ms (slower than 4 threads;
contended). Against CUDA (same model and dtype, NMA + 216, P = 1): small f32 14× slower, small
f64 5.5×, medium f32 20×, medium f64 4.7×. The CPU also runs out of host memory: the autograd
graph lives in RAM (6.0 GB for one medium f64 bead of NMA + 216, out of 11 GB). Edge building is
≤ 0.9 % of every CPU evaluation.

### 2026-11-05 17:26–17:52 Q3a classical NVT, NMA + 216, MACE-OFF23 small, CUDA float32 (`out/md/nvt_small_f32/`)

Model and dtype chosen from the CUDA grid (small f32 is the only one with a P-series in reach,
see Results); config `spikes/m8/configs/nvt_small_f32.toml` (`make_configs.py nvt`, positions
inline from `out/boxes/nma_w216.xyz`), `spikes/m8/run_md.sh` → merged `target/release/pimd-rs`,
P = 1, 300 K, dt 0.5 fs, PILE-L τ_0 = 100 fs, 20 000 steps (10 ps), seed 8101, `deterministic`
default (true), 1 libtorch thread. Overlapped with the CPU grid (17:27–17:34) and the Python CPU
timing (17:38). **Completed, exit 0, 1 524 s wall = 76 ms per step** (the bench's 76.5 ms per
evaluation; the engine's own P = 1 work is negligible). Analysis
(`pixi run -e ref python spikes/m8/analyse_md.py out/md/nvt_small_f32 --box 18.760371262 --skip-ps 2`,
1 000 frames every 10 fs, `analysis.json`):
- **NMA intact**: largest relative change of any NMA bond against the first frame 12.2 % (a C–H);
  C=O 5.8 %, C–N 6.2 %, N–H 9.4 %. Mean bonds (Å): C=O 1.246, C–N 1.334, N–H 1.012, N–CH₃ 1.455,
  C–CH₃ 1.508. Amide torsion C–C(O)–N–C stays trans: min |ω| = 145.0° over 10 ps.
- **Water intact**: longest O–H in any frame 1.130 Å; in no frame is any H closer to a foreign O
  than to its own (no proton transfer, no dissociation); largest displacement of any atom from the
  first frame 7.1 Å (diffusion, < L = 18.8 Å); no atom lost.
- **Equilibration from the TIP3P structure**: V̄ per atom −25.41097 Eh at step 10, block means
  per ps −25.411357 (0–1 ps), then −25.41147 … −25.41156 (1–10 ps): MACE relaxes the rigid-TIP3P
  start within ≈ 1 ps; no gradual or restrained approach was needed. T_est 298.0 ± 9.2 K (2–10 ps).
- **g_OO** (water O, 2–10 ps, 0.05 Å bins): first peak at 2.775 Å (bin centre), height 2.72;
  first minimum 3.775 Å, g = 0.92. Experiment (X-ray, ambient): first peak ≈ 2.80 Å, height ≈ 2.5.
  Sanity check only (classical, 216 waters, 8 ps).

### 2026-11-05 17:53 Q2 cross-check of the memory reading (`probe_mem.py`; `out/probe_mem.jsonl`)

`pixi run -e ml python spikes/m8/probe_mem.py <export> out/boxes/nma_w216.xyz P dtype` (same
export, edges built with numpy, `torch.cuda.max_memory_allocated/reserved`); run while the Q3b
drift run held the GPU, so its times are not used. Peak allocated / peak reserved (MiB) against the
Rust bench's `nvidia-smi` graph memory: small f32 P = 1: 693 / 838 vs 868; small f32 P = 4: 2 706 /
3 976 vs 3 848 (chunk 4 in the chunk pass); medium f32 P = 1: 2 718 / 3 556 vs 3 527; small f64
P = 1: 1 367 / 2 014 vs 1 542. The `nvidia-smi` reading tracks the caching allocator's reserved
memory (within −23 … +4 %); the memory actually allocated is 68–83 % of the reserved, i.e.
20–30 % is allocator slack (fragmentation). Edge counts agree with the Rust build (25 190 and
34 406 at P = 1). Bead-0 energies (eV, float32): −456 372.09375 small, −456 372.125 medium:
multiples of 1/32 eV, the float32 spacing at this magnitude.

### 2026-11-05 17:52–18:10 Q3b energy drift, float32 (`out/md/drift_small_f32_dt{0.5,1.0}/`)

Start: the last frame (step 20 000) of Q3a, positions only (`make_configs.py drift --from-traj`);
`[dynamics] method = "rpmd"` at P = 1: per cycle `decorrelation_steps` with PILE-L, a momentum
redraw, then `segment_steps` with `Thermostat::Off`; `segments.csv` gives H/N (Eh per atom) at
segment start and end. Small, CUDA float32.
- dt 0.5 fs, seed 8201, 4 cycles of 400 + 2 000 steps (segments of 1 ps): 717 s wall
  (75 ms/step). ΔH/N per segment: +1.2e-7, +1.78e-6, −1.09e-6, −3.6e-7 Eh per atom per ps.
- dt 1.0 fs, seed 8202, 4 cycles of 200 + 1 000 steps (1 ps): 355 s wall (74 ms/step). ΔH/N:
  +2.24e-6, +1.97e-6, −5.7e-7, −3.19e-6 Eh per atom per ps.
Both: no systematic sign, |ΔH/N| ≤ 3.2e-6 Eh per atom per ps (≤ 1.0 K per atom per ps). The float32
energy is quantised at 0.031 eV = 1.15e-3 Eh per bead (Q1), i.e. 1.7e-6 Eh per atom, and its error
is 2–3 spacings, so a single float32 H/N reading is uncertain at ≈ 5e-6 Eh per atom: **float32
cannot resolve a drift below that level, and none above it is seen at 0.5 or 1.0 fs.** Structure
(`analyse_md.py --skip-ps 0`, frames every 100/200 fs): largest NMA bond change 9.6 % / 9.1 %,
min |ω| 163.0° / 160.1°, longest O–H 1.089 / 1.091 Å, no proton transfer; T_est 308 ± 10 K /
295 ± 8 K. 1.0 fs is stable for classical P = 1 water on MACE (the PIMD time step is a separate
question, see M6c: 0.5 fs biases estimator consistency at P = 32).

### 2026-11-05 18:10–18:41 Q3b energy drift, float64 (`out/md/drift_small_f64_dt0.5/`)

Same start, small, CUDA float64, dt 0.5 fs, seed 8203, 2 cycles of 400 + 2 000 steps: 1 823 s wall
(380 ms/step; the bench's 387 ms per evaluation). ΔH/N per 1 ps segment: **+1.7e-7 and
+4.7e-7 Eh per atom per ps** (H4: ≤ 1e-6 holds). Structure: largest NMA bond change 12.3 %, min |ω| 156.0°, longest O–H 1.080 Å, no proton transfer, T_est 302 ± 9 K. Ran alone on the GPU.

### 2026-11-05 18:41–18:47 MACE-OFF24 medium, allocator setting, Python MACE (`out/bench_off24.jsonl`, `out/bench_expandable.jsonl`, `out/py_mace_cuda.jsonl`)

Load average 0.93, GPU background 506 MiB at the start. Run in sequence (`out/logs/bench_final.log`).
- **MACE-OFF24 medium** (r_max 6.0 Å, f32 CUDA): 89 edges per atom (58.9 k per NMA + 216 bead),
  7.4 µs and 106 kB per edge (≈ OFF23 medium per edge, 1.7× its edges). P = 1: 138 ms (NMA + 64,
  1.9 GB), 261 ms (NMA + 128, 3.7 GB), 436 ms (NMA + 216, 6.1 GB). NMA + 216, P = 32, chunks of
  1 bead: 13 969 ms per evaluation (1 warm-up, 3 timed). One NMA + 216 bead is the largest graph
  that fits.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (environment only, libtorch reads it; no
  code change): small f32 NMA + 216 P = 8 468 ms, 5 602 MiB (without: 487 ms, 5 984 MiB);
  **P = 10 582 ms = 58 ms/bead, 6 986 MiB: fits** (without: 1 421 ms, spilling); P = 12: libtorch
  error `CUDA driver error: device not ready` (not an OOM message); medium f32 NMA + 216 P = 2:
  579 ms = 290 ms/bead, fits (without: 667 ms, +12 %). Less allocator slack (Q2 cross-check:
  20–30 %) moves the usable graph from ≈ 6 GB to ≈ 7 GB.
- **Python MACE** (`py_mace_time.py`, `MACECalculator` on the model file, mace's neighbour list,
  pure water 64 = 192 atoms, one structure per call, 1 torch thread): small f64 CUDA 110.0 ms,
  small f32 CUDA 34.9 ms, medium f32 CUDA 85.9 ms; and (17:38, overlapping the Q3a run) small
  f64 CPU 920.6 ms at 1 thread, 527.9 ms at 4 threads. Energies: f64 −133 216.24045 eV on both
  devices; f32 −133 216.203 (small), −133 216.078 (medium) eV.

Measurement wall time used in total: 16:49–18:47 ≈ 2 h (of the ≈ 4 h cap); MACE-OFF24 included.

## Results

Every number is from the log entries above; the full per-case tables are regenerated by
`python3 spikes/m8/summarise.py spikes/m8/results/bench_*.jsonl` (`spikes/m8/results/bench_tables.md`;
the JSON lines in `spikes/m8/results/` are copies of `spikes/m8/out/`). CUDA = RTX 4060 8 GB under
WSL2, `deterministic = true`; CPU = 4 libtorch threads. "spill" = the graph exceeded the usable
GPU memory and WSL2 moved it to shared system memory (no error, 2–21× slower).

### R1. ms per evaluation, one graph of all P beads (CUDA grid; CPU cases; OFF24)

NMA + N_w waters; edges per bead: small 7.7 k / 14.9 k / 25.2 k, medium 10.5 k / 20.5 k / 34.4 k,
OFF24 medium 18.3 k / 35.4 k / 58.9 k for N_w = 64 / 128 / 216 (204 / 396 / 660 atoms).

| model | dtype | device | N_w | P = 1 | P = 8 | P = 16 | P = 32 |
|---|---|---|---|---|---|---|---|
| OFF23 small | f32 | CUDA | 64 | 26 | 144 | 293 | 791 |
| | | | 128 | 55 | 290 | 573 | 24 396 (spill) |
| | | | 216 | 76 | 487 | 14 856 (spill) | — |
| OFF23 small | f64 | CUDA | 64 | 120 | 928 | 2 047 | — |
| | | | 128 | 228 | 1 764 | — | — |
| | | | 216 | 387 | 14 630 (spill) | — | — |
| OFF23 medium | f32 | CUDA | 64 | 97 | 4 912 (spill) | — | — |
| | | | 128 | 183 | error | — | — |
| | | | 216 | 298 | error | — | — |
| OFF23 medium | f64 | CUDA | 64 | 634 | error | — | — |
| | | | 128 | 1 284 | error | — | — |
| | | | 216 | 2 370 | — | — | — |
| OFF24 medium | f32 | CUDA | 64 / 128 / 216 | 138 / 261 / 436 | — | — | — |
| OFF23 small | f32 | CPU | 64 / 128 / 216 | 274 / 728 / 1 083 | 2 897 (N_w 64) | | |
| OFF23 small | f64 | CPU | 64 / 128 / 216 | 575 / 1 223 / 2 133 | 5 873 (N_w 64) | | |
| OFF23 medium | f32 | CPU | 64 / 128 / 216 | 1 524 / 3 202 / 5 866 | | | |
| OFF23 medium | f64 | CPU | 64 / 128 / 216 | 3 037 / 6 398 / 11 101 | | | |

"—" = skipped because the next smaller P already needed > 6 GB. Per edge, while the graph fits
(P ≥ 8 for small): small f32 2.3–2.4 µs, small f64 15 µs, medium f32 8.4–9.3 µs, medium f64
60–69 µs, OFF24 f32 7.4 µs (CUDA); CPU 36–49 (small f32), 75–96 (small f64), 145–170 (medium
f32), 289–323 µs (medium f64).

### R2. NMA + 216 at P = 32, bead-chunked (one merged `evaluate` per chunk of c beads)

| model, dtype | c = 1 | c = 2 | c = 4 | c = 8 | c = 10 (expandable segments) |
|---|---|---|---|---|---|
| small f32 | 2 410 | 2 075 | 1 998 | **1 967** | ≈ 1 860 (58.2 ms/bead at P = 10) |
| medium f32 | **9 971** | 12 023 (spill); ≈ 9 270 with expandable segments | — | — | — |
| small f64 | 12 417 | 12 341 | 13 226 | — | — |
| medium f64 | 75 419 | — | — | — | — |
| OFF24 medium f32 | 13 969 | — | — | — | — |

`deterministic = false`: small f32 c = 4 −6 %, medium f32 c = 1 −10 %.

### R3. Peak GPU memory (graph memory = reserved by libtorch, from `nvidia-smi`)

| model, dtype | kB per edge | one NMA + 216 bead | largest graph that runs at full speed (atoms; edges) |
|---|---|---|---|
| small f32 | 30–35 | 0.87 GB | 5 280 (P = 8 × 660; 201 k) default allocator; 6 600 (P = 10; 251 k) with expandable segments |
| small f64 | 60–69 | 1.5 GB | ≈ 2 640 (c = 4 × 660 runs, 7.1 GB, +7 % per edge); ≈ 95 k edges |
| medium f32 | 105–108 (93 with expandable segments) | 3.5 GB | 1 188 (P = 3 × 396; 62 k); 1 320 (P = 2 × 660) with expandable segments |
| medium f64 | 207–216 | 7.0 GB | 660 (one bead), at the limit |
| OFF24 medium f32 | 106 | 6.1 GB | 660 (one bead) |

Usable: ≈ 6 GB of the 8 GB with the default caching allocator (≈ 1.3 GB held by the CUDA
context, the model and the Windows host's 0.5–1.4 GB), ≈ 7 GB with expandable segments. The
reserved memory exceeds the allocated memory by 20–30 % (Q2 cross-check).

### R4. Edge building (copy of the merged all-pairs code, 12 rayon threads)

| case | all-pairs ms | cell list ms | evaluation ms | all-pairs share |
|---|---|---|---|---|
| small f32 CUDA, NMA + 216, P = 1 | 6.1 | 4.1 | 76 | 8.0 % (worst case measured) |
| small f32 CUDA, NMA + 216, P = 8 | 10.5 | 7.0 | 487 | 2.2 % |
| small f32 CUDA, NMA + 216, P = 32 (c = 8) | 34.9 (all 32 beads) | 27.1 | 1 967 | 1.8 % |
| medium f32 CUDA, NMA + 216, P = 32 (c = 1) | 37.5 | 50.2 | 9 971 | 0.4 % |
| any CPU case | ≤ 9.6 | | ≥ 274 | ≤ 0.9 % |

The spike's cell list reproduces the all-pairs edge list exactly (order and shifts) in every case
and is 1.3–1.5× faster at 660 atoms for small (r_max 4.5 Å, 4 cells per side), slower below 400
atoms and slower for medium (r_max 5.0 Å: 3 cells per side, so every cell is a neighbour and
nothing is pruned). Host edge buffers: 12 MB at 200 k edges.

### R5. Stability (Q3)

| run | wall, ms/step | NMA max bond change | min abs ω | max O–H | proton transfers | ΔH/N per ps (Eh/atom) |
|---|---|---|---|---|---|---|
| NVT 10 ps, f32, 0.5 fs | 1 524 s, 76 | 12.2 % | 145.0° | 1.130 Å | 0 | (thermostatted) |
| NVE 4 × 1 ps, f32, 0.5 fs | 717 s, 75 | 9.6 % | 163.0° | 1.089 Å | 0 | +1.2e-7, +1.8e-6, −1.1e-6, −3.6e-7 |
| NVE 4 × 1 ps, f32, 1.0 fs | 355 s, 74 | 9.1 % | 160.1° | 1.091 Å | 0 | +2.2e-6, +2.0e-6, −5.7e-7, −3.2e-6 |
| NVE 2 × 1 ps, f64, 0.5 fs | 1 823 s, 380 | 12.3 % | 156.0° | 1.080 Å | 0 | +1.7e-7, +4.7e-7 |

g_OO (NVT, 2–10 ps): first peak 2.775 Å, height 2.72; first minimum 3.775 Å (0.92).

### R6. Budget projections (Q5)

M8b, NMA + 216, MACE-OFF23 small, CUDA float32, dt 0.5 fs, chunk 8; ms per step from R1/R2
(P = 1 from the engine run, R5; P = 16 = 2 × the P = 8 single graph); one run = 10 ps
equilibration + production.

| P | ms/step | 20 ps production (60 k steps) | 50 ps production (120 k steps) |
|---|---|---|---|
| 1 | 76 | 1.3 h | 2.5 h |
| 8 | 487 | 8.1 h | 16.2 h |
| 16 | ≈ 975 | 16.3 h | 32.5 h |
| 32 | 1 967 | 32.8 h | 65.6 h |
| study: P = 1, 8, 16, 32 + a second P = 32 seed | | **91 h (3.8 days)** | 182 h (7.6 days) |

Alternatives, per P = 32 run of 60 k steps: OFF23 medium f32 (c = 1) 166 h; OFF24 medium 233 h;
small f64 207 h; NMA + 128 small f32 ≈ 19 h (≈ 1.16 s/step). Medium or OFF24 at P = 1 (classical
model check): 5.0 h / 7.3 h per 60 k steps. `deterministic = false` would save 6–10 %.

M8a comparison, pure water 64 (192 atoms), MACE-OFF23 small, 2.5 ps equilibration + 20 ps
production (45 k steps at 0.5 fs) per side:

| side | P = 1 | P = 8 |
|---|---|---|
| pimd-rs in process, f32 CUDA (NMA + 64 numbers: 26 / 144 ms) | 0.3 h | 1.8 h |
| i-PI + one Python MACE client, f32 CUDA (34.9 ms per bead + ≈ 3 ms i-PI per step) | 0.5 h | 3.6 h |
| i-PI + one Python MACE client, f64 CUDA (110 ms per bead) | 1.4 h | 11 h |
| i-PI + four Python CPU clients, f64, 1 thread (921 ms per bead, 2 beads each at P = 8) | 11.5 h | 23 h |

The i-PI overhead (≈ 3 ms per step) is from the M5 water/xTB gate run (`validation/RESULTS.md`:
i-PI reference 2 195 s vs pimd-rs 914 s for the same 404 000 steps with the same clients), so it
is an estimate for a 3-atom system; at 192 atoms it is still far below the MACE cost. Python
MACE f32 CUDA costs 1.3× the in-process call per bead (34.9 ms vs 26 ms).

## Writeup and recommendations

Provenance: log entries are cited by their time and heading; R1–R6 are the tables above. The
engine was not modified; nothing under `crates/`, no manifest, `scripts/check.sh` or pixi file
changed. One deviation: a crate (`cc` 1.11.0) was fetched from crates.io by the spike's first
build before the workspace lock was copied in (16:54–16:57 entry); the bench as committed builds
offline with the workspace's versions.

### Answers to the five questions

1. **Cost** (R1, R2; 17:01–17:12, 17:12–17:26, 17:27–17:34, 18:41–18:47). The cost is per edge and
   linear in the number of edges while the graph fits: on CUDA 2.4 µs (small f32), 15 µs (small
   f64), 8.7 µs (medium f32), 65 µs (medium f64), 7.4 µs (OFF24 medium f32). The liquid has 38
   (small), 52 (medium) and 89 (OFF24) edges per atom. One NMA + 216 bead costs 61 ms (small f32,
   chunked), 300 ms (medium f32), 436 ms (OFF24 f32), 387 ms (small f64), 2.4 s (medium f64). The
   CPU is 14–20× slower than CUDA in float32 and 5× in float64. Edge building on the CPU is
   ≤ 8 % of an evaluation (small f32, P = 1) and ≤ 2.2 % at every P ≥ 8 (R4).
2. **Memory** (R3; 17:12–17:26, 17:53, 18:41–18:47). Also per edge: 31 kB (small f32), 62 kB
   (small f64), 105 kB (medium f32), 210 kB (medium f64), 106 kB (OFF24 f32). About 6 GB of the
   8 GB is usable (7 GB with `expandable_segments`). The largest graph that runs at full speed is
   8 beads of NMA + 216 for small f32 (10 with expandable segments), 1 bead for medium f32 (2 with
   expandable segments), OFF24 f32 and medium f64. A full P = 32 graph fits for no model. On this
   WSL2 machine an oversized graph usually does not fail: it spills into system memory and runs
   2–21× slower (16:57–17:00 entry); sometimes libtorch raises an error instead.
3. **Stability** (R5; 17:26–17:52, 17:52–18:10, 18:10–18:41). Classical NMA + 216 on MACE-OFF23
   small is stable for 10 ps NVT at 0.5 fs and for NVE segments at 0.5 and 1.0 fs. NMA stays intact
   and trans; there is no proton transfer and no atom is lost. The rigid-TIP3P start relaxes
   within ≈ 1 ps. Drift in float64 at 0.5 fs is ≤ 4.7e-7 Eh per atom per ps. In float32 no drift
   above the float32 energy resolution (≈ 5e-6 Eh per atom per reading) is seen. g_OO's first
   peak is at 2.775 Å, height 2.72 (experiment ≈ 2.80 Å, ≈ 2.5–2.6): slightly over-structured, as
   expected for classical MLIP water. Medium and OFF24 were not run in MD; only their costs were
   measured.
4. **Start structure** (16:49–16:52). `spikes/m8/build_box.py` (OpenMM 8.6.1 in `ref`) builds
   NMA as ACE–NME (ff14SB), adds exactly N_w TIP3P waters with `addSolvent(numAdded=…)`, rescales
   to the cube fixed by 0.997 g cm⁻³, then minimises and runs 50 ps of NVT. It needs no packmol,
   no download and no fallback route, and takes < 3 min for all boxes. MACE accepts the result
   directly. The NMA + 216 box is committed as `spikes/m8/data/nma_w216.xyz` (28 kB), and the pure
   64-water box for the M8a comparison as `spikes/m8/data/water_w64.xyz`.
5. **Budget** (R6). The M8b study at NMA + 216, small f32, chunk 8, dt 0.5 fs, P = 1/8/16/32 plus a
   second P = 32 seed, 20 ps production each, needs ≈ 91 h (3.8 days) of GPU time; 50 ps
   production doubles that. Medium or OFF24 make a P = 32 run take 7–10 days, so they are out
   except as P = 1 checks. The M8a i-PI comparison (64 waters, P = 1 and 8, 20 ps per side) needs
   ≈ 2 h on the pimd-rs side and ≈ 4 h on the i-PI side with one Python f32 CUDA client.

### Hypotheses

- H1: per-edge and linear: **holds**. "f32 CUDA ≥ 10× f64 CUDA": **fails**; it is 6× (small)
  and 7× (medium). "≥ 30× four CPU threads": holds against CPU f64 (35×); against CPU f32 it is
  14–18×. Medium 3–5× small: holds (3.9× at P = 1, 5.1× chunked at P = 32).
- H2: edge share ≤ 10 %: **holds** (≤ 8 %, ≤ 2.2 % at P ≥ 8). Cell list ≥ 5× faster at
  660 atoms: **fails**; it is 1.3–1.5× for small and slower for medium. Edge buffers tens of MB:
  holds.
- H3: memory per edge and f32 halving it: **holds**. "Full P = 32 N ≈ 660 does not fit": holds.
  Largest graph "few thousand atoms (medium) to ≈ 10 000 (small)": **partly**; small reaches
  5 300–6 600 atoms, medium ≈ 1 300. Chunking at ≥ 2 000 atoms costs ≤ 10 %: holds (c = 2…8
  within 6 %; 660-atom chunks +23 %).
- H4: **holds** for small (f32 and f64). Medium was not run in MD.
- H5: **holds**.
- H6: P = 16 at ≈ 1 s/step holds; P = 32 at ≈ 2 s/step is above the 0.3–1 s guess. "Tens of ps
  per day per P" holds marginally (22 ps/day at P = 32). "A few days": holds (3.8 days for 20 ps
  production). The i-PI comparison in hours: holds.

### Recommendations for the M8a brief and M8b

| Item | Recommendation | Evidence |
|---|---|---|
| N | **NMA + 216 waters** (660 atoms, L = 18.760 Å at 0.997 g cm⁻³), the start file from `build_box.py` | Q4; inside the outline's 200–260; 25 k small edges per bead, so 8 beads per graph fit (R3) |
| P | **P = 1, 8, 16, 32**; production quantum results at P = 32 | M6c (gas-phase water): P = 16 gives 90 % of the P = 32 quantum energy shift, P = 32 is 96 % of the harmonic P → ∞ shift; R6 cost |
| model | **MACE-OFF23 small** for every P. Optional: one classical P = 1 run with OFF23 medium (5 h) as a model check. OFF24 medium is not recommended (r_max 6 Å, 89 edges per atom, one bead per graph, 233 h per P = 32 run) | R1, R2, R6 |
| dtype | **float32** for production; float64 only for the M8a correctness ladder (energies and forces to 1e-10) and for a drift check | f64 is 6× slower (R1, R2); f32 dynamics stable (R5). The f32 bead energy is quantised at 0.031 eV (1.15e-3 Eh, 5e-6 Eh per atom; Q1 float32 note) |
| device | **CUDA** | CPU is 14–20× slower in f32 and needs up to 6 GB of RAM per medium bead (R1, CPU entry) |
| dt | **0.5 fs** for P ≥ 8 (the M6c estimator-consistency bias at P = 32 already appears at 0.5 fs); 1.0 fs is stable for P = 1 and could halve the classical run | R5; M6c writeup "Engine behaviour" |
| bead chunk | M8a must support chunks of **1 to 8 beads** (any value ≤ P). Recommended for M8b: **8** (small f32), and 1 for medium, OFF24 or f64. The 1e-12 equivalence test should cover c = 1 and an uneven split (e.g. P = 32, c = 5) | R2, R3 |
| cell-list edges | **Not needed for speed or for memory at this size.** All-pairs is ≤ 2.2 % of the time at P ≥ 8. The host edge buffers are 12 MB, and GPU memory is the model's per-edge activations, which do not depend on how the edge list is built. A cell list only pays at a few thousand atoms per bead (all-pairs is O(N²): ≈ 9× at 2 000 atoms). Recommend dropping or deferring decision 3(a) (owner decision below) | R4, R3 |
| deterministic | Keep `true` as the default; `false` saves 6–10 % | R2 |
| allocator | Document `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for M8b runs (an environment variable, no engine change). It moves the usable graph from ≈ 6 to ≈ 7 GB; with it c = 10 fits for small | 18:41–18:47 entry |
| i-PI comparison size | **64 waters (12.43 Å) is the minimum box** (32 waters would violate r_max < L/2). P = 1 and P = 8, small, f32 CUDA on both sides, 20 ps production each: ≈ 2 h pimd-rs + ≈ 4 h i-PI | R6; Models and systems |

### Points that need an owner decision

1. **Decision 3(a), cell-list edges.** The spike finds them unnecessary for N ≈ 660 for speed and
   for memory. Options: (i) drop them from M8a, so M8a = bead chunks + positions file, with the
   graph built as today; (ii) keep them for future larger systems, accepting work that M8b does
   not need. Recommendation: (i).
2. **float32 energy resolution.** Per-bead float32 energies of the 660-atom box carry ≈ 3e-3 Eh
   (2–3 × 0.031 eV) of rounding, because the total includes the atomic reference energies
   (≈ −456 000 eV). Forces and the dynamics are unaffected (R5). This is 1.4e-5 Eh per molecule in
   ⟨V̄⟩ and ⟨E_vir⟩, small against NQE shifts of ≈ 0.02 Eh per water but not negligible for
   energy differences. Options: (i) accept it and take energy observables from float64
   re-evaluation of saved frames; (ii) have the export return the interaction energy, adding E0
   back in float64 in the adapter. Option (ii) changes `export_mace.py` and P-6, and was already
   noted in the M6 spike. It was not measured here.
3. **Silent GPU-memory spill on WSL2.** A chunk that is too large does not fail; it runs up to
   21× slower. M8a cannot rely on an out-of-memory error as the signal for "chunk too large", and
   the default "all beads in one graph" would silently spill for NMA + 216 at P ≥ 10 (small).
   Options: documentation only (chunk size chosen from R3), or a startup check against free GPU
   memory (new scope). Recommendation: documentation and a recommended `chunk` in the M8b configs.
4. **M8b GPU budget**: ≈ 4 days (20 ps production) or ≈ 8 days (50 ps), with the GPU held free of
   other work.

### Problems and caveats

- GPU background from the Windows host varied between 0.46 and 1.36 GB during the spike. This
  shifts the fit limit by up to ≈ 0.8 GB, so the "largest graph" numbers carry that margin.
- The CPU cases, the 17:38 Python CPU timing and the 17:53 memory probe overlapped MD runs (logged
  where they happened). Timing repeatability on CUDA: small f32 NMA + 216 P = 8 gave 649, 608
  (non-deterministic), 487 and 468 ms (expandable) in four processes (16:54–16:57, 17:01–17:12,
  18:41–18:47). Treat single CUDA numbers as ±15–30 %; the budget uses the R1/R2 values.
- The bench truncated libtorch error messages to 300 characters during the CUDA grid, so the
  four "error" cases there show only the start of a TorchScript traceback (the later bench keeps
  the tail). They occurred at 8–17 GB of requested graph memory and are treated as "does not fit".
- The memory figures are libtorch's reserved memory read through `nvidia-smi` and checked against
  `torch.cuda.max_memory_reserved` (17:53). They include allocator slack.
- MACE-OFF23 medium and OFF24 medium were measured for cost and memory only; no MD was run with
  them.
