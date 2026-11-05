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
