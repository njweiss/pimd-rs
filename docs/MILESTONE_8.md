# Milestone 8: Solvated ligand (N-methylacetamide in water), periodic MACE with larger batches

Status: **outline (2026-11-05)**. Decisions 1–4 taken; the M8a brief is written after the spike.
Three phases, run in order:

| Phase | Flow | Output |
| --- | --- | --- |
| M8 spike | experiment (WORKFLOW "Experiments") | `docs/experiments/m8-spike.md`: cost, memory and stability measurements, recommendations for M8a and M8b |
| M8a engine | A/B/C | cell-list MACE edges, bead-chunked evaluation, positions from a file |
| M8b study | experiment | `docs/experiments/m8-nma-water.md`: ML-PIMD of N-methylacetamide in water |

## Goal

ML-PIMD of a small neutral solute in explicit periodic water, entirely on MACE-OFF, at a size that
does not fit the M6 backend's all-pairs edges and single-graph evaluation. Quantum vs classical
solvation structure and solute geometry, from merged engine code only.

## Already done

M1–M7 merged. Relevant to M8:

- `MacePotential` (SPEC P-6) handles a periodic orthorhombic cell with minimum image and requires
  r_max < L_min/2; edges are built all-pairs per bead; all P beads are one graph.
- MILESTONE_6 "Out of scope" deferred to M8: neighbour lists beyond all-pairs, periodic MACE
  production runs, bead-chunked evaluation.
- `trajectory.xyz` already writes every bead of every atom (MILESTONE_5 "Trajectory output":
  P·N lines per frame, bead-major, Å, unwrapped, full `{}` precision), so the solvation analysis
  needs no engine change.
- Models on this machine, git-ignored: `ff_models/mace_off23/` (small, medium, large, 23b medium),
  `ff_models/mace_off24/` (medium), TorchScript exports in `ff_models/exports/`.
- GPU: NVIDIA GeForce RTX 4060, 8 GB.

## Decisions taken (owner, 2026-11-05)

1. **No ML/MM split.** The whole box (solute and water) is MACE-OFF. ML/MM needs electrostatic
   coupling, which the engine does not implement (CLAUDE.md "Do not"); if it is needed later, the
   MM side would sit behind the socket backend. Cost accepted: MACE-OFF has no long-range
   electrostatics beyond its cutoff, so only neutral solutes.
2. **Spike, then M8a (engine, A/B/C), then M8b (study, experiment log)**, as M6. The spike fixes N,
   P, model size and dtype before the M8a brief is finalised.
3. **Engine scope of M8a:**
   - (a) cell-list edges for `MacePotential`, giving the same edge set in the same order as the
     all-pairs build (results without the new option bitwise unchanged);
   - (b) bead-chunked evaluation (beads per graph as a config key, default all beads); chunked
     equal to unchunked to 1e-12 relative;
   - (c) positions from a file (e.g. `system.xyz_file`; contents enter the config hash);
   - (d) solvation observables (site–site RDFs, hydrogen bonds, solute geometry) by
     post-processing the full-bead `trajectory.xyz` in the study's scripts, not in the engine.
4. **Solute: N-methylacetamide (NMA)**, the standard peptide-bond model, neutral and inside
   MACE-OFF's training domain, in about 200–260 waters (box ≈ 19–20 Å), NVT at a fixed density.

## M8 spike (plan outline; the full plan goes in `docs/experiments/m8-spike.md` before measuring)

Uses merged engine code; spike code in `spikes/m8/` (not a workspace member); nothing under
`crates/` changes. Questions:

1. **Cost.** ms per evaluation of `MacePotential` against N (NMA + 64, 128, 216 waters) and P
   (1, 8, 16, 32), float32 and float64, CUDA and CPU, MACE-OFF23 small and medium (MACE-OFF24
   medium if time allows). Share of the time spent building edges all-pairs on the CPU.
2. **Memory.** Peak GPU memory against P·N per model and dtype; the largest graph that fits in
   8 GB, hence the bead chunk size M8a has to support.
3. **Stability and sanity.** Classical (P = 1) NVT of NMA in water on MACE-OFF23 for about 10 ps:
   NMA stays intact, no atoms lost, energy drift under `Off` at the chosen dt; water g_OO against
   the experimental first peak, as a sanity check only.
4. **Start structure.** A scripted pipeline in the `ref` environment (packmol or OpenMM Modeller,
   then a short classical OpenMM equilibration at the target density) producing an xyz file.
5. **Budget.** Projected wall time for the M8b production (P convergence, quantum vs classical)
   and for an M8a comparison against i-PI driving Python MACE through the socket on a small
   periodic water box.

Recommendations at the end: N, P, model, dtype, dt, chunk size, and whether (a) is needed for
speed or only for memory.

## M8a validation ladder (outline)

1. In-process `MacePotential` vs the ASE MACE calculator with periodic boundaries on solvated
   snapshots: energies and forces to 1e-10 relative in float64.
2. Cell-list edges equal all-pairs edges (set and order) on random and solvated boxes; without the
   new options, bitwise as on main.
3. Chunked equals unchunked to 1e-12 relative; bead independence as in M6.
4. pimd-rs vs i-PI + Python MACE through the socket on a small periodic water box (32–64
   molecules) at P = 1 and P = 8: ⟨E_vir⟩ and g_OO within 5σ (size fixed by the spike budget).

## M8b study (outline)

Quantum (converged P) vs classical (P = 1) NMA in water at 300 K: NMA–water site RDFs (carbonyl O
to water H, amide H to water O), hydrogen-bond distances and counts, NMA bond lengths and the amide
torsion distribution, ⟨E_vir⟩, throughput. Trajectories stay out of the repository (size: about
0.7 MB per frame at P = 16 and N ≈ 700); scripts, configs and derived tables are committed.

## Out of scope

ML/MM; electrostatics; charged solutes; NPT or barostats; dynamics (RPMD, TRPMD, PA-CMD) of the
solvated system; non-orthorhombic cells; training or fine-tuning; multi-GPU; shipping weights or
exports.
