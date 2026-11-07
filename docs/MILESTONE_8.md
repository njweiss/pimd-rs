# Milestone 8: Solvated ligand (N-methylacetamide in water), periodic MACE with larger batches

Status: **M8a merged (2026-11-07)** (brief approved 2026-11-05; SPEC P-6 and D-3 amended). M8b study next. Spike done; decisions 1–8
taken. Three phases, run in order:

| Phase | Flow | Output |
| --- | --- | --- |
| M8 spike (done) | experiment (WORKFLOW "Experiments") | `docs/experiments/m8-spike.md`: cost, memory and stability measurements, recommendations |
| M8a engine | A/B/C | bead-chunked MACE evaluation, positions from an xyz file |
| M8b study | experiment | `docs/experiments/m8-nma-water.md`: ML-PIMD of N-methylacetamide in water |

Crates touched in M8a: `pimd-torch` (chunked evaluation), `pimd-cli` (config keys, xyz input).
Spec sections: 9 (P-6 amended), 2 (D-3 sentence). `pimd-potentials`, `pimd-integrators` and
`pimd-estimators` are unchanged.

## Goal

ML-PIMD of a small neutral solute in explicit periodic water, entirely on MACE-OFF, at a size whose
all-beads graph does not fit the GPU. Quantum vs classical solvation structure and solute
geometry, from merged engine code only.

## Already done

M1–M7 merged. Relevant to M8:

- `MacePotential` (SPEC P-6) handles a periodic orthorhombic cell with minimum image and requires
  r_max < L_min/2; edges are built all-pairs per bead; all P beads are one graph.
- `trajectory.xyz` writes every bead of every atom (MILESTONE_5 "Trajectory output": P·N lines
  per frame, bead-major, Å, unwrapped, full `{}` precision), so the solvation analysis needs no
  engine change.
- Models on this machine, git-ignored: `ff_models/mace_off23/` (small, medium, large, 23b medium),
  `ff_models/mace_off24/` (medium); float64 exports of OFF23 small and medium and OFF24 medium in
  `ff_models/exports/m8-*-f64.pt`.
- GPU: NVIDIA GeForce RTX 4060, 8 GB VRAM; WSL2 with about 11 GiB RAM.
- **Spike** (`docs/experiments/m8-spike.md`, code in `spikes/m8/`, commits 3fe3529 and 4eaa280),
  NMA + 216 waters (660 atoms, L = 18.760 Å), MACE-OFF23 small:
  - float32 CUDA 76 ms per evaluation at P = 1, 487 ms at P = 8, ≈ 2.0 s at P = 32 in chunks of
    8 beads; float64 is 6–7× slower, CPU 14–20× slower in float32.
  - GPU memory is ≈ 31 kB per edge (small, float32), 25 k edges per bead: ≈ 6 GB usable for the
    graph, so at most 8 beads per graph (10 with `expandable_segments`). On WSL2 a larger graph
    usually does not fail: it spills into shared system memory and runs 2–21× slower.
  - All-pairs edge building is ≤ 2.2 % of an evaluation at P ≥ 8; a cell list gives no gain at
    this size, in speed or memory.
  - 10 ps classical NVT at 0.5 fs (float32 CUDA) is stable: NMA intact and trans, T = 298 ± 9 K,
    g_OO first peak 2.78 Å. NVE drift in float64 ≈ 2–5e-7 Eh per atom per ps.
  - Start boxes from `spikes/m8/build_box.py` (OpenMM 8.6.1, `ref` env, ff14SB ACE–NME = NMA,
    TIP3P, 0.997 g/cm³, 50 ps classical NVT): `spikes/m8/data/nma_w216.xyz`,
    `spikes/m8/data/water_w64.xyz` (L = 12.43 Å, the smallest box with r_max < L/2).

## Decisions taken (owner, 2026-11-05)

Before the spike:

1. **No ML/MM split.** The whole box is MACE-OFF. ML/MM needs electrostatic coupling, which the
   engine does not implement (CLAUDE.md "Do not"); if needed later, the MM side would sit behind
   the socket backend. Only neutral solutes (MACE-OFF has no long-range electrostatics).
2. **Spike, then M8a (engine, A/B/C), then M8b (study)**, as M6.
3. **Engine scope:** bead-chunked evaluation, positions from a file; solvation observables by
   post-processing the full-bead `trajectory.xyz` in the study's scripts, not in the engine.
   (Cell-list edges were also listed; dropped by decision 5.)
4. **Solute: N-methylacetamide (NMA)** in about 200–260 waters, NVT at a fixed density.

After the spike:

5. **No cell-list edges.** All-pairs edges stay (≤ 2.2 % of an evaluation; GPU memory is the
   model's per-edge activations, independent of how edges are built).
6. **float32 energies are not changed in the engine.** A float32 bead energy carries ≈ 3e-3 Eh of
   rounding (the total includes the atomic reference energies), about 10× below the thermal spread
   of one sample's energy. M8b checks for bias by re-evaluating a subset of saved frames in float64
   with Python MACE; only if that shows a bias does an export change (interaction energy in
   float32, reference energies added in float64) become a follow-up.
7. **Silent spill:** no free-memory check. Safe chunk sizes are documented, every M8b config sets
   one explicitly, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is documented (an
   environment variable; the engine does not set it). The default stays one graph of all beads.
8. **M8b production: 20 ps per run first** (≈ 4 days of GPU time for P = 1, 8, 16, 32 and a second
   P = 32 seed), extended from checkpoints if the error bars need it; the GPU is kept free of other
   work during the runs.

## M8a: engine

### In scope

| ID | What |
| --- | --- |
| P-6 (amended) | bead-chunked evaluation: consecutive chunks of c beads, each one batched graph |
| D-3 (sentence) | results depend on c; c is part of the config hash |
| config | `potential.beads_per_graph`; `system.positions_xyz` |

### Public API (stubs; `pimd_torch`, feature `torch`)

```rust
impl MacePotential {
    /// Evaluates the P beads in consecutive chunks of `c` beads (the last chunk may be smaller),
    /// each chunk one batched graph, chunks in ascending bead order (P-6). Without this call,
    /// one graph of all beads (as before).
    /// Errors: c == 0 -> InvalidParameter { what: "beads per graph", value: 0.0 }.
    pub fn with_beads_per_graph(self, c: usize) -> Result<MacePotential, PotentialError>;
    /// The chunk size set by `with_beads_per_graph`, or `None` (all beads in one graph).
    pub fn beads_per_graph(&self) -> Option<usize>;
}
```

`PotentialConfig::Mace` gains `beads_per_graph: Option<usize>`. `SystemConfig` is unchanged:
`positions_xyz` fills `positions` and `symbols`.

### SPEC changes (applied with this brief's approval)

**P-6**, the second sentence ("One `evaluate` call evaluates all P beads as one batched graph
(bead j holds atoms jN … jN+N−1).") becomes:

> One `evaluate` call evaluates the P beads in consecutive chunks of c beads (default c = P; the
> last chunk holds the remaining P mod c beads), chunks in ascending bead order; each chunk is one
> batched graph in which its b-th bead holds atoms bN … bN+N−1. Energies and forces are stored by
> bead index.

**D-3**, appended to the torch paragraph:

> Results for different chunk sizes (`potential.beads_per_graph`) agree only to stated tolerances;
> the chunk size is part of the config hash.

### Config

```toml
[system]
n_atoms = 660
masses_amu = [...]                  # explicit, as before (isotopes stay under user control)
box_angstrom = 18.760
positions_xyz = "nma_w216.xyz"      # positions (Å) and symbols from an xyz file

[potential]
kind = "mace"
model = "/path/to/export.pt"
beads_per_graph = 8                 # optional, ≥ 1; default: all beads in one graph
```

- **`positions_xyz`.** Standard xyz: line 1 the atom count, line 2 a comment (ignored), then one
  line per atom `symbol x y z` in Å (extra columns are an error). Read at parse time; the path is
  taken as given (as `potential.model`). It replaces both `positions_angstrom` and `symbols`: either
  of them alongside it is `Parse` naming the key. A file that cannot be read is `CliError::Io`
  naming `system.positions_xyz` and the path (the Q24 precedent for `potential.model`). Malformed
  contents, a count line ≠ the number of atom lines, a count ≠ `n_atoms`, or a non-finite
  coordinate are `InvalidValue("system.positions_xyz")`. Symbols follow the existing rules
  (non-empty, no whitespace; Q22) and, for kind = "mace", must be known to the model.
- **Hash.** The parsed positions and symbols enter the hash exactly as if given inline, so a config
  with `positions_xyz` hashes as its inline equivalent (not the path). Runs of the two are
  byte-identical.
- **`beads_per_graph`.** Only with kind = "mace" (with another kind: `Parse`); 0 is
  `InvalidValue("potential.beads_per_graph")`. The hash appends the effective chunk size
  min(c, P) only when it is < P, so configs without the key, or with c ≥ P, hash as on main.
- **`run.log`** echoes the chunk size when set.
- Checkpoint layout unchanged.
- Documentation (README and the `PotentialConfig::Mace` doc): safe chunk sizes from the spike for
  an 8 GB card (≈ 190 k edges per graph for small float32, ≈ 95 k small float64, ≈ 60 k medium
  float32, ≈ 30 k medium float64; one 660-atom water box bead has ≈ 25 k edges with small); that
  on WSL2 an oversized graph slows down instead of failing; `expandable_segments`.

### Required tests

Deterministic (tiny model, CPU float64, `CHECK_TORCH=1`):

1. **Chunk equals separate calls (P-6).** For P = 5 and c = 1, 2, 3, 5, 7 (gas phase and the
   4-water box): each chunk's energies and forces are bitwise equal to evaluating that chunk's
   beads alone as a P = b `Beads`; c ≥ P bitwise equals no chunking. Different c agree to 1e-12
   relative.
2. **Bead mapping.** Moving bead k changes only bead k's outputs (1e-14 relative) for every c,
   including k in the last, short chunk.
3. **Determinism (D-3).** Repeated chunked calls bitwise equal at threads 1 and 4; first and
   second call equal; a P change between calls (chunk sizes change) gives correct results.
4. **Errors.** c = 0 → `InvalidParameter`; the existing P-6 errors unchanged with chunking.
5. **Config.** `beads_per_graph` parse, defaults, rejection with other kinds and at 0; hash: no key
   and c ≥ P hash as on main (value recorded with main's code), c < P changes it.
   `positions_xyz`: hand-written files (good, wrong count, extra column, non-finite, unknown
   symbol, missing file), exclusivity with `positions_angstrom` and `symbols`, Å → bohr through
   `units`, hash equal to the inline equivalent.
6. **Run.** A tiny-model run (P = 4, c = 3, positions from an xyz file) is byte-identical across
   repeats and to the inline-positions config; restart at K + K equals 2K bitwise; `run.log`
   shows the chunk size.

Long (`CHECK_LONG=1 CHECK_TORCH=1`, needs `PIMD_MACE_MODEL`, skips with a message otherwise):

7. **Periodic MACE-OFF23 vs Python MACE.** NMA + 216 waters (the spike box, regenerated by a
   committed generator), P = 4 displaced beads: in-process float64 CPU vs Python MACE through ASE
   with periodic boundaries and mace's own neighbour list, |ΔE| ≤ 1e-12·|E|, |ΔF| ≤ 1e-10·max |F|;
   float32 CUDA (chunks of 2) vs that reference to the M6 float32 tolerances. CUDA rows skip
   without CUDA.
8. **vs i-PI.** The 64-water box, MACE-OFF23 small, 300 K, P = 1 and P = 8: pimd-rs in-process
   (float32 CUDA, chunked) against i-PI driving one Python MACE client (float32 CUDA) through the
   socket, ⟨E_vir⟩ and g_OO within 5σ_combined, bias allowance 0 (same splitting and dt). Run
   lengths set so that σ is useful; spike budget ≈ 2 h for pimd-rs and ≈ 4 h for the i-PI
   generator (registered as slow; CHECK_IPI only).

Reference data: the boxes and the snapshot JSON come from generators next to them (the box
builder ported from `spikes/m8/build_box.py`, `ref` env; the Python MACE reference in the `ml`
env). If OpenMM's box is not bitwise reproducible across runs, the generator check compares
coordinates to a stated tolerance; the test author decides and records it.

### Mutant ideas

Chunk offsets off by one bead; the last short chunk dropped or padded with stale beads; chunk
outputs written to the wrong beads; per-P tensor cache reused for a chunk of a different size;
`beads_per_graph` ignored (one graph); hash including c ≥ P or the xyz path; xyz positions not
converted from Å; xyz symbols ignored in favour of an `X` default; atom order permuted on read;
count line ignored.

### Clarifications (owner, 2026-11-06; Q34–Q37, test commit ea53edc)

- **Test 7 float32 energy tolerance (Q34):** |ΔE| ≤ 0.25 eV per bead (8 float32 ulp at
  |E| ≈ 4.56e5 eV; the M6 5e-3 eV is below float32 resolution for 660 atoms; measured 0.106 eV).
  The force tolerance stays at the M6 1e-4·max |F|.
- **Boxes (Q35):** `gen_m8_boxes.py` is bitwise reproducible (Python `random` seeded before
  `addHydrogens`, OpenMM CPU platform, 1 thread, deterministic forces) and is checked with the usual
  rtol 1e-9 / atol 1e-12. `crates/pimd-cli/tests/data/m8_boxes.json` supersedes
  `spikes/m8/data/*.xyz`, which cannot be regenerated; M8b starts from its `nma_w216` box.
- **Test 8 (Q36):** periodic Python client `crates/pimd-cli/tests/clients/mace_pbc_client.py`
  (deterministic torch algorithms, profiling executor off, started with `pixi run -e ml python`);
  i-PI cell as a diagonal matrix; PILE-L τ₀ = 100 fs, dt 0.5 fs, 2.5 ps equilibration, production
  20 ps at P = 1 and 15 ps at P = 8, engine P = 8 with c = 3; σ = the largest block standard error
  over 0.5, 1 and 2 ps blocks; bias allowance 0. `gen_ipi_water_mace.py` is slow (CHECK_IPI,
  ≈ 3.3–4 h); generators needing the model run only with `PIMD_MACE_MODEL`; the engine side runs in
  the CHECK_LONG + CHECK_TORCH step (≈ 2 h). MACE-OFF23 small has r_max = 4.5 Å.
- **Config details (Q37):** xyz columns split on any run of spaces; the count line is a
  non-negative integer alone on its line; empty, comma-separated or symbol-less atom lines are
  malformed; a directory as `positions_xyz` is `CliError::Io`; an unknown symbol with kind "mace"
  may name `system.symbols` or `system.positions_xyz`; a negative, fractional, string or boolean
  `beads_per_graph` is `Parse` naming the key or `InvalidValue("potential.beads_per_graph")`;
  configs without the key (or with c ≥ P) hash as on main at 3e6a5bd (values recorded in the
  test); run.log mentions the chunk size. Test 6 asserts that a c = 3 run's trajectory differs
  from the unchunked run's (true for this libtorch build; revisit if a libtorch update makes them
  bitwise equal).

### Out of scope (M8a)

Cell-list edges (decision 5); a free-memory check or automatic chunk size (decision 7); exporting
the interaction energy (decision 6, conditional follow-up); masses from symbols; non-orthorhombic
cells; other file formats (PDB, extended xyz); setting `expandable_segments` from the engine.

## M8b study (outline; the plan goes in `docs/experiments/m8-nma-water.md` before running)

NMA + 216 waters, MACE-OFF23 small, float32 CUDA with an explicit `beads_per_graph`, 300 K,
dt 0.5 fs (P ≥ 8), P = 1, 8, 16, 32 and a second P = 32 seed, 10 ps equilibration and 20 ps
production per run (decision 8). Observables by post-processing the full-bead trajectories:
NMA–water site RDFs (carbonyl O to water H, amide H to water O), hydrogen-bond distances and
counts, NMA bond lengths and the amide torsion distribution, ⟨E_vir⟩ and throughput. A float64
re-evaluation of a subset of frames with Python MACE checks the float32 energies for bias
(decision 6). Optionally one classical P = 1 run with MACE-OFF23 medium as a model check.
Trajectories stay out of the repository (≈ 0.7 MB per frame at P = 16); scripts, configs and
derived tables are committed.

## Out of scope

ML/MM; electrostatics; charged solutes; NPT or barostats; dynamics (RPMD, TRPMD, PA-CMD) of the
solvated system; non-orthorhombic cells; training or fine-tuning; multi-GPU; shipping weights or
exports.

## Acceptance

- [x] Spike: plan, log and recommendations committed
- [x] Owner approves the M8a brief, the P-6 and D-3 changes and the public API
- [ ] M8a: all tests pass; `check.sh` and `CHECK_TORCH=1` exit 0; long validation passes; review
      done with no blocking findings; `QUESTIONS.md` has no open entries
- [ ] M8b: plan committed before running; log, results and writeup committed
