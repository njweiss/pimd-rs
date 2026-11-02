# Milestone 6: In-process torch backend, MACE-OFF23, gas-phase ML-RPMD

Status: **approved (2026-11-01)**: M6a done; M6b amended after the spike and approved; M6c as drafted. Three phases, run in order:

| Phase | Flow | Output |
| --- | --- | --- |
| M6a spike | experiment (WORKFLOW "Experiments") | `docs/experiments/m6-spike.md`: log, measurements, backend recommendation |
| M6b engine | A/B/C | `pimd-torch` crate, MACE adapter, `kind = "mace"` in the config |
| M6c study | experiment | `docs/experiments/m6-water-mace.md`: gas-phase ML-RPMD of water |

Crates touched in M6b: `pimd-torch` (new), `pimd-cli` (feature `torch`); `scripts/check.sh` (`CHECK_TORCH`); `validation/verify_reference_data.py` (torch generators). `pimd-potentials` is unchanged. Spec sections: 1 (U-3), 2 (D-1, D-3), 9 (P-1, P-4, P-5, new P-6), 12.

## Goal

pimd-rs evaluates a MACE model in-process, batched over beads, on CPU or GPU, so ML-RPMD runs without a socket round trip per bead. Validated against Python MACE on the same model, and against the M5 socket path with a Python MACE client. Then a small study: quantum effects on gas-phase water with MACE-OFF23, compared with the M5 GFN2-xTB results.

## Already done

M1–M5 merged; R1 on hold. Measured on this machine (2026-11-01):

- `pixi run -e ml`: PyTorch 2.13.0 (CUDA 13.0 build, `torch.cuda.is_available()` true), mace-torch 0.3.17, ASE.
- GPU: NVIDIA GeForce RTX 4060, 8 GB, driver 610.88.
- `tch` 0.26.0 (crates.io) expects exactly PyTorch 2.13.0, the version in the `ml` environment, and links against that environment's libtorch (`LIBTORCH_USE_PYTORCH=1` inside `pixi run -e ml`).
- MACE-OFF23 weights (small, medium, large) are in `ff_models/mace_off23/`, which is git-ignored (SPEC §11). The owner confirmed the licence (2026-11-01): the README may cite MACE-OFF23 provided weights and exports stay out of the repository.
- M6a spike done (`docs/experiments/m6-spike.md`, code in `spikes/m6-torch/`).

## Decisions taken (owner, 2026-11-01)

Before the spike:

1. M6 is split into spike, engine and study; the engine API is finalised after the spike.
2. Spike candidates: `tch` with a TorchScript export, against the M5 socket path with a Python MACE client as baseline. ONNX excluded; PyO3 only if `tch` fails.
3. The adapter supports float32 and float64 (D-1: f32 only inside model adapters). float64 is validated tightly, float32 to a looser stated tolerance.
4. Build isolation: new crate `pimd-torch`; `pimd-cli` gains an optional feature `torch`; the default `check.sh` never needs libtorch; `CHECK_TORCH=1` runs the torch checks inside `pixi run -e ml`. New dependency `tch` (0.26) approved.
5. Test models: a tiny randomly initialised MACE (committed generator script, fixed seed) for the fast tests; MACE-OFF23 only in long validation and the study, from an owner-supplied path (`PIMD_MACE_MODEL`).
6. Study: gas-phase water with MACE-OFF23 at 300 K, P convergence 1 … 32, quantum vs classical, comparison with M5 GFN2-xTB, throughput per bead on CPU and GPU; optionally ethanol for scaling. Numbers and plots, not a pass/fail gate.

After the spike:

7. **Backend: `tch` 0.26 + TorchScript** (SPEC §12 closed). Default float64 on CPU; float32 on CUDA is the documented fast option. Energy observables in M6c come from float64 runs (the float32 energy error, ≈ 5e-5 Eh per bead, is the size of the M5 error bars).
8. **CUDA determinism: optional, on by default.** `pimd-torch` ships a small C++ file (`setDeterministicAlgorithms`, which `tch` does not bind), compiled by `build.rs` with `g++` via `std::process::Command` (no new crate). The config key `potential.deterministic` (default `true`) switches it; with `false`, CUDA results are checked to stated tolerances only. Calling it needs one `unsafe extern "C"` block, allowed only in `pimd-torch`'s `ffi` module with a `// SAFETY:` comment; it is the only `unsafe` in the workspace.
9. **Tiny test model committed** (decision 5 refined): the tiny model's pickled file (≈ 274 kB) and its float64 export (≈ 0.37 MB) go under `crates/pimd-torch/tests/data/` with the generator, locked with the tests. They are generated from random initialisation, not third-party weights. MACE-OFF23 weights and exports are never committed.
10. **API as below** (adds `threads` and `deterministic` to `load`, `atomic_number`, `cutoff`; rule P-6; config keys `threads` and `deterministic`).

## M6a: spike (done)

Questions, measurements and results: `docs/experiments/m6-spike.md`. Code: `spikes/m6-torch/`, not a workspace member, kept as the record and not maintained.

## M6b: engine

### Build isolation

`pimd-torch` is a workspace member whose `tch` dependency is optional behind its feature `torch`. Without the feature the crate compiles to an empty library (`#![cfg(feature = "torch")]` on its modules), so `cargo test --workspace`, `clippy --workspace --all-targets` and `fmt` never touch libtorch and torch-sys's build script never runs. `pimd-cli`'s feature `torch` enables `pimd-torch/torch`. Never `--all-features` (already the rule in `check.sh`).

`pimd-torch/build.rs` acts only when `CARGO_FEATURE_TORCH` is set: rpath `$CONDA_PREFIX/lib` (pixi sets no `LD_LIBRARY_PATH`), `-Wl,--no-as-needed,-ltorch_cuda` (otherwise CUDA is not found), and compiling the C++ file (decision 8) against the environment's ATen headers. The built binary is tied to the `ml` environment's absolute path; moving the repository needs a rebuild.

### Public API (stubs; pimd_torch, feature `torch`)

```rust
/// Where the model runs.
pub enum Device { Cpu, Cuda(usize) }
/// Precision inside the model (D-1: f32 only inside the adapter; inputs and outputs are f64).
pub enum Dtype { F32, F64 }

/// Atomic number of an element symbol ("H" -> 1, ...), H through I; `None` if unknown.
pub fn atomic_number(symbol: &str) -> Option<u8>;

pub struct MacePotential { /* module, species index per atom, r_max (bohr), device, dtype */ }
impl MacePotential {
    /// Loads a TorchScript export (export_mace.py) for N = `atomic_numbers.len()` atoms, sets the
    /// libtorch intra-op thread count to `threads`, disables the TorchScript profiling executor,
    /// and sets libtorch's deterministic-algorithms switch to `deterministic` (process-wide).
    /// Errors: missing or invalid file, or a module that is not an export -> Backend;
    /// an atomic number the model does not know -> InvalidParameter { what: "atomic number" };
    /// threads == 0 -> InvalidParameter { what: "threads" }; CUDA unavailable -> Backend.
    pub fn load(path: &std::path::Path, atomic_numbers: &[u8], device: Device, dtype: Dtype,
                threads: usize, deterministic: bool) -> Result<MacePotential, PotentialError>;
    /// The model cutoff r_max in bohr.
    pub fn cutoff(&self) -> f64;
}
impl Potential for MacePotential { /* P-6 */ }
```

The thread count, the profiling executor and the deterministic switch are libtorch process-wide settings; the CLI creates one `MacePotential` per process.

Export script: `crates/pimd-torch/scripts/export_mace.py` (from the spike, unchanged in substance): `pixi run -e ml python crates/pimd-torch/scripts/export_mace.py MACE-OFF23_small.model --out small.pt`. It writes a float64 export; the adapter casts it for `Dtype::F32`. Exports of MACE-OFF23 live next to the weights (e.g. `ff_models/exports/`).

### SPEC changes (applied with this amendment's approval)

**P-6** (new):

> **P-6** `MacePotential` (crate `pimd-torch`, feature `torch`) evaluates a TorchScript MACE export (`crates/pimd-torch/scripts/export_mace.py`) in-process. One `evaluate` call evaluates all P beads as one batched graph (bead j holds atoms jN … jN+N−1). Edges: every ordered pair i ≠ j of the same bead with |q_j − q_i + s| < r_max, where s = 0 without a cell and s is the minimum-image correction of C-2 with a cell; with a cell, r_max < L_min/2 is required, otherwise `InvalidParameter { what: "cutoff", value: r_max }` (bohr). Positions are converted bohr → Å, energies eV → Eh and forces eV/Å → Eh/bohr with the `units` constants only (U-3); with `Dtype::F32` the cast to f32 happens on the way into the model and back to f64 on the way out (D-1). Energies are per bead (S-4), forces −∇V from the model's autograd (S-3). `subset = Some` → `Unsupported` (P-4); `positions.n_atoms() ≠ N` → `ShapeMismatch`; any libtorch error or a non-finite output → `Backend(String)`. `virial` is `None` (P-5). The TorchScript profiling executor is disabled and the intra-op thread count fixed at load (D-3).

**D-3** (appended):

> With the in-process torch backend (P-6) the guarantees hold only where the model evaluation is bitwise reproducible: for a fixed model file, device, dtype, libtorch intra-op thread count (`potential.threads`) and libtorch build. The adapter disables the TorchScript profiling executor, so every call, including the first one after a restart, runs the same graph. On CPU, results for a fixed thread count are bitwise reproducible, including restart. On CUDA they are bitwise reproducible only with `potential.deterministic = true` (the default), which enables libtorch's deterministic algorithms; with `false`, CUDA results, including restart, agree only to stated tolerances. Results at different libtorch thread counts, CPU vs CUDA, float32 vs float64, and batched vs per-bead evaluation agree only to stated tolerances.

**§12**: remove "In-process backend choice (`tch-rs` expected), decided by the M6 spike." (decided: `tch` 0.26 + TorchScript, MILESTONE_6 decision 7).

### Config

```toml
[system]
symbols = ["O", "H", "H"]          # required for kind = "mace"

[potential]
kind = "mace"                      # only with feature `torch`; otherwise a config error naming the feature
model = "/path/to/export.pt"       # TorchScript export (export_mace.py)
device = "cpu"                     # "cpu" | "cuda" | "cuda:N"; default "cpu"
dtype = "float64"                  # "float64" | "float32"; default "float64"
threads = 1                        # libtorch intra-op threads, ≥ 1; default 1
deterministic = true               # libtorch deterministic algorithms (bitwise CUDA); default true
```

Config hash: covers the model file by an FNV-1a hash of its bytes (the existing `fnv1a` in config.rs; not the path), the device kind (cpu vs cuda; not the index), `dtype`, `threads`, `deterministic`, and for `kind = "mace"` also `symbols` (they select the species, unlike for the socket). Validation (errors name the field as in Q15): unknown symbol, `threads = 0`, bad device string, `model`/`device`/`dtype`/`threads`/`deterministic` with another kind, missing `symbols`.

### Required tests

Tiny model; reference JSON from `crates/pimd-torch/tests/data/gen_mace_ref.py`, which evaluates the committed pickled tiny model with Python MACE and mace's own neighbour list (`pixi run -e ml`; locked like all reference data; `verify_reference_data.py` runs the torch generators with the `ml` interpreter). The committed export must be the export of the committed pickled model (`gen_tiny_mace.py` writes both).

- **Per evaluation vs Python MACE** (water, ethanol, 4-water box with boundary-crossing edges; 8 displaced beads): float64 CPU, |ΔE| ≤ 1e-12·|E| and |ΔF| ≤ 1e-10·max |F| (measured: ΔE = 0, ΔF ≤ 2e-15·max |F|); float32 vs the float64 reference, |ΔE| ≤ 5e-3 eV (converted to Eh) and |ΔF| ≤ 1e-4·max |F| (measured 1.5e-3 eV, 4e-6). CUDA rows run only if CUDA is available (skip with a message).
- **Units:** hand-converted values (eV·`1/HARTREE_IN_EV`, Å·`1/BOHR_IN_ANGSTROM`) on one bead.
- **Batching:** batched equals per-bead to 1e-12 relative (not bitwise; measured 5e-15); bead independence (moving bead k leaves the other beads' outputs unchanged to 1e-14 relative).
- **Edges:** a box geometry with unwrapped positions gives the same result as wrapped positions (1e-12 relative); r_max ≥ L_min/2 → `InvalidParameter`.
- **Errors:** `subset = Some` → `Unsupported`; wrong atom count → `ShapeMismatch`; unknown atomic number; `threads = 0`; missing file and non-TorchScript file → `Backend`.
- **Determinism:** first and second call bitwise equal (catches the profiling executor left on); 20 repeated calls bitwise equal at threads 1 and 4; CUDA with `deterministic = true`: repeated calls bitwise equal at ethanol P = 32 (288 atoms; non-deterministic without the switch in the spike); restart K + checkpoint + K vs 2K bitwise on CPU float64 and on CUDA with `deterministic = true`.
- **Config:** keys, defaults, validation, hash coverage (model bytes, dtype, device kind, threads, deterministic, symbols; not the model path or the device index).
- **Long** (`CHECK_LONG=1 CHECK_TORCH=1`, needs `PIMD_MACE_MODEL`, skips with a message otherwise): MACE-OFF23 water PIMD in-process vs the socket path with Python MACE clients (`mace_client.py` moved to `crates/pimd-torch/tests/clients/`), per evaluation (ΔE ≤ 1e-12·|E|, ΔF ≤ 1e-10·max |F|; measured ΔE bitwise, ΔF ≤ 4e-16·max |F|) and over a short trajectory with the M5 bridge's tolerances.

### `CHECK_TORCH` mechanics

```bash
if [ "${CHECK_TORCH:-0}" = "1" ]; then
  step "torch backend (pixi ml env)"
  pixi run -e ml env LIBTORCH_USE_PYTORCH=1 bash -c '
    cargo clippy -p pimd-torch -p pimd-cli --all-targets --features torch -- -D warnings &&
    RAYON_NUM_THREADS=1 cargo test -p pimd-torch -p pimd-cli --features torch &&
    RAYON_NUM_THREADS=4 cargo test -p pimd-torch -p pimd-cli --features torch'
  # with CHECK_LONG=1: --release --features torch,long-validation (MACE-OFF23 via PIMD_MACE_MODEL)
fi
```

### Clarifications (Phase A, owner 2026-11-01)

- **Config variant (Q24):** `#[cfg(feature = "torch")] PotentialConfig::Mace { model: PathBuf, device: pimd_torch::Device, dtype: pimd_torch::Dtype, threads: usize, deterministic: bool }`; `Device` and `Dtype` derive `Debug, Clone, Copy, PartialEq, Eq`; `"cuda"` is `Device::Cuda(0)`. Bad `device`, `dtype` or `symbols` strings are `InvalidValue` naming the field or `Parse` mentioning its key; `threads = 0` is `InvalidValue { what: "potential.threads" }`; without the feature, a `kind = "mace"` config is an error whose message names `torch`. Whether the model bytes are hashed at parse time (an extra field) or at hash time is the implementer's choice.
- **Build scripts (Q25):** `pimd-cli` has its own `build.rs` (std only, no-op without the feature) emitting the rpath and `--no-as-needed` link arguments, since `cargo:rustc-link-arg` does not reach dependent crates. Both `build.rs` files and `src/deterministic.cpp` are real code from Phase A; the implementer may change them.
- **Long test model (Q26):** `PIMD_MACE_MODEL` names the MACE `.model` file (e.g. `ff_models/mace_off23/MACE-OFF23_small.model`); the long test exports it into a temporary directory with `export_mace.py`, so both sides use the same weights and nothing derived is written into the repository.

### Mutant ideas

eV/Å conversion missing or inverted; forces taken as the gradient (sign); beads mixed up in the batch (ptr offsets off by one bead); edge vector q_i − q_j (sign); edge shifts with the wrong sign; float32 cast applied to the outputs only; cutoff in Å used as bohr; species index by atomic number instead of position in the model's list; profiling executor left on; `deterministic` ignored; hash missing the model bytes or covering the path.

## M6c: study

Plan first (`docs/experiments/m6-water-mace.md`), written before running: question, hypothesis, system, protocol, and what would count as agreement or disagreement with M5 GFN2-xTB and with the literature. Uses only merged engine code (WORKFLOW: the engine is not modified during a study).

Protocol (to be fixed in the plan): gas-phase water, MACE-OFF23, 300 K, BAOAB + PILE-L, P = 1, 4, 8, 16, 32; ⟨E_vir⟩, ⟨E_prim⟩, ⟨V̄⟩, bead-averaged ⟨r_OH⟩ and ⟨θ_HOH⟩, r_OH distribution; block-averaged errors (WORKFLOW policy); quantum (P = 32 or converged P) vs classical (P = 1); comparison with the M5 GFN2-xTB numbers in `validation/RESULTS.md`; throughput per bead on CPU and GPU, float32 and float64 (energy observables from float64, decision 7). Optional: ethanol for scaling. Scripts and configs committed; tables and figures regenerated by script; a short writeup whose claims cite the log.

## Out of scope

Training or fine-tuning (CLAUDE.md "Do not"); shipping MACE-OFF23 weights or exports; other ML architectures (only what the adapter needs for MACE); bead-chunked evaluation on several Rust threads (measured 10–20 % faster on CPU in the spike; an option for later); neighbour lists beyond all-pairs (M8); periodic MACE production runs (M8); dynamics (M7); electrostatics; multi-GPU.

## Acceptance

- [x] M6a: spike log and writeup committed
- [x] M6a: owner decision on the backend recorded (SPEC §12 closed); this brief amended and approved
- [ ] M6b: all tests pass; `check.sh` exits 0 (default and `CHECK_TORCH=1`), including thread-count runs
- [ ] M6b: long validation passes; agreement levels added to `validation/RESULTS.md`
- [ ] M6b: review done; no blocking findings; `QUESTIONS.md` has no open entries
- [ ] M6c: study plan, log and writeup committed; README cites MACE-OFF23 (licence confirmed 2026-11-01)
