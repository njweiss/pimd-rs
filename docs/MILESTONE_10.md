# Milestone 10a: supported MACE socket client (e3nn or cuEquivariance)

Status: **merged (2026-11-13)**; brief approved (owner, 2026-11-13); the spike's cueq requirements lock removed from the repo (decision 2). Flow A/B/C (docs/WORKFLOW.md).

Crates touched: none in library code. New script `crates/pimd-torch/scripts/mace_client.py`
(beside `export_mace.py`); tests in `crates/pimd-cli/tests/` (that crate already depends on
`pimd-socket` and `pimd-torch`); a `CHECK_CUEQ` step in `scripts/check.sh`; `.gitignore`; README;
SPEC: new rule K-7 and one D-3 sentence. The engine is unchanged. No new Rust dependencies; no
new pixi dependencies.

## Goal

Run MACE through the merged socket backend (K-1 … K-6) with a supported client, so production
studies can use NVIDIA's cuEquivariance (cueq) kernels: 4.5× faster per PIMD step at P = 32 with
four clients on the M8b box, the same float32 force errors (docs/experiments/m10-profiling-spike.md,
Q3). The client must reproduce the in-process adapter's energies and forces (P-6, v2) to stated
tolerances, using the engine's unit constants.

## Already done

- Spike (docs/experiments/m10-profiling-spike.md, `spikes/m10/cueq_client.py`): a working client;
  measured speed, memory (1.2–2.3 GB host RAM per client, K ≤ 4 on this host) and determinism
  (not bitwise repeatable on CUDA; runs agree to ≈ 1e-8 Eh). Its units come from ase (a 2e-7 Eh
  offset on the M8b box); the supported client fixes that.
- Two locked test clients show the conventions: `crates/pimd-torch/tests/clients/mace_client.py`
  (gas phase) and `crates/pimd-cli/tests/clients/mace_pbc_client.py` (orthorhombic). Both import
  the locked M5 `ipi_client.py` (`connect`, `serve`) by path, copy the U-2 constants, return a zero
  virial and keep no state between calls. They use mace's ASE calculator, which sums E0 in the
  model dtype (the M9 offset); the supported client does not.
- Spike environment: `spikes/m10/cueq/.venv` (cuequivariance 0.13.0, layered on the `ml` env with
  `--system-site-packages`).

## Decisions (owner, 2026-11-13)

1. **M10a approved** as recommended by the spike writeup.
2. **Licence.** The cueq kernel packages (`cuequivariance-ops-cu13`, `cuequivariance-ops-torch-cu13`)
   are under NVIDIA's proprietary licence; using them is fine **as long as the licence does not
   extend into the repository**. So: no cueq package in `pixi.toml` or `pixi.lock`, nothing of
   theirs vendored or committed. The client imports `cuequivariance_torch` only with
   `--backend cueq`, and the e3nn backend needs nothing beyond the `ml` env. The README gives
   install steps into a gitignored venv (`.venv-cueq/`), as it does for model weights.
3. **Reproducibility.** Agreement to stated tolerances (≈ 1e-8 Eh between repeat runs) is
   acceptable for production studies on the socket + cueq route. The in-process path stays
   bitwise reproducible (D-3).

## In scope

| ID | What |
| --- | --- |
| K-7 (new) | The supported MACE client: inputs, edges, energy, units, cells, errors, statelessness |
| D-3 (sentence) | Reproducibility status of the client route |
| docs | README: running the client, installing cueq out of the repo, the K ≤ RAM limit |
| check | `CHECK_CUEQ=1 ./scripts/check.sh` runs the cueq tests (needs the venv and a GPU) |

## SPEC text to add

**K-7** `crates/pimd-torch/scripts/mace_client.py` is an i-PI client (K-2) for a MACE model file
(the pickled `torch.nn.Module` MACE writes, not a TorchScript export). For each `POSDATA` it
evaluates one bead:
- Species: from the engine config named by `--config` (its `[system]` `symbols`, or the symbols of
  its `positions_xyz` file), in atom order; read once at start-up. An element the model does not
  know, or an atom count that differs from the config's, exits non-zero before connecting or on
  that request.
- Cell: h all zeros → gas phase; h diagonal → orthorhombic box L = diag(h); anything else exits
  non-zero (K-5 then reports `Backend` on the engine side).
- Edges: as P-6 (every ordered pair i ≠ j with |q_j − q_i + s| < r_max, minimum image of C-2 with a
  cell; with a cell, r_max < L_min/2 or exit non-zero), built in float64.
- Energy: MACE's interaction energy in the model dtype, cast to float64, plus Σ_i E0(Z_i) summed
  in float64 from the model's `atomic_energies_fn` (as the P-6 v2 adapter). Forces −∇ of the
  interaction energy by autograd.
- Units: bohr → Å, eV → Eh, eV/Å → Eh/bohr with the U-2 constants, copied (the client is external).
- Precision: `--dtype float64` (default) or `float32`; with float32 the model is cast before any
  conversion and positions enter in float32, outputs leave as float64 (D-1).
- Backend: `--backend e3nn` (default) or `cueq` (`mace.cli.convert_e3nn_cueq`, CUDA only).
- Virial: zeros (K-6; the engine reads it only for P = 1, and no estimator uses it).
- Stateless between calls; torch runs one intra-op thread.

**D-3 (added sentence)** With the MACE socket client (K-7), evaluations on CUDA are not bitwise
reproducible with either backend; such runs agree with each other and with the in-process adapter
only to stated tolerances.

## Command line

```bash
pixi run -e ml python crates/pimd-torch/scripts/mace_client.py MODEL --config CONFIG.toml \
    --address unix:/path [--backend e3nn|cueq] [--dtype float64|float32] [--device cpu|cuda] \
    [--patience 120]
# cueq: same command with .venv-cueq/bin/python in place of python
```

The engine side is the existing `kind = "socket"` potential with `n_clients = K`; start K clients.

## Required tests (Phase A; `crates/pimd-cli/tests/`, feature `torch`, run by `CHECK_TORCH=1`)

Each test starts the client(s) from the `ml` env on the committed `tiny_mace.model` (or, for the
long tests, MACE-OFF23 small) and drives them through `SocketPotential`; the expectation is
`MacePotential` on the v2 export of the same model (`tiny_mace_v2_f64.pt`), evaluated in-process.
Tolerances are the test author's, each with a derivation or measurement in a comment (WORKFLOW
tolerance policy).

1. K-7, U-2: e3nn, float64, CPU, gas phase and an orthorhombic box, P = 4, K = 1 and K = 2: energies
   and forces equal the adapter's to near round-off (edge order differs; state the bound).
2. K-7, D-1: e3nn, float32, CPU: equal to the adapter at `Dtype::F32` to float32 tolerance; the
   energy is not offset (Σ E0 in float64: use `tiny_mace_bige0_v2_f64.pt`'s model or an equivalent
   large-E0 check, so a float32 E0 sum would fail).
3. K-7: the U-2 constants in the script equal `pimd_core::units` (parsed from the script, compared
   bitwise).
4. K-7, K-5: errors reach the engine as `Backend`: non-diagonal cell; r_max ≥ L_min/2; atom count
   mismatch; unknown element in the config (client exits before connecting → bind/accept timeout
   or `Backend`; state which).
5. K-7, D-3: statelessness: the same positions after different ones give bitwise the same result
   (CPU, float64).
6. `CHECK_CUEQ=1` only (env `CUEQ_PYTHON` names the venv's python; GPU): cueq float64 CUDA equals
   the adapter float64 to near round-off; cueq float32 CUDA equals the adapter float32 CUDA to
   float32 tolerance.
7. Long (`long-validation`, skipped with a message when `ff_models/` lacks the model): MACE-OFF23
   small on the M8b start box, P = 4, K = 2, e3nn float32 CUDA, and cueq when `CHECK_CUEQ=1`, against
   the in-process float32 adapter.

Mutants (at least): ase constants in place of U-2; Σ E0 in float32; E0 omitted; minimum image
dropped; `<=` for the cutoff; species order from `atomic_numbers` mixed up; virial-path or cell
transposed for a non-cubic orthorhombic box; state cached between calls.

## Acceptance

- [x] Tests 1–5 and 7 pass under `CHECK_TORCH=1 CHECK_LONG=1`; test 6 under `CHECK_CUEQ=1`.
- [x] No cueq package in `pixi.toml` / `pixi.lock`; `.venv-cueq/` gitignored; README install steps.
- [x] Measurement (appended to docs/experiments/m10-profiling-spike.md): the spike's Q3 (v) run at
      P = 32, K = 4 with the supported cueq client is within 10 % of 423.5 ms per step, and its
      energies agree with the engine's in-process run to the test 7 tolerance.
- [x] SPEC K-7 and the D-3 sentence added; CLAUDE.md status row.

## Out of scope

A batched socket request (several beads per message; spike recommendation 2), pyo3 embedding,
non-orthorhombic cells, real virials, starting clients from the engine.

## Clarifications (after Phase A, 2026-11-13)

- **Q41.** Test 6 uses `tiny_mace_cueq.model` (the tiny model with `use_reduced_cg = False`): mace's
  cueq converter trusts that flag, and models built without cuequivariance installed carry a wrong
  `True`. The README notes this failure mode (einsum size error) and its cause.
- **Q42.** Accepted as pinned by the tests. Item 4: a relative `positions_xyz` in the config is
  resolved as the engine resolves it (against the working directory); K-7 says so.
- The measurement (Acceptance) uses `.venv-cueq/` created by the README's own install steps.

## After review (2026-11-13)

Phase C found nothing blocking. Fixed before merge: the client refuses multi-head models, as the
v2 export does (K-7 "as the P-6 v2 adapter"); config errors (no `[system]`, malformed or short xyz,
no atoms) exit with a message instead of a traceback; the README says the cuEquivariance 0.13.0 pin
is deliberate; Q41 and Q42 moved to the resolved index. Not changed: a NaN position yields finite
results, as in the in-process adapter (pre-existing, P-6); dense O(N²) distance tensors (fine at
N = 660).
