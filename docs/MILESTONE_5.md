# Milestone 5: Socket backend and GFN2-xTB

Crates touched: `pimd-socket` (new), `pimd-potentials` (`PotentialError::Backend`), `pimd-integrators` (`RingPolymerSim::from_parts`), `pimd-cli` (socket potential, molecular config, gas phase, trajectory output, restart without re-evaluation), plus validation clients and scripts. Spec sections: 2 (D-3), 9 (P-1, P-4, P-5), 9a (K-1 … K-6).

## Goal

pimd-rs drives any i-PI-protocol force client over a socket, so external codes and Python models plug in without touching the engine. Validated by a bridge test (Lennard-Jones served from Python equals the in-process LJ) and by gas-phase water on GFN2-xTB against i-PI driving the same client.

## Already done

M1–M4 merged; R1 on hold. LAMMPS 2025.07.22, i-PI 3.3.1, tblite (Python, `tblite.interface`) and ASE in `pixi run -e ref`. Measured on this machine: a GFN2-xTB single point on water takes about 0.3 ms from scratch, single-threaded; with OpenMP threads under background load it can take seconds, so clients run single-threaded.

## In scope

| ID | Item |
| --- | --- |
| K-1 … K-6 | `SocketPotential`: i-PI protocol server, bead assignment, errors, cell and virial conventions |
| P-1, P-4, P-5 | `SocketPotential` as a `Potential` (shape errors, `subset`, virial) |
| D-3 | Determinism with deterministic, stateless clients; thread-count independence |
| (Q18) | Restart restores saved forces and energies instead of re-evaluating |
| (new) | Molecular config (per-atom masses, explicit positions, optional box), gas phase, trajectory output |
| (new) | Validation clients (Python LJ and GFN2-xTB), bridge test, water/xTB vs i-PI |

## Public API (stubs)

```rust
// pimd_potentials::PotentialError (addition)
Backend(String),                       // K-5: external backend failure (I/O, protocol, timeout)

// pimd_socket (new crate)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Address { Unix(std::path::PathBuf), Tcp(String /* host:port */) }
impl std::str::FromStr for Address { type Err = PotentialError; }   // "unix:/path" or "host:port"

pub struct SocketPotential { /* listener, clients, n_atoms, timeout */ }
impl SocketPotential {
    /// Binds `address`, then waits up to `timeout` for `n_clients` clients (K-1).
    pub fn bind_and_accept(address: &Address, n_atoms: usize, n_clients: usize,
        timeout: std::time::Duration) -> Result<SocketPotential, PotentialError>;
}
impl Potential for SocketPotential { /* K-2 … K-6 */ }
impl Drop for SocketPotential { /* EXIT to every client; remove own Unix socket file */ }

// pimd_integrators (addition, Q18)
impl RingPolymerSim {
    /// As `new`, but takes the forces and energies at `state.q` instead of evaluating them
    /// (restart). Validates masses (S-2) and shapes (P-1) like `new`; no `evaluate` call.
    pub fn from_parts(state: RingState, cell: Option<Cell>, integrator: RingPolymerIntegrator,
        potential: Box<dyn Potential>, forces: Beads, energies: Vec<f64>)
        -> Result<Self, IntegratorError>;
}
```

`pimd_cli::Config` changes (public fields, Q15 layout extended): `SystemConfig { n_atoms, masses: Vec<f64> /* m_e */, cell: Option<Cell>, positions: Option<Vec<[f64; 3]>> /* bohr */, symbols: Option<Vec<String>> }`; `potential: PotentialConfig` with `PotentialConfig::Lj(LjParams)` and `PotentialConfig::Socket { address: Address, n_clients: usize, timeout_s: f64 }`; `OutputConfig` gains `trajectory_every: Option<usize>`. `CliError` gains no variant (socket failures arrive as `CliError::Potential(PotentialError::Backend(..))`).

Allowed new dependencies: none for the Rust crates (std sockets and threads only). Python validation clients use only what `pixi -e ref` has (numpy, tblite).

## Config format (additions)

```toml
[system]
n_atoms = 3
masses_amu = [15.999, 1.008, 1.008]       # or mass_amu = 20.1797 for all atoms (exactly one of the two)
symbols = ["O", "H", "H"]                  # optional, output only
positions_angstrom = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]  # optional
# box_angstrom optional: absent = gas phase (C-1 None); then positions_angstrom is required

[potential]
kind = "socket"
address = "unix:/tmp/pimd-water.sock"      # or "localhost:31415"
n_clients = 4
timeout_s = 60.0

[output]
trajectory_every = 10                      # optional
```

## Clarifications

- **Restart (Q18, owner decision):** `run_from_checkpoint` restores the saved forces and energies with `RingPolymerSim::from_parts`; the M4 bitwise re-evaluation check is removed. The checkpoint format is unchanged (version 2). Restart is bitwise for in-process potentials and for socket clients that are deterministic and stateless (SPEC D-3 note). M4 tests that pinned the re-evaluation check are updated in Phase A.
- **Config hash:** covers the new `[system]` fields (masses, cell presence and lengths, initial positions) and the potential kind and its physical parameters; it excludes `symbols`, the socket `address`, `n_clients` and `timeout_s` (connection details that do not change results), besides `n_steps` and `[output]` as in M4.
- **Validation of the config:** exactly one of `mass_amu`/`masses_amu`; `masses_amu`, `positions_angstrom` and `symbols` have length `n_atoms`; gas phase requires `positions_angstrom`; RDF settings require a box; `n_clients ≥ 1`, `timeout_s > 0`; `address` must parse. Errors name the field as in Q15 (`InvalidValue { what: "section.key" }`, or `Parse` for type and syntax errors).
- **Initial state:** with `positions_angstrom`, every bead of atom i starts at that position; otherwise the M4 lattice. Momenta and draw order as in M4.
- **Trajectory output:** `trajectory.xyz`, one frame at every production step s with s % `trajectory_every` == 0, independent of `stride` (Q19): line 1 `P·N`, line 2 `step=<s> beads=<P> atoms=<N> units=angstrom`, then P·N lines `symbol x y z` (bead-major, symbol `X` without `symbols`), positions unwrapped, written with `{}` for f64. After a restart it holds only frames after the checkpoint, like `observables.csv`.
- **Socket lifecycle:** clients connect after the server binds and retry until it accepts; the server waits up to `timeout_s` for all `n_clients`. Every read and write uses `timeout_s`. The binary never starts clients. Results depend only on what the clients return, not on which client finishes first (K-4).
- **Validation clients (locked with the tests):** `crates/pimd-socket/tests/clients/ipi_client.py`, a minimal i-PI protocol client written from K-2 in Python/numpy, atomic units throughout, single-threaded, stateless per call, with backends `lj` (P-3 formula, shift option, open or orthorhombic minimum image) and `xtb` (GFN2-xTB via `tblite.interface.Calculator`, a fresh single point every call, `OMP_NUM_THREADS=1`, forces = −gradient). Used by the engine's tests and by the i-PI reference generator alike.

## Required tests

Cargo tests (fast; no Python):

- **Protocol (K-2, K-3):** a mock client written in Rust inside the test (its own encoder/decoder from K-2) checks byte-exact `POSDATA` (cell, inverse, positions) for a periodic and a gas-phase call, returns hand-made forces, energies and virial, and the `Evaluation` equals them bitwise.
- **Bead assignment (K-4):** with 1, 2 and 3 mock clients and P = 1, 2, 5, 8 the mock records which beads each client saw (j mod n_clients, ascending); results are bitwise identical for any n_clients and when clients answer with randomised delays; `NEEDINIT` is answered with `INIT` (bead index, empty string).
- **Errors (K-5):** wrong atom count, NaN force, unexpected header, client disconnect mid-message, no client within the timeout, `subset = Some` → the documented `PotentialError` variant; lowest failing bead index reported; `Drop` sends `EXIT` and removes the socket file.
- **Address parsing:** `unix:/path`, `host:port`, malformed strings.
- **`from_parts` (Q18):** equals `new` bitwise over a run when given the forces and energies `new` computed; validates masses and shapes; performs no `evaluate` (counting potential).
- **Config:** molecular fields, gas phase, socket section, every validation rule above, hash coverage (included fields change it, excluded ones do not).
- **Restart with a mock client:** K + checkpoint + K equals 2K bitwise, socket potential with a deterministic stateless mock; gas phase.
- **Trajectory output:** format, recording rule, restart concatenation.

Long validation (`long-validation`; needs `pixi run -e ref`):

- **Bridge:** the Python `lj` client against the in-process `LennardJones` on the M4 neon system (P = 1 and P = 4, periodic) and a gas-phase cluster: per-evaluation energies and forces agree to 1e-12 relative over a 200-step trajectory driven by the in-process forces; a 50-step PIMD run through the socket matches the in-process run to a stated tolerance (chaos horizon documented).
- **Water on GFN2-xTB vs i-PI (feature `socket-validation`, run only with `CHECK_LONG=1 CHECK_IPI=1`):** gas-phase water, 300 K, P = 16, PILE-L (`pile_lambda = 1`), same `xtb` client for both codes. Compare ⟨E_vir⟩, ⟨E_prim⟩, ⟨V̄⟩, the bead-averaged ⟨r_OH⟩ and ⟨θ_HOH⟩, and the r_OH histogram within 5σ_combined (block averaged on both sides, WORKFLOW policy). Reference: `crates/pimd-cli/tests/data/gen_ipi_water_xtb.py` (accepts `--out`) → `ipi_water_xtb.json`, bit-reproducible, added to the slow set of `validation/verify_reference_data.py`. Budget: about 1 hour per side on this machine; the author states the run times.

## Mutant ideas (physics-bug set from WORKFLOW.md applies)

Header not padded to 12 bytes; cell sent as h⁻¹ or transposed for a non-cubic box; forces taken as the gradient (sign); bead assignment by finish order; `NEEDINIT` ignored; atom count not checked; timeout not applied; restart re-evaluating instead of restoring; hash including `address` or `n_clients`; gas phase sending a cell; trajectory written in bohr.

## Out of scope

Starting or supervising clients from the engine; non-orthorhombic cells; i-PI's batched protocol extension; ML potentials (M6); dynamics (M7); electrostatics.

## Acceptance

- [ ] All tests pass; `check.sh` exits 0, including thread-count runs
- [ ] Bridge and water/xTB validation pass; agreement levels added to `validation/RESULTS.md`
- [ ] Review done; no blocking findings; `QUESTIONS.md` has no open entries
