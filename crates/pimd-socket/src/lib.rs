//! i-PI protocol socket backend (SPEC §9a, K-1 … K-6): `SocketPotential` acts as the i-PI
//! *server*, sends bead positions to external force-provider clients and receives their energies,
//! forces and virials. All quantities on the wire are Hartree atomic units (U-1), so the engine
//! converts nothing. The engine never starts a client or knows its model (P-1, K-1).

use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use pimd_core::{Beads, Cell};
use pimd_potentials::{Evaluation, Potential, PotentialError};

/// Where the server listens: a Unix-domain socket path or a TCP `host:port`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Address {
    /// Unix-domain socket at this path (`"unix:/path"`).
    Unix(PathBuf),
    /// TCP socket, `"host:port"` as written.
    Tcp(String),
}

impl FromStr for Address {
    type Err = PotentialError;

    /// Parses `"unix:/path"` into `Address::Unix` and `"host:port"` into `Address::Tcp`.
    ///
    /// Errors: a `PotentialError` for malformed strings (empty path, missing or invalid port).
    fn from_str(_s: &str) -> Result<Address, PotentialError> {
        unimplemented!("M5: Address::from_str")
    }
}

/// An i-PI protocol server serving `n_clients` connected force clients (K-1).
pub struct SocketPotential {
    _private: (),
}

impl SocketPotential {
    /// Binds `address`, then waits up to `timeout` for `n_clients` clients (K-1), numbered in
    /// accept order (K-4). `n_atoms` is the atom count N every client must return (K-5).
    /// `timeout` also bounds every later read and write.
    ///
    /// Errors: `PotentialError::Backend` if binding fails or fewer than `n_clients` clients
    /// connect within `timeout`.
    pub fn bind_and_accept(
        _address: &Address,
        _n_atoms: usize,
        _n_clients: usize,
        _timeout: Duration,
    ) -> Result<SocketPotential, PotentialError> {
        unimplemented!("M5: SocketPotential::bind_and_accept")
    }
}

impl Potential for SocketPotential {
    /// Sends bead j (positions `[P, N, 3]`, bohr) to client j mod n_clients, beads in ascending
    /// order per client, clients concurrently (K-4); cell as in K-3. Returns energies (Eh) and
    /// forces (Eh/bohr) stored by bead index; `virial` is the client's for P = 1, else `None`
    /// (K-6).
    ///
    /// Errors: `ShapeMismatch` if `positions.n_atoms()` differs from N (P-1); `Unsupported` for
    /// `subset = Some` (P-4); `Backend` for protocol failures, the lowest failing bead's (K-5).
    fn evaluate(
        &mut self,
        _positions: &Beads,
        _cell: Option<&Cell>,
        _subset: Option<&[usize]>,
    ) -> Result<Evaluation, PotentialError> {
        unimplemented!("M5: SocketPotential::evaluate")
    }
}

impl Drop for SocketPotential {
    /// Sends `EXIT` to every client and removes the Unix socket file this potential created (K-6).
    fn drop(&mut self) {
        unimplemented!("M5: SocketPotential::drop")
    }
}
