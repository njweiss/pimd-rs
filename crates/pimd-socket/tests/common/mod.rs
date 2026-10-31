//! Helpers for the `pimd-socket` tests (MILESTONE_5 "Required tests"). The mock client lives in
//! `mock.rs`; everything here is written from SPEC K-1 … K-6 and P-1, independently of the
//! server under test.
#![allow(dead_code)]

pub mod mock;

use std::sync::Arc;
use std::time::Duration;

use mock::{Action, Endpoint, Model, Reply, Request};
use pimd_core::Beads;
use pimd_socket::{Address, SocketPotential};

/// Generous timeout for tests that are not about timeouts (slow, loaded machines).
pub const PATIENT: Duration = Duration::from_secs(20);

/// The server-side address for a mock endpoint (built directly, not parsed).
pub fn address(ep: &Endpoint) -> Address {
    match ep {
        Endpoint::Unix(p) => Address::Unix(p.clone()),
        Endpoint::Tcp(a) => Address::Tcp(a.clone()),
    }
}

/// Bead-coded positions: q_{j,i,d} = 0.75 (j + 1) − 0.125 i + 0.0625 d + 0.001 j i, distinct for
/// every bead, so a mock can tell from the positions which bead it was sent.
pub fn coded_positions(p: usize, n: usize) -> Beads {
    let mut q = Beads::zeros(p, n);
    for j in 0..p {
        for i in 0..n {
            let v = [0, 1, 2].map(|d| {
                0.75 * (j + 1) as f64 - 0.125 * i as f64
                    + 0.0625 * d as f64
                    + 0.001 * (j * i) as f64
            });
            q.set(j, i, v);
        }
    }
    q
}

/// Row j of `q` (3N values, atom-major), as sent in `POSDATA` (K-2).
pub fn row(q: &Beads, j: usize) -> Vec<f64> {
    let n3 = q.n_atoms() * 3;
    q.as_slice()[j * n3..(j + 1) * n3].to_vec()
}

/// The bead of `q` whose row equals `positions` bitwise.
pub fn bead_of(q: &Beads, positions: &[f64]) -> usize {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let want = bits(positions);
    (0..q.n_beads())
        .find(|&j| bits(&row(q, j)) == want)
        .unwrap_or_else(|| panic!("positions match no bead: {positions:?}"))
}

/// A hand-made reply that depends only on the positions: E = Σ_k (k + 1)·x_k·10⁻³ − 0.5,
/// F_k = 0.25 k − x_k, W_m = (m + 1)·x_0 − 0.5 m (no physics, just distinct numbers).
pub fn hand_reply(x: &[f64]) -> Reply {
    let energy = x
        .iter()
        .enumerate()
        .map(|(k, v)| (k + 1) as f64 * v * 1e-3)
        .sum::<f64>()
        - 0.5;
    let forces = x
        .iter()
        .enumerate()
        .map(|(k, v)| 0.25 * k as f64 - v)
        .collect();
    let virial = std::array::from_fn(|m| (m + 1) as f64 * x[0] - 0.5 * m as f64);
    Reply {
        energy,
        forces,
        virial,
        extra: Vec::new(),
    }
}

/// `hand_reply` as a mock model.
pub fn hand_model() -> Model {
    Arc::new(|r: &Request| Action::Reply(hand_reply(&r.positions)))
}

/// Binds `ep` and accepts `n_clients` (the mocks must already be connecting).
pub fn server(ep: &Endpoint, n_atoms: usize, n_clients: usize) -> SocketPotential {
    SocketPotential::bind_and_accept(&address(ep), n_atoms, n_clients, PATIENT)
        .unwrap_or_else(|e| panic!("bind_and_accept failed: {e:?}"))
}

pub fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}
