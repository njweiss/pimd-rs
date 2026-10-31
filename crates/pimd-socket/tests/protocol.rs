//! Wire protocol (MILESTONE_5 "Required tests: Protocol"; SPEC K-1, K-2, K-3, K-6, P-1, P-5).
//! A mock client written in the test from K-2 (`common/mock.rs`) records the exact bytes the
//! server sends and answers with hand-made energies, forces and virials; the `Evaluation` must
//! equal them bitwise. Expected `POSDATA` payloads are assembled here byte by byte from K-2/K-3:
//! h = diag(L) row-major, h⁻¹ = diag(1/L), int32 N, 3N float64 positions (bead row j, atom-major),
//! little-endian; gas phase: h and h⁻¹ all zeros.

mod common;

use std::sync::Arc;

use common::mock::{
    Action, Endpoint, Event, Reply, Request, Spec, free_port, spawn_clients, unix_endpoint,
};
use common::{bits, coded_positions, hand_model, hand_reply, row, server};
use pimd_core::Cell;
use pimd_potentials::Potential;

/// The `POSDATA` payload of K-2/K-3 for one bead, built by hand.
fn expected_payload(cell: Option<[f64; 3]>, positions: &[f64]) -> Vec<u8> {
    let mut h = [0.0f64; 9];
    let mut hi = [0.0f64; 9];
    if let Some(l) = cell {
        for d in 0..3 {
            h[4 * d] = l[d];
            hi[4 * d] = 1.0 / l[d];
        }
    }
    let mut w = Vec::new();
    for v in h.iter().chain(&hi) {
        w.extend_from_slice(&v.to_le_bytes());
    }
    w.extend_from_slice(&((positions.len() / 3) as i32).to_le_bytes());
    for v in positions {
        w.extend_from_slice(&v.to_le_bytes());
    }
    w
}

/// K-2, K-3, K-6, P-5: periodic, non-cubic cell (10, 12.5, 17 bohr), P = 1, N = 3, Unix socket.
/// The payload is byte-exact (h = diag(L), h⁻¹ = diag(1/L), not swapped); energies, forces and
/// the virial equal the mock's hand-made reply bitwise (virial `Some` at P = 1). A second call at
/// other positions works too (the stream stays in sync, including a non-empty extra string).
#[test]
fn posdata_bytes_and_reply_periodic_unix() {
    let (_d, path) = unix_endpoint();
    let ep = Endpoint::Unix(path.clone());
    let model = Arc::new(|r: &Request| {
        let mut rep = hand_reply(&r.positions);
        rep.extra = br#"{"note": "extra strings are read and ignored"}"#.to_vec();
        Action::Reply(rep)
    });
    let mocks = spawn_clients(ep.clone(), vec![Spec::new(model)]);
    let l = [10.0, 12.5, 17.0];
    let cell = Cell::new(l).unwrap();
    let mut pot = server(&ep, 3, 1);
    let q1 = coded_positions(1, 3);
    let mut q2 = coded_positions(1, 3);
    q2.as_mut_slice()[4] = -3.25;
    for q in [&q1, &q2] {
        let ev = pot.evaluate(q, Some(&cell), None).unwrap();
        let want = hand_reply(&row(q, 0));
        assert_eq!(bits(&ev.energies), bits(&[want.energy]), "energy");
        assert_eq!(bits(ev.forces.as_slice()), bits(&want.forces), "forces");
        assert_eq!((ev.forces.n_beads(), ev.forces.n_atoms()), (1, 3));
        let w = ev.virial.expect("K-6: virial is Some for P = 1");
        let flat: Vec<f64> = w.iter().flatten().copied().collect();
        assert_eq!(bits(&flat), bits(&want.virial), "virial (row-major)");
    }
    drop(pot);
    let logs = mocks.join().unwrap();
    assert!(logs[0].violations.is_empty(), "{:?}", logs[0].violations);
    let reqs = logs[0].requests();
    assert_eq!(reqs.len(), 2);
    for (r, q) in reqs.iter().zip([&q1, &q2]) {
        assert_eq!(
            r.payload,
            expected_payload(Some(l), &row(q, 0)),
            "POSDATA payload not byte-exact (K-2, K-3)"
        );
    }
    assert!(logs[0].got_exit(), "K-6: Drop sends EXIT");
    assert!(
        !path.exists(),
        "K-6: Drop removes the Unix socket file it created"
    );
}

/// K-2, K-3, K-6: gas phase (cell `None`) over TCP, P = 3, N = 2: h and h⁻¹ are all zeros, every
/// bead's positions are sent as its row; the result holds each bead's reply at its own index and
/// the virial is `None` (P > 1).
#[test]
fn posdata_bytes_and_reply_gas_phase_tcp() {
    let addr = format!("127.0.0.1:{}", free_port());
    let ep = Endpoint::Tcp(addr);
    let mocks = spawn_clients(ep.clone(), vec![Spec::new(hand_model())]);
    let mut pot = server(&ep, 2, 1);
    let q = coded_positions(3, 2);
    let ev = pot.evaluate(&q, None, None).unwrap();
    assert!(ev.virial.is_none(), "K-6: virial None for P > 1");
    assert_eq!((ev.forces.n_beads(), ev.forces.n_atoms()), (3, 2));
    for j in 0..3 {
        let want = hand_reply(&row(&q, j));
        assert_eq!(ev.energies[j].to_bits(), want.energy.to_bits(), "bead {j}");
        assert_eq!(bits(&row(&ev.forces, j)), bits(&want.forces), "bead {j}");
    }
    drop(pot);
    let logs = mocks.join().unwrap();
    assert!(logs[0].violations.is_empty(), "{:?}", logs[0].violations);
    let reqs = logs[0].requests();
    assert_eq!(reqs.len(), 3);
    for (j, r) in reqs.iter().enumerate() {
        assert_eq!(r.payload, expected_payload(None, &row(&q, j)), "bead {j}");
        assert!(r.cell.iter().chain(&r.inv).all(|v| v.to_bits() == 0));
    }
    assert!(logs[0].got_exit(), "K-6: Drop sends EXIT over TCP too");
}

/// K-2: every request follows the protocol: the server polls `STATUS` before its first
/// `POSDATA`, and each `POSDATA` is followed by `GETFORCE` before the next `POSDATA`; the mock
/// sees no violation over several calls; `EXIT` comes last.
#[test]
fn message_sequence() {
    let (_d, path) = unix_endpoint();
    let ep = Endpoint::Unix(path);
    let mocks = spawn_clients(ep.clone(), vec![Spec::new(hand_model())]);
    let mut pot = server(&ep, 4, 1);
    let q = coded_positions(4, 4);
    for _ in 0..3 {
        pot.evaluate(&q, None, None).unwrap();
    }
    drop(pot);
    let log = &mocks.join().unwrap()[0];
    assert!(log.violations.is_empty(), "{:?}", log.violations);
    let first_pos = log
        .events
        .iter()
        .position(|e| matches!(e, Event::PosData(_)))
        .unwrap();
    assert!(
        log.events[..first_pos]
            .iter()
            .any(|e| matches!(e, Event::Status)),
        "no STATUS before the first POSDATA"
    );
    let mut holding = false;
    let mut n_pos = 0;
    for e in &log.events {
        match e {
            Event::PosData(_) => {
                assert!(!holding, "POSDATA before GETFORCE");
                holding = true;
                n_pos += 1;
            }
            Event::GetForce => {
                assert!(holding, "GETFORCE without POSDATA");
                holding = false;
            }
            _ => {}
        }
    }
    assert_eq!(n_pos, 12);
    assert!(matches!(log.events.last(), Some(Event::Exit)));
}

/// P-1, K-2: the result has shape `[P, N, 3]` and P energies even when the reply carries the
/// energy only (zero forces); values pass through without unit conversion (U-1): an energy of
/// exactly 1.0 Eh and a force of exactly 1.0 Eh/bohr come back as 1.0.
#[test]
fn no_unit_conversion() {
    let (_d, path) = unix_endpoint();
    let ep = Endpoint::Unix(path);
    let model = Arc::new(|r: &Request| {
        let mut forces = vec![0.0; r.positions.len()];
        forces[0] = 1.0;
        Action::Reply(Reply {
            energy: 1.0,
            forces,
            virial: [0.0; 9],
            extra: Vec::new(),
        })
    });
    let mocks = spawn_clients(ep.clone(), vec![Spec::new(model)]);
    let mut pot = server(&ep, 2, 1);
    let ev = pot.evaluate(&coded_positions(2, 2), None, None).unwrap();
    assert_eq!(ev.energies, vec![1.0, 1.0]);
    for j in 0..2 {
        assert_eq!(row(&ev.forces, j), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }
    drop(pot);
    mocks.join().unwrap();
}
