//! Errors (MILESTONE_5 "Required tests: Errors"; SPEC K-5, K-6, P-1, P-4). Each fault is produced
//! by a mock client written from K-2: a returned atom count ≠ N, a non-finite number, an
//! unexpected header, a connection closed mid-message, a client that stops answering (timeout),
//! no client at all within the accept timeout; all are `PotentialError::Backend`, and with several
//! failing beads the error names the lowest bead index. `subset = Some` is `Unsupported` and a
//! wrong atom count in the positions is `ShapeMismatch` (P-1). Timeouts are short (0.3–0.5 s) and
//! the checks allow generous slack for a loaded machine.
//!
//! Assumption (report): the `Backend` message contains `bead <j>` for the failing bead j (K-5
//! "the error of the lowest bead index is returned").

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::mock::{
    Action, Endpoint, Model, Request, Spec, hdr, spawn_clients, toy_model, unix_endpoint,
};
use common::{address, bead_of, coded_positions, hand_model, hand_reply, server};
use pimd_core::Beads;
use pimd_potentials::{Potential, PotentialError};
use pimd_socket::SocketPotential;

/// A model that answers normally except for bead `bad` of `q`, where it takes `fault`.
fn faulty(q: &Beads, bad: usize, fault: Action) -> Model {
    let q = q.clone();
    Arc::new(move |r: &Request| {
        if bead_of(&q, &r.positions) == bad {
            fault.clone()
        } else {
            Action::Reply(hand_reply(&r.positions))
        }
    })
}

/// Evaluates `q` (gas phase) with mocks built from `models` and a server timeout `timeout`;
/// returns the result and the elapsed time of the evaluate call.
fn evaluate_with(
    models: Vec<Model>,
    q: &Beads,
    timeout: Duration,
) -> (Result<(), PotentialError>, Duration) {
    let (_d, path) = unix_endpoint();
    let ep = Endpoint::Unix(path);
    let n_clients = models.len();
    let mocks = spawn_clients(ep.clone(), models.into_iter().map(Spec::new).collect());
    let mut pot = SocketPotential::bind_and_accept(&address(&ep), q.n_atoms(), n_clients, timeout)
        .unwrap_or_else(|e| panic!("bind_and_accept: {e:?}"));
    let t0 = Instant::now();
    let r = pot.evaluate(q, None, None).map(|_| ());
    let dt = t0.elapsed();
    drop(pot);
    mocks.join().unwrap();
    (r, dt)
}

/// The `Backend` message, or a panic naming what was expected.
fn backend(r: Result<(), PotentialError>, what: &str) -> String {
    match r {
        Err(PotentialError::Backend(m)) => m,
        other => panic!("{what}: expected PotentialError::Backend, got {other:?}"),
    }
}

/// K-5: a reply announcing N + 1 atoms (and sending 3(N + 1) forces), or N − 1, is `Backend`.
#[test]
fn wrong_atom_count() {
    let q = coded_positions(2, 3);
    for n in [4usize, 2] {
        let reply = hand_reply(&common::row(&q, 1));
        let (r, _) = evaluate_with(
            vec![faulty(&q, 1, Action::WrongAtomCount(reply, n))],
            &q,
            Duration::from_secs(10),
        );
        backend(r, &format!("{n} atoms returned for 3"));
    }
}

/// K-5: a NaN force, an infinite energy, or a NaN virial component at P = 1 is `Backend`.
#[test]
fn non_finite_values() {
    let q2 = coded_positions(2, 3);
    let mut nan_force = hand_reply(&common::row(&q2, 1));
    nan_force.forces[4] = f64::NAN;
    let mut inf_energy = hand_reply(&common::row(&q2, 0));
    inf_energy.energy = f64::INFINITY;
    for (bad, reply, what) in [(1, nan_force, "NaN force"), (0, inf_energy, "inf energy")] {
        let (r, _) = evaluate_with(
            vec![faulty(&q2, bad, Action::Reply(reply))],
            &q2,
            Duration::from_secs(10),
        );
        backend(r, what);
    }
    let q1 = coded_positions(1, 3);
    let mut nan_virial = hand_reply(&common::row(&q1, 0));
    nan_virial.virial[4] = f64::NAN;
    let (r, _) = evaluate_with(
        vec![faulty(&q1, 0, Action::Reply(nan_virial))],
        &q1,
        Duration::from_secs(10),
    );
    backend(r, "NaN virial at P = 1");
}

/// K-5: an unexpected header in place of `FORCEREADY`, or an unknown answer to `STATUS`, is
/// `Backend`.
#[test]
fn unexpected_header() {
    let q = coded_positions(2, 3);
    for h in [hdr("HAVEDATA"), hdr("FORCEREADX"), hdr("READY")] {
        let (r, _) = evaluate_with(
            vec![faulty(&q, 0, Action::BadHeader(h))],
            &q,
            Duration::from_secs(10),
        );
        backend(
            r,
            &format!("{:?} instead of FORCEREADY", String::from_utf8_lossy(&h)),
        );
    }
    let (_d, path) = unix_endpoint();
    let ep = Endpoint::Unix(path);
    let mut spec = Spec::new(hand_model());
    spec.bad_status = Some(hdr("WHATEVER"));
    let mocks = spawn_clients(ep.clone(), vec![spec]);
    let mut pot = server(&ep, 3, 1);
    let r = pot.evaluate(&q, None, None).map(|_| ());
    drop(pot);
    mocks.join().unwrap();
    backend(r, "unknown STATUS answer");
}

/// K-5: a client that sends `FORCEREADY` and the energy, then closes the connection, is `Backend`
/// (not a hang, not a panic).
#[test]
fn disconnect_mid_message() {
    let q = coded_positions(3, 2);
    let (r, dt) = evaluate_with(
        vec![faulty(&q, 2, Action::Disconnect(-1.0))],
        &q,
        Duration::from_secs(10),
    );
    backend(r, "disconnect mid-message");
    assert!(dt < Duration::from_secs(5), "took {dt:?}");
}

/// K-5, MILESTONE_5 "Socket lifecycle" (every read and write uses the timeout): a client that
/// stops answering for 3 s is `Backend` after about the 0.4 s timeout, well before it would have
/// answered.
#[test]
fn stalled_client_times_out() {
    let q = coded_positions(2, 2);
    let (r, dt) = evaluate_with(
        vec![faulty(&q, 1, Action::Hang(Duration::from_secs(3)))],
        &q,
        Duration::from_millis(400),
    );
    backend(r, "stalled client");
    assert!(
        dt < Duration::from_millis(2500),
        "timeout not applied: evaluate took {dt:?}"
    );
}

/// K-1, K-5: no client within the accept timeout (0.3 s) is `Backend`, returned after about the
/// timeout; two required clients and only one connecting is `Backend` too.
#[test]
fn no_client_within_timeout() {
    let (_d, path) = unix_endpoint();
    let ep = Endpoint::Unix(path);
    let t0 = Instant::now();
    let r = SocketPotential::bind_and_accept(&address(&ep), 3, 1, Duration::from_millis(300));
    let dt = t0.elapsed();
    assert!(
        matches!(r, Err(PotentialError::Backend(_))),
        "no client: {:?}",
        r.err()
    );
    assert!(
        dt >= Duration::from_millis(250) && dt < Duration::from_millis(2500),
        "accept wait {dt:?} for a 0.3 s timeout"
    );

    let (_d2, path2) = unix_endpoint();
    let ep2 = Endpoint::Unix(path2);
    let mocks = spawn_clients(ep2.clone(), vec![Spec::new(hand_model())]);
    let r = SocketPotential::bind_and_accept(&address(&ep2), 3, 2, Duration::from_millis(500));
    assert!(
        matches!(r, Err(PotentialError::Backend(_))),
        "one of two clients: {:?}",
        r.err()
    );
    drop(r);
    // The lone mock sees its connection closed (or never served) and finishes.
    mocks.join().unwrap();
}

/// K-1, K-5: binding an address that cannot be bound (a directory that does not exist) is
/// `Backend`.
#[test]
fn bind_failure() {
    let ep = Endpoint::Unix("/nonexistent-dir-pimd-rs/x.sock".into());
    let r = SocketPotential::bind_and_accept(&address(&ep), 3, 1, Duration::from_millis(300));
    assert!(
        matches!(r, Err(PotentialError::Backend(_))),
        "{:?}",
        r.err()
    );
}

/// K-5: with several failing beads the error of the lowest bead index is returned. Two clients,
/// P = 4: client 0 has beads 0, 2; client 1 has beads 1, 3. Bead 2 (client 0) returns a wrong atom
/// count and bead 1 (client 1) a NaN force; the error names bead 1. The mirror case (bead 2 fails
/// on client 0 only, bead 3 on client 1) names bead 2. The bead-2 failure sits on the client that
/// finishes first, so finish order does not decide.
#[test]
fn lowest_failing_bead_reported() {
    let q = coded_positions(4, 2);
    let mk = |fails: Vec<(usize, Action)>| -> Model {
        let q = q.clone();
        Arc::new(move |r: &Request| {
            let j = bead_of(&q, &r.positions);
            match fails.iter().find(|(b, _)| *b == j) {
                Some((_, a)) => a.clone(),
                None => Action::Reply(hand_reply(&r.positions)),
            }
        })
    };
    let slow = |m: Model| -> Model {
        Arc::new(move |r: &Request| {
            std::thread::sleep(Duration::from_millis(100));
            m(r)
        })
    };
    let mut nan = hand_reply(&common::row(&q, 1));
    nan.forces[0] = f64::NAN;
    let wrong_n = Action::WrongAtomCount(hand_reply(&common::row(&q, 2)), 3);
    let (r, _) = evaluate_with(
        vec![
            mk(vec![(2, wrong_n.clone())]),
            slow(mk(vec![(1, Action::Reply(nan.clone()))])),
        ],
        &q,
        Duration::from_secs(10),
    );
    let m = backend(r, "beads 1 and 2 fail");
    assert!(
        m.contains("bead 1") && !m.contains("bead 2"),
        "K-5: lowest failing bead is 1: {m:?}"
    );

    let mut nan3 = hand_reply(&common::row(&q, 3));
    nan3.forces[0] = f64::NAN;
    let (r, _) = evaluate_with(
        vec![
            slow(mk(vec![(2, wrong_n)])),
            mk(vec![(3, Action::Reply(nan3))]),
        ],
        &q,
        Duration::from_secs(10),
    );
    let m = backend(r, "beads 2 and 3 fail");
    assert!(
        m.contains("bead 2") && !m.contains("bead 3"),
        "K-5: lowest failing bead is 2: {m:?}"
    );
}

/// P-4, K-5: `subset = Some` is `Unsupported`; P-1: positions with N + 1 atoms are
/// `ShapeMismatch { expected: N, actual: N + 1 }`. Neither disturbs the connection: a valid call
/// afterwards succeeds.
#[test]
fn subset_and_shape_mismatch() {
    let (_d, path) = unix_endpoint();
    let ep = Endpoint::Unix(path);
    let mocks = spawn_clients(ep.clone(), vec![Spec::new(toy_model())]);
    let mut pot = server(&ep, 3, 1);
    let q = coded_positions(2, 3);
    assert_eq!(
        pot.evaluate(&q, None, Some(&[0, 1])).err(),
        Some(PotentialError::Unsupported)
    );
    assert_eq!(
        pot.evaluate(&coded_positions(2, 4), None, None).err(),
        Some(PotentialError::ShapeMismatch {
            expected: 3,
            actual: 4
        })
    );
    pot.evaluate(&q, None, None)
        .expect("valid call after rejected ones");
    drop(pot);
    let logs = mocks.join().unwrap();
    assert!(logs[0].violations.is_empty(), "{:?}", logs[0].violations);
    assert_eq!(
        logs[0].requests().len(),
        2,
        "rejected calls must not reach the client"
    );
}

/// K-6: dropping the potential sends `EXIT` to every client (three clients, after use and also
/// without any evaluation) and removes the Unix socket file.
#[test]
fn drop_sends_exit_to_all_and_removes_socket_file() {
    for evaluate in [true, false] {
        let (_d, path) = unix_endpoint();
        let ep = Endpoint::Unix(path.clone());
        let mocks = spawn_clients(ep.clone(), (0..3).map(|_| Spec::new(toy_model())).collect());
        let mut pot = server(&ep, 2, 3);
        assert!(path.exists(), "socket file exists while serving");
        if evaluate {
            pot.evaluate(&coded_positions(5, 2), None, None).unwrap();
        }
        drop(pot);
        let logs = mocks.join().unwrap();
        for (k, l) in logs.iter().enumerate() {
            assert!(
                l.got_exit(),
                "client {k} got no EXIT (evaluate = {evaluate})"
            );
        }
        assert!(!path.exists(), "socket file left behind");
    }
}
