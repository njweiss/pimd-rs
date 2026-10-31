//! Bead assignment and determinism (MILESTONE_5 "Required tests: Bead assignment"; SPEC K-4,
//! D-3). Bead j goes to client j mod n_clients (clients numbered in accept order; the mocks
//! connect one after the other, so accept order = spawn order), each client sees its beads in
//! ascending j, results are stored by bead index and are bitwise identical for any n_clients and
//! any answer timing. A client answering `NEEDINIT` gets `INIT` with its current bead index and an
//! empty string.

mod common;

use std::time::{Duration, Instant};

use common::mock::{
    Endpoint, Event, NeedInit, Spec, free_port, spawn_clients, toy_model, unix_endpoint,
};
use common::{bead_of, bits, coded_positions, hand_model, server};
use pimd_core::Cell;
use pimd_potentials::{Evaluation, Potential};

/// Evaluates `q` twice through `n_clients` mocks built by `spec(k)`; checks the assignment rule
/// on every mock and returns the two evaluations.
fn run(
    ep: Endpoint,
    n_clients: usize,
    q: &pimd_core::Beads,
    cell: Option<&Cell>,
    spec: impl Fn(usize) -> Spec,
) -> (Evaluation, Evaluation, Vec<common::mock::Log>) {
    let p = q.n_beads();
    let mocks = spawn_clients(ep.clone(), (0..n_clients).map(spec).collect());
    let mut pot = server(&ep, q.n_atoms(), n_clients);
    let a = pot.evaluate(q, cell, None).unwrap();
    let b = pot.evaluate(q, cell, None).unwrap();
    drop(pot);
    let logs = mocks.join().unwrap();
    for (k, log) in logs.iter().enumerate() {
        assert!(
            log.violations.is_empty(),
            "client {k}: {:?}",
            log.violations
        );
        let seen: Vec<usize> = log
            .requests()
            .iter()
            .map(|r| bead_of(q, &r.positions))
            .collect();
        let mine: Vec<usize> = (0..p).filter(|j| j % n_clients == k).collect();
        let want: Vec<usize> = mine.iter().chain(&mine).copied().collect();
        assert_eq!(
            seen, want,
            "K-4: client {k} of {n_clients}, P = {p}: beads seen (two calls)"
        );
        assert!(log.got_exit(), "client {k}: no EXIT");
    }
    (a, b, logs)
}

fn same(a: &Evaluation, b: &Evaluation, what: &str) {
    assert_eq!(bits(&a.energies), bits(&b.energies), "{what}: energies");
    assert_eq!(
        bits(a.forces.as_slice()),
        bits(b.forces.as_slice()),
        "{what}: forces"
    );
    assert_eq!(a.virial, b.virial, "{what}: virial");
}

/// K-4, D-3: n_clients ∈ {1, 2, 3} × P ∈ {1, 2, 5, 8} (including P < n_clients): each client sees
/// exactly beads j ≡ k (mod n_clients) in ascending order, and the results are bitwise identical
/// for every n_clients and on repeated calls.
#[test]
fn round_robin_assignment_and_identical_results() {
    for p in [1usize, 2, 5, 8] {
        let q = coded_positions(p, 3);
        let mut reference: Option<Evaluation> = None;
        for n_clients in 1..=3 {
            let (_d, path) = unix_endpoint();
            let (a, b, _) = run(Endpoint::Unix(path), n_clients, &q, None, |_| {
                Spec::new(hand_model())
            });
            same(&a, &b, "repeated call");
            match &reference {
                None => reference = Some(a),
                Some(r) => same(r, &a, &format!("P = {p}, {n_clients} clients vs 1")),
            }
        }
    }
}

/// K-4, D-3: with randomised answer delays (0–3 ms per bead, different seeds per client and per
/// repetition) the results are still bitwise those of a single prompt client, and the assignment
/// is unchanged (beads are stored by index, not by finish order). P = 8, 3 clients, TCP.
#[test]
fn random_delays_do_not_change_results() {
    let q = coded_positions(8, 4);
    let cell = Cell::new([9.0, 10.0, 11.0]).unwrap();
    let (_d, path) = unix_endpoint();
    let (base, _, _) = run(Endpoint::Unix(path), 1, &q, Some(&cell), |_| {
        Spec::new(toy_model())
    });
    for rep in 0..3u64 {
        let ep = Endpoint::Tcp(format!("127.0.0.1:{}", free_port()));
        let (a, b, _) = run(ep, 3, &q, Some(&cell), |k| {
            let mut s = Spec::new(toy_model());
            s.max_delay_us = 3000;
            s.seed = 17 * rep + k as u64 + 1;
            s
        });
        same(&base, &a, &format!("delays, repetition {rep}"));
        same(&base, &b, &format!("delays, repetition {rep}, second call"));
    }
}

/// K-4: a client that answers `NEEDINIT` (before its first bead, or before every bead) is sent
/// `INIT` with the index of the bead it is about to receive and an empty string, then works
/// normally; results are unchanged. P = 5, 2 clients.
#[test]
fn needinit_answered_with_init() {
    let q = coded_positions(5, 2);
    let (_d, path) = unix_endpoint();
    let (base, _, _) = run(Endpoint::Unix(path), 1, &q, None, |_| {
        Spec::new(hand_model())
    });
    for mode in [NeedInit::Once, NeedInit::Always] {
        let (_d, path) = unix_endpoint();
        let (a, _, logs) = run(Endpoint::Unix(path), 2, &q, None, |_| {
            let mut s = Spec::new(hand_model());
            s.needinit = mode;
            s
        });
        same(&base, &a, &format!("{mode:?}"));
        for (k, log) in logs.iter().enumerate() {
            let mut n_init = 0;
            for (idx, e) in log.events.iter().enumerate() {
                if let Event::Init { bead, text } = e {
                    n_init += 1;
                    assert!(text.is_empty(), "K-4: INIT string must be empty");
                    // The next POSDATA after this INIT is for the bead announced.
                    let pos = log.events[idx..]
                        .iter()
                        .find_map(|x| match x {
                            Event::PosData(r) => Some(bead_of(&q, &r.positions)),
                            _ => None,
                        })
                        .expect("INIT not followed by POSDATA");
                    assert_eq!(
                        *bead as usize, pos,
                        "K-4: client {k}: INIT bead index vs the bead sent next"
                    );
                }
            }
            let n_beads = log.requests().len();
            match mode {
                NeedInit::Once => assert_eq!(n_init, 1, "client {k}: one INIT"),
                NeedInit::Always => assert_eq!(n_init, n_beads, "client {k}: INIT per bead"),
                NeedInit::Never => unreachable!(),
            }
        }
    }
}

/// K-4: different clients run concurrently. Four clients that each take 250 ms per bead, P = 4:
/// one evaluation takes about 250 ms when the clients work in parallel and 1 s when served one
/// after the other; it must finish within 750 ms.
#[test]
fn clients_run_concurrently() {
    let q = coded_positions(4, 2);
    let (_d, path) = unix_endpoint();
    let ep = Endpoint::Unix(path);
    let slow = || {
        let m = hand_model();
        std::sync::Arc::new(move |r: &common::mock::Request| {
            std::thread::sleep(Duration::from_millis(250));
            m(r)
        }) as common::mock::Model
    };
    let mocks = spawn_clients(ep.clone(), (0..4).map(|_| Spec::new(slow())).collect());
    let mut pot = server(&ep, 2, 4);
    let t0 = Instant::now();
    pot.evaluate(&q, None, None).unwrap();
    let dt = t0.elapsed();
    drop(pot);
    mocks.join().unwrap();
    assert!(
        dt < Duration::from_millis(750),
        "K-4: four 250 ms beads on four clients took {dt:?}"
    );
}
