//! Runs through the socket potential with mock clients (MILESTONE_5 "Required tests: Restart with
//! a mock client", "Config" hash coverage, "Socket lifecycle"; SPEC K-1, K-4, D-3, Q18).
//!
//! The mocks (`pimd-socket/tests/common/mock.rs`, written from K-2) serve the toy model
//! `mock::toy_energy_forces`, deterministic and stateless, so (D-3 note) a socket run is bitwise
//! reproducible, independent of n_clients and thread count, and equal to an in-process run with
//! the same model (`common::ToyPotential`), since the socket carries f64 values exactly.
//! The system is the brief's gas-phase water (`common::water_toml`, P = 4).

mod common;

use std::path::Path;

use common::{
    OBS_HEADER, ToyPotential, bytes, observables, parse, pool, read_csv, reference_run_with, rel,
    set, text, tmp, toy_clients, water_toml,
};
use pimd_cli::{CliError, run, run_from_checkpoint};
use pimd_potentials::PotentialError;

/// The water config at the Unix socket `path` with `n` clients and `n_steps` production steps.
fn cfg_text(path: &Path, n: usize, n_steps: usize) -> String {
    let t = water_toml(&format!("unix:{}", path.display()), n);
    set(&t, "sampling", "n_steps", &n_steps.to_string())
}

/// Runs `toml` (socket at `path`) with `n` toy clients; returns the clients' POSDATA counts.
fn run_socket(toml: &str, path: &Path, n: usize, out: &Path, restart: Option<&Path>) -> usize {
    let clients = toy_clients(path, n);
    let cfg = parse(toml);
    match restart {
        None => run(&cfg, out).unwrap(),
        Some(ck) => run_from_checkpoint(&cfg, ck, out).unwrap(),
    };
    let logs = clients.join().unwrap();
    for (k, l) in logs.iter().enumerate() {
        assert!(l.violations.is_empty(), "client {k}: {:?}", l.violations);
        assert!(l.got_exit(), "client {k}: the run must end with EXIT (K-6)");
    }
    logs.iter().map(|l| l.requests().len()).sum()
}

/// K-1, K-3, I-3, D-3: a socket run (2 clients) records the same observables as an in-process
/// run of the same model (reference loop written from the brief, `common::reference_run_with`),
/// to 1e-12 relative per value (the M4 reference-loop tolerance); its trajectory too. The clients
/// receive the gas-phase cell (all zeros, K-3) and (n_equil + n_steps + 1)·P beads in total
/// (one evaluation per step plus the initial one).
#[test]
fn socket_run_equals_in_process_reference() {
    let sock = tmp();
    let path = sock.path().join("w.sock");
    let toml = cfg_text(&path, 2, 30);
    let out = tmp();
    let clients = toy_clients(&path, 2);
    pool(1).install(|| run(&parse(&toml), out.path()).unwrap());
    let logs = clients.join().unwrap();
    let n_req: usize = logs.iter().map(|l| l.requests().len()).sum();
    assert_eq!(
        n_req,
        (10 + 30 + 1) * 4,
        "evaluations: one per step plus the initial one"
    );
    for l in &logs {
        for r in l.requests() {
            assert!(
                r.cell.iter().chain(&r.inv).all(|v| v.to_bits() == 0),
                "K-3: gas phase sends zero h and h⁻¹"
            );
        }
    }
    let cfg = parse(&toml);
    let r = reference_run_with(&cfg, Box::new(ToyPotential), Some(10));
    let rows = observables(out.path());
    assert_eq!(rows.len(), r.rows.len());
    for (got, want) in rows.iter().zip(&r.rows) {
        assert_eq!(got[0] as usize, want.step);
        for (c, w) in [
            (1, want.e_prim),
            (2, want.e_vir),
            (3, want.v),
            (4, want.t_est),
        ] {
            assert!(
                rel(got[c], w, 1e-300) <= 1e-12,
                "step {}: column {c}: {} vs reference {w}",
                want.step,
                got[c]
            );
        }
    }
    let frames = common::read_trajectory(&out.path().join("trajectory.xyz"), 4, 3);
    assert_eq!(frames.len(), r.frames.len());
    for ((s, rows), (s_ref, q)) in frames.iter().zip(&r.frames) {
        assert_eq!(s, s_ref);
        for j in 0..4 {
            for i in 0..3 {
                assert_eq!(rows[j * 3 + i].0, ["O", "H", "H"][i]);
                for d in 0..3 {
                    let w = q.get(j, i)[d] * 0.529177210903;
                    assert!(rel(rows[j * 3 + i].1[d], w, 1e-300) <= 1e-15);
                }
            }
        }
    }
}

/// D-3, K-4: the outputs do not depend on n_clients (1, 2, 3) or on the thread count (1, 4):
/// byte-identical observables, trajectory and checkpoint.
#[test]
fn socket_results_independent_of_clients_and_threads() {
    let mut reference: Option<(Vec<u8>, Vec<u8>, Vec<u8>)> = None;
    for (n, threads) in [(1usize, 1usize), (2, 4), (3, 1), (3, 4)] {
        let sock = tmp();
        let path = sock.path().join("w.sock");
        let out = tmp();
        pool(threads).install(|| run_socket(&cfg_text(&path, n, 20), &path, n, out.path(), None));
        let got = (
            bytes(&out.path().join("observables.csv")),
            bytes(&out.path().join("trajectory.xyz")),
            bytes(&out.path().join("checkpoint.bin")),
        );
        match &reference {
            None => reference = Some(got),
            Some(r) => assert!(
                r == &got,
                "{n} clients, {threads} threads differ from 1 client"
            ),
        }
    }
}

/// D-3, K-4, I-3 (Q18, (new) config hash): K + checkpoint + K equals 2K bitwise through the socket (gas phase):
/// final checkpoint, observables and trajectory concatenations. The restart uses a different
/// address, n_clients (1 → 3) and timeout_s, which the hash excludes, and evaluates no bead before
/// its first step (restored forces and energies): its clients see exactly K·P beads.
#[test]
fn socket_restart_bitwise_without_reevaluation() {
    let (s1, s2, s3) = (tmp(), tmp(), tmp());
    let (p1, p2, p3) = (
        s1.path().join("a.sock"),
        s2.path().join("b.sock"),
        s3.path().join("c.sock"),
    );
    let (full, a, b) = (tmp(), tmp(), tmp());
    pool(1).install(|| {
        run_socket(&cfg_text(&p1, 2, 30), &p1, 2, full.path(), None);
        run_socket(&cfg_text(&p2, 1, 15), &p2, 1, a.path(), None);
        let restart = set(&cfg_text(&p3, 3, 30), "potential", "timeout_s", "7.5");
        let n = run_socket(
            &restart,
            &p3,
            3,
            b.path(),
            Some(&a.path().join("checkpoint.bin")),
        );
        assert_eq!(
            n,
            15 * 4,
            "Q18: the restart must not re-evaluate the saved state"
        );
    });
    assert_eq!(
        bytes(&full.path().join("checkpoint.bin")),
        bytes(&b.path().join("checkpoint.bin")),
        "final checkpoint"
    );
    let (h1, r1) = read_csv(&a.path().join("observables.csv"));
    let (h2, r2) = read_csv(&b.path().join("observables.csv"));
    let (hf, rf) = read_csv(&full.path().join("observables.csv"));
    assert_eq!(
        (h1.as_str(), h2.as_str(), hf.as_str()),
        (OBS_HEADER, OBS_HEADER, OBS_HEADER)
    );
    let joined: Vec<Vec<String>> = r1.into_iter().chain(r2).collect();
    assert_eq!(joined, rf, "observables concatenation");
    let traj = text(&a.path().join("trajectory.xyz")) + &text(&b.path().join("trajectory.xyz"));
    assert_eq!(
        traj,
        text(&full.path().join("trajectory.xyz")),
        "trajectory concatenation"
    );
}

/// D-3, K-1 ((new) config hash): the potential kind is covered: a socket checkpoint does not restart under LJ,
/// and an LJ checkpoint does not restart under the socket potential (`CliError::Checkpoint`).
/// The gas-phase LJ system has the water's atoms (LJ parameters only to have a valid config).
#[test]
fn potential_kind_in_hash() {
    let sock = tmp();
    let path = sock.path().join("w.sock");
    let socket_toml = cfg_text(&path, 1, 10);
    let lj_section = "kind = \"lj\"\nepsilon_kelvin = 35.6\nsigma_angstrom = 1.0\ncutoff_sigma = 3.0\nskin_sigma = 0.25\nshift = true\n";
    let start = socket_toml.find("kind = \"socket\"").unwrap();
    let end = socket_toml.find("[sampling]").unwrap();
    let lj_toml = format!(
        "{}{}\n{}",
        &socket_toml[..start],
        lj_section,
        &socket_toml[end..]
    );
    let lj_toml_20 = set(&lj_toml, "sampling", "n_steps", "20");

    let (a, b) = (tmp(), tmp());
    pool(1).install(|| run_socket(&socket_toml, &path, 1, a.path(), None));
    let r = pool(1).install(|| {
        run_from_checkpoint(
            &parse(&lj_toml_20),
            &a.path().join("checkpoint.bin"),
            tmp().path(),
        )
    });
    assert!(
        matches!(r, Err(CliError::Checkpoint(_))),
        "socket → LJ: {r:?}"
    );

    pool(1).install(|| run(&parse(&lj_toml), b.path()).unwrap());
    // The implementation may bind (and wait for a client) before reading the checkpoint, so a
    // client is offered; it is not joined, since it may never be served.
    let sock2 = tmp();
    let path2 = sock2.path().join("x.sock");
    let _clients = toy_clients(&path2, 1);
    let r = pool(1).install(|| {
        run_from_checkpoint(
            &parse(&cfg_text(&path2, 1, 20)),
            &b.path().join("checkpoint.bin"),
            tmp().path(),
        )
    });
    assert!(
        matches!(r, Err(CliError::Checkpoint(_))),
        "LJ → socket: {r:?}"
    );
}

/// K-1, K-5 (MILESTONE_5 "CliError gains no variant"): no client within `timeout_s` makes `run` fail with
/// `CliError::Potential(PotentialError::Backend(_))`, after about the timeout.
#[test]
fn no_client_is_backend_error() {
    let sock = tmp();
    let path = sock.path().join("w.sock");
    let toml = set(&cfg_text(&path, 1, 10), "potential", "timeout_s", "0.3");
    let t0 = std::time::Instant::now();
    let r = run(&parse(&toml), tmp().path());
    assert!(
        matches!(r, Err(CliError::Potential(PotentialError::Backend(_)))),
        "{r:?}"
    );
    assert!(t0.elapsed() < std::time::Duration::from_secs(5));
}
