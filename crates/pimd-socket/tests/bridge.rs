//! Bridge validation (MILESTONE_5 "Long validation: Bridge"; SPEC K-1 … K-6, P-3, P-5, E-8, C-2,
//! D-3 note). Feature `long-validation`; needs the pixi `ref` environment (numpy) for the Python
//! client `tests/clients/ipi_client.py` (backend `lj`), which is written from K-2 and P-3
//! independently of the engine.
//!
//! 1. Per-evaluation agreement: an in-process `LennardJones` drives a 200-step PIMD trajectory
//!    (BAOAB + PILE-L); at every step the same bead positions are evaluated through the socket.
//!    Energies, forces and (P = 1) the virial agree to 1e-12 relative (energy: |ΔE| ≤ 1e-12 |E| per
//!    bead; forces: max |ΔF| ≤ 1e-12 max |F| per bead; virial: max |ΔW| ≤ 1e-12 max |W|). The two
//!    codes sum the same pair terms in different orders, so they differ by rounding only
//!    (observed with the reference implementation: energies ≤ 1.1e-15, forces ≤ 1.2e-15, virial
//!    3.1e-15 relative).
//!    Systems: the M4 liquid-neon system (N = 108, 14.1 Å cube, ε = 35.6 K, σ = 2.749 Å,
//!    r_c = 2.2σ shifted, 30 K) at P = 1 (one client, virial compared) and P = 4 (two clients);
//!    a 13-atom gas-phase neon cluster (cuboctahedron at the LJ minimum distance, unshifted,
//!    r_c = 30 bohr, no cell), P = 4, three clients.
//! 2. Trajectory agreement: 50 PIMD steps (dt = 2 fs, PILE-L τ_0 = 100 fs) through the socket vs
//!    in-process from the same state and seed. The runs differ by the rounding differences of
//!    (1), amplified by the dynamics. Tolerance: max |Δq| ≤ 1e-10 bohr and |ΔΣV_j| ≤ 1e-12 |ΣV_j|
//!    after 50 steps. Observed with the reference implementation: max |Δq| = 1.4e-14 bohr (neon,
//!    P = 4) and 4e-16 bohr (cluster), |ΔV|/|V| ≤ 4e-16. Chaos horizon: with the same noise
//!    sequence on both sides the Langevin friction keeps the trajectories together; over 3 000
//!    steps (6 ps) the deviation grew only to 2.9e-13 bohr (neon) and 3.9e-14 bohr (cluster),
//!    roughly linearly, with no exponential divergence visible, so 50 steps are far inside the
//!    horizon and the tolerance has a margin of ~10⁴.
//!
//! Units: all parameters are converted here with the U-2 literals (CODATA 2018).
#![cfg(feature = "long-validation")]

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use pimd_core::{Beads, Cell, Rng};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat};
use pimd_potentials::{LennardJones, LjParams, Potential};
use pimd_socket::{Address, SocketPotential};

const K_B: f64 = 3.1668115634556e-6; // Eh/K
const AMU: f64 = 1822.888486209; // m_e
const BOHR_A: f64 = 0.529177210903; // Å/bohr
const FS: f64 = 41.341373336561; // ħ/Eh per fs

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Python clients that are killed when dropped.
struct Clients(Vec<Child>);

impl Drop for Clients {
    fn drop(&mut self) {
        for c in &mut self.0 {
            c.kill().ok();
            c.wait().ok();
        }
    }
}

/// Starts `n` `ipi_client.py lj` processes (single-threaded) for `address` (`unix:/path`). Uses the
/// pixi `ref` interpreter directly when present, else `pixi run -e ref python`.
fn lj_clients(address: &str, n: usize, lj: &LjParams, pbc: bool) -> Clients {
    let script = root().join("crates/pimd-socket/tests/clients/ipi_client.py");
    let py = root().join(".pixi/envs/ref/bin/python");
    let mut out = Vec::new();
    for _ in 0..n {
        let mut cmd = if py.exists() {
            Command::new(&py)
        } else {
            let mut c = Command::new("pixi");
            c.args(["run", "-e", "ref", "python"]);
            c
        };
        cmd.current_dir(root())
            .arg(&script)
            .args([
                "--address",
                address,
                "--pbc",
                if pbc { "auto" } else { "none" },
            ])
            .args(["lj", "--epsilon", &format!("{:e}", lj.epsilon)])
            .args(["--sigma", &format!("{:e}", lj.sigma)])
            .args(["--cutoff", &format!("{:e}", lj.cutoff)])
            .env("OMP_NUM_THREADS", "1")
            .stdin(Stdio::null());
        if lj.shift {
            cmd.arg("--shift");
        }
        out.push(
            cmd.spawn()
                .expect("cannot start the Python client (pixi ref env)"),
        );
    }
    Clients(out)
}

/// A system: initial state, cell, LJ parameters, β.
struct System {
    state: RingState,
    cell: Option<Cell>,
    lj: LjParams,
    beta: f64,
}

/// MILESTONE_4 neon (N = 108, 14.1 Å cube, simple-cubic start, M4 lattice rule), P beads, 30 K,
/// thermal momenta N(0, m P/β) (RP-3).
fn neon(p: usize) -> System {
    let n = 108;
    let l = 14.1 / BOHR_A;
    let sigma = 2.749 / BOHR_A;
    let m = 20.1797 * AMU;
    let beta = 1.0 / (30.0 * K_B);
    let side = 5; // ⌈108^(1/3)⌉
    let a = l / side as f64;
    let mut q = Beads::zeros(p, n);
    let mut mom = Beads::zeros(p, n);
    let mut rng = Rng::from_seed(2024);
    let sd = (m * p as f64 / beta).sqrt();
    for i in 0..n {
        let site = [i % side, (i / side) % side, i / (side * side)].map(|k| (k as f64 + 0.5) * a);
        for j in 0..p {
            q.set(j, i, site);
            mom.set(
                j,
                i,
                [sd * rng.normal(), sd * rng.normal(), sd * rng.normal()],
            );
        }
    }
    System {
        state: RingState {
            q,
            p: mom,
            masses: vec![m; n],
        },
        cell: Some(Cell::new([l; 3]).unwrap()),
        lj: LjParams {
            epsilon: 35.6 * K_B,
            sigma,
            cutoff: 2.2 * sigma,
            skin: 0.3 * sigma,
            shift: true,
        },
        beta,
    }
}

/// 13-atom neon cuboctahedron (centre + 12 nearest neighbours at r_min = 2^(1/6) σ), gas phase,
/// unshifted LJ with r_c = 30 bohr (covers every pair), P beads at 20 K.
fn cluster(p: usize) -> System {
    let sigma = 2.749 / BOHR_A;
    let r = 2f64.powf(1.0 / 6.0) * sigma;
    let m = 20.1797 * AMU;
    let beta = 1.0 / (20.0 * K_B);
    let mut sites = vec![[0.0; 3]];
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        for sa in [-1.0, 1.0] {
            for sb in [-1.0, 1.0] {
                let mut x = [0.0; 3];
                x[a] = sa * r / 2f64.sqrt();
                x[b] = sb * r / 2f64.sqrt();
                sites.push(x);
            }
        }
    }
    let n = sites.len();
    let mut q = Beads::zeros(p, n);
    let mut mom = Beads::zeros(p, n);
    let mut rng = Rng::from_seed(77);
    let sd = (m * p as f64 / beta).sqrt();
    for (i, s) in sites.iter().enumerate() {
        for j in 0..p {
            q.set(j, i, *s);
            mom.set(
                j,
                i,
                [sd * rng.normal(), sd * rng.normal(), sd * rng.normal()],
            );
        }
    }
    System {
        state: RingState {
            q,
            p: mom,
            masses: vec![m; n],
        },
        cell: None,
        lj: LjParams {
            epsilon: 35.6 * K_B,
            sigma,
            cutoff: 30.0,
            skin: 1.0,
            shift: false,
        },
        beta,
    }
}

fn integrator(sys: &System) -> RingPolymerIntegrator {
    RingPolymerIntegrator::new(
        2.0 * FS,
        sys.beta,
        sys.state.q.n_beads(),
        Thermostat::PileL {
            tau0: Some(100.0 * FS),
        },
    )
    .unwrap()
}

/// Binds a socket potential in a fresh directory and starts `n_clients` Python clients.
fn socket(sys: &System, n_clients: usize) -> (tempfile::TempDir, Clients, SocketPotential) {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("bridge.sock");
    let clients = lj_clients(
        &format!("unix:{}", path.display()),
        n_clients,
        &sys.lj,
        sys.cell.is_some(),
    );
    let pot = SocketPotential::bind_and_accept(
        &Address::Unix(path),
        sys.state.q.n_atoms(),
        n_clients,
        Duration::from_secs(120),
    )
    .expect("Python clients did not connect");
    (d, clients, pot)
}

fn max_abs(v: impl Iterator<Item = f64>) -> f64 {
    v.fold(0.0, |m, x| m.max(x.abs()))
}

/// Part 1: per-evaluation agreement over 200 in-process-driven steps (worst relative errors of
/// energy, force and virial against 1e-12).
fn per_evaluation(sys: System, n_clients: usize, what: &str) {
    let (_d, _clients, mut sock) = socket(&sys, n_clients);
    let pot = LennardJones::new(sys.lj).unwrap();
    let mut sim =
        RingPolymerSim::new(sys.state.clone(), sys.cell, integrator(&sys), Box::new(pot)).unwrap();
    let mut inproc = LennardJones::new(sys.lj).unwrap();
    let mut rng = Rng::from_seed(5);
    let (mut we, mut wf, mut ww) = (0.0f64, 0.0f64, 0.0f64);
    let p = sys.state.q.n_beads();
    let n3 = sys.state.q.n_atoms() * 3;
    for step in 0..=200 {
        if step > 0 {
            sim.step(&mut rng).unwrap();
        }
        let q = &sim.state().q;
        let s = sock.evaluate(q, sys.cell.as_ref(), None).unwrap();
        let r = inproc.evaluate(q, sys.cell.as_ref(), None).unwrap();
        for j in 0..p {
            let e = (s.energies[j] - r.energies[j]).abs() / r.energies[j].abs();
            let fs = &s.forces.as_slice()[j * n3..(j + 1) * n3];
            let fr = &r.forces.as_slice()[j * n3..(j + 1) * n3];
            let f = max_abs(fs.iter().zip(fr).map(|(a, b)| a - b)) / max_abs(fr.iter().copied());
            we = we.max(e);
            wf = wf.max(f);
        }
        match (s.virial, r.virial) {
            (Some(a), Some(b)) => {
                let fa: Vec<f64> = a.iter().flatten().copied().collect();
                let fb: Vec<f64> = b.iter().flatten().copied().collect();
                let w =
                    max_abs(fa.iter().zip(&fb).map(|(x, y)| x - y)) / max_abs(fb.iter().copied());
                ww = ww.max(w);
            }
            (None, None) => assert!(p > 1, "P-5/K-6: virial expected at P = 1"),
            (a, b) => panic!("{what}: virial presence differs: socket {a:?}, in-process {b:?}"),
        }
    }
    eprintln!("{what}: worst relative error energy {we:.2e}, force {wf:.2e}, virial {ww:.2e}");
    assert!(we <= 1e-12, "{what}: energy {we:e}");
    assert!(wf <= 1e-12, "{what}: forces {wf:e}");
    assert!(ww <= 1e-12, "{what}: virial {ww:e}");
}

/// Part 2: 50 steps through the socket vs in-process from the same state and seed.
fn trajectory(sys: System, n_clients: usize, what: &str) {
    let (_d, _clients, sock) = socket(&sys, n_clients);
    let pot = LennardJones::new(sys.lj).unwrap();
    let mut a =
        RingPolymerSim::new(sys.state.clone(), sys.cell, integrator(&sys), Box::new(pot)).unwrap();
    let mut b = RingPolymerSim::new(
        sys.state.clone(),
        sys.cell,
        integrator(&sys),
        Box::new(sock),
    )
    .unwrap();
    let (mut ra, mut rb) = (Rng::from_seed(9), Rng::from_seed(9));
    for _ in 0..50 {
        a.step(&mut ra).unwrap();
        b.step(&mut rb).unwrap();
    }
    let dq = max_abs(
        a.state()
            .q
            .as_slice()
            .iter()
            .zip(b.state().q.as_slice())
            .map(|(x, y)| x - y),
    );
    let va: f64 = a.energies().iter().sum::<f64>();
    let vb: f64 = b.energies().iter().sum::<f64>();
    let dv = (va - vb).abs() / va.abs();
    eprintln!("{what}: after 50 steps max |dq| = {dq:.2e} bohr, |dV|/|V| = {dv:.2e}");
    assert!(dq <= 1e-10, "{what}: positions differ by {dq:e} bohr");
    assert!(dv <= 1e-12, "{what}: potential differs by {dv:e} relative");
}

/// K-2, K-3, K-6, P-3, P-5, E-8: neon liquid, P = 1, one client: energies, forces and the virial
/// per evaluation to 1e-12 relative.
#[test]
fn bridge_neon_p1_per_evaluation() {
    per_evaluation(neon(1), 1, "neon P = 1");
}

/// K-2 … K-4, P-3: neon liquid, P = 4, two clients: per evaluation to 1e-12 relative.
#[test]
fn bridge_neon_p4_per_evaluation() {
    per_evaluation(neon(4), 2, "neon P = 4");
}

/// K-3 (gas phase: zero cell → open boundaries in the client), P-3: 13-atom cluster, P = 4, three
/// clients: per evaluation to 1e-12 relative.
#[test]
fn bridge_cluster_gas_phase_per_evaluation() {
    per_evaluation(cluster(4), 3, "cluster P = 4");
}

/// K-1 … K-4, D-3 note: 50 PIMD steps through the socket match the in-process run (neon P = 4,
/// two clients; cluster P = 4, three clients) within the stated tolerances.
#[test]
fn bridge_pimd_trajectories_match() {
    trajectory(neon(4), 2, "neon P = 4");
    trajectory(cluster(4), 3, "cluster P = 4");
}
