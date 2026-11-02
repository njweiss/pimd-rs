//! Long validation with MACE-OFF23 (MILESTONE_6 "Required tests: Long"; SPEC P-6, U-3, K-1 …
//! K-6, D-3 note). Features `torch` and `long-validation` (CHECK_LONG=1 CHECK_TORCH=1).
//!
//! Needs `PIMD_MACE_MODEL`: the path of a MACE-OFF23 model file as distributed (e.g.
//! `ff_models/mace_off23/MACE-OFF23_small.model`; never committed, never downloaded here). Without
//! it the test prints a message and returns. The test exports the model to a temporary directory
//! with `crates/pimd-torch/scripts/export_mace.py` (float64), so the in-process adapter and the
//! Python clients see the same weights; Python comes from the pixi `ml` environment
//! (`.pixi/envs/ml/bin/python` when present, else `pixi run -e ml python`).
//!
//! System: gas-phase water at the M5 start geometry (O 0 0 0.1173; H 0 ±0.7572 −0.4692 Å), masses
//! 15.999, 1.008, 1.008 amu, P = 8, 300 K, dt = 0.5 fs, BAOAB + PILE-L with τ_0 = 20 fs, thermal
//! momenta N(0, m P/β) (RP-3). In-process: `MacePotential`, CPU, float64, 1 thread. Socket path:
//! `SocketPotential` with four `tests/clients/mace_client.py` clients (MACECalculator on the model
//! file, float64, CPU, 1 thread).
//!
//! 1. Per evaluation (brief): the in-process potential drives 100 steps; at every step the same
//!    bead positions go through the socket: |ΔE| ≤ 1e-12·|E| and max|ΔF| ≤ 1e-10·max|F| per bead
//!    (spike Q7: ΔE bitwise, ΔF ≤ 4e-16·max|F| over 160 evaluations).
//! 2. Trajectory (brief: "with the M5 bridge's tolerances", `pimd-socket/tests/bridge.rs`): 50
//!    steps through the socket vs in-process from the same state and seed: max|Δq| ≤ 1e-10 bohr and
//!    |ΔΣ_j V_j| ≤ 1e-12·|Σ_j V_j|.
//!
//! Unit literals (U-2, CODATA 2018) are written out here.
#![cfg(all(feature = "torch", feature = "long-validation"))]

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use pimd_core::{Beads, Rng};
use pimd_integrators::{RingPolymerIntegrator, RingPolymerSim, RingState, Thermostat};
use pimd_potentials::Potential;
use pimd_socket::{Address, SocketPotential};
use pimd_torch::{Device, Dtype, MacePotential};

const K_B: f64 = 3.1668115634556e-6; // Eh/K
const AMU: f64 = 1822.888486209; // m_e
const BOHR_A: f64 = 0.529177210903; // Å/bohr
const FS: f64 = 41.341373336561; // ħ/Eh per fs
const P: usize = 8;
const N_CLIENTS: usize = 4;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The `ml` environment's Python (direct when present, else through pixi).
fn ml_python() -> Command {
    let py = root().join(".pixi/envs/ml/bin/python");
    if py.exists() {
        Command::new(py)
    } else {
        let mut c = Command::new("pixi");
        c.args(["run", "-e", "ml", "python"]);
        c
    }
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

/// The model file from `PIMD_MACE_MODEL`, or `None` (with a message) when unset.
fn model_file() -> Option<PathBuf> {
    match std::env::var_os("PIMD_MACE_MODEL") {
        Some(p) => {
            let p = PathBuf::from(p);
            assert!(p.is_file(), "PIMD_MACE_MODEL={} is not a file", p.display());
            Some(p)
        }
        None => {
            eprintln!(
                "SKIP long MACE-OFF23 validation: set PIMD_MACE_MODEL to a MACE-OFF23 model file \
                 (e.g. ff_models/mace_off23/MACE-OFF23_small.model)"
            );
            None
        }
    }
}

/// Exports `model` (float64) into `dir`; returns the export's path.
fn export(model: &Path, dir: &Path) -> PathBuf {
    let out = dir.join("export.pt");
    let st = ml_python()
        .current_dir(root())
        .arg(root().join("crates/pimd-torch/scripts/export_mace.py"))
        .arg(model)
        .arg("--out")
        .arg(&out)
        .stdin(Stdio::null())
        .status()
        .expect("cannot run export_mace.py (pixi ml env)");
    assert!(st.success(), "export_mace.py failed");
    out
}

/// Binds a socket and starts `N_CLIENTS` MACE clients on `model`.
fn socket(model: &Path) -> (tempfile::TempDir, Clients, SocketPotential) {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("mace.sock");
    let address = format!("unix:{}", path.display());
    let script = root().join("crates/pimd-torch/tests/clients/mace_client.py");
    let clients = Clients(
        (0..N_CLIENTS)
            .map(|_| {
                ml_python()
                    .current_dir(root())
                    .arg(&script)
                    .args(["--address", &address])
                    .arg(model)
                    .args(["--symbols", "O", "H", "H"])
                    .env("OMP_NUM_THREADS", "1")
                    .stdin(Stdio::null())
                    .spawn()
                    .expect("cannot start mace_client.py (pixi ml env)")
            })
            .collect(),
    );
    let pot = SocketPotential::bind_and_accept(
        &Address::Unix(path),
        3,
        N_CLIENTS,
        Duration::from_secs(300),
    )
    .expect("MACE clients did not connect");
    (d, clients, pot)
}

fn initial_state(beta: f64) -> RingState {
    let x = [
        [0.0, 0.0, 0.1173],
        [0.0, 0.7572, -0.4692],
        [0.0, -0.7572, -0.4692],
    ];
    let masses = [15.999 * AMU, 1.008 * AMU, 1.008 * AMU];
    let mut q = Beads::zeros(P, 3);
    let mut mom = Beads::zeros(P, 3);
    let mut rng = Rng::from_seed(606);
    for i in 0..3 {
        let sd = (masses[i] * P as f64 / beta).sqrt();
        for j in 0..P {
            q.set(j, i, x[i].map(|c| c / BOHR_A));
            mom.set(j, i, [0, 1, 2].map(|_| sd * rng.normal()));
        }
    }
    RingState {
        q,
        p: mom,
        masses: masses.to_vec(),
    }
}

fn integrator(beta: f64) -> RingPolymerIntegrator {
    RingPolymerIntegrator::new(
        0.5 * FS,
        beta,
        P,
        Thermostat::PileL {
            tau0: Some(20.0 * FS),
        },
    )
    .unwrap()
}

fn mace(export: &Path) -> MacePotential {
    MacePotential::load(export, &[8, 1, 1], Device::Cpu, Dtype::F64, 1, true).unwrap()
}

fn max_abs(v: impl Iterator<Item = f64>) -> f64 {
    v.fold(0.0, |m, x| m.max(x.abs()))
}

/// P-6, U-3, K-1 … K-6, D-3 note: MACE-OFF23 water, in-process vs socket with Python MACE
/// clients, per evaluation (100 steps) and over a 50-step trajectory, within the brief's
/// tolerances.
#[test]
fn mace_off23_in_process_matches_socket_clients() {
    let Some(model) = model_file() else {
        return;
    };
    let tmp = tempfile::tempdir().unwrap();
    let exported = export(&model, tmp.path());
    let beta = 1.0 / (300.0 * K_B);

    // 1. Per evaluation.
    let (_d, _clients, mut sock) = socket(&model);
    let mut sim = RingPolymerSim::new(
        initial_state(beta),
        None,
        integrator(beta),
        Box::new(mace(&exported)),
    )
    .unwrap();
    let mut inproc = mace(&exported);
    let mut rng = Rng::from_seed(5);
    let (mut we, mut wf) = (0.0f64, 0.0f64);
    for step in 0..=100 {
        if step > 0 {
            sim.step(&mut rng).unwrap();
        }
        let q = &sim.state().q;
        let s = sock.evaluate(q, None, None).unwrap();
        let r = inproc.evaluate(q, None, None).unwrap();
        for j in 0..P {
            we = we.max((s.energies[j] - r.energies[j]).abs() / s.energies[j].abs());
            let fs = &s.forces.as_slice()[j * 9..(j + 1) * 9];
            let fr = &r.forces.as_slice()[j * 9..(j + 1) * 9];
            let df = max_abs(fs.iter().zip(fr).map(|(a, b)| a - b));
            wf = wf.max(df / max_abs(fs.iter().copied()));
        }
    }
    eprintln!("per evaluation: worst |dE|/|E| {we:.2e}, |dF|/max|F| {wf:.2e}");
    assert!(we <= 1e-12, "energy {we:e}");
    assert!(wf <= 1e-10, "forces {wf:e}");
    drop(sock);

    // 2. Trajectory.
    let (_d2, _clients2, sock) = socket(&model);
    let mut a = RingPolymerSim::new(
        initial_state(beta),
        None,
        integrator(beta),
        Box::new(mace(&exported)),
    )
    .unwrap();
    let mut b =
        RingPolymerSim::new(initial_state(beta), None, integrator(beta), Box::new(sock)).unwrap();
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
    let va: f64 = a.energies().iter().sum();
    let vb: f64 = b.energies().iter().sum();
    let dv = (va - vb).abs() / vb.abs();
    eprintln!("trajectory: after 50 steps max |dq| {dq:.2e} bohr, |dV|/|V| {dv:.2e}");
    assert!(dq <= 1e-10, "positions differ by {dq:e} bohr");
    assert!(dv <= 1e-12, "potential differs by {dv:e} relative");
}
