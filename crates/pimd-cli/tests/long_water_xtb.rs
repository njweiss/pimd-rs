//! Gas-phase water on GFN2-xTB through the socket backend vs i-PI (MILESTONE_5 "Long validation:
//! Water on GFN2-xTB vs i-PI"; SPEC K-1 … K-6, I-1, T-1, T-2, E-1 … E-4, D-3 note). Feature
//! `socket-validation`, run only by `CHECK_LONG=1 CHECK_IPI=1 ./scripts/check.sh` (release);
//! needs the pixi `ref` environment (tblite) for the force clients.
//!
//! Reference: `data/ipi_water_xtb.json`, written by `data/gen_ipi_water_xtb.py` (i-PI 3.3.1 driving
//! the same client, `crates/pimd-socket/tests/clients/ipi_client.py` backend `xtb`, with four
//! single-threaded, stateless clients). The system, protocol, block lengths and observables are
//! read from that file, so both sides run the same thing: one water molecule (O, H, H; 15.999,
//! 1.008, 1.008 amu; the brief's start geometry, every bead at it), 300 K, P = 16, dt = 0.5 fs,
//! PILE-L with τ_0 = 20 fs and γ_k = 2ω_k, gas phase (engine: no box, so the clients receive the
//! zero cell of K-3). The engine runs `pimd_cli::run` with a socket potential and four clients.
//!
//! Compared (per molecule, Eh; geometry bead-averaged): ⟨E_vir⟩, ⟨E_prim⟩, ⟨V̄⟩, ⟨r_OH⟩ (bohr),
//! ⟨θ_HOH⟩ (degrees), and the r_OH histogram per bin, each within 5σ_combined (WORKFLOW tolerance
//! policy; σ from block averages of the generator's block length and 4× that length, the larger,
//! same procedure on both sides). Energies come from `observables.csv` (per atom × 3), geometry
//! from `trajectory.xyz` (Å, converted back with the U-2 literal).
//!
//! Discretisation-bias allowance: 0. Both codes use the same splitting (B A O A B with the exact
//! free ring-polymer propagator; i-PI `splitting="baoab"`) at the same dt, so their O(dt²) bias is
//! common. The size of the bias was measured with the reference implementation at dt = 0.5 fs and
//! dt/2 = 0.25 fs (same protocol, twice the steps at dt/2): see "Results" below; it is below a
//! third of the tolerance.
//!
//! Results with the throwaway reference implementation (release, this protocol; per molecule):
//! ⟨E_vir⟩ = −5.049908 ± 0.000023 Eh vs i-PI −5.049943 ± 0.000022 Eh (1.10 σ_c); ⟨E_prim⟩ =
//! −5.049985 ± 0.000047 vs −5.050054 ± 0.000048 Eh (1.03 σ_c); ⟨V̄⟩ = −5.061468 ± 0.000017 vs
//! −5.061498 ± 0.000015 Eh (1.32 σ_c); ⟨r_OH⟩ = 1.849247 ± 0.000422 vs 1.849065 ± 0.000432 bohr
//! (0.30 σ_c); ⟨θ_HOH⟩ = 107.0878 ± 0.0149° vs 107.0927 ± 0.0154° (−0.23 σ_c); r_OH histogram
//! worst bin 2.19 σ_c of 25. dt/2 check (engine, 0.25 fs, 808 000 steps, same 2-ps blocks):
//! E_vir(dt) − E_vir(dt/2) = +6.2e-5 ± 3.3e-5 Eh (1.9 σ), V̄ +3.4e-5 ± 2.4e-5 Eh (1.5 σ), E_prim
//! −0.5e-5 ± 6.2e-5 Eh, r_OH +0.0006 ± 0.0006 bohr (1.1 σ), θ −0.006 ± 0.022° : no resolved bias
//! (the E_vir difference is 0.39 of the 5σ_c tolerance, 1.6e-4 Eh, but not significant; and it is
//! common to both codes). Engine run time: 30 min (4 xTB clients, 4-thread pool; about 4.5 ms per
//! step, with other validation runs sharing the machine; the i-PI reference takes about 55 min).
#![cfg(feature = "socket-validation")]

#[path = "../../pimd-potentials/tests/common/lammps.rs"]
mod lammps;

mod common;

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;

use common::{mean_and_sigma, observables, parse, pool, read_trajectory, tmp};
use lammps::Json;
use pimd_cli::run;

const REF_JSON: &str = include_str!("data/ipi_water_xtb.json");
const BOHR_A: f64 = 0.529177210903; // Å per bohr (U-2)
const N_CLIENTS: usize = 4;
const SEED: u64 = 31337;

fn reference() -> &'static Json {
    static J: OnceLock<Json> = OnceLock::new();
    J.get_or_init(|| Json::parse(REF_JSON))
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn protocol(key: &str) -> usize {
    reference().get("protocol").get(key).num() as usize
}

/// Python clients, killed when dropped.
struct Clients(Vec<Child>);

impl Drop for Clients {
    fn drop(&mut self) {
        for c in &mut self.0 {
            c.kill().ok();
            c.wait().ok();
        }
    }
}

/// `n` `ipi_client.py xtb` clients (single-threaded) for `address`.
fn xtb_clients(address: &str, n: usize) -> Clients {
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
            .args(["--address", address, "xtb", "--symbols", "O", "H", "H"])
            .env("OMP_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .stdin(Stdio::null());
        out.push(
            cmd.spawn()
                .expect("cannot start the xtb client (pixi ref env)"),
        );
    }
    Clients(out)
}

/// The engine config of the reference system, with the socket at `address`.
fn water_toml(address: &str, dt_fs: f64, n_equil: usize, n_steps: usize) -> String {
    let s = reference().get("system");
    let list = |v: &Json| {
        v.arr()
            .iter()
            .map(|x| format!("{}", x.num()))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let pos = s
        .get("positions_angstrom")
        .arr()
        .iter()
        .map(|p| format!("[{}]", list(p)))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"
[system]
n_atoms = 3
masses_amu = [{}]
symbols = ["O", "H", "H"]
positions_angstrom = [{pos}]

[potential]
kind = "socket"
address = "{address}"
n_clients = {N_CLIENTS}
timeout_s = 120.0

[sampling]
temperature_kelvin = {}
n_beads = {}
dt_fs = {dt_fs}
tau0_fs = {}
n_equil = {n_equil}
n_steps = {n_steps}
stride = {}
seed = {SEED}

[output]
checkpoint_every = {n_steps}
trajectory_every = {}
"#,
        list(s.get("masses_amu")),
        s.get("temperature_kelvin").num(),
        s.get("n_beads").num() as usize,
        s.get("tau0_fs").num(),
        protocol("stride"),
        protocol("traj_stride"),
    )
}

/// Per-sample series of one engine run.
struct Series {
    e_vir: Vec<f64>,
    e_prim: Vec<f64>,
    v: Vec<f64>,
    r_oh: Vec<f64>,
    theta: Vec<f64>,
    /// Per frame: the 2P O–H distances (bohr).
    r_all: Vec<Vec<f64>>,
}

/// Runs the engine at the reference dt (production length from the reference protocol).
fn engine_run() -> &'static Series {
    static S: OnceLock<Series> = OnceLock::new();
    S.get_or_init(|| {
        let dt = reference().get("system").get("dt_fs").num();
        let (n_equil, n_prod) = (protocol("n_equil"), protocol("n_prod"));
        let sock = tmp();
        let path = sock.path().join("water.sock");
        let address = format!("unix:{}", path.display());
        let cfg = parse(&water_toml(&address, dt, n_equil, n_prod));
        let out = tmp();
        let _clients = xtb_clients(&address, N_CLIENTS);
        let t0 = std::time::Instant::now();
        pool(4).install(|| run(&cfg, out.path())).unwrap();
        eprintln!("engine run: {:.0} s", t0.elapsed().as_secs_f64());
        series(out.path(), cfg.sampling.n_beads)
    })
}

/// Reads observables (per atom → per molecule) and geometry from a run directory.
fn series(dir: &Path, p: usize) -> Series {
    let rows = observables(dir);
    let col = |c: usize| rows.iter().map(|r| 3.0 * r[c]).collect::<Vec<f64>>();
    let frames = read_trajectory(&dir.join("trajectory.xyz"), p, 3);
    let (mut r_oh, mut theta, mut r_all) = (Vec::new(), Vec::new(), Vec::new());
    for (_, rows) in &frames {
        let (mut rs, mut ts) = (Vec::new(), 0.0);
        for j in 0..p {
            let x = |i: usize| rows[j * 3 + i].1.map(|v| v / BOHR_A);
            let (o, h1, h2) = (x(0), x(1), x(2));
            let d1 = [0, 1, 2].map(|d| h1[d] - o[d]);
            let d2 = [0, 1, 2].map(|d| h2[d] - o[d]);
            let n = |d: [f64; 3]| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let (r1, r2) = (n(d1), n(d2));
            let c = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / (r1 * r2);
            ts += c.clamp(-1.0, 1.0).acos().to_degrees();
            rs.push(r1);
            rs.push(r2);
        }
        // Same order as the generator: the P first bonds, then the P second bonds.
        let ordered: Vec<f64> = (0..p)
            .map(|j| rs[2 * j])
            .chain((0..p).map(|j| rs[2 * j + 1]))
            .collect();
        r_oh.push(ordered.iter().sum::<f64>() / (2 * p) as f64);
        theta.push(ts / p as f64);
        r_all.push(ordered);
    }
    Series {
        e_vir: col(2),
        e_prim: col(1),
        v: col(3),
        r_oh,
        theta,
        r_all,
    }
}

/// Per-bin (mean, σ) of the r_OH density, σ from blocks of `len` and `4·len` frames (larger).
fn histogram_stats(r_all: &[Vec<f64>], len: usize) -> (Vec<f64>, Vec<f64>) {
    let h = reference().get("protocol");
    let (lo, hi) = (h.get("hist_lo_bohr").num(), h.get("hist_hi_bohr").num());
    let nb = h.get("hist_bins").num() as usize;
    let width = (hi - lo) / nb as f64;
    let dens: Vec<Vec<f64>> = r_all
        .iter()
        .map(|r| {
            let mut c = vec![0.0; nb];
            for &x in r {
                let k = ((x - lo) / width).floor();
                if k >= 0.0 && (k as usize) < nb {
                    c[k as usize] += 1.0;
                }
            }
            c.iter().map(|v| v / (r.len() as f64 * width)).collect()
        })
        .collect();
    let sigma = |l: usize, b: usize| {
        let n = dens.len() / l;
        let means: Vec<f64> = (0..n)
            .map(|k| (0..l).map(|i| dens[k * l + i][b]).sum::<f64>() / l as f64)
            .collect();
        let m = means.iter().sum::<f64>() / n as f64;
        let v = means.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64;
        (v / n as f64).sqrt()
    };
    (0..nb)
        .map(|b| {
            let m = dens.iter().map(|d| d[b]).sum::<f64>() / dens.len() as f64;
            (m, sigma(len, b).max(sigma(4 * len, b)))
        })
        .unzip()
}

/// Asserts |engine − reference| ≤ 5σ_combined for `key` in the reference JSON.
fn compare(key: &str, x: &[f64], len: usize, unit: &str) {
    let (m, s) = mean_and_sigma(x, len, key);
    let r = reference().get(key);
    let (mr, sr) = (r.get("mean").num(), r.get("sigma").num());
    let sc = (s * s + sr * sr).sqrt();
    eprintln!(
        "{key}: engine {m:.6} ± {s:.6} {unit}, i-PI {mr:.6} ± {sr:.6} {unit}, diff {:.2} σ_c",
        (m - mr) / sc
    );
    assert!(
        (m - mr).abs() <= 5.0 * sc,
        "{key}: engine {m} ± {s} vs i-PI {mr} ± {sr} ({:.1} σ_c)",
        (m - mr).abs() / sc
    );
}

/// U-2, U-3: the generator's atomic-unit values (its own CODATA 2018 conversion) equal the
/// engine's parsed config to 1e-15 relative, so both codes simulate the same system.
#[test]
fn same_system_as_ipi() {
    let d = reference().get("derived_atomic_units");
    let dt = reference().get("system").get("dt_fs").num();
    let c = parse(&water_toml("unix:/tmp/unused.sock", dt, 1, 1));
    let close = |a: f64, b: f64, w: &str| {
        assert!(
            (a - b).abs() <= 1e-15 * b.abs().max(1e-300) || a == b,
            "{w}: engine {a} vs generator {b}"
        )
    };
    for (m, r) in c.system.masses.iter().zip(d.get("masses_me").nums()) {
        close(*m, r, "mass");
    }
    let pos = c.system.positions.as_ref().unwrap();
    for (p, r) in pos.iter().zip(d.get("positions_bohr").arr()) {
        for (a, b) in p.iter().zip(r.nums()) {
            close(*a, b, "position");
        }
    }
    assert!(c.system.cell.is_none());
    close(c.sampling.beta, d.get("beta").num(), "beta");
    close(c.sampling.dt, d.get("dt").num(), "dt");
    close(c.sampling.tau0, d.get("tau0").num(), "tau0");
    assert_eq!(c.sampling.n_beads, 16);
}

/// E-1 … E-4, K-1 … K-6 ((new) validation vs i-PI): ⟨E_vir⟩, ⟨E_prim⟩ and ⟨V̄⟩ per molecule within 5σ_combined.
#[test]
fn water_energies_match_ipi() {
    let s = engine_run();
    let len = protocol("block_steps") / protocol("stride");
    compare("e_vir", &s.e_vir, len, "Eh");
    compare("e_prim", &s.e_prim, len, "Eh");
    compare("v", &s.v, len, "Eh");
}

/// K-1 … K-6, I-1, T-1 ((new) validation, structure vs i-PI): bead-averaged ⟨r_OH⟩ and ⟨θ_HOH⟩ within 5σ_combined, and the r_OH
/// histogram in every bin within 5σ_combined (bins empty on both sides must agree exactly).
#[test]
fn water_geometry_matches_ipi() {
    let s = engine_run();
    let len = protocol("block_steps") / protocol("traj_stride");
    compare("r_oh", &s.r_oh, len, "bohr");
    compare("theta_hoh_deg", &s.theta, len, "deg");
    let (m, sg) = histogram_stats(&s.r_all, len);
    let h = reference().get("r_oh_hist");
    let (mr, sr) = (h.get("density").nums(), h.get("sigma").nums());
    let mut worst = 0.0f64;
    for b in 0..m.len() {
        let sc = (sg[b] * sg[b] + sr[b] * sr[b]).sqrt();
        if sc == 0.0 {
            assert_eq!(m[b], mr[b], "bin {b}: no samples on one side only");
            continue;
        }
        let z = (m[b] - mr[b]).abs() / sc;
        worst = worst.max(z);
        assert!(
            z <= 5.0,
            "r_OH histogram bin {b}: engine {} ± {} vs i-PI {} ± {} ({z:.1} σ_c)",
            m[b],
            sg[b],
            mr[b],
            sr[b]
        );
    }
    eprintln!("r_OH histogram: worst bin {worst:.2} σ_c of {}", m.len());
}
