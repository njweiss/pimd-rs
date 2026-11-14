//! Shared by the tests of the supported MACE socket client (MILESTONE_10 "M10a: Required tests";
//! SPEC K-7): starting `crates/pimd-torch/scripts/mace_client.py`, driving it through
//! `SocketPotential`, the expected values from the in-process `MacePotential` on the v2 export of
//! the same model, and the test systems.
//!
//! Systems (written here; geometry only, not physics): small molecules built from standard
//! templates (Å), placed in a cluster (gas phase) or in a non-cubic orthorhombic box
//! L = 21.0 × 23.5 × 26.0 bohr (L_min/2 = 10.5 bohr > r_max = 5 Å = 9.449 bohr, P-6/K-7). In the
//! box every molecule is within r_max of another one only through a periodic image along x, y or
//! z (one pair per axis), and three molecules are shifted by whole box lengths (unwrapped
//! positions, C-5), so a missing minimum image, a cubic box or permuted axes all change the edges.
//! P = 4 beads: the base geometry plus a deterministic uniform displacement (SplitMix64, ±0.2 bohr
//! per component) per bead.
//!
//! Memory (host RAM ≈ 11 GB): at most two client processes are alive at a time. Tests of one
//! binary that start clients hold `lock()` while their clients live; test binaries run one after
//! the other.
#![allow(dead_code)]

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use pimd_core::units::BOHR_IN_ANGSTROM;
use pimd_core::{Beads, Cell};
use pimd_potentials::{Evaluation, Potential, PotentialError};
use pimd_socket::{Address, SocketPotential};
use pimd_torch::{Device, Dtype, MacePotential};

/// Accept, read and write timeout of the engine side: covers the client's start-up (Python,
/// torch and mace imports, model load: ≈ 5–15 s measured on CPU; cueq conversion on CUDA ≈ 20 s).
pub const TIMEOUT: Duration = Duration::from_secs(300);

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The script under test (K-7).
pub fn client_script() -> PathBuf {
    root().join("crates/pimd-torch/scripts/mace_client.py")
}

fn data(name: &str) -> PathBuf {
    root().join("crates/pimd-torch/tests/data").join(name)
}

/// The committed tiny MACE test model (pickled, gen_tiny_mace.py; H C N O, r_max 5 Å).
pub fn tiny_model() -> PathBuf {
    data("tiny_mace.model")
}

/// Its committed v2 float64 export (gen_tiny_mace_v2.py): the adapter's expectation.
pub fn tiny_export() -> PathBuf {
    data("tiny_mace_v2_f64.pt")
}

/// The tiny model with the E0 table BIG_E0 (≈ ±1e6 eV), pickled (gen_tiny_mace_client_models.py).
pub fn bige0_model() -> PathBuf {
    data("tiny_mace_bige0.model")
}

/// Its committed v2 float64 export (gen_tiny_mace_v2.py; the generator above checks that the
/// pickled file and this export give the same interaction energies and forces).
pub fn bige0_export() -> PathBuf {
    data("tiny_mace_bige0_v2_f64.pt")
}

/// `tiny_mace.model` with `use_reduced_cg = False`, which its weights' basis requires for mace's
/// cueq conversion (gen_tiny_mace_client_models.py, QUESTIONS.md Q41); its export is
/// `tiny_export()`.
pub fn cueq_model() -> PathBuf {
    data("tiny_mace_cueq.model")
}

/// BIG_E0 of gen_tiny_mace_v2.py (eV; H, C, N, O).
pub const BIG_E0: [(u8, f64); 4] = [
    (1, 1000000.03),
    (6, -2000000.05),
    (7, -1500000.3),
    (8, -2002069.23),
];

static LOCK: Mutex<()> = Mutex::new(());

/// Serialises the tests of one binary that start clients (at most two clients alive).
pub fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The `ml` environment's Python (direct when present, else through pixi).
pub fn ml_python() -> Command {
    let py = root().join(".pixi/envs/ml/bin/python");
    if py.exists() {
        Command::new(py)
    } else {
        let mut c = Command::new("pixi");
        c.args(["run", "-e", "ml", "python"]);
        c
    }
}

/// How to start a client: interpreter, model file and the optional flags (None = the default).
#[derive(Clone)]
pub struct ClientOpts {
    pub python: PathBuf,
    pub model: PathBuf,
    pub backend: Option<&'static str>,
    pub dtype: Option<&'static str>,
    pub device: Option<&'static str>,
}

impl ClientOpts {
    /// e3nn, float64, CPU, all by default (no flags given) on `model`, `ml` env Python.
    pub fn defaults(model: PathBuf) -> ClientOpts {
        let py = root().join(".pixi/envs/ml/bin/python");
        ClientOpts {
            python: py,
            model,
            backend: None,
            dtype: None,
            device: None,
        }
    }
}

/// A running client process, killed (if still running) on drop; its stderr goes to `log`.
pub struct ClientProc {
    pub child: Child,
    pub log: PathBuf,
}

impl ClientProc {
    /// The client's stderr so far.
    pub fn stderr(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap_or_default()
    }

    /// Waits up to `limit` for the process to exit.
    pub fn wait_for(&mut self, limit: Duration) -> Option<ExitStatus> {
        let end = Instant::now() + limit;
        loop {
            if let Some(s) = self.child.try_wait().expect("try_wait") {
                return Some(s);
            }
            if Instant::now() >= end {
                return None;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

impl Drop for ClientProc {
    fn drop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            self.child.kill().ok();
        }
        self.child.wait().ok();
    }
}

/// Starts the client: `python mace_client.py MODEL --config CONFIG --address unix:SOCK [flags]`
/// (brief "Command line"), from the repository root, stdin closed, stderr to `dir/<tag>.log`.
pub fn spawn(opts: &ClientOpts, config: &Path, sock: &Path, dir: &Path, tag: &str) -> ClientProc {
    let mut cmd = if opts.python.exists() {
        Command::new(&opts.python)
    } else {
        ml_python()
    };
    cmd.current_dir(root())
        .arg(client_script())
        .arg(&opts.model)
        .arg("--config")
        .arg(config)
        .arg("--address")
        .arg(format!("unix:{}", sock.display()));
    for (flag, v) in [
        ("--backend", opts.backend),
        ("--dtype", opts.dtype),
        ("--device", opts.device),
    ] {
        if let Some(v) = v {
            cmd.arg(flag).arg(v);
        }
    }
    let log = dir.join(format!("{tag}.log"));
    let child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(File::create(&log).unwrap())
        .spawn()
        .unwrap_or_else(|e| panic!("cannot start {}: {e}", opts.python.display()));
    ClientProc { child, log }
}

/// A complete engine config (kind = "socket", K clients at `sock`) for `symbols` with inline
/// symbols: the file the client reads its species from (K-7). Masses are placeholders (unused).
pub fn socket_config(symbols: &[&str], sock: &Path, k: usize) -> String {
    let pos: Vec<String> = (0..symbols.len())
        .map(|i| format!("[{}.0, 0.0, 0.0]", 2 * i))
        .collect();
    let sym: Vec<String> = symbols.iter().map(|s| format!("\"{s}\"")).collect();
    format!(
        r#"# MILESTONE_10 test config: the client reads [system] symbols (K-7)
[system]
n_atoms = {n}
mass_amu = 1.0
symbols = [{sym}]
positions_angstrom = [{pos}]

[potential]
kind = "socket"
address = "unix:{addr}"
n_clients = {k}
timeout_s = 300.0

[sampling]
temperature_kelvin = 300.0
n_beads = 4
dt_fs = 0.5
tau0_fs = 20.0
n_equil = 0
n_steps = 1
stride = 1
seed = 1

[output]
checkpoint_every = 1
trajectory_every = 1
"#,
        n = symbols.len(),
        sym = sym.join(", "),
        pos = pos.join(", "),
        addr = sock.display(),
    )
}

/// Writes `socket_config` to `dir/config.toml` (checked to parse as an engine config).
pub fn write_config(dir: &Path, symbols: &[&str], sock: &Path, k: usize) -> PathBuf {
    let text = socket_config(symbols, sock, k);
    pimd_cli::Config::from_toml_str(&text)
        .unwrap_or_else(|e| panic!("test config rejected by the engine: {e:?}"));
    let path = dir.join("config.toml");
    std::fs::write(&path, text).unwrap();
    path
}

/// A complete engine config like `socket_config`, but with the atoms (symbols and positions, Å)
/// from an xyz file `dir/atoms.xyz` given by its absolute path as `positions_xyz` (the client
/// takes the symbols from that file, K-7); written to `dir/config_xyz.toml` and checked to parse.
pub fn write_config_xyz(
    dir: &Path,
    symbols: &[&str],
    positions_angstrom: &[[f64; 3]],
    sock: &Path,
    k: usize,
) -> PathBuf {
    let xyz = dir.join("atoms.xyz");
    let mut t = format!("{}\nMILESTONE_10 test atoms\n", symbols.len());
    for (s, x) in symbols.iter().zip(positions_angstrom) {
        t.push_str(&format!("{s} {} {} {}\n", x[0], x[1], x[2]));
    }
    std::fs::write(&xyz, t).unwrap();
    let inline = socket_config(symbols, sock, k);
    let mut text = String::new();
    for line in inline.lines() {
        if line.starts_with("symbols") || line.starts_with("positions_angstrom") {
            continue;
        }
        text.push_str(line);
        text.push('\n');
        if line == "mass_amu = 1.0" {
            text.push_str(&format!("positions_xyz = \"{}\"\n", xyz.display()));
        }
    }
    pimd_cli::Config::from_toml_str(&text)
        .unwrap_or_else(|e| panic!("test config rejected by the engine: {e:?}"));
    let path = dir.join("config_xyz.toml");
    std::fs::write(&path, text).unwrap();
    path
}

/// K clients of `opts` on a fresh socket in `dir`, accepted by a `SocketPotential` for
/// `symbols.len()` atoms (K-1, K-4: client k serves beads k, k + K, …).
pub struct Session {
    pub pot: SocketPotential,
    pub clients: Vec<ClientProc>,
}

impl Session {
    /// Config with inline `symbols` (written to `dir`).
    pub fn start(opts: &ClientOpts, symbols: &[&str], k: usize, dir: &Path) -> Session {
        let sock = dir.join("k7.sock");
        let config = write_config(dir, symbols, &sock, k);
        Session::start_with(opts, &config, &sock, symbols.len(), k, dir)
    }

    /// Clients reading `config`, socket `sock`, N = `n_atoms`.
    pub fn start_with(
        opts: &ClientOpts,
        config: &Path,
        sock: &Path,
        n_atoms: usize,
        k: usize,
        dir: &Path,
    ) -> Session {
        let clients: Vec<ClientProc> = (0..k)
            .map(|c| spawn(opts, config, sock, dir, &format!("client{c}")))
            .collect();
        let address = Address::Unix(sock.to_path_buf());
        let pot =
            SocketPotential::bind_and_accept(&address, n_atoms, k, TIMEOUT).unwrap_or_else(|e| {
                let logs: Vec<String> = clients.iter().map(ClientProc::stderr).collect();
                panic!("clients did not connect: {e:?}\nclient stderr: {logs:#?}")
            });
        Session { pot, clients }
    }

    /// Evaluates, panicking with the error and the clients' stderr on failure.
    pub fn eval(&mut self, q: &Beads, cell: Option<&Cell>) -> Evaluation {
        match self.pot.evaluate(q, cell, None) {
            Ok(e) => e,
            Err(e) => {
                let logs: Vec<String> = self.clients.iter().map(ClientProc::stderr).collect();
                panic!("socket evaluation failed: {e:?}\nclient stderr: {logs:#?}")
            }
        }
    }

    /// Drops the potential (EXIT to every client, K-6) and checks that every client exits 0.
    pub fn finish(self) {
        let Session { pot, mut clients } = self;
        drop(pot);
        for (k, c) in clients.iter_mut().enumerate() {
            let st = c.wait_for(Duration::from_secs(60));
            assert!(
                st.is_some_and(|s| s.success()),
                "client {k} must exit 0 after EXIT, got {st:?}; stderr:\n{}",
                c.stderr()
            );
        }
    }
}

/// The in-process expectation: `MacePotential` on `export` for `symbols` (P-6), deterministic,
/// one libtorch thread.
pub fn adapter(export: &Path, symbols: &[&str], device: Device, dtype: Dtype) -> MacePotential {
    let z: Vec<u8> = symbols
        .iter()
        .map(|s| pimd_torch::atomic_number(s).unwrap())
        .collect();
    MacePotential::load(export, &z, device, dtype, 1, true)
        .unwrap_or_else(|e| panic!("adapter load {}: {e:?}", export.display()))
}

pub fn eval_adapter(pot: &mut MacePotential, q: &Beads, cell: Option<&Cell>) -> Evaluation {
    pot.evaluate(q, cell, None)
        .unwrap_or_else(|e| panic!("adapter evaluate: {e:?}"))
}

/// The box of the tests (bohr).
pub const BOX: [f64; 3] = [21.0, 23.5, 26.0];

pub fn cell() -> Cell {
    Cell::new(BOX).unwrap()
}

/// Molecule templates, (symbol, Å).
const WATER: [(&str, [f64; 3]); 3] = [
    ("O", [0.0, 0.0, 0.1173]),
    ("H", [0.0, 0.7572, -0.4692]),
    ("H", [0.0, -0.7572, -0.4692]),
];
const FORMALDEHYDE: [(&str, [f64; 3]); 4] = [
    ("C", [0.0, 0.0, -0.53]),
    ("O", [0.0, 0.0, 0.68]),
    ("H", [0.0, 0.94, -1.12]),
    ("H", [0.0, -0.94, -1.12]),
];
const METHANOL: [(&str, [f64; 3]); 6] = [
    ("C", [-0.75, 0.0, 0.0]),
    ("O", [0.68, 0.0, 0.0]),
    ("H", [1.0, 0.9, 0.0]),
    ("H", [-1.11, 1.03, 0.0]),
    ("H", [-1.11, -0.51, 0.89]),
    ("H", [-1.11, -0.51, -0.89]),
];
const AMMONIA: [(&str, [f64; 3]); 4] = [
    ("N", [0.0, 0.0, 0.0]),
    ("H", [0.94, 0.0, -0.33]),
    ("H", [-0.47, 0.81, -0.33]),
    ("H", [-0.47, -0.81, -0.33]),
];

/// One placed molecule: template, axis permutation (new axis a takes template axis perm[a]),
/// centre in the cluster (bohr), centre in the box (bohr, wrapped) and the whole box lengths
/// added to it (unwrapped image).
type Placement = (
    &'static [(&'static str, [f64; 3])],
    [usize; 3],
    [f64; 3],
    [f64; 3],
    [f64; 3],
);

/// Test system 1 (19 atoms; H, C, O and no N, so a species index taken from the elements present
/// differs from the model's): water A, formaldehyde, water C, water B, methanol. Box contacts, all
/// only through an image: water A – formaldehyde across x (image distance 6.0 bohr between
/// centres), water A – water C across z (6.5), water B – methanol across y (6.5).
const SYSTEM1: [Placement; 5] = [
    (
        &WATER,
        [1, 0, 2],
        [0.0, 0.0, 0.0],
        [0.5, 6.0, 1.0],
        [0.0, 0.0, 0.0],
    ),
    (
        &FORMALDEHYDE,
        [2, 1, 0],
        [5.8, 0.5, 0.0],
        [15.5, 6.0, 1.5],
        [1.0, 1.0, 0.0],
    ),
    (
        &WATER,
        [0, 2, 1],
        [0.0, 5.8, 0.5],
        [1.0, 6.5, 20.5],
        [0.0, 0.0, 1.0],
    ),
    (
        &WATER,
        [0, 1, 2],
        [0.5, 0.0, 5.8],
        [10.0, 1.0, 12.0],
        [0.0, 0.0, 0.0],
    ),
    (
        &METHANOL,
        [1, 2, 0],
        [5.8, 5.8, 3.0],
        [10.5, 18.0, 12.5],
        [1.0, 0.0, 0.0],
    ),
];

/// Test system 2 (11 atoms; H, C, N, O): water, ammonia, formaldehyde. Box contacts, only through
/// an image: water – ammonia across x and z (6.0 and 2.5 bohr between centres), ammonia –
/// formaldehyde across y (7.5).
const SYSTEM2: [Placement; 3] = [
    (
        &WATER,
        [1, 0, 2],
        [0.0, 0.0, 0.0],
        [0.5, 6.0, 24.5],
        [0.0, 0.0, 0.0],
    ),
    (
        &AMMONIA,
        [0, 1, 2],
        [5.8, 0.0, 0.5],
        [15.5, 6.0, 1.0],
        [1.0, 0.0, 0.0],
    ),
    (
        &FORMALDEHYDE,
        [2, 0, 1],
        [2.5, 5.8, 0.0],
        [15.0, 22.0, 1.5],
        [0.0, 1.0, -1.0],
    ),
];

/// A test system: symbols in atom order, cluster and box positions (bohr, one bead).
pub struct System {
    pub symbols: Vec<&'static str>,
    pub cluster: Vec<[f64; 3]>,
    pub boxed: Vec<[f64; 3]>,
}

fn build(placements: &[Placement]) -> System {
    let mut s = System {
        symbols: vec![],
        cluster: vec![],
        boxed: vec![],
    };
    for (atoms, perm, c, b, img) in placements {
        for (sym, x) in atoms.iter() {
            let r = [0, 1, 2].map(|a| x[perm[a]] / BOHR_IN_ANGSTROM);
            s.symbols.push(sym);
            s.cluster.push([0, 1, 2].map(|a| c[a] + r[a]));
            s.boxed
                .push([0, 1, 2].map(|a| b[a] + r[a] + img[a] * BOX[a]));
        }
    }
    s
}

pub fn system1() -> System {
    build(&SYSTEM1)
}

pub fn system2() -> System {
    build(&SYSTEM2)
}

/// SplitMix64 (the test's own generator).
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// P beads: `base` plus a uniform displacement in [−a, a) per component (seeded).
pub fn beads(base: &[[f64; 3]], p: usize, a: f64, seed: u64) -> Beads {
    let n = base.len();
    let mut rng = SplitMix64(seed);
    let mut q = Beads::zeros(p, n);
    for j in 0..p {
        for (i, x) in base.iter().enumerate() {
            q.set(j, i, x.map(|v| v + a * (2.0 * rng.next_f64() - 1.0)));
        }
    }
    q
}

/// Largest |ΔE| (Eh) over beads and largest |ΔF| (Eh/bohr) over all components.
pub fn max_diffs(got: &Evaluation, want: &Evaluation) -> (f64, f64) {
    assert_eq!(got.energies.len(), want.energies.len(), "bead count");
    assert_eq!(
        got.forces.as_slice().len(),
        want.forces.as_slice().len(),
        "force count"
    );
    let de = got
        .energies
        .iter()
        .zip(&want.energies)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let df = got
        .forces
        .as_slice()
        .iter()
        .zip(want.forces.as_slice())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    (de, df)
}

/// max |F| over all components (Eh/bohr).
pub fn max_abs(e: &Evaluation) -> f64 {
    e.forces
        .as_slice()
        .iter()
        .map(|f| f.abs())
        .fold(0.0, f64::max)
}

/// Bitwise equality of energies and forces.
pub fn bits_equal(a: &Evaluation, b: &Evaluation) -> bool {
    a.energies.len() == b.energies.len()
        && a.energies
            .iter()
            .zip(&b.energies)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.forces.as_slice().len() == b.forces.as_slice().len()
        && a.forces
            .as_slice()
            .iter()
            .zip(b.forces.as_slice())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The minimum-image distance (C-2, written here) between atoms i and j of one bead, bohr.
pub fn min_image_distance(x: [f64; 3], y: [f64; 3], l: [f64; 3]) -> f64 {
    let mut r2 = 0.0;
    for a in 0..3 {
        let d = y[a] - x[a];
        let v = d - l[a] * (d / l[a] + 0.5).floor();
        r2 += v * v;
    }
    r2.sqrt()
}

/// Unused-result guard for `PotentialError` matches.
pub fn is_backend(e: &PotentialError) -> bool {
    matches!(e, PotentialError::Backend(_))
}
