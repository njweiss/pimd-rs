//! A mock i-PI protocol *client* written in the test from SPEC K-2 (wire format of i-PI 3.3.1,
//! `ipi/interfaces/sockets.py`), independent of the server under test. Shared by the
//! `pimd-socket` and `pimd-cli` tests (`#[path]` include).
//!
//! Wire format (K-2): 12-byte ASCII headers, upper case, right-padded with spaces; int32 and
//! float64 little-endian.
//! - server → client: `STATUS`; `INIT` + i32 bead + i32 len + bytes; `POSDATA` + h (9 f64,
//!   row-major) + h⁻¹ (9 f64) + i32 N + 3N f64; `GETFORCE`; `EXIT`.
//! - client → server: `READY`, `NEEDINIT`, `HAVEDATA` (answers to `STATUS`); after `GETFORCE`:
//!   `FORCEREADY` + f64 energy + i32 N + 3N f64 forces + 9 f64 virial + i32 len + extra bytes.
//!
//! The client answers `STATUS` with `NEEDINIT` while it wants an `INIT`, `HAVEDATA` after
//! `POSDATA` until `GETFORCE`, `READY` otherwise (as i-PI's drivers do), and accepts `GETFORCE`
//! right after `POSDATA` without a `STATUS` in between (i-PI's consolidated sends). Anything else
//! is recorded as a protocol violation and the client hangs up.
#![allow(dead_code)]

use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// The 12-byte padded header of K-2, built here by hand.
pub fn hdr(name: &str) -> [u8; 12] {
    let mut h = [b' '; 12];
    h[..name.len()].copy_from_slice(name.as_bytes());
    h
}

/// Where the mock connects (its own type, so the mock does not depend on `Address::from_str`).
#[derive(Debug, Clone)]
pub enum Endpoint {
    Unix(PathBuf),
    Tcp(String),
}

enum Conn {
    Unix(UnixStream),
    Tcp(TcpStream),
}

impl Read for Conn {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Conn::Unix(s) => s.read(b),
            Conn::Tcp(s) => s.read(b),
        }
    }
}

impl Write for Conn {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        match self {
            Conn::Unix(s) => s.write(b),
            Conn::Tcp(s) => s.write(b),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Conn::Unix(s) => s.flush(),
            Conn::Tcp(s) => s.flush(),
        }
    }
}

impl Conn {
    /// Connects, retrying until the server has bound (clients connect after the server binds and
    /// retry until it accepts, MILESTONE_5 "Socket lifecycle"). Gives up after `patience`.
    fn connect(ep: &Endpoint, patience: Duration) -> Option<Conn> {
        let t0 = Instant::now();
        loop {
            let c = match ep {
                Endpoint::Unix(p) => UnixStream::connect(p).map(Conn::Unix),
                Endpoint::Tcp(a) => TcpStream::connect(a).map(|s| {
                    s.set_nodelay(true).ok();
                    Conn::Tcp(s)
                }),
            };
            match c {
                Ok(c) => return Some(c),
                Err(_) if t0.elapsed() < patience => thread::sleep(Duration::from_millis(5)),
                Err(_) => return None,
            }
        }
    }

    fn shutdown(&self) {
        match self {
            Conn::Unix(s) => {
                s.shutdown(std::net::Shutdown::Both).ok();
            }
            Conn::Tcp(s) => {
                s.shutdown(std::net::Shutdown::Both).ok();
            }
        }
    }
}

/// One `POSDATA` message as received.
#[derive(Debug, Clone)]
pub struct Request {
    /// h, 9 values row-major (K-2, K-3).
    pub cell: [f64; 9],
    /// h⁻¹, 9 values row-major.
    pub inv: [f64; 9],
    /// The atom count sent.
    pub n_atoms: usize,
    /// 3N positions, bohr.
    pub positions: Vec<f64>,
    /// The bytes after the 12-byte header, exactly as received.
    pub payload: Vec<u8>,
}

/// What the mock returns for a request.
#[derive(Debug, Clone)]
pub struct Reply {
    pub energy: f64,
    pub forces: Vec<f64>,
    pub virial: [f64; 9],
    pub extra: Vec<u8>,
}

/// The mock's reaction to a request (faults for K-5).
#[derive(Debug, Clone)]
pub enum Action {
    /// Answer normally.
    Reply(Reply),
    /// Answer with the forces of `reply` but announce and send `n` atoms (pads or truncates the
    /// force list to 3n values).
    WrongAtomCount(Reply, usize),
    /// Answer `GETFORCE` with this 12-byte header instead of `FORCEREADY`, then hang up.
    BadHeader([u8; 12]),
    /// Send `FORCEREADY` and the energy, then hang up (disconnect mid-message).
    Disconnect(f64),
    /// Stall for this long after `POSDATA` (no answer), then hang up.
    Hang(Duration),
}

/// When the mock asks for `INIT`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeedInit {
    Never,
    /// Before its first `POSDATA`.
    Once,
    /// Before every `POSDATA`.
    Always,
}

/// Something the mock saw.
#[derive(Debug, Clone)]
pub enum Event {
    Status,
    Init { bead: i32, text: Vec<u8> },
    PosData(Request),
    GetForce,
    Exit,
}

/// Everything one mock client saw.
#[derive(Debug, Clone, Default)]
pub struct Log {
    pub events: Vec<Event>,
    pub violations: Vec<String>,
    /// The client could not connect within its patience.
    pub never_connected: bool,
}

impl Log {
    /// The `POSDATA` requests in order.
    pub fn requests(&self) -> Vec<&Request> {
        self.events
            .iter()
            .filter_map(|e| match e {
                Event::PosData(r) => Some(r),
                _ => None,
            })
            .collect()
    }

    pub fn got_exit(&self) -> bool {
        self.events.iter().any(|e| matches!(e, Event::Exit))
    }
}

pub type Model = Arc<dyn Fn(&Request) -> Action + Send + Sync>;

/// One mock client's behaviour.
#[derive(Clone)]
pub struct Spec {
    pub model: Model,
    pub needinit: NeedInit,
    /// Pseudo-random compute delay of up to this many µs after each `POSDATA` (seeded LCG).
    pub max_delay_us: u64,
    pub seed: u64,
    /// Answer the first `STATUS` with this header instead of a valid status.
    pub bad_status: Option<[u8; 12]>,
}

impl Spec {
    pub fn new(model: Model) -> Spec {
        Spec {
            model,
            needinit: NeedInit::Never,
            max_delay_us: 0,
            seed: 1,
            bad_status: None,
        }
    }
}

fn read_exact(c: &mut Conn, n: usize) -> Option<Vec<u8>> {
    let mut b = vec![0u8; n];
    match c.read_exact(&mut b) {
        Ok(()) => Some(b),
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => None,
        Err(_) => None,
    }
}

fn f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn i32_at(b: &[u8]) -> i32 {
    i32::from_le_bytes(b[..4].try_into().unwrap())
}

/// The bytes of a `FORCEREADY` message (K-2) announcing `n` atoms.
pub fn forceready(r: &Reply, n: usize) -> Vec<u8> {
    let mut w = Vec::new();
    w.extend_from_slice(&hdr("FORCEREADY"));
    w.extend_from_slice(&r.energy.to_le_bytes());
    w.extend_from_slice(&(n as i32).to_le_bytes());
    for k in 0..3 * n {
        let f = r.forces.get(k).copied().unwrap_or(0.0);
        w.extend_from_slice(&f.to_le_bytes());
    }
    for v in r.virial {
        w.extend_from_slice(&v.to_le_bytes());
    }
    w.extend_from_slice(&(r.extra.len() as i32).to_le_bytes());
    w.extend_from_slice(&r.extra);
    w
}

/// Serves one connection until `EXIT`, end of stream or a violation.
fn serve(mut c: Conn, spec: Spec) -> Log {
    let mut log = Log::default();
    let mut want_init = spec.needinit != NeedInit::Never;
    let mut pending: Option<Action> = None;
    let mut first_status = true;
    let mut lcg = spec.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    while let Some(h) = read_exact(&mut c, 12) {
        if h == hdr("STATUS") {
            log.events.push(Event::Status);
            let reply = match (first_status, spec.bad_status) {
                (true, Some(b)) => b,
                _ if pending.is_some() => hdr("HAVEDATA"),
                _ if want_init => hdr("NEEDINIT"),
                _ => hdr("READY"),
            };
            first_status = false;
            if c.write_all(&reply).is_err() {
                break;
            }
            if spec.bad_status.is_some() && reply == spec.bad_status.unwrap() {
                c.shutdown();
                break;
            }
        } else if h == hdr("INIT") {
            let Some(b) = read_exact(&mut c, 8) else {
                log.violations.push("INIT truncated".into());
                break;
            };
            let (bead, len) = (i32_at(&b[..4]), i32_at(&b[4..]));
            if len < 0 {
                log.violations.push(format!("INIT length {len}"));
                break;
            }
            let Some(text) = read_exact(&mut c, len as usize) else {
                log.violations.push("INIT string truncated".into());
                break;
            };
            log.events.push(Event::Init { bead, text });
            want_init = false;
        } else if h == hdr("POSDATA") {
            if want_init {
                log.violations
                    .push("POSDATA before the requested INIT".into());
                break;
            }
            if pending.is_some() {
                log.violations.push("POSDATA while holding data".into());
                break;
            }
            let Some(b) = read_exact(&mut c, 18 * 8 + 4) else {
                log.violations.push("POSDATA truncated".into());
                break;
            };
            let n = i32_at(&b[144..]);
            if n < 0 {
                log.violations.push(format!("POSDATA atom count {n}"));
                break;
            }
            let n = n as usize;
            let Some(x) = read_exact(&mut c, 3 * n * 8) else {
                log.violations.push("POSDATA positions truncated".into());
                break;
            };
            let v = f64s(&b[..144]);
            let mut payload = b.clone();
            payload.extend_from_slice(&x);
            let req = Request {
                cell: v[..9].try_into().unwrap(),
                inv: v[9..18].try_into().unwrap(),
                n_atoms: n,
                positions: f64s(&x),
                payload,
            };
            let action = (spec.model)(&req);
            log.events.push(Event::PosData(req));
            if spec.max_delay_us > 0 {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                thread::sleep(Duration::from_micros((lcg >> 33) % spec.max_delay_us));
            }
            if let Action::Hang(d) = action {
                thread::sleep(d);
                c.shutdown();
                break;
            }
            pending = Some(action);
        } else if h == hdr("GETFORCE") {
            log.events.push(Event::GetForce);
            let Some(action) = pending.take() else {
                log.violations.push("GETFORCE without data".into());
                break;
            };
            let bytes = match action {
                Action::Reply(r) => forceready(&r, r.forces.len() / 3),
                Action::WrongAtomCount(r, n) => forceready(&r, n),
                Action::BadHeader(b) => {
                    c.write_all(&b).ok();
                    c.shutdown();
                    break;
                }
                Action::Disconnect(e) => {
                    let mut w = hdr("FORCEREADY").to_vec();
                    w.extend_from_slice(&e.to_le_bytes());
                    c.write_all(&w).ok();
                    c.flush().ok();
                    c.shutdown();
                    break;
                }
                Action::Hang(_) => unreachable!(),
            };
            if c.write_all(&bytes).is_err() {
                break;
            }
            if spec.needinit == NeedInit::Always {
                want_init = true;
            }
        } else if h == hdr("EXIT") {
            log.events.push(Event::Exit);
            break;
        } else {
            log.violations.push(format!(
                "unexpected header {:?}",
                String::from_utf8_lossy(&h)
            ));
            break;
        }
    }
    log
}

/// Connects one client per spec to `ep`, one after the other (each connection completes before
/// the next starts, so the server's accept order is the spec order, K-4), then serves them
/// concurrently. Returns the logs in spec order when every client has finished.
pub fn spawn_clients(ep: Endpoint, specs: Vec<Spec>) -> JoinHandle<Vec<Log>> {
    thread::spawn(move || {
        let mut handles = Vec::new();
        for spec in specs {
            match Conn::connect(&ep, Duration::from_secs(20)) {
                Some(c) => handles.push(Some(thread::spawn(move || serve(c, spec)))),
                None => handles.push(None),
            }
        }
        handles
            .into_iter()
            .map(|h| match h {
                Some(h) => h.join().expect("mock client panicked"),
                None => Log {
                    never_connected: true,
                    ..Log::default()
                },
            })
            .collect()
    })
}

/// A Unix endpoint in a fresh temporary directory, and the directory (keep it alive).
pub fn unix_endpoint() -> (tempfile::TempDir, PathBuf) {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("pimd.sock");
    (d, p)
}

/// A TCP port that was free a moment ago on 127.0.0.1.
pub fn free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap().port()
}

/// A deterministic, stateless toy model (any N, gas phase or not): harmonic springs between all
/// atom pairs, V = Σ_{i<j} ½ k (r_ij − r0)², plus a weak trap ½ κ |q_i|² per atom, with
/// F = −∇V. Used both by the mock clients and by in-process reference potentials, so a run
/// through the socket and a run in-process perform the same floating-point operations.
pub fn toy_energy_forces(x: &[f64]) -> (f64, Vec<f64>) {
    const K: f64 = 0.4; // Eh/bohr²
    const R0: f64 = 1.8; // bohr
    const KAPPA: f64 = 1e-3; // Eh/bohr²
    let n = x.len() / 3;
    let mut e = 0.0;
    let mut f = vec![0.0; 3 * n];
    for i in 0..n {
        for d in 0..3 {
            let q = x[3 * i + d];
            e += 0.5 * KAPPA * q * q;
            f[3 * i + d] -= KAPPA * q;
        }
    }
    for i in 0..n {
        for j in i + 1..n {
            let r = [0, 1, 2].map(|d| x[3 * i + d] - x[3 * j + d]);
            let rl = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            let dr = rl - R0;
            e += 0.5 * K * dr * dr;
            for d in 0..3 {
                let g = K * dr * r[d] / rl;
                f[3 * i + d] -= g;
                f[3 * j + d] += g;
            }
        }
    }
    (e, f)
}

/// `toy_energy_forces` as a mock model (virial zeros, no extra string).
pub fn toy_model() -> Model {
    Arc::new(|r: &Request| {
        let (energy, forces) = toy_energy_forces(&r.positions);
        Action::Reply(Reply {
            energy,
            forces,
            virial: [0.0; 9],
            extra: Vec::new(),
        })
    })
}
