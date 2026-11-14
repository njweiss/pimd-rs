//! Errors of the supported MACE socket client (MILESTONE_10 "M10a: Required tests" 4; SPEC K-7,
//! K-5, K-3, C-2).
//!
//! K-7: a non-diagonal cell, r_max ≥ L_min/2, an atom count other than the config's and an
//! element the model does not know make the client exit non-zero (before connecting, or on that
//! request). On the engine side a client that closes the connection, or never connects, is
//! `PotentialError::Backend` (K-5, tested with mock clients in M5).
//!
//! Every case has a positive control in the same test (the same client serves a valid request
//! before the bad one, or the same command with a known element serves one), so a client that
//! fails every request does not pass. Tiny model (e3nn, float64, CPU), test system 1 (19 atoms).
//!
//! `SocketPotential` sends only h = diag(L) or zeros (K-3) and always the N it was built for, so
//! the non-diagonal cell and the atom count mismatch are sent by a minimal i-PI server written
//! here from K-2 (`RawServer`); "reaches the engine as `Backend`" is then the closed connection
//! (EOF on the reply), which K-5 maps to `Backend`.
#![cfg(feature = "torch")]

mod mace_client_common;

use std::io::{ErrorKind, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::time::{Duration, Instant};

use mace_client_common::{
    ClientOpts, ClientProc, Session, TIMEOUT, beads, cell, is_backend, lock, spawn, system1,
    tiny_model, write_config,
};
use pimd_core::Cell;
use pimd_core::units::BOHR_IN_ANGSTROM;
use pimd_potentials::Potential;
use pimd_socket::{Address, SocketPotential};

const P: usize = 2;

fn header(name: &str) -> [u8; 12] {
    let mut h = [b' '; 12];
    h[..name.len()].copy_from_slice(name.as_bytes());
    h
}

/// A one-client i-PI server (K-2) that sends arbitrary h and atom counts.
struct RawServer {
    stream: UnixStream,
}

/// The reply to one request: `Ok((energy, N, forces))` for FORCEREADY, `Err(description)` for
/// a closed connection or anything else.
type Reply = Result<(f64, usize, Vec<f64>), String>;

impl RawServer {
    /// Binds `sock`, then waits up to `TIMEOUT` for one client.
    fn accept(sock: &Path, client: &mut ClientProc) -> RawServer {
        let listener = UnixListener::bind(sock).unwrap();
        listener.set_nonblocking(true).unwrap();
        let end = Instant::now() + TIMEOUT;
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(false).unwrap();
                    stream.set_read_timeout(Some(TIMEOUT)).unwrap();
                    stream.set_write_timeout(Some(TIMEOUT)).unwrap();
                    return RawServer { stream };
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => {
                    assert!(
                        Instant::now() < end,
                        "client did not connect; stderr:\n{}",
                        client.stderr()
                    );
                    if let Ok(Some(st)) = client.child.try_wait() {
                        panic!(
                            "client exited ({st}) before connecting:\n{}",
                            client.stderr()
                        );
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(e) => panic!("accept: {e}"),
            }
        }
    }

    fn read_exact(&mut self, n: usize) -> Result<Vec<u8>, String> {
        let mut b = vec![0u8; n];
        self.stream
            .read_exact(&mut b)
            .map_err(|e| format!("{:?}: {e}", e.kind()))?;
        Ok(b)
    }

    fn f64s(b: &[u8]) -> Vec<f64> {
        b.chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// STATUS (INIT on NEEDINIT) until READY, then POSDATA with `h` (row-major, bohr), h⁻¹ and
    /// the positions (bohr, `[n, 3]` flat), then GETFORCE and the reply.
    fn request(&mut self, h: [f64; 9], hinv: [f64; 9], pos: &[f64]) -> Reply {
        let mut sent_init = false;
        loop {
            self.stream
                .write_all(&header("STATUS"))
                .map_err(|e| e.to_string())?;
            let r = self.read_exact(12)?;
            if r == header("READY") {
                break;
            } else if r == header("NEEDINIT") && !sent_init {
                let mut w = header("INIT").to_vec();
                w.extend_from_slice(&0i32.to_le_bytes());
                w.extend_from_slice(&0i32.to_le_bytes());
                self.stream.write_all(&w).map_err(|e| e.to_string())?;
                sent_init = true;
            } else {
                return Err(format!("reply {:?} to STATUS", String::from_utf8_lossy(&r)));
            }
        }
        let n = pos.len() / 3;
        let mut w = header("POSDATA").to_vec();
        for v in h.iter().chain(&hinv) {
            w.extend_from_slice(&v.to_le_bytes());
        }
        w.extend_from_slice(&(n as i32).to_le_bytes());
        for v in pos {
            w.extend_from_slice(&v.to_le_bytes());
        }
        w.extend_from_slice(&header("GETFORCE"));
        // The client may already have closed the connection; a write error is a refusal too.
        self.stream.write_all(&w).map_err(|e| e.to_string())?;
        let r = self.read_exact(12)?;
        if r != header("FORCEREADY") {
            return Err(format!(
                "reply {:?} to GETFORCE",
                String::from_utf8_lossy(&r)
            ));
        }
        let e = Self::f64s(&self.read_exact(8)?)[0];
        let got = i32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()) as usize;
        let f = Self::f64s(&self.read_exact(24 * got)?);
        self.read_exact(72)?;
        let len = i32::from_le_bytes(self.read_exact(4)?.try_into().unwrap());
        self.read_exact(len.max(0) as usize)?;
        Ok((e, got, f))
    }
}

/// The client exits non-zero within 60 s (after its connection closed or before connecting).
fn assert_exits_nonzero(c: &mut ClientProc, what: &str) {
    let st = c.wait_for(Duration::from_secs(60));
    assert!(
        st.is_some_and(|s| !s.success()),
        "{what}: the client must exit non-zero, got {st:?}; stderr:\n{}",
        c.stderr()
    );
}

/// Bead 0 of `q` as a flat position list.
fn flat(q: &pimd_core::Beads) -> Vec<f64> {
    q.as_slice()[..3 * q.n_atoms()].to_vec()
}

/// The gas-phase control request: FORCEREADY with N atoms and finite values.
fn assert_control(srv: &mut RawServer, c: &ClientProc, pos: &[f64]) {
    let r = srv.request([0.0; 9], [0.0; 9], pos);
    let (e, n, f) = r.unwrap_or_else(|e| {
        panic!(
            "control request (gas phase) failed: {e}; stderr:\n{}",
            c.stderr()
        )
    });
    assert_eq!(n, pos.len() / 3, "control: atom count");
    assert!(
        e.is_finite() && f.iter().all(|x| x.is_finite()),
        "control: finite"
    );
}

/// K-7, K-3, K-5: a cell h with a non-zero off-diagonal element (h_xy = 0.5 bohr, otherwise the
/// test box) makes the client close the connection without FORCEREADY and exit non-zero, after
/// it served a gas-phase request (control).
#[test]
fn non_diagonal_cell_closes_the_connection() {
    let _g = lock();
    let s = system1();
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("k7.sock");
    let config = write_config(dir.path(), &s.symbols, &sock, 1);
    let mut c = spawn(
        &ClientOpts::defaults(tiny_model()),
        &config,
        &sock,
        dir.path(),
        "c",
    );
    let mut srv = RawServer::accept(&sock, &mut c);
    assert_control(&mut srv, &c, &flat(&beads(&s.cluster, 1, 0.0, 0)));
    let l = cell().lengths();
    let h = [l[0], 0.5, 0.0, 0.0, l[1], 0.0, 0.0, 0.0, l[2]];
    // h⁻¹ of the upper-triangular h (row-major).
    let hinv = [
        1.0 / l[0],
        -0.5 / (l[0] * l[1]),
        0.0,
        0.0,
        1.0 / l[1],
        0.0,
        0.0,
        0.0,
        1.0 / l[2],
    ];
    let r = srv.request(h, hinv, &flat(&beads(&s.boxed, 1, 0.0, 0)));
    assert!(r.is_err(), "non-diagonal h must be refused, got a reply");
    eprintln!("non-diagonal h: {}", r.unwrap_err());
    drop(srv);
    assert_exits_nonzero(&mut c, "non-diagonal h");
}

/// K-7, K-5: an atom count other than the config's (N − 1 = 18 of 19 atoms) makes the client close
/// the connection without FORCEREADY and exit non-zero, after it served a request with N atoms
/// (control).
#[test]
fn atom_count_mismatch_closes_the_connection() {
    let _g = lock();
    let s = system1();
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("k7.sock");
    let config = write_config(dir.path(), &s.symbols, &sock, 1);
    let mut c = spawn(
        &ClientOpts::defaults(tiny_model()),
        &config,
        &sock,
        dir.path(),
        "c",
    );
    let mut srv = RawServer::accept(&sock, &mut c);
    let pos = flat(&beads(&s.cluster, 1, 0.0, 0));
    assert_control(&mut srv, &c, &pos);
    let r = srv.request([0.0; 9], [0.0; 9], &pos[..pos.len() - 3]);
    assert!(r.is_err(), "N − 1 atoms must be refused, got a reply");
    eprintln!("N - 1 atoms: {}", r.unwrap_err());
    drop(srv);
    assert_exits_nonzero(&mut c, "atom count");
}

/// K-7, K-5, C-2: with a cell, r_max < L_min/2 is required. The box L = (L₀, 23.5, 26.0) bohr
/// with L₀ = fl(10 Å / BOHR_IN_ANGSTROM) has L_min/2 = r_max exactly (5 Å; checked: L₀ converts
/// back to 10.0 Å exactly, and fl(5/B)·2 = L₀ since doubling is exact, so the equality holds in
/// bohr and in Å). It is refused: `SocketPotential::evaluate` returns `Backend` (closed
/// connection) and the client exits non-zero, after the same client served the gas phase
/// (control). A client that checks `≤` instead of `<`, or not at all, evaluates it.
#[test]
fn cutoff_at_half_the_box_is_backend() {
    let _g = lock();
    let s = system1();
    let l0 = 10.0 / BOHR_IN_ANGSTROM;
    assert_eq!(l0 * BOHR_IN_ANGSTROM, 10.0, "premise: L0 is 10 Å exactly");
    assert_eq!(
        2.0 * (5.0 / BOHR_IN_ANGSTROM),
        l0,
        "premise: r_max = L0/2 in bohr"
    );
    let dir = tempfile::tempdir().unwrap();
    let mut sess = Session::start(
        &ClientOpts::defaults(tiny_model()),
        &s.symbols,
        1,
        dir.path(),
    );
    sess.eval(&beads(&s.cluster, P, 0.1, 3), None);
    let small = Cell::new([l0, 23.5, 26.0]).unwrap();
    let r = sess
        .pot
        .evaluate(&beads(&s.boxed, P, 0.1, 4), Some(&small), None);
    match &r {
        Err(e) if is_backend(e) => eprintln!("L_min/2 = r_max: {e:?}"),
        other => panic!("L_min/2 = r_max must be Backend, got {other:?}"),
    }
    let Session { pot, mut clients } = sess;
    drop(pot);
    assert_exits_nonzero(&mut clients[0], "r_max >= L_min/2");
}

/// K-7, K-5: an element the model does not know (Si for the H C N O tiny model, the 4th atom of
/// test system 1) makes the client exit non-zero, either before connecting (then the engine's
/// accept times out: `Backend`, "0 of 1 clients connected"), or on the first request (`Backend`,
/// closed connection). The test waits up to 60 s for an exit without binding the socket; if the
/// client is still running it binds and sends one request. Control: the same command with C in
/// place of Si connects and serves a request.
#[test]
fn unknown_element_is_backend() {
    let _g = lock();
    let s = system1();
    let mut bad = s.symbols.clone();
    assert_eq!(bad[3], "C", "premise: atom 3 is the formaldehyde carbon");
    bad[3] = "Si";
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("k7.sock");
    let config = write_config(dir.path(), &bad, &sock, 1);
    let mut c = spawn(
        &ClientOpts::defaults(tiny_model()),
        &config,
        &sock,
        dir.path(),
        "c",
    );
    let address = Address::Unix(sock.clone());
    match c.wait_for(Duration::from_secs(60)) {
        Some(st) => {
            assert!(!st.success(), "unknown element: exit status {st}");
            let r =
                SocketPotential::bind_and_accept(&address, bad.len(), 1, Duration::from_secs(2));
            match r {
                Err(e) if is_backend(&e) => eprintln!("exited before connecting ({st}): {e:?}"),
                Err(e) => panic!("expected Backend, got {e:?}"),
                Ok(_) => panic!("a client that exited cannot connect"),
            }
        }
        None => {
            let r = SocketPotential::bind_and_accept(&address, bad.len(), 1, TIMEOUT);
            let mut pot = match r {
                Ok(p) => p,
                Err(e) => panic!("client neither exited nor connected: {e:?}"),
            };
            let r = pot.evaluate(&beads(&s.cluster, 1, 0.0, 0), None, None);
            match &r {
                Err(e) if is_backend(e) => eprintln!("refused on the request: {e:?}"),
                other => panic!("unknown element must be Backend, got {other:?}"),
            }
            drop(pot);
            assert_exits_nonzero(&mut c, "unknown element");
        }
    }
    drop(c);
    let dir = tempfile::tempdir().unwrap();
    let mut ctl = Session::start(
        &ClientOpts::defaults(tiny_model()),
        &s.symbols,
        1,
        dir.path(),
    );
    ctl.eval(&beads(&s.cluster, 1, 0.0, 0), None);
    ctl.finish();
}
