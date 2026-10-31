//! i-PI protocol socket backend (SPEC §9a, K-1 … K-6): `SocketPotential` acts as the i-PI
//! *server*, sends bead positions to external force-provider clients and receives their energies,
//! forces and virials. All quantities on the wire are Hartree atomic units (U-1), so the engine
//! converts nothing. The engine never starts a client or knows its model (P-1, K-1).

use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::str::FromStr;
use std::thread;
use std::time::{Duration, Instant};

use pimd_core::{Beads, Cell};
use pimd_potentials::{Evaluation, Potential, PotentialError};

/// Where the server listens: a Unix-domain socket path or a TCP `host:port`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Address {
    /// Unix-domain socket at this path (`"unix:/path"`).
    Unix(PathBuf),
    /// TCP socket, `"host:port"` as written.
    Tcp(String),
}

impl FromStr for Address {
    type Err = PotentialError;

    /// Parses `"unix:/path"` into `Address::Unix` and `"host:port"` into `Address::Tcp`.
    ///
    /// Errors: a `PotentialError` for malformed strings (empty path, missing or invalid port).
    fn from_str(s: &str) -> Result<Address, PotentialError> {
        let bad = |why: &str| PotentialError::Backend(format!("invalid address {s:?}: {why}"));
        if let Some(path) = s.strip_prefix("unix:") {
            if path.is_empty() {
                return Err(bad("empty socket path"));
            }
            return Ok(Address::Unix(PathBuf::from(path)));
        }
        let (host, port) = s
            .rsplit_once(':')
            .ok_or_else(|| bad("expected \"unix:/path\" or \"host:port\""))?;
        if host.is_empty() {
            return Err(bad("empty host"));
        }
        if !port.bytes().all(|b| b.is_ascii_digit()) || port.parse::<u16>().is_err() {
            return Err(bad("port must be an integer in 0 ..= 65535"));
        }
        Ok(Address::Tcp(s.to_string()))
    }
}

/// The i-PI message headers (K-2): 12 bytes, upper case, right-padded with spaces.
fn header(name: &str) -> [u8; 12] {
    let mut h = [b' '; 12];
    h[..name.len()].copy_from_slice(name.as_bytes());
    h
}

/// One connected client stream.
enum Stream {
    Unix(UnixStream),
    Tcp(TcpStream),
}

impl Read for Stream {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Stream::Unix(s) => s.read(b),
            Stream::Tcp(s) => s.read(b),
        }
    }
}

impl Write for Stream {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        match self {
            Stream::Unix(s) => s.write(b),
            Stream::Tcp(s) => s.write(b),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Stream::Unix(s) => s.flush(),
            Stream::Tcp(s) => s.flush(),
        }
    }
}

impl Stream {
    /// Blocking mode with `timeout` on every read and write.
    fn configure(&self, timeout: Duration) -> std::io::Result<()> {
        match self {
            Stream::Unix(s) => {
                s.set_nonblocking(false)?;
                s.set_read_timeout(Some(timeout))?;
                s.set_write_timeout(Some(timeout))
            }
            Stream::Tcp(s) => {
                s.set_nonblocking(false)?;
                s.set_nodelay(true)?;
                s.set_read_timeout(Some(timeout))?;
                s.set_write_timeout(Some(timeout))
            }
        }
    }
}

/// A connected client and whether an earlier failure left its stream out of sync.
struct Client {
    stream: Stream,
    broken: bool,
}

/// An i-PI protocol server serving `n_clients` connected force clients (K-1).
pub struct SocketPotential {
    clients: Vec<Client>,
    n_atoms: usize,
    /// Accept, read and write timeout.
    timeout: Duration,
    /// The Unix socket file this potential created, removed on drop (K-6).
    socket_file: Option<PathBuf>,
}

/// Describes an I/O error on the wire (timeouts and closed connections named as such).
fn io_message(e: &std::io::Error, timeout: Duration) -> String {
    match e.kind() {
        ErrorKind::WouldBlock | ErrorKind::TimedOut => format!("timed out after {timeout:?}"),
        ErrorKind::UnexpectedEof => "connection closed by the client".to_string(),
        _ => format!("I/O error: {e}"),
    }
}

/// Accepts `n_clients` connections from `accept` (non-blocking, `Ok(None)` when none is waiting)
/// until `deadline`.
fn accept_all(
    n_clients: usize,
    deadline: Instant,
    timeout: Duration,
    mut accept: impl FnMut() -> std::io::Result<Option<Stream>>,
) -> Result<Vec<Client>, PotentialError> {
    let mut clients = Vec::with_capacity(n_clients);
    while clients.len() < n_clients {
        match accept() {
            Ok(Some(stream)) => {
                stream.configure(timeout).map_err(|e| {
                    PotentialError::Backend(format!("configuring a client connection: {e}"))
                })?;
                clients.push(Client {
                    stream,
                    broken: false,
                });
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    return Err(PotentialError::Backend(format!(
                        "{} of {n_clients} clients connected within {timeout:?}",
                        clients.len()
                    )));
                }
                thread::sleep(Duration::from_millis(1));
            }
            Err(e) => {
                return Err(PotentialError::Backend(format!("accepting a client: {e}")));
            }
        }
    }
    Ok(clients)
}

/// `Ok(None)` for a would-block accept.
fn nonblocking<T>(r: std::io::Result<T>) -> std::io::Result<Option<T>> {
    match r {
        Ok(v) => Ok(Some(v)),
        Err(e) if e.kind() == ErrorKind::WouldBlock => Ok(None),
        Err(e) => Err(e),
    }
}

impl SocketPotential {
    /// Binds `address`, then waits up to `timeout` for `n_clients` clients (K-1), numbered in
    /// accept order (K-4). `n_atoms` is the atom count N every client must return (K-5).
    /// `timeout` also bounds every later read and write.
    ///
    /// Errors: `PotentialError::Backend` if binding fails or fewer than `n_clients` clients
    /// connect within `timeout`.
    pub fn bind_and_accept(
        address: &Address,
        n_atoms: usize,
        n_clients: usize,
        timeout: Duration,
    ) -> Result<SocketPotential, PotentialError> {
        let deadline = Instant::now() + timeout;
        let bind_err =
            |e: std::io::Error| PotentialError::Backend(format!("binding {address:?}: {e}"));
        match address {
            Address::Unix(path) => {
                let listener = UnixListener::bind(path).map_err(bind_err)?;
                let accepted = listener
                    .set_nonblocking(true)
                    .map_err(bind_err)
                    .and_then(|()| {
                        accept_all(n_clients, deadline, timeout, || {
                            nonblocking(listener.accept()).map(|o| o.map(|(s, _)| Stream::Unix(s)))
                        })
                    });
                match accepted {
                    Ok(clients) => Ok(SocketPotential {
                        clients,
                        n_atoms,
                        timeout,
                        socket_file: Some(path.clone()),
                    }),
                    Err(e) => {
                        std::fs::remove_file(path).ok();
                        Err(e)
                    }
                }
            }
            Address::Tcp(addr) => {
                let listener = TcpListener::bind(addr.as_str()).map_err(bind_err)?;
                listener.set_nonblocking(true).map_err(bind_err)?;
                let clients = accept_all(n_clients, deadline, timeout, || {
                    nonblocking(listener.accept()).map(|o| o.map(|(s, _)| Stream::Tcp(s)))
                })?;
                Ok(SocketPotential {
                    clients,
                    n_atoms,
                    timeout,
                    socket_file: None,
                })
            }
        }
    }
}

/// One bead's reply: energy (Eh), 3N forces (Eh/bohr), virial (9, row-major, Eh).
type BeadResult = (f64, Vec<f64>, [f64; 9]);

/// The request that is the same for every bead of a call: h and h⁻¹ (K-3) and N.
struct Request<'a> {
    cell_bytes: Vec<u8>,
    positions: &'a [f64],
    n_atoms: usize,
}

fn read_bytes(s: &mut Stream, n: usize) -> std::io::Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    s.read_exact(&mut b)?;
    Ok(b)
}

fn read_i32(s: &mut Stream) -> std::io::Result<i32> {
    let mut b = [0u8; 4];
    s.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn read_f64s(s: &mut Stream, n: usize) -> std::io::Result<Vec<f64>> {
    Ok(read_bytes(s, 8 * n)?
        .chunks_exact(8)
        .map(|c| {
            let mut a = [0u8; 8];
            a.copy_from_slice(c);
            f64::from_le_bytes(a)
        })
        .collect())
}

/// Wire errors of one exchange: I/O (with the timeout for the message) or protocol.
enum Fault {
    Io(std::io::Error),
    Protocol(String),
}

impl From<std::io::Error> for Fault {
    fn from(e: std::io::Error) -> Fault {
        Fault::Io(e)
    }
}

/// The 12-byte header as text, for error messages.
fn show(h: &[u8]) -> String {
    format!("{:?}", String::from_utf8_lossy(h))
}

/// One bead on one client (K-2, K-4): `STATUS` (answering `NEEDINIT` with `INIT` for bead `j` and
/// an empty string, once), then `POSDATA` + `GETFORCE`, then the `FORCEREADY` reply.
fn exchange(s: &mut Stream, req: &Request, j: usize) -> Result<BeadResult, Fault> {
    let mut sent_init = false;
    loop {
        s.write_all(&header("STATUS"))?;
        let h = read_bytes(s, 12)?;
        if h == header("READY") {
            break;
        } else if h == header("NEEDINIT") && !sent_init {
            let mut w = header("INIT").to_vec();
            w.extend_from_slice(&(j as i32).to_le_bytes());
            w.extend_from_slice(&0i32.to_le_bytes());
            s.write_all(&w)?;
            sent_init = true;
        } else {
            return Err(Fault::Protocol(format!(
                "unexpected reply {} to STATUS",
                show(&h)
            )));
        }
    }
    let n = req.n_atoms;
    let n3 = 3 * n;
    let mut w = Vec::with_capacity(12 + req.cell_bytes.len() + 4 + 8 * n3 + 12);
    w.extend_from_slice(&header("POSDATA"));
    w.extend_from_slice(&req.cell_bytes);
    w.extend_from_slice(&(n as i32).to_le_bytes());
    for x in &req.positions[j * n3..(j + 1) * n3] {
        w.extend_from_slice(&x.to_le_bytes());
    }
    w.extend_from_slice(&header("GETFORCE"));
    s.write_all(&w)?;
    let h = read_bytes(s, 12)?;
    if h != header("FORCEREADY") {
        return Err(Fault::Protocol(format!(
            "unexpected reply {} to GETFORCE",
            show(&h)
        )));
    }
    let energy = read_f64s(s, 1)?[0];
    let got = read_i32(s)?;
    if got < 0 || got as usize != n {
        return Err(Fault::Protocol(format!(
            "client returned {got} atoms, expected {n}"
        )));
    }
    let forces = read_f64s(s, n3)?;
    let virial = read_f64s(s, 9)?;
    let len = read_i32(s)?;
    if len < 0 {
        return Err(Fault::Protocol(format!(
            "negative extra-string length {len}"
        )));
    }
    // Q21: discard the extra string in bounded chunks instead of allocating `len` bytes.
    let len = len as u64;
    if std::io::copy(&mut (&mut *s).take(len), &mut std::io::sink())? != len {
        return Err(Fault::Io(ErrorKind::UnexpectedEof.into()));
    }
    if !energy.is_finite() {
        return Err(Fault::Protocol(format!("non-finite energy {energy}")));
    }
    if forces.iter().any(|f| !f.is_finite()) {
        return Err(Fault::Protocol("non-finite force".to_string()));
    }
    if virial.iter().any(|v| !v.is_finite()) {
        return Err(Fault::Protocol("non-finite virial".to_string()));
    }
    let mut vir = [0.0; 9];
    vir.copy_from_slice(&virial);
    Ok((energy, forces, vir))
}

impl Potential for SocketPotential {
    /// Sends bead j (positions `[P, N, 3]`, bohr) to client j mod n_clients, beads in ascending
    /// order per client, clients concurrently (K-4); cell as in K-3. Returns energies (Eh) and
    /// forces (Eh/bohr) stored by bead index; `virial` is the client's for P = 1, else `None`
    /// (K-6).
    ///
    /// Errors: `ShapeMismatch` if `positions.n_atoms()` differs from N (P-1); `Unsupported` for
    /// `subset = Some` (P-4); `Backend` for protocol failures, the lowest failing bead's (K-5).
    fn evaluate(
        &mut self,
        positions: &Beads,
        cell: Option<&Cell>,
        subset: Option<&[usize]>,
    ) -> Result<Evaluation, PotentialError> {
        if subset.is_some() {
            return Err(PotentialError::Unsupported);
        }
        if positions.n_atoms() != self.n_atoms {
            return Err(PotentialError::ShapeMismatch {
                expected: self.n_atoms,
                actual: positions.n_atoms(),
            });
        }
        let (p, n) = (positions.n_beads(), self.n_atoms);
        // K-3: h = diag(L), h⁻¹ = diag(1/L), row-major; all zeros in the gas phase.
        let mut h = [0.0f64; 9];
        let mut hi = [0.0f64; 9];
        if let Some(c) = cell {
            for (d, l) in c.lengths().into_iter().enumerate() {
                h[4 * d] = l;
                hi[4 * d] = 1.0 / l;
            }
        }
        let cell_bytes: Vec<u8> = h.iter().chain(&hi).flat_map(|v| v.to_le_bytes()).collect();
        let req = Request {
            cell_bytes,
            positions: positions.as_slice(),
            n_atoms: n,
        };
        let n_clients = self.clients.len();
        let timeout = self.timeout;
        // K-4: client k serves beads k, k + n_clients, … in ascending order; clients run
        // concurrently; a client stops at its first failure.
        let per_client: Vec<Vec<(usize, Result<BeadResult, String>)>> = thread::scope(|sc| {
            let handles: Vec<_> = self
                .clients
                .iter_mut()
                .enumerate()
                .filter(|(k, _)| *k < p)
                .map(|(k, client)| {
                    let req = &req;
                    sc.spawn(move || {
                        let mut out = Vec::new();
                        for j in (k..p).step_by(n_clients) {
                            if client.broken {
                                out.push((
                                    j,
                                    Err(format!("client {k} is unusable after an earlier failure")),
                                ));
                                break;
                            }
                            match exchange(&mut client.stream, req, j) {
                                Ok(r) => out.push((j, Ok(r))),
                                Err(f) => {
                                    client.broken = true;
                                    let msg = match f {
                                        Fault::Io(e) => io_message(&e, timeout),
                                        Fault::Protocol(m) => m,
                                    };
                                    out.push((j, Err(format!("client {k}: {msg}"))));
                                    break;
                                }
                            }
                        }
                        out
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| match h.join() {
                    Ok(v) => v,
                    Err(e) => std::panic::resume_unwind(e),
                })
                .collect()
        });
        let mut results: Vec<Option<Result<BeadResult, String>>> = (0..p).map(|_| None).collect();
        for (j, r) in per_client.into_iter().flatten() {
            results[j] = Some(r);
        }
        let mut energies = Vec::with_capacity(p);
        let mut forces = Vec::with_capacity(p * n * 3);
        let mut virial = None;
        // K-5: the error of the lowest failing bead.
        for (j, r) in results.into_iter().enumerate() {
            match r {
                Some(Ok((e, f, w))) => {
                    energies.push(e);
                    forces.extend_from_slice(&f);
                    if p == 1 {
                        virial = Some([[w[0], w[1], w[2]], [w[3], w[4], w[5]], [w[6], w[7], w[8]]]);
                    }
                }
                Some(Err(m)) => return Err(PotentialError::Backend(format!("bead {j}: {m}"))),
                // Unreachable: every bead is either served or after a failed one on its client,
                // and a lower bead's failure has already returned.
                None => {
                    return Err(PotentialError::Backend(format!("bead {j}: not evaluated")));
                }
            }
        }
        let forces = Beads::from_vec(p, n, forces)
            .map_err(|e| PotentialError::Backend(format!("assembling forces: {e}")))?;
        Ok(Evaluation {
            energies,
            forces,
            virial,
        })
    }
}

impl Drop for SocketPotential {
    /// Sends `EXIT` to every client and removes the Unix socket file this potential created (K-6).
    fn drop(&mut self) {
        for c in &mut self.clients {
            c.stream.write_all(&header("EXIT")).ok();
        }
        if let Some(path) = &self.socket_file {
            std::fs::remove_file(path).ok();
        }
    }
}
