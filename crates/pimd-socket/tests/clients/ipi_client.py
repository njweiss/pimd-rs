"""Minimal i-PI protocol force client for the M5 validation (MILESTONE_5 "Validation clients";
SPEC K-2, K-3, P-3). Locked with the tests.

Written from SPEC K-2 (wire format of i-PI 3.3.1, ipi/interfaces/sockets.py): 12-byte ASCII
headers, upper case, right-padded with spaces; int32 and float64 little-endian. The client answers
STATUS with NEEDINIT until it has received INIT, with HAVEDATA while it holds a result, READY
otherwise; reads POSDATA (h, h^-1 row-major, N, 3N positions), computes, and answers GETFORCE with
FORCEREADY (energy, N, 3N forces, 9 virial, extra-string length 0). It accepts GETFORCE right after
POSDATA (i-PI sends both at once) and stops on EXIT or when the server closes the connection.

Atomic units throughout (Hartree, bohr), so neither side converts anything. Single-threaded
(BLAS/OpenMP thread counts are pinned to 1 before numpy or tblite are imported) and stateless:
every POSDATA is a fresh calculation from the positions and cell alone, so results do not depend
on which client computes which bead or in what order (SPEC D-3 note).

Backends:
  lj   SPEC P-3: V(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6] for r < r_c, minus V(r_c) with --shift;
       forces -dV/dr; virial W = sum_{i<j} r_ij (x) f_ij (SPEC E-8, row-major). Periodic with the
       minimum image of SPEC C-2 when h is non-zero (orthorhombic only, L = diag(h)); open
       boundaries when h is all zeros (SPEC K-3) or with --pbc none.
  xtb  GFN2-xTB single point via tblite.interface.Calculator (a new calculator every call),
       forces = -gradient, virial zeros (not used: no pressure is computed in M5). Gas phase only:
       h must be zero unless --pbc none (i-PI always sends its cell, so the i-PI reference runs
       pass --pbc none).

Usage (from the repository root):
    pixi run -e ref python crates/pimd-socket/tests/clients/ipi_client.py \
        --address unix:/tmp/x.sock lj --epsilon 1.1e-4 --sigma 5.19 --cutoff 11.4 --shift
    pixi run -e ref python crates/pimd-socket/tests/clients/ipi_client.py \
        --address localhost:31415 xtb --symbols O H H
--address takes "unix:/path" or "host:port" (as the engine's config). For i-PI's own Unix sockets
(/tmp/ipi_<name>) pass the full path. The client retries connecting for --patience seconds.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_MAX_ACTIVE_LEVELS"):
    os.environ[_v] = "1"
os.environ.setdefault("OMP_STACKSIZE", "64M")

import argparse  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HDRLEN = 12


def hdr(name):
    return name.upper().ljust(HDRLEN).encode()


STATUS, READY, NEEDINIT, HAVEDATA = hdr("status"), hdr("ready"), hdr("needinit"), hdr("havedata")
INIT, POSDATA, GETFORCE, FORCEREADY, EXIT = (
    hdr("init"),
    hdr("posdata"),
    hdr("getforce"),
    hdr("forceready"),
    hdr("exit"),
)


def connect(address, patience):
    t0 = time.time()
    while True:
        try:
            if address.startswith("unix:"):
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(address[len("unix:"):])
            else:
                host, port = address.rsplit(":", 1)
                s = socket.create_connection((host, int(port)))
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return s
        except OSError:
            if time.time() - t0 > patience:
                raise
            time.sleep(0.02)


def recvall(s, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = s.recv_into(view[got:], n - got)
        if k == 0:
            raise EOFError
        got += k
    return bytes(buf)


class LennardJones:
    def __init__(self, eps, sigma, cutoff, shift, pbc):
        self.eps, self.sigma, self.rc, self.pbc = eps, sigma, cutoff, pbc
        sr6 = (sigma / cutoff) ** 6
        self.vshift = 4.0 * eps * (sr6 * sr6 - sr6) if shift else 0.0

    def compute(self, pos, h):
        n = len(pos)
        periodic = self.pbc == "auto" and np.any(h != 0.0)
        if periodic:
            if np.any(h - np.diag(np.diag(h)) != 0.0):
                sys.exit("ipi_client lj: only orthorhombic cells are supported")
            length = np.diag(h)
        i, j = np.triu_indices(n, 1)
        d = pos[i] - pos[j]  # r_ij = r_i - r_j
        if periodic:
            d -= length * np.floor(d / length + 0.5)  # SPEC C-2: [-L/2, L/2)
        r2 = (d * d).sum(axis=1)
        inside = r2 < self.rc * self.rc
        d, r2, i, j = d[inside], r2[inside], i[inside], j[inside]
        sr2 = self.sigma * self.sigma / r2
        sr6 = sr2 * sr2 * sr2
        energy = float((4.0 * self.eps * (sr6 * sr6 - sr6) - self.vshift).sum())
        fscal = 24.0 * self.eps * (2.0 * sr6 * sr6 - sr6) / r2  # -dV/dr / r
        fij = d * fscal[:, None]  # force on i from j
        forces = np.zeros((n, 3))
        np.add.at(forces, i, fij)
        np.add.at(forces, j, -fij)
        virial = np.einsum("ka,kb->ab", d, fij)
        return energy, forces, virial


class Xtb:
    def __init__(self, symbols, charge, uhf, pbc):
        from ase.data import atomic_numbers
        from tblite.interface import Calculator

        self.Calculator = Calculator
        self.numbers = np.array([atomic_numbers[s] for s in symbols])
        self.charge, self.uhf, self.pbc = charge, uhf, pbc

    def compute(self, pos, h):
        if self.pbc == "auto" and np.any(h != 0.0):
            sys.exit("ipi_client xtb: gas phase only (h must be zero, or pass --pbc none)")
        if len(pos) != len(self.numbers):
            sys.exit(f"ipi_client xtb: {len(pos)} atoms sent, {len(self.numbers)} symbols given")
        calc = self.Calculator("GFN2-xTB", self.numbers, pos, charge=self.charge, uhf=self.uhf)
        calc.set("verbosity", 0)
        res = calc.singlepoint()
        return float(res.get("energy")), -np.asarray(res.get("gradient")), np.zeros((3, 3))


def serve(sock, model):
    need_init, result = True, None
    while True:
        try:
            msg = recvall(sock, HDRLEN)
        except (EOFError, OSError):
            return 0
        if msg == STATUS:
            sock.sendall(HAVEDATA if result is not None else NEEDINIT if need_init else READY)
        elif msg == INIT:
            _bead, length = np.frombuffer(recvall(sock, 8), "<i4")
            recvall(sock, int(length))
            need_init = False
        elif msg == POSDATA:
            cells = np.frombuffer(recvall(sock, 144), "<f8")
            h = cells[:9].reshape(3, 3)
            n = int(np.frombuffer(recvall(sock, 4), "<i4")[0])
            pos = np.frombuffer(recvall(sock, 24 * n), "<f8").reshape(n, 3).copy()
            result = model.compute(pos, h)
        elif msg == GETFORCE:
            if result is None:
                sys.exit("ipi_client: GETFORCE without POSDATA")
            energy, forces, virial = result
            n = len(forces)
            sock.sendall(
                FORCEREADY
                + np.float64(energy).astype("<f8").tobytes()
                + np.int32(n).astype("<i4").tobytes()
                + np.ascontiguousarray(forces, dtype="<f8").tobytes()
                + np.ascontiguousarray(virial, dtype="<f8").tobytes()
                + np.int32(0).astype("<i4").tobytes()
            )
            result = None
        elif msg == EXIT:
            return 0
        else:
            sys.exit(f"ipi_client: unexpected header {msg!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--address", required=True, help='"unix:/path" or "host:port"')
    ap.add_argument("--patience", type=float, default=60.0, help="seconds to retry connecting")
    ap.add_argument("--pbc", choices=["auto", "none"], default="auto")
    sub = ap.add_subparsers(dest="backend", required=True)
    lj = sub.add_parser("lj")
    lj.add_argument("--epsilon", type=float, required=True, help="Eh")
    lj.add_argument("--sigma", type=float, required=True, help="bohr")
    lj.add_argument("--cutoff", type=float, required=True, help="bohr")
    lj.add_argument("--shift", action="store_true")
    xtb = sub.add_parser("xtb")
    xtb.add_argument("--symbols", nargs="+", required=True)
    xtb.add_argument("--charge", type=float, default=0.0)
    xtb.add_argument("--uhf", type=int, default=0)
    a = ap.parse_args()
    if a.backend == "lj":
        model = LennardJones(a.epsilon, a.sigma, a.cutoff, a.shift, a.pbc)
    else:
        model = Xtb(a.symbols, a.charge, a.uhf, a.pbc)
    sock = connect(a.address, a.patience)
    try:
        return serve(sock, model)
    finally:
        sock.close()


if __name__ == "__main__":
    sys.exit(main())
