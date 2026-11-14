"""i-PI force client for the M11 tier-1 Morse systems (docs/experiments/m11-dynamics-failure-modes.md).

    pixi run -e ref python studies/m11-dynamics-failure/morse_client.py --address unix:/path {s1|s2}

Morse V(r) = D (1 - exp(-a (r - r_e)))^2, OH-like parameters (atomic units throughout, K-2):
  s1  separable: V = sum over particles and Cartesian components of V(q_{i,d}) (independent 1D
      oscillators, as the M7 double-well rung);
  s2  radial: V = sum over particles of V(|q_i|) (a particle bound to a fixed centre).
Gas phase only (h must be zero, K-3). Stateless; virial zeros (not read for P > 1, K-6).
Protocol from the locked M5 client (crates/pimd-socket/tests/clients/ipi_client.py: connect,
serve), imported by path, unchanged; importing it pins BLAS threads to 1.

    ... morse_client.py --check   finite-difference check of both systems, then exit.
"""

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "ipi_client", os.path.join(HERE, "../../crates/pimd-socket/tests/clients/ipi_client.py"))
ipi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ipi)

import numpy as np  # noqa: E402

D = 0.18748  # Eh
A = 1.1605  # 1/bohr
R_E = 1.8324  # bohr


def morse(r):
    """V(r) and dV/dr."""
    e = np.exp(-A * (r - R_E))
    return D * (1.0 - e) ** 2, 2.0 * D * A * e * (1.0 - e)


class Morse:
    def __init__(self, system):
        self.system = system

    def energy_forces(self, pos):
        if self.system == "s1":
            v, dv = morse(pos)
            return float(v.sum()), -dv
        r = np.sqrt((pos * pos).sum(axis=1))
        v, dv = morse(r)
        return float(v.sum()), -(dv / r)[:, None] * pos

    def compute(self, pos, h):
        if np.any(h != 0.0):
            sys.exit("morse_client: gas phase only (h must be zero)")
        e, f = self.energy_forces(pos)
        return e, f, np.zeros((3, 3))


def check():
    rng = np.random.default_rng(1)
    for system in ("s1", "s2"):
        m = Morse(system)
        pos = rng.normal(0.0, 0.3, (5, 3)) + (R_E if system == "s1" else R_E / np.sqrt(3.0))
        e, f = m.energy_forces(pos)
        h = 1e-5
        fd = np.zeros_like(pos)
        for i in range(pos.shape[0]):
            for d in range(3):
                p, q = pos.copy(), pos.copy()
                p[i, d] += h
                q[i, d] -= h
                fd[i, d] = -(m.energy_forces(p)[0] - m.energy_forces(q)[0]) / (2 * h)
        rel = float(np.abs(fd - f).max() / np.abs(f).max())
        print(f"{system}: E = {e:.10f} Eh, max |F_fd - F| / max |F| = {rel:.2e}")
        if rel > 1e-7:
            sys.exit(f"morse_client: {system} finite-difference check failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address")
    ap.add_argument("--patience", type=float, default=60.0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("system", nargs="?", choices=["s1", "s2"])
    a = ap.parse_args()
    if a.check:
        return check()
    if not a.address or not a.system:
        ap.error("--address and a system are required")
    sock = ipi.connect(a.address, a.patience)
    try:
        return ipi.serve(sock, Morse(a.system))
    finally:
        sock.close()


if __name__ == "__main__":
    sys.exit(main())
