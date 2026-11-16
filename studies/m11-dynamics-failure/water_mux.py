"""Batched i-PI client for M11 tier 2: one process, K connections, one MACE call per engine step.

    pixi run -e ml python studies/m11-dynamics-failure/water_mux.py MODEL --config CONFIG.toml \
        --address unix:/path --k P [--device cuda] [--dtype float32|float64]
    ... water_mux.py MODEL --check [--device cuda] [--dtype ...]   checks, then exit

The engine's socket potential runs one thread per client, each doing STATUS → READY, then
POSDATA + GETFORCE → FORCEREADY for its beads (K-2, K-4; crates/pimd-socket/src/lib.rs `exchange`).
With n_clients = P every connection carries one bead per step. This process opens K = P
connections, answers each STATUS with READY, collects the K POSDATA blocks, and evaluates all
P × N molecules in one MACE call (molecules never interact: edges within a molecule only, as
water_client.py). Each connection gets its bead's energy and forces.

Per-bead energy: Σ over the bead's atoms of MACE's per-node interaction energy (node energy minus
node E0, both as MACE returns them in float64) plus Σ E0 in float64 (as the M10a client). Forces:
autograd of the total, which is the per-bead force because beads share no edges. Gas phase only.
"""

import argparse
import os
import sys
import tomllib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import water_client as wc  # noqa: E402  (imports the M5 protocol helpers and the M10a client)

ipi = wc.ipi


def recv(s, n):
    return ipi.recvall(s, n)


class Batched:
    def __init__(self, model, symbols, k, device, dtype, backend="e3nn"):
        import torch
        from ase.data import atomic_numbers

        self.torch = torch
        self.k, self.n = k, len(symbols)
        if self.n % 3:
            sys.exit(f"water_mux: {self.n} atoms is not a whole number of molecules")
        self.client, _ = wc.client_for(model, k * self.n // 3, device, dtype, backend)
        c = self.client
        # client_for built O, H, H per molecule; the config must name the atoms that way.
        if [atomic_numbers[s] for s in symbols] != [8, 1, 1] * (self.n // 3):
            sys.exit("water_mux: the config's atoms must be O, H, H per molecule")
        table = torch.load(model, map_location="cpu", weights_only=False).atomic_energies_fn.atomic_energies
        e0 = table.detach().reshape(-1)
        # E0 per node as MACE adds it inside the model (model dtype, then float64), to subtract.
        self.node_e0 = e0.to(c.dtype).to(torch.float64)[c.species.cpu()].to(c.device)
        # Σ E0 of one bead in float64, as the M10a client.
        self.e0_bead = sum(float(e0.to(torch.float64)[s]) for s in c.species[: self.n].tolist()) / wc.mc.HARTREE_IN_EV

    def compute(self, pos):
        """pos `[k, N, 3]` (bohr) -> energies `[k]` (Eh), forces `[k, N, 3]` (Eh/bohr)."""
        torch, c = self.torch, self.client
        q = torch.tensor(pos.reshape(-1, 3), dtype=torch.float64, device=c.device)
        v = q[c.cj] - q[c.ci]
        inside = (v * v).sum(-1) < c.r_max * c.r_max
        i, j = c.ci[inside], c.cj[inside]
        zeros = torch.zeros(i.numel(), 3, dtype=c.dtype, device=c.device)
        data = {"positions": (q * wc.mc.BOHR_IN_ANGSTROM).to(c.dtype), "node_attrs": c.node_attrs,
                "edge_index": torch.stack([i, j]), "shifts": zeros, "unit_shifts": zeros,
                "cell": c.cell, "batch": c.batch, "ptr": c.ptr}
        out = c.model(data, training=False, compute_force=True, compute_virials=False)
        inter = out["node_energy"].detach().to(torch.float64) - self.node_e0
        e = inter.reshape(self.k, self.n).sum(1).cpu().numpy() / wc.mc.HARTREE_IN_EV + self.e0_bead
        f = out["forces"].detach().to(torch.float64).cpu().numpy().reshape(self.k, self.n, 3)
        return e, f * wc.mc.BOHR_IN_ANGSTROM / wc.mc.HARTREE_IN_EV


def serve(socks, model):
    """Protocol loop over K connections; returns when the engine sends EXIT or closes."""
    k = len(socks)
    while True:
        pos = np.empty((k, model.n, 3))
        for b, s in enumerate(socks):
            while True:
                try:
                    msg = recv(s, ipi.HDRLEN)
                except (EOFError, OSError):
                    return 0
                if msg == ipi.STATUS:
                    s.sendall(ipi.READY)
                elif msg == ipi.INIT:
                    _bead, length = np.frombuffer(recv(s, 8), "<i4")
                    recv(s, int(length))
                elif msg == ipi.POSDATA:
                    h = np.frombuffer(recv(s, 144), "<f8")[:9]
                    if np.any(h != 0.0):
                        sys.exit("water_mux: gas phase only (h must be zero)")
                    n = int(np.frombuffer(recv(s, 4), "<i4")[0])
                    if n != model.n:
                        sys.exit(f"water_mux: {n} atoms sent, the config has {model.n}")
                    pos[b] = np.frombuffer(recv(s, 24 * n), "<f8").reshape(n, 3)
                    if recv(s, ipi.HDRLEN) != ipi.GETFORCE:
                        sys.exit("water_mux: POSDATA not followed by GETFORCE")
                    break
                elif msg == ipi.EXIT:
                    return 0
                else:
                    sys.exit(f"water_mux: unexpected header {msg!r}")
        e, f = model.compute(pos)
        for b, s in enumerate(socks):
            s.sendall(ipi.FORCEREADY + np.float64(e[b]).astype("<f8").tobytes()
                      + np.int32(model.n).astype("<i4").tobytes()
                      + np.ascontiguousarray(f[b], dtype="<f8").tobytes()
                      + np.zeros(9, "<f8").tobytes() + np.int32(0).astype("<i4").tobytes())


def check(a):
    """Batched energies and forces against water_client.py bead by bead."""
    rng = np.random.default_rng(3)
    r, th = 0.9572 / wc.mc.BOHR_IN_ANGSTROM, np.radians(104.52)
    mol = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0], [r * np.cos(th), r * np.sin(th), 0.0]])
    n_mol, k = 4, 3
    pos = np.array([np.concatenate([mol + rng.normal(0, 0.05, mol.shape) + [20.0 * m, 0, 0]
                                    for m in range(n_mol)]) for _ in range(k)])
    bat = Batched(a.model, ["O", "H", "H"] * n_mol, k, a.device, a.dtype, a.backend)
    # Reference: the bead-by-bead client with e3nn (the batched route may use cueq).
    one, _ = wc.client_for(a.model, n_mol, a.device, a.dtype)
    e, f = bat.compute(pos)
    de = df = 0.0
    for b in range(k):
        e1, f1 = one.energy_forces(pos[b])
        de, df = max(de, abs(e[b] - e1)), max(df, float(np.abs(f[b] - f1).max()))
    print(f"{a.dtype} {a.backend}: batched vs bead-by-bead (e3nn), {k} beads × {n_mol} molecules: max |dE| = {de:.2e} Eh, "
          f"max |dF| = {df:.2e} Eh/bohr")
    tol = (1e-10, 1e-10) if a.dtype == "float64" else (2e-6, 1e-5)
    return 0 if de < tol[0] and df < tol[1] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--config")
    ap.add_argument("--address")
    ap.add_argument("--k", type=int)
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float32")
    ap.add_argument("--backend", choices=["e3nn", "cueq"], default="e3nn")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--patience", type=float, default=120.0)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check(a)
    if not (a.config and a.address and a.k):
        ap.error("--config, --address and --k are required")
    with open(a.config, "rb") as f:
        symbols = tomllib.load(f)["system"]["symbols"]
    model = Batched(a.model, symbols, a.k, a.device, a.dtype, a.backend)
    socks = [ipi.connect(a.address, a.patience) for _ in range(a.k)]
    try:
        return serve(socks, model)
    finally:
        for s in socks:
            try:
                s.close()
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
