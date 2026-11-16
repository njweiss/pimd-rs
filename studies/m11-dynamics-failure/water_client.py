"""i-PI client for M11 tier 2: N independent H₂O molecules on a MACE model (docs/experiments/m11-tier2-water.md).

    pixi run -e ml python studies/m11-dynamics-failure/water_client.py MODEL --config CONFIG.toml \
        --address unix:/path [--device cuda] [--dtype float64] [--backend e3nn]
    ... water_client.py MODEL --check [--device cuda]    client checks, then exit

The supported M10a client (crates/pimd-torch/scripts/mace_client.py, K-7), imported by path, with
one change: an edge i → j is kept only when i and j are in the same molecule (atoms 3m, 3m+1, 3m+2,
the config's atom order), so molecules never interact at any distance. Within a molecule the edges
are those of the M10a client (P-6: f64, bohr, r < r_max, row-major order). Gas phase only.
"""

import argparse
import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The M5 protocol client first: importing it pins OMP/BLAS threads to 1 before torch is imported.
ipi = _load("ipi_client", os.path.join(ROOT, "crates/pimd-socket/tests/clients/ipi_client.py"))
mc = _load("mace_client", os.path.join(ROOT, "crates/pimd-torch/scripts/mace_client.py"))


class WaterClient(mc.MaceClient):
    def __init__(self, args):
        super().__init__(args)
        if self.n % 3:
            sys.exit(f"water_client: {self.n} atoms is not a whole number of molecules")
        torch = self.torch
        # Candidate intramolecular pairs (i != j), row-major as torch.nonzero gives them.
        m = torch.arange(self.n // 3, device=self.device)
        a, b = torch.meshgrid(torch.arange(3, device=self.device), torch.arange(3, device=self.device),
                              indexing="ij")
        keep = a != b
        self.ci = (3 * m[:, None] + a[keep][None, :]).reshape(-1)
        self.cj = (3 * m[:, None] + b[keep][None, :]).reshape(-1)

    def energy_forces(self, pos):
        """Positions `[N, 3]` (bohr) -> energy (Eh), forces `[N, 3]` (Eh/bohr)."""
        torch = self.torch
        q = torch.tensor(pos, dtype=torch.float64, device=self.device)
        v = q[self.cj] - q[self.ci]
        inside = (v * v).sum(-1) < self.r_max * self.r_max
        i, j = self.ci[inside], self.cj[inside]
        zeros = torch.zeros(i.numel(), 3, dtype=self.dtype, device=self.device)
        data = {
            "positions": (q * mc.BOHR_IN_ANGSTROM).to(self.dtype),
            "node_attrs": self.node_attrs,
            "edge_index": torch.stack([i, j]),
            "shifts": zeros,
            "unit_shifts": zeros,
            "cell": self.cell,
            "batch": self.batch,
            "ptr": self.ptr,
        }
        out = self.model(data, training=False, compute_force=True, compute_virials=False)
        e_int = float(out["interaction_energy"].detach().to(torch.float64)[0])
        forces = out["forces"].detach().to(torch.float64).cpu().numpy()
        return (e_int / mc.HARTREE_IN_EV + self.e0_sum,
                forces * mc.BOHR_IN_ANGSTROM / mc.HARTREE_IN_EV)

    def compute(self, pos, h):
        import numpy as np

        if pos.shape[0] != self.n:
            sys.exit(f"water_client: {pos.shape[0]} atoms sent, the config has {self.n}")
        if np.any(h != 0.0):
            sys.exit("water_client: gas phase only (h must be zero)")
        e, f = self.energy_forces(pos)
        return e, f, np.zeros((3, 3))


def client_for(model, n_molecules, device="cpu", dtype="float64", backend="e3nn"):
    """A WaterClient for n_molecules H₂O (atom order O, H, H per molecule), without a socket."""
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write("[system]\nsymbols = [" + ", ".join(['"O", "H", "H"'] * n_molecules) + "]\n")
        path = f.name
    try:
        args = argparse.Namespace(model=model, config=path, device=device, dtype=dtype,
                                  backend=backend)
        return WaterClient(args), mc.MaceClient(args)
    finally:
        os.unlink(path)


def check(a):
    """Finite differences; identity with the M10a client for one molecule; no cross terms."""
    import numpy as np

    rng = np.random.default_rng(2)
    # M6-like water geometry (bohr): O at the origin, H at ~0.96 Å, angle ~104.5°.
    r, th = 0.9572 / mc.BOHR_IN_ANGSTROM, np.radians(104.52)
    mol = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0], [r * np.cos(th), r * np.sin(th), 0.0]])
    ok = True

    one, m10a = client_for(a.model, 1, a.device, a.dtype, a.backend)
    p1 = mol + rng.normal(0.0, 0.05, mol.shape)
    e_w, f_w = one.energy_forces(p1)
    e_m, f_m, _ = m10a.compute(p1, np.zeros((3, 3)))
    same = e_w == e_m and np.array_equal(f_w, f_m)
    print(f"one molecule vs M10a client: E {e_w!r} vs {e_m!r}, identical: {same}")
    ok &= bool(same)

    h = 1e-4
    fd = np.zeros_like(p1)
    for i in range(3):
        for d in range(3):
            p, q = p1.copy(), p1.copy()
            p[i, d] += h
            q[i, d] -= h
            fd[i, d] = -(one.energy_forces(p)[0] - one.energy_forces(q)[0]) / (2 * h)
    rel = float(np.abs(fd - f_w).max() / np.abs(f_w).max())
    print(f"finite differences (h = {h} bohr): max |F_fd - F| / max |F| = {rel:.2e}")
    ok &= rel < 1e-6 if a.dtype == "float64" else rel < 1e-2

    n = 4
    many, _ = client_for(a.model, n, a.device, a.dtype, a.backend)
    mols = [mol + rng.normal(0.0, 0.05, mol.shape) for _ in range(n)]
    # Molecules 0 and 1 overlap (1.5 bohr apart): the M10a client would couple them; this must not.
    offsets = [np.zeros(3), np.array([1.5, 0.0, 0.0]), np.array([30.0, 0.0, 0.0]), np.array([0.0, 40.0, 0.0])]
    pos = np.concatenate([m + o for m, o in zip(mols, offsets)])
    e_n, f_n = many.energy_forces(pos)
    e_s, f_s = 0.0, []
    for m, o in zip(mols, offsets):
        e, f = one.energy_forces(m + o)
        e_s += e
        f_s.append(f)
    de, df = abs(e_n - e_s), float(np.abs(f_n - np.concatenate(f_s)).max())
    print(f"{n} molecules (two overlapping) vs sum of single molecules: |dE| = {de:.2e} Eh, "
          f"max |dF| = {df:.2e} Eh/bohr")
    ok &= de < 1e-9 and df < 1e-9
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--config")
    ap.add_argument("--address")
    ap.add_argument("--backend", choices=["e3nn", "cueq"], default="e3nn")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--patience", type=float, default=120.0)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check(a)
    if not a.config or not a.address:
        ap.error("--config and --address are required")
    client = WaterClient(a)
    sock = ipi.connect(a.address, a.patience)
    try:
        return ipi.serve(sock, client)
    finally:
        sock.close()


if __name__ == "__main__":
    sys.exit(main())
