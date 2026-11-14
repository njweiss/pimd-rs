"""Supported i-PI socket client for MACE models (SPEC K-7, MILESTONE_10 "M10a").

    pixi run -e ml python crates/pimd-torch/scripts/mace_client.py MODEL --config CONFIG.toml \
        --address unix:/path [--backend e3nn|cueq] [--dtype float64|float32] [--device cpu|cuda] \
        [--patience 120]
    # cueq: the same command with .venv-cueq/bin/python in place of python

MODEL is a MACE model file as MACE writes it (a pickled torch.nn.Module), not a TorchScript
export. The species come from the engine config CONFIG.toml (`[system]` `symbols`, or the symbols
of its `positions_xyz` file, a relative path resolved against the working directory as the engine
does), in atom order. The engine side is the `kind = "socket"` potential with `n_clients = K`;
start K clients.

Each POSDATA is one bead, evaluated from the positions and the cell alone (no state between
calls): edges as the in-process adapter builds them (SPEC P-6, float64, bohr), MACE's interaction
energy in the model dtype cast to float64 plus Σ E0 summed in float64 from the model's
`atomic_energies_fn` (as the v2 export), forces from autograd, units converted with the U-2
constants below. The protocol loop is the M5 client's `serve` (imported by path). Bad input (an
unknown element, a non-orthorhombic cell, r_max ≥ L_min/2, a wrong atom count) exits non-zero; the
engine then sees a closed connection or no client (`Backend`, K-5).
"""

import argparse
import importlib.util
import os
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
IPI = os.path.join(HERE, "../../pimd-socket/tests/clients/ipi_client.py")

BOHR_IN_ANGSTROM = 0.529177210903  # SPEC U-2
HARTREE_IN_EV = 27.211386245988  # SPEC U-2


def config_symbols(path):
    """The atom symbols of the engine config at `path`, in atom order."""
    with open(path, "rb") as f:
        system = tomllib.load(f).get("system")
    if not isinstance(system, dict):
        sys.exit(f"mace_client: {path}: no [system] table")
    if "symbols" in system:
        return list(system["symbols"])
    if "positions_xyz" not in system:
        sys.exit(f"mace_client: {path}: [system] has neither `symbols` nor `positions_xyz`")
    xyz = system["positions_xyz"]
    with open(xyz) as f:
        lines = f.read().splitlines()
    try:
        n = int(lines[0].strip())
        symbols = [lines[2 + i].split()[0] for i in range(n)]
    except (IndexError, ValueError):
        sys.exit(f"mace_client: {xyz}: not an xyz file (count line, comment line, then one atom per line)")
    if n == 0:
        sys.exit(f"mace_client: {xyz}: no atoms")
    return symbols


class MaceClient:
    def __init__(self, args):
        import mace  # noqa: F401  (first: sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, which e3nn needs)
        import torch
        from ase.data import atomic_numbers

        self.torch = torch
        torch.set_num_threads(1)
        # Every call runs the same graph (as the adapter, P-6) for any scripted submodules.
        torch._C._jit_set_profiling_mode(False)
        self.dtype = torch.float64 if args.dtype == "float64" else torch.float32
        self.device = torch.device(args.device)
        if args.backend == "cueq" and self.device.type != "cuda":
            sys.exit("mace_client: --backend cueq needs --device cuda")

        symbols = config_symbols(args.config)
        model = torch.load(args.model, map_location="cpu", weights_only=False)
        z_model = [int(z) for z in model.atomic_numbers]
        species = []
        for i, s in enumerate(symbols):
            z = atomic_numbers.get(s)
            if z not in z_model:
                sys.exit(f"mace_client: atom {i} ({s}) is not an element of the model (Z {z_model})")
            species.append(z_model.index(z))
        self.n = len(symbols)
        # E0 table in float64 before any cast; Σ E0 in atom order, as the adapter (P-6 v2).
        table = model.atomic_energies_fn.atomic_energies
        if table.dim() > 1 and table.shape[0] > 1:  # as export_mace.py (v2)
            sys.exit(f"mace_client: the model has {table.shape[0]} heads (E0 table "
                     f"{tuple(table.shape)}); only single-head models are supported")
        e0 = table.detach().to(torch.float64).reshape(-1)
        e0_ev = 0.0
        for s in species:
            e0_ev += float(e0[s])
        self.e0_sum = e0_ev / HARTREE_IN_EV
        self.r_max = float(model.r_max) / BOHR_IN_ANGSTROM  # bohr

        model = model.to(self.dtype)
        if args.backend == "cueq":
            from mace.cli.convert_e3nn_cueq import run as convert

            model = convert(model, device=args.device, return_model=True)
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        dev = self.device
        self.species = torch.tensor(species, dtype=torch.int64, device=dev)
        self.node_attrs = torch.nn.functional.one_hot(self.species, len(z_model)).to(self.dtype)
        self.batch = torch.zeros(self.n, dtype=torch.int64, device=dev)
        self.ptr = torch.tensor([0, self.n], dtype=torch.int64, device=dev)
        self.cell = torch.zeros(3, 3, dtype=self.dtype, device=dev)

    def compute(self, pos, h):
        """One bead: positions `[N, 3]` and cell h (bohr) -> energy (Eh), forces `[N, 3]` (Eh/bohr),
        virial (zeros, K-6)."""
        import numpy as np

        torch = self.torch
        if pos.shape[0] != self.n:
            sys.exit(f"mace_client: {pos.shape[0]} atoms sent, the config has {self.n}")
        if np.all(h == 0.0):
            lengths = None
        elif np.all(h - np.diag(np.diag(h)) == 0.0) and np.all(np.diag(h) > 0.0):
            lengths = np.diag(h).copy()
            if self.r_max >= 0.5 * lengths.min():
                sys.exit(f"mace_client: r_max {self.r_max} bohr >= L_min/2 = {0.5 * lengths.min()} bohr")
        else:
            sys.exit(f"mace_client: an orthorhombic cell (diagonal h) or zeros is required, got {h.tolist()}")

        # Edges as P-6 (adapter's Edges::build): float64, bohr; shifts in Å.
        q = torch.tensor(pos, dtype=torch.float64, device=self.device)
        d = q[None, :, :] - q[:, None, :]  # d[i, j] = q_j - q_i
        if lengths is None:
            v = d
        else:
            L = torch.tensor(lengths, dtype=torch.float64, device=self.device)
            v = d - L * torch.round(d / L)  # C-2 minimum image (ties are beyond r_max)
        r2 = (v * v).sum(-1)
        r2.fill_diagonal_(float("inf"))
        i, j = torch.nonzero(r2 < self.r_max * self.r_max, as_tuple=True)
        if lengths is None:
            unit = torch.zeros(i.numel(), 3, dtype=torch.float64, device=self.device)
            shifts = unit
        else:
            unit = torch.round((v - d)[i, j] / L)
            shifts = unit * L * BOHR_IN_ANGSTROM

        data = {
            "positions": (q * BOHR_IN_ANGSTROM).to(self.dtype),
            "node_attrs": self.node_attrs,
            "edge_index": torch.stack([i, j]),
            "shifts": shifts.to(self.dtype),
            "unit_shifts": unit.to(self.dtype),
            "cell": self.cell,
            "batch": self.batch,
            "ptr": self.ptr,
        }
        out = self.model(data, training=False, compute_force=True, compute_virials=False)
        e_int = float(out["interaction_energy"].detach().to(torch.float64)[0])
        forces = out["forces"].detach().to(torch.float64).cpu().numpy()
        energy = e_int / HARTREE_IN_EV + self.e0_sum
        return energy, forces * BOHR_IN_ANGSTROM / HARTREE_IN_EV, np.zeros((3, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", help="MACE model file (pickled torch.nn.Module)")
    ap.add_argument("--config", required=True, help="engine config (TOML) naming the atoms")
    ap.add_argument("--address", required=True, help='"unix:/path" or "host:port"')
    ap.add_argument("--backend", choices=["e3nn", "cueq"], default="e3nn")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--patience", type=float, default=120.0, help="seconds to retry connecting")
    a = ap.parse_args()
    # Importing the M5 client pins OMP/BLAS threads to 1 before torch is imported.
    spec = importlib.util.spec_from_file_location("ipi_client", IPI)
    ipi_client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ipi_client)
    client = MaceClient(a)
    sock = ipi_client.connect(a.address, a.patience)
    try:
        return ipi_client.serve(sock, client)
    finally:
        sock.close()


if __name__ == "__main__":
    sys.exit(main())
