"""M10 spike Q3 (v): i-PI socket client running MACE-OFF23 small with cuEquivariance (float32, CUDA).

    pixi run -e ml spikes/m10/cueq/.venv/bin/python spikes/m10/cueq_client.py --address unix:/path [--e3nn]

The protocol loop is the locked M5 test client's `serve` (crates/pimd-socket/tests/clients/
ipi_client.py, imported read-only). One bead per POSDATA (SPEC K-4): edges are built on the GPU in
float64 (all ordered pairs within r_max under the minimum image, as P-6), the model is the eager
cuEquivariance conversion (or, with --e3nn, the e3nn model) in the export's v2 wrapper, and the
energy is the interaction energy plus Σ E0 in float64, as the v2 adapter does. Spike measurement
only: unit constants from ase.units, not pimd_core::units.
"""

import argparse
import importlib.util
import os
import sys

import common  # noqa: F401  (imports mace first)
import torch
from ase.units import Bohr, Hartree

spec = importlib.util.spec_from_file_location(
    "ipi_client", os.path.join(common.ROOT, "crates/pimd-socket/tests/clients/ipi_client.py"))
ipi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ipi)

ap = argparse.ArgumentParser()
ap.add_argument("--address", required=True)
ap.add_argument("--e3nn", action="store_true")
ap.add_argument("--xyz", default=common.BOX, help="species order (positions come from the server)")
a = ap.parse_args()
torch.set_num_threads(1)
dt = torch.float32
dev = torch.device("cuda")
model = common.load_small()
e0_table = model.atomic_energies_fn.atomic_energies.detach().clone().double().reshape(-1)
z = model.atomic_numbers.tolist()
r_max = float(model.r_max)
if not a.e3nn:
    from mace.cli.convert_e3nn_cueq import run as convert
    dflt = torch.get_default_dtype()
    model = convert(model.to(dt), device="cuda", return_model=True)
    torch.set_default_dtype(dflt)
mod = common.wrap_eager(model, dt)
from ase.data import atomic_numbers as an  # noqa: E402
sym, _, _ = common.read_box(a.xyz)
species = torch.tensor([z.index(an[s]) for s in sym], device=dev)
e0_sum = float(e0_table[species.cpu()].sum())  # eV, float64, atom order
n = len(sym)
ptr = torch.tensor([0, n], device=dev)
cell0 = torch.zeros(1, 3, 3, dtype=dt, device=dev)


class Cueq:
    def compute(self, pos, h):
        q = torch.tensor(pos * Bohr, dtype=torch.float64, device=dev)
        L = float(h[0, 0]) * Bohr
        d = q[None, :, :] - q[:, None, :]  # d[i, j] = q_j - q_i
        v = d - L * torch.round(d / L) if L > 0 else d
        r2 = (v * v).sum(-1)
        r2.fill_diagonal_(float("inf"))
        i, j = torch.nonzero(r2 < r_max ** 2, as_tuple=True)
        ei = torch.stack([i, j])
        shifts = (v - d)[i, j]
        unit = torch.round(shifts / L) if L > 0 else torch.zeros_like(shifts)
        e, f, _ = mod(q.to(dt), species, ei, shifts.to(dt), ptr, False, unit.to(dt), cell0)
        energy = (float(e[0]) + e0_sum) / Hartree
        forces = f.double().cpu().numpy() * Bohr / Hartree
        return energy, forces, torch.zeros(9, dtype=torch.float64).numpy()


sock = ipi.connect(a.address, 120.0)
sys.exit(ipi.serve(sock, Cueq()))
