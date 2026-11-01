"""Spike question 2: is the virial available through the export? Calls the exported module with
compute_virials=True on bead 0 of a periodic geometry and compares -virial/V with the stress of
mace's MACECalculator (ASE Voigt order xx yy zz yz xz xy, eV/Å^3).

    pixi run -e ml python spikes/m6-torch/check_virial.py MODEL EXPORTED_F64 GEOM
"""
import sys

import mace  # noqa: F401
import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ref_eval import read_geom  # noqa: E402

model, exported, geom = sys.argv[1:4]
symbols, pos, cell = read_geom(geom)
q = pos[0]
m = torch.jit.load(exported)
z = m.atomic_numbers().tolist()
from ase.data import atomic_numbers as AN  # noqa: E402

species = torch.tensor([z.index(AN[s]) for s in symbols])
n = len(q)
send, recv, sh, un = [], [], [], []
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        d = q[j] - q[i]
        u = -np.floor(d / cell + 0.5)
        if np.linalg.norm(d + u * cell) < float(m.r_max()):
            send.append(i); recv.append(j); sh.append(u * cell); un.append(u)
t = lambda x: torch.tensor(np.array(x), dtype=torch.float64)  # noqa: E731
e, f, w = m(t(q), species, torch.tensor([send, recv]), t(sh), torch.tensor([0, n]), True, t(un),
            t(np.diag(cell))[None])
vol = float(np.prod(cell))
s_exp = (-w[0] / vol).detach().numpy()
atoms = Atoms(symbols, positions=q, cell=np.diag(cell), pbc=True)
atoms.calc = MACECalculator(model_paths=model, device="cpu", default_dtype="float64")
s_ref = atoms.get_stress(voigt=False)
print("virial (eV):", w[0].detach().numpy().round(6).tolist())
print(f"max|stress_export - stress_mace| = {np.abs(s_exp - s_ref).max():.3e} eV/A^3 "
      f"(max|stress| {np.abs(s_ref).max():.3e}); energy diff {abs(float(e[0]) - atoms.get_potential_energy()):.3e} eV")
