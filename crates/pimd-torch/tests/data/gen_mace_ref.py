"""Python MACE reference for the pimd-torch tests (MILESTONE_6 "M6b: Required tests").

    pixi run -e ml python crates/pimd-torch/tests/data/gen_mace_ref.py --out crates/pimd-torch/tests/data

Evaluates the committed tiny model `tiny_mace.model` (next to this script; gen_tiny_mace.py) with
mace's own ASE calculator (`MACECalculator`, float64, CPU, torch 1 thread) and mace's own
neighbour list (`get_neighborhood`), bead by bead, and writes `mace_ref.json` to --out.

Systems (from the M6a spike's make_geoms.py, seed 7; Å before conversion):
  water         gas-phase water, M5 start geometry (O 0 0 0.1173; H 0 ±0.7572 -0.4692), P = 8
  ethanol       ase.build.molecule("CH3CH2OH") (G2 geometry), P = 8
  waterbox      four water molecules in a 10.5 Å cubic box, random orientations and centres
                (numpy default_rng(7)), molecule 1 shifted by +L along x and molecule 2 by -L
                along y, so positions are unwrapped as ring-polymer positions are (SPEC C-5);
                P = 8; many edges cross the boundary
  water_single  the water base geometry undisplaced, P = 1 (units test)
Bead j of a P-bead system = base + 0.05 Å · N(0, 1) per component (default_rng(7 + P)), so beads
differ (bead 0 too).

Units: positions are generated in Å, converted to bohr with the SPEC U-2 constant
(q_bohr = x / 0.529177210903) and stored in bohr; the model is evaluated at q_bohr · 0.529177210903
Å, which is what the adapter computes from the stored bohr values (P-6), so both sides see the same
Å input. Energies are stored in eV and forces in eV/Å, as MACE returns them (the tests convert).
The cell (waterbox) is stored in bohr likewise and given to ASE as L_bohr · 0.529177210903 Å.

Consistency checks (the generator fails otherwise):
  - the committed export tiny_mace_f64.pt holds exactly the committed model's parameters
    (every state_dict tensor equal), so it is the export of the committed model;
  - the export, run in Python with mace's edge list, reproduces MACECalculator per bead
    (|ΔE| <= 1e-12 |E|, max |ΔF| <= 1e-10 max |F|).
"""

import argparse
import hashlib
import json
import os
import sys

import mace  # noqa: F401  (first: sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, which e3nn 0.4.4 needs)
import numpy as np
import torch
from ase import Atoms
from ase.build import molecule
from mace.calculators import MACECalculator
from mace.data.neighborhood import get_neighborhood

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "tiny_mace.model")
EXPORT = os.path.join(HERE, "tiny_mace_f64.pt")

BOHR_IN_ANGSTROM = 0.529177210903  # SPEC U-2 (CODATA 2018)
SEED = 7
SIGMA = 0.05  # Å
L_BOX = 10.5  # Å
P = 8
WATER = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
Z_OF = {"H": 1, "C": 6, "N": 7, "O": 8}


def rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def waterbox(seed):
    rng = np.random.default_rng(seed)
    centres = []
    while len(centres) < 4:
        c = rng.uniform(0, L_BOX, 3)
        d = [np.abs(c - o - L_BOX * np.round((c - o) / L_BOX)) for o in centres]
        if all(np.linalg.norm(x) > 3.0 for x in d):
            centres.append(c)
    pos = []
    for k, c in enumerate(centres):
        mol = (WATER - WATER[0]) @ rot(rng).T + c
        if k == 1:
            mol += [L_BOX, 0, 0]
        if k == 2:
            mol -= [0, L_BOX, 0]
        pos.append(mol)
    return ["O", "H", "H"] * 4, np.concatenate(pos)


def beads(base, p, seed):
    if p == 1:
        return base[None].copy()
    rng = np.random.default_rng(seed + p)
    return base[None] + SIGMA * rng.normal(size=(p, len(base), 3))


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_export_is_model(model, scripted):
    msd = model.state_dict()
    esd = scripted.state_dict()
    for k, v in msd.items():
        e = esd.get("model." + k)
        assert e is not None, f"export lacks parameter {k}"
        assert e.dtype == v.dtype and torch.equal(e, v), f"{k} differs"
    assert len(esd) == len(msd) + 2, "export has unexpected extra state"  # + z, rmax buffers


def export_eval(scripted, q_ang, z_index, ei, shifts, unit, cell):
    pos = torch.tensor(q_ang, dtype=torch.float64)
    out = scripted(
        pos,
        torch.tensor(z_index, dtype=torch.int64),
        torch.tensor(ei, dtype=torch.int64),
        torch.tensor(shifts, dtype=torch.float64),
        torch.tensor([0, len(q_ang)], dtype=torch.int64),
        False,
        torch.tensor(unit, dtype=torch.float64),
        torch.tensor(cell, dtype=torch.float64).reshape(1, 3, 3),
    )
    return float(out[0][0]), out[1].detach().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)
    calc = MACECalculator(model_paths=MODEL, device="cpu", default_dtype="float64")
    r_max = float(calc.r_max)
    model = torch.load(MODEL, map_location="cpu", weights_only=False)
    z_model = [int(z) for z in model.atomic_numbers]
    scripted = torch.jit.load(EXPORT, map_location="cpu")
    check_export_is_model(model, scripted)
    assert float(scripted.r_max()) == r_max
    assert [int(z) for z in scripted.atomic_numbers()] == z_model

    eth = molecule("CH3CH2OH")
    wb_sym, wb_pos = waterbox(SEED)
    systems = {
        "water": (["O", "H", "H"], WATER, None, P),
        "ethanol": (eth.get_chemical_symbols(), eth.get_positions(), None, P),
        "waterbox": (wb_sym, wb_pos, L_BOX, P),
        "water_single": (["O", "H", "H"], WATER, None, 1),
    }
    out_sys = {}
    for name, (sym, base, l_ang, p) in systems.items():
        q_bohr = beads(base, p, SEED) / BOHR_IN_ANGSTROM
        base_bohr = base / BOHR_IN_ANGSTROM
        periodic = l_ang is not None
        l_bohr = l_ang / BOHR_IN_ANGSTROM if periodic else None
        cell_ang = np.diag([l_bohr * BOHR_IN_ANGSTROM] * 3) if periodic else np.zeros((3, 3))
        z_index = [z_model.index(Z_OF[s]) for s in sym]
        energies, forces, n_edges, n_cross = [], [], [], []
        for qb in q_bohr:
            q = qb * BOHR_IN_ANGSTROM
            atoms = Atoms(sym, positions=q, cell=cell_ang if periodic else None, pbc=periodic)
            atoms.calc = calc
            e = float(atoms.get_potential_energy())
            f = atoms.get_forces()
            ei, shifts, unit, _ = get_neighborhood(
                q, r_max, pbc=(periodic,) * 3, cell=cell_ang if periodic else None)
            e2, f2 = export_eval(scripted, q, z_index, ei, shifts, unit, cell_ang)
            assert abs(e2 - e) <= 1e-12 * abs(e), (name, e, e2)
            assert np.max(np.abs(f2 - f)) <= 1e-10 * np.max(np.abs(f)), name
            energies.append(e)
            forces.extend(float(x) for x in f.reshape(-1))
            n_edges.append(int(ei.shape[1]))
            n_cross.append(int(np.sum(np.any(unit != 0, axis=1))))
        if periodic:
            assert sum(n_cross) > 0, "waterbox must have boundary-crossing edges"
        out_sys[name] = {
            "symbols": sym,
            "atomic_numbers": [Z_OF[s] for s in sym],
            "n_beads": p,
            "n_atoms": len(sym),
            "cell_bohr": [l_bohr] * 3 if periodic else None,
            "base_positions_bohr": [float(x) for x in base_bohr.reshape(-1)],
            "positions_bohr": [float(x) for x in q_bohr.reshape(-1)],
            "energies_ev": energies,
            "forces_ev_per_angstrom": forces,
            "n_edges_mace": n_edges,
            "n_boundary_edges_mace": n_cross,
        }
        print(f"{name}: P = {p}, N = {len(sym)}, edges {n_edges}, crossing {n_cross}, "
              f"E[0] = {energies[0]:.6f} eV, max|F| = {max(abs(x) for x in forces):.4f} eV/Å")

    import ase
    import e3nn
    import mace as mace_pkg

    out = {
        "versions": {
            "torch": torch.__version__,
            "mace-torch": mace_pkg.__version__,
            "e3nn": e3nn.__version__,
            "ase": ase.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "model": {
            "file": "tiny_mace.model",
            "export": "tiny_mace_f64.pt",
            "sha256_model": sha256(MODEL),
            "sha256_export": sha256(EXPORT),
            "r_max_angstrom": r_max,
            "atomic_numbers": z_model,
        },
        "bohr_in_angstrom": BOHR_IN_ANGSTROM,
        "seed": SEED,
        "sigma_angstrom": SIGMA,
        "systems": out_sys,
    }
    with open(os.path.join(a.out, "mace_ref.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
