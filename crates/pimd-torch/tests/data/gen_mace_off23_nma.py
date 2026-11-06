"""Python MACE reference for the periodic MACE-OFF23 long validation (MILESTONE_8 "M8a: Required
tests" 7; SPEC P-6, U-3, C-2).

    PIMD_MACE_MODEL=ff_models/mace_off23/MACE-OFF23_small.model \
        pixi run -e ml python crates/pimd-torch/tests/data/gen_mace_off23_nma.py [--out DIR]

Writes mace_off23_nma.json next to this script (or into DIR). Needs the MACE-OFF23 small model file
named by PIMD_MACE_MODEL (as distributed; never committed); validation/verify_reference_data.py
runs it only when PIMD_MACE_MODEL is set. The model's SHA-256 and 64-bit FNV-1a are recorded; the
engine test long_mace_off23_pbc.rs checks the FNV-1a of its own PIMD_MACE_MODEL against it.

System: N-methylacetamide + 216 waters, the box `nma_w216` of
crates/pimd-cli/tests/data/m8_boxes.json (gen_m8_boxes.py; 660 atoms, cubic L = 18.760371262 Å),
P = 4 beads. Positions are converted to bohr with the SPEC U-2 constant (x_bohr = x_A /
0.529177210903, the config parser's conversion) and bead j of atom i, component k, is
x_bohr + AMP_BOHR (2 u - 1), with u from SplitMix64(SEED) (Steele, Lea, Flood 2014; the generator
of the pimd-torch tests' common::SplitMix64, u = (z >> 11) 2^-53) drawn in the order bead j, atom
i, component k: exactly common::displaced(base, 4, AMP_BOHR, SEED) of the engine test, so the
positions need not be stored. Beads are not wrapped: displaced atoms near a face lie outside
[0, L) (unwrapped, as ring-polymer positions are, SPEC C-5).

Reference: mace's ASE `MACECalculator` on the model file, float64, CPU, torch 1 thread, TorchScript
profiling executor off, periodic boundaries (pbc = True, cell L Å = L_bohr * 0.529177210903) and
mace's own neighbour list, one bead per call at the Å positions x_bohr * 0.529177210903 (what the
adapter computes from bohr, P-6). Energies in eV and forces in eV/Å, as MACE returns them (the test
converts). Consistency check (the generator fails otherwise): every bead evaluated again with all
atoms wrapped into the cell gives the same energy (1e-12 relative) and forces (1e-10 max|F|).

Run time about 1 min on this machine (12 CPU threads available, 1 used); about 2 GB of RAM.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import mace  # noqa: E402,F401  (first: sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase import Atoms  # noqa: E402
from mace.calculators import MACECalculator  # noqa: E402
from mace.data.neighborhood import get_neighborhood  # noqa: E402

HERE = Path(__file__).resolve().parent
BOXES = HERE.parent.parent.parent / "pimd-cli" / "tests" / "data" / "m8_boxes.json"
BOX = "nma_w216"
BOHR_IN_ANGSTROM = 0.529177210903  # SPEC U-2 (CODATA 2018)
P = 4
SEED = 20261105
AMP_BOHR = 0.1
MASK = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed):
        self.s = seed

    def next_f64(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z ^= z >> 31
        return float(z >> 11) * (1.0 / float(1 << 53))

    def sym(self, a):
        return a * (2.0 * self.next_f64() - 1.0)


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & MASK
    return h


def displaced(base_bohr, p, amp, seed):
    rng = SplitMix64(seed)
    out = np.empty((p, len(base_bohr), 3))
    for j in range(p):
        for i, b in enumerate(base_bohr):
            for k in range(3):
                out[j, i, k] = b[k] + rng.sym(amp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()
    m = os.environ.get("PIMD_MACE_MODEL")
    if not m or not Path(m).is_file():
        sys.exit("gen_mace_off23_nma.py: set PIMD_MACE_MODEL to the MACE-OFF23 small model file")
    model = Path(m).resolve()
    raw = model.read_bytes()

    box = json.loads(BOXES.read_text())["boxes"][BOX]
    sym = box["symbols"]
    base_bohr = [[x / BOHR_IN_ANGSTROM for x in r] for r in box["positions_angstrom"]]
    l_bohr = box["box_angstrom"] / BOHR_IN_ANGSTROM
    cell = np.diag([l_bohr * BOHR_IN_ANGSTROM] * 3)
    q_bohr = displaced(base_bohr, P, AMP_BOHR, SEED)

    torch.set_num_threads(1)
    torch._C._jit_set_profiling_mode(False)
    calc = MACECalculator(model_paths=str(model), device="cpu", default_dtype="float64")
    r_max = float(calc.r_max)
    assert r_max < 0.5 * cell[0, 0], (r_max, cell[0, 0])

    energies, forces, n_edges, n_cross, n_outside = [], [], [], [], []
    for qb in q_bohr:
        q = qb * BOHR_IN_ANGSTROM
        atoms = Atoms(sym, positions=q, cell=cell, pbc=True)
        atoms.calc = calc
        e = float(atoms.get_potential_energy())
        f = atoms.get_forces().copy()
        wrapped = Atoms(sym, positions=q, cell=cell, pbc=True)
        wrapped.wrap()
        wrapped.calc = calc
        e2 = float(wrapped.get_potential_energy())
        f2 = wrapped.get_forces()
        assert abs(e2 - e) <= 1e-12 * abs(e), (e, e2)
        assert np.max(np.abs(f2 - f)) <= 1e-10 * np.max(np.abs(f))
        ei, _, unit, _ = get_neighborhood(q, r_max, pbc=(True,) * 3, cell=cell)
        energies.append(e)
        forces.extend(float(x) for x in f.reshape(-1))
        n_edges.append(int(ei.shape[1]))
        n_cross.append(int(np.sum(np.any(unit != 0, axis=1))))
        n_outside.append(int(np.sum((q < 0) | (q >= cell[0, 0]))))
        print(f"bead: E = {e:.6f} eV, max|F| = {np.max(np.abs(f)):.4f} eV/Å, edges {ei.shape[1]}")
    assert min(n_cross) > 0 and min(n_outside) > 0

    import ase
    import e3nn

    out = {
        "description": "Python MACE (MACECalculator, float64, CPU, periodic, mace's neighbour list) "
        "energies (eV) and forces (eV/Angstrom) of 4 displaced beads of NMA + 216 water",
        "generator": "gen_mace_off23_nma.py",
        "versions": {
            "torch": torch.__version__,
            "mace-torch": mace.__version__,
            "e3nn": e3nn.__version__,
            "ase": ase.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "model": {
            "file": model.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "fnv1a64": f"{fnv1a64(raw):#018x}",
            "r_max_angstrom": r_max,
        },
        "box": BOX,
        "n_atoms": len(sym),
        "n_beads": P,
        "box_angstrom": box["box_angstrom"],
        "bohr_in_angstrom": BOHR_IN_ANGSTROM,
        "seed": SEED,
        "amp_bohr": AMP_BOHR,
        "energies_ev": energies,
        "forces_ev_per_angstrom": forces,
        "n_edges_mace": n_edges,
        "n_boundary_edges_mace": n_cross,
        "n_coordinates_outside_cell": n_outside,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "mace_off23_nma.json"
    path.write_text(json.dumps(out) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
