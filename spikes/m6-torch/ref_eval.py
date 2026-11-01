"""Python MACE reference for the M6a spike (questions 3 and 4).

    pixi run -e ml python spikes/m6-torch/ref_eval.py MODEL GEOM OUT [--dtype float64|float32]
        [--device cpu|cuda] [--edges EDGES_OUT] [--exported EXPORTED.pt --rust-edges FILE]

Evaluates every bead of GEOM (spike text format, make_geoms.py) with mace's own ASE calculator
(`MACECalculator` on the original model file; mace's neighbour list, matscipy) and writes
    E b energy_eV
    F b i fx fy fz         (eV/Å)
to OUT. With --edges, also writes mace's edge list (get_neighborhood) per bead:
    edge b sender receiver sx sy sz ux uy uz      (shift in Å, unit shift integers)
Gas phase: pbc False; cell given: pbc True with that orthorhombic cell, positions as given
(unwrapped).
"""

import argparse

import mace  # noqa: F401  (first: sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, which e3nn 0.4.4 needs)
import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator
from mace.data.neighborhood import get_neighborhood


def read_geom(path):
    with open(path) as f:
        lines = f.read().split("\n")
    n = int(lines[0].split()[1])
    p = int(lines[1].split()[1])
    cell = np.array([float(x) for x in lines[2].split()[1:4]])
    symbols = lines[3].split()[1:]
    pos = np.array([[float(x) for x in ln.split()] for ln in lines[4:4 + n * p]]).reshape(p, n, 3)
    return symbols, pos, cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("geom")
    ap.add_argument("out")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--edges")
    a = ap.parse_args()
    torch.set_num_threads(1)
    symbols, pos, cell = read_geom(a.geom)
    periodic = bool(np.any(cell != 0))
    calc = MACECalculator(model_paths=a.model, device=a.device, default_dtype=a.dtype)
    r_max = float(calc.r_max)
    with open(a.out, "w") as f:
        for b, q in enumerate(pos):
            atoms = Atoms(symbols, positions=q, cell=np.diag(cell) if periodic else None,
                          pbc=periodic)
            atoms.calc = calc
            e = atoms.get_potential_energy()
            fr = atoms.get_forces()
            f.write(f"E {b} {e:.17g}\n")
            for i, x in enumerate(fr):
                f.write(f"F {b} {i} {x[0]:.17g} {x[1]:.17g} {x[2]:.17g}\n")
    if a.edges:
        with open(a.edges, "w") as f:
            for b, q in enumerate(pos):
                ei, shifts, unit, _ = get_neighborhood(
                    q, r_max, pbc=(periodic,) * 3, cell=np.diag(cell) if periodic else None)
                for k in range(ei.shape[1]):
                    s, u = shifts[k], unit[k]
                    f.write(f"edge {b} {ei[0, k]} {ei[1, k]} {s[0]:.17g} {s[1]:.17g} {s[2]:.17g} "
                            f"{int(u[0])} {int(u[1])} {int(u[2])}\n")


if __name__ == "__main__":
    main()
