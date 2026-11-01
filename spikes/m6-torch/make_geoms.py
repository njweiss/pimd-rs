"""Bead geometries for the M6a spike, in the spike's plain-text format (read by the Rust spike and
ref_eval.py):

    natoms N
    nbeads P
    cell Lx Ly Lz          (Å; 0 0 0 = gas phase)
    symbols S1 ... SN
    x y z                  (P*N lines, Å, bead-major as SPEC S-1)

    pixi run -e ml python spikes/m6-torch/make_geoms.py --outdir spikes/m6-torch/out/geoms

Bead j = base geometry + Gaussian displacement (sigma 0.05 Å, numpy default_rng(seed + P)), so the
beads differ (bead 0 too). Systems:
  water    M5 start geometry (crates/pimd-cli/tests/data/ipi_water_xtb.json, system.positions_angstrom)
  ethanol  ase.build.molecule("CH3CH2OH") (G2 geometry; coordinates are printed into the log)
  waterbox four water molecules in a 10.5 Å cubic box, random orientation and centre
           (default_rng(seed)), then two molecules shifted by +-L along x and y so that positions are
           unwrapped as ring-polymer positions are in pimd-rs (SPEC C-5).
"""

import argparse
import os

import numpy as np

WATER = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
L_BOX = 10.5
SIGMA = 0.05


def ethanol():
    from ase.build import molecule

    a = molecule("CH3CH2OH")
    return a.get_chemical_symbols(), a.get_positions()


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


def write(path, symbols, beads, cell):
    p, n, _ = beads.shape
    with open(path, "w") as f:
        f.write(f"natoms {n}\nnbeads {p}\ncell {cell[0]:.17g} {cell[1]:.17g} {cell[2]:.17g}\n")
        f.write("symbols " + " ".join(symbols) + "\n")
        for x in beads.reshape(-1, 3):
            f.write(f"{x[0]:.17g} {x[1]:.17g} {x[2]:.17g}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    sym_e, pos_e = ethanol()
    systems = {
        "water": (["O", "H", "H"], WATER, (0.0, 0.0, 0.0)),
        "ethanol": (sym_e, pos_e, (0.0, 0.0, 0.0)),
        "waterbox": (*waterbox(a.seed), (L_BOX,) * 3),
    }
    for name, (sym, base, cell) in systems.items():
        print(f"{name}: base geometry (Å)")
        for s, x in zip(sym, base):
            print(f"  {s} {x[0]: .6f} {x[1]: .6f} {x[2]: .6f}")
        for p in (1, 8, 16, 32):
            rng = np.random.default_rng(a.seed + p)
            beads = base[None] + SIGMA * rng.normal(size=(p, len(sym), 3))
            write(os.path.join(a.outdir, f"{name}_p{p}.txt"), sym, beads, cell)


if __name__ == "__main__":
    main()
