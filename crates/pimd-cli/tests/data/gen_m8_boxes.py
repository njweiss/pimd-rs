"""Start boxes for the M8a long validation (MILESTONE_8 "M8a: Required tests" 7 and 8), ported from
the M8 spike's box builder (spikes/m8/build_box.py, docs/experiments/m8-spike.md Q4).

    pixi run -e ref python crates/pimd-cli/tests/data/gen_m8_boxes.py [--out DIR]

Writes m8_boxes.json next to this script (or into DIR) with two boxes:

  nma_w216   N-methylacetamide + 216 TIP3P waters (660 atoms), test 7 (pimd-torch
             long_mace_off23_pbc.rs, through gen_mace_off23_nma.py)
  water_w64  64 TIP3P waters (192 atoms), test 8 (pimd-cli long_water_mace_ipi.rs and
             gen_ipi_water_mace.py)

Route (unchanged in substance from the spike): N-methylacetamide is the capped "dipeptide" ACE-NME
(Amber ff14SB has both caps), heavy atoms written here in a trans geometry, hydrogens by
Modeller.addHydrogens. Modeller.addSolvent(numAdded=N_w) gives exactly N_w waters (in its own box);
molecule centres are then scaled affinely into the cube of edge L fixed by the mass density
(0.997 g/cm^3, masses below), and the box is minimised and equilibrated in NVT (ff14SB + TIP3P, PME,
HBonds constraints, LangevinMiddle 300 K, 1/ps, 2 fs, 50 ps). Molecules are written whole (the
centre of each molecule wrapped into [0, L)), solute first (12 atoms: OpenMM's atom order), then
O H H per water.

Changes from the spike, for reproducibility only: Python's `random` is seeded (SEED) before each
box is built (Modeller.addHydrogens draws the initial hydrogen positions from it; unseeded, two
runs of the spike's script gave NMA boxes differing by up to 19 Å after equilibration); OpenMM's
CPU platform with one thread (`Threads` = 1) and deterministic forces (`DeterministicForces` =
true), so the force sums do not depend on scheduling.
Two runs of this script give byte-identical JSON (checked during test authoring, 2026-11-05; see
QUESTIONS.md Q35), so validation/verify_reference_data.py compares it with its usual
rtol 1e-9 / atol 1e-12.

Positions are rounded to 1e-10 Angstrom (as the spike's %.10f), in Angstrom; the box edge L is
rounded likewise and the positions are those of the rounded file (what the tests read).
"""

import argparse
import io
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as u

AVOGADRO = 6.02214076e23
MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}
DENSITY = 0.997  # g/cm^3
EQUIL_PS = 50.0
SEED = 8000
HERE = Path(__file__).resolve().parent

# Trans NMA heavy atoms, Å: C=O 1.23, C-N 1.33, C-CH3 1.52, N-CH3 1.46; O and the N-H anti.
NMA_PDB = """\
ATOM      1  CH3 ACE A   1      -0.666  -1.366   0.000  1.00  0.00           C
ATOM      2  C   ACE A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      3  O   ACE A   1      -0.633   1.054   0.000  1.00  0.00           O
ATOM      4  N   NME A   2       1.330   0.000   0.000  1.00  0.00           N
ATOM      5  C   NME A   2       2.104   1.238   0.000  1.00  0.00           C
END
"""

BOXES = [("nma_w216", 216, True), ("water_w64", 64, False)]


def build(n_waters, solute):
    # Modeller.addHydrogens places the new hydrogens at positions drawn from Python's `random`
    # before minimising them, and addSolvent then depends on them: seeded for reproducibility.
    random.seed(SEED)
    ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    if solute:
        pdb = app.PDBFile(io.StringIO(NMA_PDB))
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)
    else:
        modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model="tip3p", numAdded=n_waters, neutralize=False)

    top = modeller.topology
    pos = np.array(modeller.positions.value_in_unit(u.angstrom))
    elems = [at.element.symbol for at in top.atoms()]
    n_w = sum(1 for r in top.residues() if r.name == "HOH")
    assert n_w == n_waters, (n_w, n_waters)
    mass = sum(MASS[e] for e in elems)
    side = round((mass / AVOGADRO / DENSITY * 1e24) ** (1.0 / 3.0), 10)  # Å

    # Affine scaling of molecule centres into the target cube, molecules kept rigid.
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    centre_old = 0.5 * (lo + hi)
    span = (hi - lo).max() + 1.0
    new = pos.copy()
    for res in top.residues():
        idx = [at.index for at in res.atoms()]
        c = pos[idx].mean(axis=0)
        c_new = (c - centre_old) * (side / span) + 0.5 * side
        new[idx] = pos[idx] - c + c_new
    top.setPeriodicBoxVectors(np.eye(3) * side * 0.1 * u.nanometer)  # nm

    system = ff.createSystem(
        top,
        nonbondedMethod=app.PME,
        nonbondedCutoff=min(0.9, 0.49 * side * 0.1) * u.nanometer,
        constraints=app.HBonds,
    )
    integ = openmm.LangevinMiddleIntegrator(300 * u.kelvin, 1.0 / u.picosecond, 2.0 * u.femtosecond)
    integ.setRandomNumberSeed(SEED)
    plat = openmm.Platform.getPlatformByName("CPU")
    props = {"Threads": "1", "DeterministicForces": "true"}
    sim = app.Simulation(top, system, integ, plat, props)
    sim.context.setPositions(new * 0.1 * u.nanometer)
    sim.minimizeEnergy(maxIterations=2000)
    sim.context.setVelocitiesToTemperature(300 * u.kelvin, SEED)
    sim.step(int(round(EQUIL_PS / 0.002)))
    st = sim.context.getState(getPositions=True, getEnergy=True)
    q = np.array(st.getPositions().value_in_unit(u.angstrom))

    # Whole molecules, centre of each molecule wrapped into [0, L).
    for res in top.residues():
        idx = [at.index for at in res.atoms()]
        first = q[idx[0]]
        d = q[idx] - first
        q[idx] = first + d - side * np.round(d / side)
        c = q[idx].mean(axis=0)
        q[idx] -= side * np.floor(c / side)

    order = [at.index for at in top.atoms() if at.residue.name != "HOH"]
    order += [at.index for at in top.atoms() if at.residue.name == "HOH"]
    return {
        "n_atoms": len(order),
        "n_waters": n_w,
        "solute": "NMA" if solute else "none",
        "box_angstrom": side,
        "symbols": [elems[k] for k in order],
        "masses_amu": [MASS[elems[k]] for k in order],
        "positions_angstrom": [[round(float(x), 10) for x in q[k]] for k in order],
        "openmm_epot_kjmol": round(
            st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole), 3
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()
    t0 = time.time()
    boxes = {name: build(n_w, solute) for name, n_w, solute in BOXES}
    data = {
        "description": "Start boxes for the M8a long validation (ported from spikes/m8/build_box.py): "
        "OpenMM ff14SB NMA (ACE-NME) + TIP3P water, 0.997 g/cm^3, minimised + 50 ps NVT at 300 K; "
        "positions in Angstrom, whole molecules, solute first, then O H H per water",
        "generator": "gen_m8_boxes.py",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "openmm": openmm.__version__,
        },
        "protocol": {
            "density_g_cm3": DENSITY,
            "equil_ps": EQUIL_PS,
            "dt_fs": 2.0,
            "temperature_kelvin": 300.0,
            "friction_per_ps": 1.0,
            "seed": SEED,
            "platform": "CPU, Threads = 1, DeterministicForces = true",
            "forcefield": "amber14/protein.ff14SB.xml, amber14/tip3p.xml, PME, HBonds",
            "masses_amu": MASS,
        },
        "boxes": boxes,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "m8_boxes.json"
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {path} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
