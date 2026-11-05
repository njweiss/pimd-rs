"""M8 spike Q4: NMA in a cubic box of TIP3P water at a fixed density, built offline with OpenMM.

    pixi run -e ref python spikes/m8/build_box.py --waters 216 --out spikes/m8/out/boxes/nma_w216.xyz
    pixi run -e ref python spikes/m8/build_box.py --waters 64 --no-solute --out .../water_w64.xyz

Route (docs/experiments/m8-spike.md, Q4): N-methylacetamide is the capped "dipeptide" ACE-NME
(Amber ff14SB has both caps), heavy atoms written here in a trans geometry, hydrogens by
Modeller.addHydrogens. Modeller.addSolvent(numAdded=N_w) gives exactly N_w waters (its own box);
molecule centres are then scaled affinely into the cube of edge L fixed by the mass density, and
the box is minimised and equilibrated in NVT (ff14SB + TIP3P, PME, HBonds constraints, Langevin
300 K, 1/ps, 2 fs). Molecules are written whole (not wrapped atom by atom) as extended xyz in Å,
NMA first (12 atoms), then O H H per water.
"""

import argparse
import io
import math
import os

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as u

AVOGADRO = 6.02214076e23
MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}

# Trans NMA heavy atoms, Å: C=O 1.23, C-N 1.33, C-CH3 1.52, N-CH3 1.46; O and the N-H anti.
NMA_PDB = """\
ATOM      1  CH3 ACE A   1      -0.666  -1.366   0.000  1.00  0.00           C
ATOM      2  C   ACE A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      3  O   ACE A   1      -0.633   1.054   0.000  1.00  0.00           O
ATOM      4  N   NME A   2       1.330   0.000   0.000  1.00  0.00           N
ATOM      5  C   NME A   2       2.104   1.238   0.000  1.00  0.00           C
END
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waters", type=int, required=True)
    ap.add_argument("--no-solute", action="store_true", help="pure water (M8a comparison box)")
    ap.add_argument("--density", type=float, default=0.997, help="g/cm^3")
    ap.add_argument("--equil-ps", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=8000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    if a.no_solute:
        # One water as the seed topology, removed again below (addSolvent needs a topology).
        modeller = app.Modeller(app.Topology(), [])
    else:
        pdb = app.PDBFile(io.StringIO(NMA_PDB))
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff)
    modeller.addSolvent(ff, model="tip3p", numAdded=a.waters, neutralize=False)

    top = modeller.topology
    pos = np.array(modeller.positions.value_in_unit(u.angstrom))
    elems = [at.element.symbol for at in top.atoms()]
    n_w = sum(1 for r in top.residues() if r.name == "HOH")
    assert n_w == a.waters, (n_w, a.waters)
    mass = sum(MASS[e] for e in elems)
    side = (mass / AVOGADRO / a.density * 1e24) ** (1.0 / 3.0)  # Å

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
    integ.setRandomNumberSeed(a.seed)
    platform = openmm.Platform.getPlatformByName("CPU")
    sim = app.Simulation(top, system, integ, platform)
    sim.context.setPositions(new * 0.1 * u.nanometer)
    sim.minimizeEnergy(maxIterations=2000)
    sim.context.setVelocitiesToTemperature(300 * u.kelvin, a.seed)
    n_steps = int(round(a.equil_ps / 0.002))
    sim.step(n_steps)
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
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"{len(order)}\n")
        f.write(
            f'Lattice="{side:.10f} 0.0 0.0 0.0 {side:.10f} 0.0 0.0 0.0 {side:.10f}" '
            f'Properties=species:S:1:pos:R:3 pbc="T T T" waters={n_w} '
            f"solute={'none' if a.no_solute else 'NMA'} density={a.density} "
            f"openmm={openmm.__version__} equil_ps={a.equil_ps} seed={a.seed} "
            f"epot_kjmol={st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole):.3f}\n"
        )
        for k in order:
            f.write(f"{elems[k]} {q[k, 0]:.10f} {q[k, 1]:.10f} {q[k, 2]:.10f}\n")
    print(f"{a.out}: {len(order)} atoms, {n_w} waters, L = {side:.4f} A, "
          f"E_pot = {st.getPotentialEnergy()}")


if __name__ == "__main__":
    main()
