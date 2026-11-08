"""Predicted BAOAB end-of-step kinetic bias of T_est at P = 1 from MACE forces on saved frames
(docs/experiments/m8-nma-water.md, log 2026-11-08). To O(Δt²) at P = 1,
⟨p_end²/m⟩ = kT − (Δt²/4)⟨|F|²/m⟩ per degree of freedom, so
ΔT = −(Δt²/4) ⟨Σ_i |F_i|²/m_i⟩ / (3 N k_B). Forces: mace's MACECalculator, MACE-OFF23 small,
float64, CUDA, on 50 frames spread over production (pixi `ml` env):

    pixi run -e ml python studies/m8-nma-water/tbias_forces.py studies/m8-nma-water/out/p1/p1

For P > 1 the force form does not hold (the springs enter the configurational identity); see
tbias_harmonic.py. The script prints the P > 1 number of the same formula only for reference.
"""
import sys; sys.path.insert(0,"studies/m8-nma-water")
import numpy as np; from pathlib import Path
from health import read_xyz_frames, box_length
from ase import Atoms
from mace.calculators import MACECalculator
calc = MACECalculator(model_paths="ff_models/mace_off23/MACE-OFF23_small.model", device="cuda", default_dtype="float64")
for run in sys.argv[1:]:
    run = Path(run); L = box_length(run)
    m = np.array(eval([l for l in (run/"config.toml").read_text().splitlines() if l.startswith("masses_amu")][0].split("=")[1]))
    fr = list(read_xyz_frames(run/"trajectory.xyz")); fr = fr[::len(fr)//50][:50]
    vals = []
    for c, sym, x in fr:
        n = int(c.split("atoms=")[1].split()[0])
        for b in x.reshape(-1, n, 3):
            a = Atoms(sym[:n], positions=b, cell=[L]*3, pbc=True); a.calc = calc
            f = a.get_forces(); vals.append(((f**2).sum(1)/m).sum())
    v = np.array(vals); dt = 0.5
    dT = -(dt**2/4)*v*9.648533e-3/8.617333e-5/(3*n)
    print(run.name, len(v), "samples, predicted dT at 0.5 fs:", dT.mean(), "+-", dT.std(ddof=1)/np.sqrt(len(v)))
