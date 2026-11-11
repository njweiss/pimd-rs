"""float64 re-evaluation of the M8b runs (docs/experiments/m8-nma-water.md, Protocol step 3, Q4).

    pixi run -e ml python studies/m8-nma-water/reeval_f64.py [run ...]   # default: every finished run

For 50 frames of each run (every 20th trajectory frame, 400 fs apart, parts joined as health.py):
every bead through mace's ASE `MACECalculator` on the MACE-OFF23 small model file, float64, CUDA,
periodic (as gen_mace_off23_nma.py). From those energies and forces, per atom: V̄ (E-3) and
E_vir = 3N/(2β) − (1/(2P)) Σ_j Σ_i (q_ji − q̄_i)·F_ji + V̄ (E-2, E-4; centroid of the unwrapped
beads, as the engine), compared with the engine's float32 `observables.csv` row of the same step.
Q4 holds for a run if |mean(float32 − float64)| < σ(⟨·⟩)/5 for V̄ and E_vir, σ(⟨·⟩) the run's
block error (results/summary.json, analyse.py). Constants as pimd_core::units. Writes
results/reeval_f64.json.
"""

import json
import sys

import numpy as np

from health import HERE, frames, box_length, observables, read_xyz_frames

OUT, RES = HERE / "out", HERE / "results"
RUNS = ["p1/p1", "quantum/p8", "quantum/p16", "quantum/p32", "quantum/p32_seed2"]
MODEL = HERE.parents[1] / "ff_models/mace_off23/MACE-OFF23_small.model"
K_B = 3.1668115634556e-6        # Eh/K
BOHR = 0.529177210903           # Å
HARTREE_EV = 27.211386245988
T = 300.0
N_MOL = 217


def reeval(name, calc):
    from ase import Atoms

    run = OUT / name
    L = box_length(run)
    _, sym, _ = next(read_xyz_frames(HERE / "out/start/nma_w216.xyz"))
    obs = observables(run)
    row = {int(s): i for i, s in enumerate(obs["step"])}
    beta = 1.0 / (K_B * T)
    out = []
    for k, (step, beads) in enumerate(frames(run)):
        if k % 20 != 19:
            continue
        p, n = beads.shape[:2]
        e, f = [], []
        for b in beads:
            a = Atoms(sym, positions=b, cell=[L] * 3, pbc=True)
            a.calc = calc
            e.append(a.get_potential_energy() / HARTREE_EV)
            f.append(a.get_forces() / HARTREE_EV * BOHR)       # Eh/bohr
        q = beads / BOHR
        vbar = np.mean(e)
        k_vir = 3 * n / (2 * beta) - np.sum((q - q.mean(0)) * np.array(f)) / (2 * p)
        i = row[step]
        out.append((step, vbar / n, (k_vir + vbar) / n, obs["v_eh"][i], obs["e_vir_eh"][i]))
    a = np.array(out)
    res = {"P": int(p), "frames": len(a), "steps": a[:, 0].astype(int).tolist()}
    summary = json.loads((RES / "summary.json").read_text()).get(name, {})
    for j, key, skey in ((1, "v", "v_eh_per_mol"), (2, "e_vir", "e_vir_eh_per_mol")):
        d = a[:, j + 2] - a[:, j]                                   # float32 − float64, Eh per atom
        sig_run = summary[skey]["s"] * N_MOL / 660 if skey in summary else float("nan")
        res[key] = {"mean_diff_eh_per_atom": float(d.mean()), "sem_diff": float(d.std(ddof=1) / np.sqrt(len(d))),
                    "sd_diff": float(d.std(ddof=1)), "max_abs_diff": float(np.abs(d).max()),
                    "sigma_run_eh_per_atom": float(sig_run), "no_bias": bool(abs(d.mean()) < sig_run / 5)}
    return res


def main():
    from mace.calculators import MACECalculator

    names = sys.argv[1:] or [r for r in RUNS if (OUT / r / "wall.txt").exists()]
    calc = MACECalculator(model_paths=str(MODEL), device="cuda", default_dtype="float64")
    path = RES / "reeval_f64.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        res[name] = reeval(name, calc)
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "steps"}), flush=True)
        path.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
