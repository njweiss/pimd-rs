"""M11 tier 2 pilot step 3: PIMD energies per molecule against H4 (σ from 10 blocks).

    pixi run -e ref python studies/m11-dynamics-failure/statics_t2.py [RUN_DIR ...]
    (default: out/tier2/pilot/statics/*)

H4: E per molecule = E_min + ZPE + 3 k_B T + E_vib(T), with E_vib the thermal vibrational energy
above the ground state from the J = 0 levels (results/tier2/reference.json); translation and rotation
classical. Deviations are given relative to E − E_min (the part above the minimum), since a
percentage of the total (−76 Eh) means nothing. The engine reports energies per atom (×3 here).
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
K_B = 3.1668115634556e-6  # Eh/K, U-2
CM = 219474.6313632


def h4(ref, t):
    beta = 1.0 / (K_B * t)
    e = np.array([lv["E_cm"] for lv in ref["levels"]]) / CM
    w = np.exp(-beta * e)
    e_vib = float((w * e).sum() / w.sum())
    return ref["zpe_eh"] + 3 * K_B * t + e_vib  # above E_min


def main():
    with open(os.path.join(HERE, "results/tier2/reference.json")) as f:
        ref = json.load(f)
    dirs = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "out/tier2/pilot/statics/*")))
    print(f"{'run':16s} {'P':>4s} {'E_vir−E_min':>12s} {'dEvir %':>8s} {'±':>5s} {'dEprim %':>9s} {'±':>5s}  (H4 above minimum, Eh/molecule)")
    for d in dirs:
        f = os.path.join(d, "observables.csv")
        if not os.path.exists(f):
            continue
        txt = open(os.path.join(d, "config.toml")).read()
        t = float(txt.split("temperature_kelvin = ")[1].split("\n")[0])
        p = int(txt.split("n_beads = ")[1].split("\n")[0])
        target = h4(ref, t)
        data = np.genfromtxt(f, delimiter=",", names=True)

        def rel(col):
            x = 3.0 * data[col] - ref["E_min_eh"]
            means = np.array([b.mean() for b in np.array_split(x, 10)])
            return x.mean(), 100 * (x.mean() / target - 1), 100 * means.std(ddof=1) / np.sqrt(10) / target

        ev, ep = rel("e_vir_eh"), rel("e_prim_eh")
        print(f"{os.path.basename(d):16s} {p:4d} {ev[0]:12.6f} {ev[1]:8.3f} {ev[2]:5.3f} {ep[1]:9.3f} {ep[2]:5.3f}  ({target:.6f})")


if __name__ == "__main__":
    main()
