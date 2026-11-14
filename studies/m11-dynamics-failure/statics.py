"""M11 pilot step 3: PIMD static averages vs exact (per particle; σ from 10 blocks).

    pixi run -e ref python studies/m11-dynamics-failure/statics.py [OUT_GLOB]
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact  # noqa: E402

pattern = sys.argv[1] if len(sys.argv) > 1 else "studies/m11-dynamics-failure/out/pilot/statics/*"
cache = {}
print(f"{'run':24s} {'dV/V %':>9s} {'±':>6s} {'dEvir/E %':>10s} {'±':>6s} {'dEprim/E %':>11s} {'±':>6s}")
for d in sorted(glob.glob(pattern)):
    f = os.path.join(d, "observables.csv")
    if not os.path.exists(f):
        continue
    name = os.path.basename(d)
    system, t = name.split("_")[0], float(name.split("_")[1][1:])
    if (system, t) not in cache:
        cache[system, t] = exact.reference(system, t, [0.0])
    ref = cache[system, t]
    data = np.genfromtxt(f, delimiter=",", names=True)

    def rel(col, exact_value):
        x = data[col]
        blocks = np.array_split(x, 10)
        means = np.array([b.mean() for b in blocks])
        return 100 * (x.mean() / exact_value - 1), 100 * means.std(ddof=1) / np.sqrt(10) / abs(exact_value)

    v = rel("v_eh", ref["V_per_particle"])
    ev = rel("e_vir_eh", ref["E_per_particle"])
    ep = rel("e_prim_eh", ref["E_per_particle"])
    print(f"{name:24s} {v[0]:9.3f} {v[1]:6.3f} {ev[0]:10.3f} {ev[1]:6.3f} {ep[0]:11.3f} {ep[1]:6.3f}")
