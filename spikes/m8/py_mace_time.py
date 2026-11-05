"""M8 spike Q5: time of one Python MACE evaluation (MACECalculator, ASE, periodic) of a box.

    pixi run -e ml python spikes/m8/py_mace_time.py MODEL BOX.xyz cpu|cuda float64|float32 [threads]

This is what an i-PI socket client with Python MACE does per bead (the M8a comparison): ASE
Atoms with the box, MACECalculator on the original model file (mace's own neighbour list),
energy and forces. Median of 10 calls after 2 warm-up calls; positions jittered by 1e-4 Å per
call so no result is cached.
"""

import json
import sys
import time

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator

model, box, dev, dtype = sys.argv[1:5]
threads = int(sys.argv[5]) if len(sys.argv) > 5 else 1
torch.set_num_threads(threads)
atoms = read(box)
atoms.pbc = True
calc = MACECalculator(model_paths=model, device=dev, default_dtype=dtype)
atoms.calc = calc
rng = np.random.default_rng(1)
p0 = atoms.get_positions().copy()
times = []
for k in range(12):
    atoms.set_positions(p0 + 1e-4 * rng.standard_normal(p0.shape))
    t = time.perf_counter()
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    if dev == "cuda":
        torch.cuda.synchronize()
    if k >= 2:
        times.append((time.perf_counter() - t) * 1e3)
print(json.dumps({"model": model, "box": box, "device": dev, "dtype": dtype, "threads": threads,
                  "n_atoms": len(atoms), "ms_median": float(np.median(times)), "e_ev": float(e)}))
