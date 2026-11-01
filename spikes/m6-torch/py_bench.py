"""Spike question 5, cross-check: time the exported TorchScript module (and the eager MACE model)
from Python on the same inputs as the Rust bench, to tell tch overhead from libtorch/driver cost.

    pixi run -e ml python spikes/m6-torch/py_bench.py MODEL_FILE EXPORTED.pt GEOM DEVICE [profiling 0|1]
Prints the median ms per call (all beads batched, includes host->device copies and the copy back).
"""
import sys
import time

import mace  # noqa: F401
import numpy as np
import torch
from ase.data import atomic_numbers as AN

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ref_eval import read_geom  # noqa: E402

model_file, exported, geom, device = sys.argv[1:5]
prof = len(sys.argv) < 6 or sys.argv[5] == "1"
torch.set_num_threads(1)
torch._C._jit_set_profiling_mode(prof)  # = tch::jit::set_profiling_mode; NOT _jit_set_profiling_executor(False): the legacy executor breaks the in-module autograd forces
symbols, pos, cell = read_geom(geom)
P, N, _ = pos.shape
m = torch.jit.load(exported, map_location=device)
z = m.atomic_numbers().tolist()
r = float(m.r_max())
s, rc = [], []
for b in range(P):
    for i in range(N):
        for j in range(N):
            if i != j and np.linalg.norm(pos[b, j] - pos[b, i]) < r:
                s.append(b * N + i); rc.append(b * N + j)
sp = [z.index(AN[x]) for x in symbols] * P


def call(mod):
    dev = torch.device(device)
    args = (torch.tensor(pos.reshape(-1, 3), device=dev), torch.tensor(sp, device=dev),
            torch.tensor([s, rc], device=dev), torch.zeros(len(s), 3, dtype=torch.float64, device=dev),
            torch.tensor([b * N for b in range(P + 1)], device=dev), False,
            torch.zeros(len(s), 3, dtype=torch.float64, device=dev),
            torch.zeros(P, 3, 3, dtype=torch.float64, device=dev))
    e, f, _ = mod(*args)
    return e.cpu(), f.cpu()


def bench(mod, n=10, w=3):
    for _ in range(w):
        call(mod)
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); call(mod); t.append(1e3 * (time.perf_counter() - t0))
    return np.median(t)


print(f"python torchscript {device} P={P} profiling={prof}: {bench(m):.1f} ms")
# eager: same wrapper class, not scripted
from export_mace import PimdMace  # noqa: E402
eager = PimdMace(torch.load(model_file, map_location=device, weights_only=False).double(),
                 torch.float64).to(device)
print(f"python eager       {device} P={P}: {bench(eager):.1f} ms")
