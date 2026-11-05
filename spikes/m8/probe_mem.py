"""M8 spike Q2 cross-check: peak GPU memory and time of one batched MACE evaluation in Python.

    pixi run -e ml python spikes/m8/probe_mem.py EXPORT.pt BOX.xyz P f32|f64 [--eager MODEL]

Loads the same TorchScript export the Rust bench uses (or, with --eager, the original MACE model
file through the same wrapper class without scripting), builds the P-6 edges (all ordered pairs of
one bead within r_max under the minimum image) with numpy, runs forward with autograd forces and
prints torch.cuda.max_memory_allocated / max_memory_reserved and the median time of 5 calls after
2 warm-up calls. Beads: P copies + Gaussian σ = 0.05 Å (numpy seed 8; not the Rust bench's RNG).
"""

import argparse
import importlib.util
import json
import os
import time

import mace  # noqa: F401  (sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD first)
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("export")
ap.add_argument("xyz")
ap.add_argument("p", type=int)
ap.add_argument("dtype", choices=["f32", "f64"])
ap.add_argument("--eager", help="original .model file: run eager instead of TorchScript")
a = ap.parse_args()

lines = open(a.xyz).read().splitlines()
n = int(lines[0])
L = float(lines[1].split('Lattice="')[1].split()[0])
sym = [ln.split()[0] for ln in lines[2:2 + n]]
pos = np.array([[float(x) for x in ln.split()[1:4]] for ln in lines[2:2 + n]])
rng = np.random.default_rng(8)
beads = [pos if a.p == 1 else pos + 0.05 * rng.standard_normal(pos.shape) for _ in range(a.p)]

dev = torch.device("cuda")
dt = torch.float32 if a.dtype == "f32" else torch.float64
mod = torch.jit.load(a.export, map_location=dev)
z = mod.atomic_numbers().tolist()
r_max = float(mod.r_max())
if a.eager:
    spec = importlib.util.spec_from_file_location(
        "export_mace", os.path.join(os.path.dirname(__file__), "../../crates/pimd-torch/scripts/export_mace.py"))
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)
    model = torch.load(a.eager, map_location="cpu", weights_only=False).to(dt).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    mod = em.PimdMace(model, dt).to(dev)
else:
    mod = mod.to(dt)
from ase.data import atomic_numbers as an  # noqa: E402
species = torch.tensor([z.index(an[s]) for s in sym] * a.p, device=dev)

snd, rcv, sh = [], [], []
for b, q in enumerate(beads):
    d = q[None, :, :] - q[:, None, :]  # d[i, j] = q_j - q_i
    v = d - L * np.round(d / L)
    r2 = (v ** 2).sum(-1)
    np.fill_diagonal(r2, np.inf)
    i, j = np.nonzero(r2 < r_max ** 2)
    snd.append(i + b * n)
    rcv.append(j + b * n)
    sh.append((v - d)[i, j])
ei = torch.tensor(np.stack([np.concatenate(snd), np.concatenate(rcv)]), device=dev)
shifts = torch.tensor(np.concatenate(sh), dtype=dt, device=dev)
unit = torch.round(shifts / L)
positions = torch.tensor(np.concatenate(beads), dtype=dt, device=dev)
ptr = torch.arange(a.p + 1, device=dev) * n
cell = torch.zeros(a.p, 3, 3, dtype=dt, device=dev)

torch.cuda.synchronize()
base = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()
times = []
for k in range(7):
    torch.cuda.synchronize()
    t = time.perf_counter()
    e, f, _ = mod(positions, species, ei, shifts, ptr, False, unit, cell)
    e = e.detach().cpu()
    torch.cuda.synchronize()
    if k >= 2:
        times.append((time.perf_counter() - t) * 1e3)
print(json.dumps({
    "export": a.export, "eager": bool(a.eager), "xyz": a.xyz, "p": a.p, "dtype": a.dtype,
    "edges": int(ei.shape[1]), "ms_median": float(np.median(times)),
    "base_alloc_mib": base / 2**20,
    "peak_alloc_mib": torch.cuda.max_memory_allocated() / 2**20,
    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    "e0_ev": float(e[0]), "torch": torch.__version__,
}))
