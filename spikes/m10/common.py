"""Shared inputs for the M10 spike Python probes (docs/experiments/m10-profiling-spike.md).

Builds the P-6 inputs exactly as spikes/m8/probe_mem.py does (P copies of a box + Gaussian
σ = 0.05 Å, numpy seed 8; all ordered pairs of one bead within r_max under the minimum image), and
wraps an eager MACE model (e3nn or cuEquivariance) in the export's PimdMaceV2 class so every probe
calls the same forward as the TorchScript export the engine loads.
"""

import importlib.util
import os
import statistics
import time

import mace  # noqa: F401  (first: sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, which e3nn 0.4.4 needs)
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SMALL = os.path.join(ROOT, "ff_models/mace_off23/MACE-OFF23_small.model")
EXPORT = os.path.join(ROOT, "ff_models/exports/m8b-off23-small-f64.pt")
# The M8b start box with a Lattice= header added (spikes/m10/out/boxes, made in the log, Q1).
BOX = os.path.join(ROOT, "spikes/m10/out/boxes/nma_w216_m8b.xyz")


def export_module():
    spec = importlib.util.spec_from_file_location(
        "export_mace", os.path.join(ROOT, "crates/pimd-torch/scripts/export_mace.py"))
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)
    return em


def read_box(path=BOX):
    lines = open(path).read().splitlines()
    n = int(lines[0])
    lat = lines[1].split('Lattice="')
    L = float(lat[1].split()[0]) if len(lat) > 1 else None
    sym = [ln.split()[0] for ln in lines[2:2 + n]]
    pos = np.array([[float(x) for x in ln.split()[1:4]] for ln in lines[2:2 + n]])
    return sym, pos, L


def inputs(p, dtype, z, r_max, path=BOX, device="cuda"):
    """The forward arguments for P beads of the box (positions Å, edges, shifts, ptr, cell)."""
    from ase.data import atomic_numbers as an
    sym, pos, L = read_box(path)
    n = len(sym)
    rng = np.random.default_rng(8)
    beads = [pos if p == 1 else pos + 0.05 * rng.standard_normal(pos.shape) for _ in range(p)]
    snd, rcv, sh = [], [], []
    for b, q in enumerate(beads):
        d = q[None, :, :] - q[:, None, :]  # d[i, j] = q_j - q_i
        v = d - L * np.round(d / L) if L else d
        r2 = (v ** 2).sum(-1)
        np.fill_diagonal(r2, np.inf)
        i, j = np.nonzero(r2 < r_max ** 2)
        snd.append(i + b * n)
        rcv.append(j + b * n)
        sh.append((v - d)[i, j])
    dev = torch.device(device)
    ei = torch.tensor(np.stack([np.concatenate(snd), np.concatenate(rcv)]), device=dev)
    shifts = torch.tensor(np.concatenate(sh), dtype=dtype, device=dev)
    unit = torch.round(shifts / L) if L else torch.zeros_like(shifts)
    species = torch.tensor([z.index(an[s]) for s in sym] * p, device=dev)
    positions = torch.tensor(np.concatenate(beads), dtype=dtype, device=dev)
    ptr = torch.arange(p + 1, device=dev) * n
    cell = torch.zeros(p, 3, 3, dtype=dtype, device=dev)
    return (positions, species, ei, shifts, ptr, False, unit, cell)


def wrap_eager(model, dtype, device="cuda"):
    """`model` (eager MACE, any backend) inside the export's v2 wrapper, frozen, on `device`."""
    em = export_module()
    e0 = model.atomic_energies_fn.atomic_energies.detach().clone().to(torch.float64).reshape(-1)
    model = model.to(dtype).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    return em.PimdMaceV2(model, dtype, e0).to(device)


def load_small(device="cpu"):
    return torch.load(SMALL, map_location=device, weights_only=False)


def timed(fn, warm=3, n=5):
    """Median and min wall ms of `fn()` (CUDA synchronised) after `warm` calls, and the last result."""
    out = None
    for _ in range(warm):
        out = fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts), min(ts), out
