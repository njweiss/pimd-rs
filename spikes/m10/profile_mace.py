"""M10 spike Q2: where the time of one MACE evaluation goes (small, float32, CUDA).

    pixi run -e ml python spikes/m10/profile_mace.py P [--out OUT.json]

Two passes on the same inputs (common.inputs, NMA + 216):
- the TorchScript export as the engine loads it (cast to float32), timed and profiled: total GPU
  kernel time, idle fraction, top kernels;
- the eager model in the same wrapper, with record_function ranges around each top-level stage of
  the forward (embedding, radial, spherical harmonics, interaction i, product i, readout i) and
  around torch.autograd.grad (the force backward), to attribute GPU time to stages.
"""

import argparse
import json
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import common

ap = argparse.ArgumentParser()
ap.add_argument("p", type=int)
ap.add_argument("--out")
a = ap.parse_args()
dt = torch.float32
res = {"p": a.p, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}


def kernel_stats(prof, wall_ms, n_calls):
    """Sum of CUDA kernel time per call, idle fraction and the top 15 kernels (ms per call)."""
    per = defaultdict(float)
    tot = 0.0
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and not e.name.startswith("gpu_user_annotation"):
            us = e.device_time if hasattr(e, "device_time") else e.cuda_time
            per[e.name] += us
            tot += us
    top = sorted(per.items(), key=lambda kv: -kv[1])[:15]
    return {
        "kernel_ms_per_call": tot / 1e3 / n_calls,
        "wall_ms_per_call": wall_ms,
        "gpu_busy_frac": tot / 1e3 / n_calls / wall_ms,
        "top_kernels_ms": [(k[:110], v / 1e3 / n_calls) for k, v in top],
    }


def run_profiled(fn, n=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    import time
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t) * 1e3 / n
    return prof, wall


# Pass 1: the TorchScript export.
mod = torch.jit.load(common.EXPORT, map_location="cuda").to(dt)
z = mod.atomic_numbers().tolist()
r_max = float(mod.r_max())
args = common.inputs(a.p, dt, z, r_max)
res["edges"] = int(args[2].shape[1])
med, mn, _ = common.timed(lambda: mod(*args))
res["script_ms_median"], res["script_ms_min"] = med, mn
prof, wall = run_profiled(lambda: mod(*args))
res["script"] = kernel_stats(prof, wall, 5)
del mod
torch.cuda.empty_cache()

# Pass 2: eager, with stage ranges.
model = common.load_small()
wrapped = common.wrap_eager(model, dt)
m = wrapped.model
stack = []


def pre(name):
    def h(_mod, _inp):
        r = record_function(f"stage::{name}")
        r.__enter__()
        stack.append(r)
    return h


def post(_mod, _inp, _out):
    stack.pop().__exit__(None, None, None)


for name in ["node_embedding", "radial_embedding", "spherical_harmonics", "scale_shift"]:
    sub = getattr(m, name)
    sub.register_forward_pre_hook(pre(name))
    sub.register_forward_hook(post)
for lst in ["interactions", "products", "readouts"]:
    for i, sub in enumerate(getattr(m, lst)):
        sub.register_forward_pre_hook(pre(f"{lst}[{i}]"))
        sub.register_forward_hook(post)
_grad = torch.autograd.grad


def grad(*x, **k):
    with record_function("stage::backward(forces)"):
        return _grad(*x, **k)


torch.autograd.grad = grad
med, mn, _ = common.timed(lambda: wrapped(*args))
res["eager_ms_median"], res["eager_ms_min"] = med, mn
prof, wall = run_profiled(lambda: wrapped(*args))
res["eager"] = kernel_stats(prof, wall, 5)
stages = {}
for ev in prof.key_averages():
    if ev.key.startswith("stage::"):
        us = getattr(ev, "device_time_total", None)
        if us is None:
            us = ev.cuda_time_total
        stages[ev.key[7:]] = {"gpu_ms_per_call": us / 1e3 / 5, "cpu_ms_per_call": ev.cpu_time_total / 1e3 / 5}
res["eager_stages"] = stages
txt = json.dumps(res, indent=1)
print(txt)
if a.out:
    open(a.out, "w").write(txt)
