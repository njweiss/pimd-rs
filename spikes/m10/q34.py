"""M10 spike Q3/Q4: speed, memory and accuracy of model variants on the same inputs.

    <python> spikes/m10/q34.py VARIANT DTYPE P [--tf32] [--save-script OUT.pt]

VARIANT: script (the v2 TorchScript export, cast as the engine casts it), e3nn (eager e3nn model
in the export wrapper), cueq (the same model converted by mace.cli.convert_e3nn_cueq, in the same
wrapper). DTYPE f32 | f64. One JSON line per run: median/min ms of the full forward with autograd
forces (5 timed after 3 warm-up), peak allocated/reserved GPU memory, and, against the float64
e3nn reference (computed one bead at a time and cached in spikes/m10/out/ref_P<P>.npz), the
largest per-bead energy error (eV) and the force RMS and max-abs errors (eV/Å).
The cueq variant needs spikes/m10/cueq/.venv/bin/python; the others run in the `ml` env too.
"""

import argparse
import json
import os

import numpy as np
import torch

import common

ap = argparse.ArgumentParser()
ap.add_argument("variant", choices=["script", "e3nn", "cueq"])
ap.add_argument("dtype", choices=["f32", "f64"])
ap.add_argument("p", type=int)
ap.add_argument("--tf32", action="store_true")
ap.add_argument("--export", default=common.EXPORT, help="script variant: the TorchScript file to load")
ap.add_argument("--save-script", help="also torch.jit.script the wrapper and save it here")
a = ap.parse_args()
dt = torch.float32 if a.dtype == "f32" else torch.float64
torch.backends.cuda.matmul.allow_tf32 = a.tf32
torch.backends.cudnn.allow_tf32 = a.tf32
OUT = os.path.join(common.ROOT, "spikes/m10/out")
res = {"variant": a.variant, "dtype": a.dtype, "p": a.p, "tf32": a.tf32, "torch": torch.__version__}


def per_bead(mod, args, p):
    """Energies [P] and forces [P·N, 3] from one call per bead (float64 reference path)."""
    pos, species, ei, shifts, ptr, cv, unit, cell = args
    n = int(ptr[1])
    es, fs = [], []
    for b in range(p):
        m = (ei[0] >= b * n) & (ei[0] < (b + 1) * n)
        sl = slice(b * n, (b + 1) * n)
        e, f, _ = mod(pos[sl], species[sl], ei[:, m] - b * n, shifts[m],
                      torch.tensor([0, n], device=pos.device), cv, unit[m], cell[b:b + 1])
        es.append(e.detach().cpu().numpy())
        fs.append(f.detach().cpu().numpy())
    return np.concatenate(es), np.concatenate(fs)


exp = torch.jit.load(common.EXPORT, map_location="cuda")
z = exp.atomic_numbers().tolist()
r_max = float(exp.r_max())
ref_path = os.path.join(OUT, f"ref_P{a.p}.npz")
if not os.path.exists(ref_path):
    ref_mod = common.wrap_eager(common.load_small(), torch.float64)
    e_ref, f_ref = per_bead(ref_mod, common.inputs(a.p, torch.float64, z, r_max), a.p)
    np.savez(ref_path, e=e_ref, f=f_ref)
    del ref_mod
    torch.cuda.empty_cache()
ref = np.load(ref_path)

if a.variant == "script":
    if "cueq" in a.export:  # the op is a Python custom op: register it before loading (Q3)
        import cuequivariance_ops_torch.uniform_1d  # noqa: F401
    mod = torch.jit.load(a.export, map_location="cuda").to(dt)
    res["export"] = a.export
else:
    model = common.load_small()
    if a.variant == "cueq":
        from mace.cli.convert_e3nn_cueq import run as convert
        dflt = torch.get_default_dtype()
        model = convert(model.to(dt), device="cuda", return_model=True)
        torch.set_default_dtype(dflt)
    mod = common.wrap_eager(model, dt)
    if a.save_script:
        from e3nn.util import jit
        try:
            jit.compile(mod).save(a.save_script)
            res["scripted"] = a.save_script
        except Exception as e:  # a result, not a crash
            res["script_error"] = f"{type(e).__name__}: {str(e)[-600:]}"
del exp
args = common.inputs(a.p, dt, z, r_max)
res["edges"] = int(args[2].shape[1])
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
base = torch.cuda.memory_allocated()
try:
    med, mn, (e, f, _) = common.timed(lambda: mod(*args))
except Exception as ex:
    res["error"] = f"{type(ex).__name__}: {str(ex)[:1500]} ... {str(ex)[-300:]}"
    print(json.dumps(res))
    raise SystemExit(0)
res.update(ms_median=med, ms_min=mn, ms_per_bead=med / a.p,
           us_per_edge=med * 1e3 / res["edges"],
           graph_peak_alloc_mib=(torch.cuda.max_memory_allocated() - base) / 2**20,
           peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
e = e.detach().double().cpu().numpy()
f = f.detach().double().cpu().numpy()
df = f - ref["f"]
res.update(e_err_max_ev=float(np.abs(e - ref["e"]).max()),
           f_rms_err=float(np.sqrt((df ** 2).mean())), f_max_err=float(np.abs(df).max()),
           f_rms=float(np.sqrt((ref["f"] ** 2).mean())))
print(json.dumps(res))
